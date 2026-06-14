"""
Multi-pass voting inference for conveyor belt QC.

Now supports both persona styles via a CLI flag:
    --personas specialized   (each voter has a single focus area; default)
    --personas complete      (every voter checks the full A/B/C checklist)

Usage:
    python vote_qwen.py data/clip.mp4
    python vote_qwen.py data/clip.mp4 --personas complete
    python vote_qwen.py data/clip.mp4 --personas specialized
"""

import argparse
import atexit
import os
import re
import sys
import tempfile
import time
from collections import Counter
from urllib.parse import urlparse

import requests
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("video", nargs="?",
                    default="../data/r01_clip06_angle_anomaly.mp4",
                    help="Path or URL to the video clip")
parser.add_argument("--personas", choices=["specialized", "complete"],
                    default="specialized",
                    help="Which persona set to use (default: specialized)")
parser.add_argument("--no-consensus", action="store_true",
                    help="Skip the consensus prompt (5 voters instead of 6)")
parser.add_argument("--report", default=None,
                    help="Output JSON path (default: vote_report_<personas>.json)")
args = parser.parse_args()

VIDEO_SOURCE = args.video
PERSONA_MODE = args.personas
USE_CONSENSUS_AS_6TH = not args.no_consensus
REPORT_PATH = args.report or f"vote_report_{PERSONA_MODE}.json"

# Import the chosen persona module
if PERSONA_MODE == "specialized":
    from personas_specialized import VOTING_SUBSET, CONSENSUS_PROMPT
else:
    from personas_complete import VOTING_SUBSET, CONSENSUS_PROMPT


# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = os.getenv("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
DOWNLOADED_VIDEO_PATHS = []


# =============================================================================
# 1) LOAD MODEL (once)
# =============================================================================
print(f"Video source: {VIDEO_SOURCE}")
print(f"Model:        {MODEL_NAME}")
print(f"Persona mode: {PERSONA_MODE}")
print(f"CUDA:         {torch.cuda.is_available()}")
print(f"Script path:  {os.path.abspath(__file__)}")
print(f"Working dir:  {os.getcwd()}")
print("\nLoading model...")
t0 = time.time()

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
)
model.eval()
print(f"Model loaded in {time.time() - t0:.1f}s\n")


def prepare_video_source(video_source: str) -> str:
    parsed = urlparse(video_source)
    if parsed.scheme not in ("http", "https"):
        return video_source

    suffix = os.path.splitext(parsed.path)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()

    try:
        with requests.get(video_source, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(tmp.name, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except Exception:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        raise

    DOWNLOADED_VIDEO_PATHS.append(tmp.name)
    return tmp.name


def cleanup_temp_videos() -> None:
    for path in DOWNLOADED_VIDEO_PATHS:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


atexit.register(cleanup_temp_videos)
VIDEO_SOURCE_FOR_MODEL = prepare_video_source(VIDEO_SOURCE)
if VIDEO_SOURCE_FOR_MODEL != VIDEO_SOURCE:
    print(f"Downloaded remote video to: {VIDEO_SOURCE_FOR_MODEL}\n")


# =============================================================================
# 2) INFERENCE FOR A SINGLE PERSONA
# =============================================================================
def run_inference(video_source: str, prompt_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_source,
                    "max_pixels": 360 * 420,
                    "fps": 2.0,
                    "max_frames": 32,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if video_inputs is not None:
        video_tensors, video_metadatas = zip(*video_inputs)
        video_inputs = list(video_tensors)
        video_metadatas = list(video_metadatas)
    else:
        video_metadatas = None

    for k, v in list(video_kwargs.items()):
        if isinstance(v, list) and len(v) == 1:
            video_kwargs[k] = v[0]

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        padding=True,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    generated_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


# =============================================================================
# 3) PARSE VERDICT
# =============================================================================
def parse_verdict(response_text: str) -> str:
    """Returns 'ANOMALY', 'NORMAL', or 'UNCLEAR'."""
    text = response_text.upper()
    tail = "\n".join(
        [line for line in text.splitlines() if line.strip()][-15:]
    )

    # NORMAL patterns first (so "NO ANOMALY DETECTED" is not misread)
    normal_patterns = [
        r"\bNO\s+ANOMALY\s+DETECTED\b",
        r"\bRESULT\s*:\s*NORMAL\b",
        r"\bDECISION\s*:\s*CONTINUE\b",
        r"\bPART\s+STATUS\s*:\s*ACCEPT\b",
        r"\bROBOTIC\s+FITNESS\s*:\s*PICK-READY\b",
        r"\bANDON\s+CALL\s*:\s*NONE\b",
        r"\bMUDA\s+TYPE\s*:\s*NONE\b",
        r"\bDEFECT\s+TYPE\s*:\s*IN-SPEC\b",
        r"\bDOWNSTREAM\s+IMPACT\s*:\s*ABSENT\b",
    ]
    for pat in normal_patterns:
        if re.search(pat, tail, flags=re.DOTALL):
            return "NORMAL"

    # ANOMALY patterns - negative lookbehind blocks "NO ANOMALY"
    anomaly_patterns = [
        r"(?<!NO\s)\bANOMALY\s+DETECTED\b",
        r"\bDECISION\s*:\s*STOP\s+LINE\b",
        r"\bPART\s+STATUS\s*:\s*REJECT\b",
        r"\bROBOTIC\s+FITNESS\s*:\s*PICK-FAIL\b",
        r"\bANDON\s+CALL\s*:\s*(YELLOW|RED)\b",
        r"\bMUDA\s+TYPE\s*:\s*(DEFECT|WAITING|MOTION)\b",
        r"\bDOWNSTREAM\s+IMPACT\s*:\s*PRESENT\b",
    ]
    for pat in anomaly_patterns:
        if re.search(pat, tail, flags=re.DOTALL):
            return "ANOMALY"

    if re.search(
        r"\b(INSUFFICIENT(\s+OBSERVATION)?|INDETERMINATE|DATA\s+INSUFFICIENT)\b",
        tail,
    ):
        return "UNCLEAR"
    return "UNCLEAR"


# =============================================================================
# 4) MAJORITY VOTE
# =============================================================================
def majority_vote(verdicts: list[str]) -> tuple[str, float, dict]:
    counts = Counter(verdicts)
    total = len(verdicts)
    decisive = [v for v in verdicts if v != "UNCLEAR"]
    if not decisive:
        return "UNCLEAR", 0.0, dict(counts)
    decisive_counts = Counter(decisive)
    final, top_count = decisive_counts.most_common(1)[0]
    return final, top_count / total, dict(counts)


def confidence_label(pct: float) -> str:
    if pct >= 0.8:
        return "HIGH"
    if pct >= 0.6:
        return "MEDIUM"
    return "LOW"


# =============================================================================
# 5) MAIN LOOP
# =============================================================================
voters = list(VOTING_SUBSET)
if USE_CONSENSUS_AS_6TH:
    voters.append({
        "id": "CONS",
        "role": "Consensus prompt (distilled from 20 personas)",
        "focus": "Synthesis",
        "prompt": CONSENSUS_PROMPT,
    })

print("=" * 80)
print(f"MULTI-PASS VOTING ({len(voters)} voters, mode={PERSONA_MODE})")
print("=" * 80)

verdicts = []
all_responses = []

for i, voter in enumerate(voters, 1):
    print(f"\n[{i}/{len(voters)}] Voter: {voter['id']} - {voter['role']}")
    print(f"    Focus: {voter['focus']}")
    print(f"    Running inference...", end=" ", flush=True)

    t0 = time.time()
    try:
        response = run_inference(VIDEO_SOURCE_FOR_MODEL, voter["prompt"])
        elapsed = time.time() - t0
        verdict = parse_verdict(response)
        print(f"done ({elapsed:.1f}s) -> {verdict}")
    except Exception as e:
        print(f"FAILED: {e}")
        response = f"[ERROR: {e}]"
        verdict = "UNCLEAR"

    verdicts.append(verdict)
    all_responses.append({
        "voter_id": voter["id"],
        "role": voter["role"],
        "verdict": verdict,
        "response": response,
    })


# =============================================================================
# 6) FINAL DECISION
# =============================================================================
final, conf_pct, breakdown = majority_vote(verdicts)
conf_label = confidence_label(conf_pct)

print("\n" + "=" * 80)
print(f"FINAL DECISION (mode={PERSONA_MODE})")
print("=" * 80)
print(f"Verdict       : {final}")
print(f"Confidence    : {conf_label} ({conf_pct:.0%})")
print(f"Vote breakdown: {breakdown}")
print(f"Voter details :")
for r in all_responses:
    print(f"  - {r['voter_id']:5s} ({r['role'][:40]:40s}) -> {r['verdict']}")


# =============================================================================
# 7) FULL RESPONSES (debug)
# =============================================================================
DUMP_FULL_RESPONSES = True
if DUMP_FULL_RESPONSES:
    print("\n" + "=" * 80)
    print("FULL RESPONSES")
    print("=" * 80)
    for r in all_responses:
        print(f"\n--- {r['voter_id']} ({r['role']}) -> {r['verdict']} ---")
        print(r["response"])


# =============================================================================
# 8) JSON REPORT
# =============================================================================
import json
report = {
    "video": VIDEO_SOURCE,
    "model": MODEL_NAME,
    "persona_mode": PERSONA_MODE,
    "final_verdict": final,
    "confidence_pct": conf_pct,
    "confidence_label": conf_label,
    "vote_breakdown": breakdown,
    "voters": all_responses,
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\nReport saved: {REPORT_PATH}")
