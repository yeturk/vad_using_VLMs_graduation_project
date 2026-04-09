import dashscope
from dashscope import MultiModalConversation
import os, time, json, base64
import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)


def _normalize_api_key(raw_key: str | None) -> str:
    if not raw_key:
        return ""
    key = raw_key.strip().strip('"').strip("'")
    if key and not key.startswith("sk-"):
        key = f"sk-{key}"
    return key


api_key = _normalize_api_key(os.getenv("DASHSCOPE_API_KEY"))
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY is missing. Check your .env file.")

dashscope.api_key = api_key
dashscope.base_http_api_url = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/api/v1"
).strip()

# --- CONFIG ---
FRAMES_DIR = "/home/yunus/projects/vad_using_one_shot_learning/dataset/IPAD_dataset/S08/testing/frames/12"
LABEL_PATH = "/home/yunus/projects/vad_using_one_shot_learning/dataset/IPAD_dataset/S08/test_label/012.npy"
SAMPLE_STEP = 20       # her 20. frame'i al
SEQUENCE_LEN = 8      # kaç frame'den bir sekans oluşur
SKIP_FIRST_N = 5       # ilk N sampled frame'i atla
MODEL = "qwen-vl-plus"

PROMPT = """You are monitoring a conveyor belt sorting system.

The belt carries red and blue boxes. In normal operation:
- Each box moves continuously along the belt and falls into a bin at the end.
- Red boxes fall into the red bin, blue boxes fall into the blue bin.

There are two anomaly types:
- Sorting Error: a box falls into the wrong colored bin (red into blue bin or vice versa).
- Clogging: a box gets stuck at the edge of a bin and does not fall in — boxes stop moving or pile up.

You are given sequential frames sampled from a video. Look for these specific signs:
- A box entering the wrong bin → Sorting Error
- A box frozen/stuck at the bin edge, or boxes piling up without moving → Clogging

Reply with exactly one of:
- NORMAL
- SORTING ERROR
- CLOGGING"""


def get_sequence_ground_truth(frame_indices: list[int], labels: np.ndarray) -> int:
    """Sekans içindeki herhangi bir frame anomaliyse sekans anomali sayilir."""
    return int(any(labels[i] == 1.0 for i in frame_indices if i < len(labels)))


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def call_api(frame_paths: list[str]) -> tuple[str, float]:
    content = [{"image": _encode_image(p)} for p in frame_paths]
    content.append({"text": PROMPT})

    start = time.perf_counter()
    response = MultiModalConversation.call(
        model=MODEL,
        messages=[{"role": "user", "content": content}]
    )
    elapsed = time.perf_counter() - start

    if response.get("status_code") == 200:
        items = response["output"]["choices"][0]["message"]["content"]
        text = next((item["text"] for item in items if "text" in item), "")
    else:
        text = f"ERROR: {response.get('code')} — {response.get('message')}"

    return text, elapsed


def main():
    labels = np.load(LABEL_PATH)
    all_frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg"))
    sampled_frames = all_frames[::SAMPLE_STEP][SKIP_FIRST_N:]

    print(f"Total frames: {len(all_frames)}")
    print(f"Sampled (every {SAMPLE_STEP}th, skipping first {SKIP_FIRST_N}): {len(sampled_frames)}")
    print(f"Sequences of {SEQUENCE_LEN} (sliding window): {len(sampled_frames) - SEQUENCE_LEN + 1}")
    print(f"Model: {MODEL}")
    print("=" * 80)

    correct = 0
    total = 0

    for seq_idx in range(len(sampled_frames) - SEQUENCE_LEN + 1):
        seq_frames = sampled_frames[seq_idx:seq_idx + SEQUENCE_LEN]
        frame_indices = [int(f.replace(".jpg", "")) for f in seq_frames]
        frame_paths = [os.path.join(FRAMES_DIR, f) for f in seq_frames]

        gt = get_sequence_ground_truth(frame_indices, labels)
        gt_str = "ANOMALY" if gt else "NORMAL"

        print(f"\nSequence {seq_idx + 1:02d} | frames {frame_indices[0]:03d}–{frame_indices[-1]:03d} | GT: {gt_str}")
        print(f"  Sending {len(frame_paths)} frames to API...")

        response_text, elapsed = call_api(frame_paths)

        prediction = response_text.strip().split("\n")[0].upper()
        predicted_anomaly = "NORMAL" not in prediction
        match = predicted_anomaly == bool(gt)
        correct += match
        total += 1

        print(f"  Inference time: {elapsed:.2f}s")
        print(f"  Prediction: {prediction} | Match: {'✓' if match else '✗'}")

    print("\n" + "=" * 80)
    print(f"RESULTS: {correct}/{total} correct ({100 * correct / total:.1f}%)")


if __name__ == "__main__":
    main()
