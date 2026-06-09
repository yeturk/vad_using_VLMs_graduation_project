"""Dense-frame probe for the intra-object temporal anomaly.

Instead of letting the model sample the video internally, we extract N frames in
time order and send them as a labeled image sequence, asking the model to track
each pen RELATIVE to its neighbors (board translation is invariant to this).
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vera_lite.dashscope_client import call_qwen, extract_json_object


def extract_frames(video: str, n: int, start_frac: float, end_frac: float) -> list[str]:
    """Return n evenly spaced frames as JPEG base64 data URIs, in time order.

    Sampling is restricted to the central [start_frac, end_frac] window so that the
    board's entry/exit (pens leaving the frame) is excluded - those edges otherwise
    look like relative pen movement and cause false positives.
    """
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo = int(total * start_frac)
    hi = int(total * end_frac)
    indices = [int(round(lo + i * (hi - lo) / (n - 1))) for i in range(n)]
    uris: list[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            uris.append("data:image/jpeg;base64," + base64.b64encode(buf).decode())
    cap.release()
    return uris


def build_prompt(n: int) -> str:
    return f"""
You are an industrial quality-control vision system.

A white board slides LEFT to RIGHT carrying three pens in a row. Because the whole
board translates, all three pens move together across the frame from left to right
- that is normal and expected.

You are given {n} frames sampled in time order from one short clip
(frame 1 = earliest, frame {n} = latest).

Your task: decide whether the three pens stay RIGID as a group on the board, or
whether any pen moves ON ITS OWN relative to the others / to the board.

Focus on RELATIVE cues that are independent of the board's overall left-to-right travel:
- Spacing: does the gap between adjacent pens stay constant across the frames, or
  does the spacing between any two pens visibly change (one pen drifting closer to
  or farther from a neighbor)?
- Arrangement: do all three pens keep the same relative positions, or does one pen
  shift, slide, rotate, or jump relative to the other two?
- A pen that moves differently from the group (not merely carried along by the
  board) is an ANOMALY of type unexpected_pen_movement.

IMPORTANT - do NOT mistake these normal effects for relative movement:
- The whole group translating left to right (the board carrying all pens together).
- Pens near the left or right edge being partially cut off, entering, or leaving
  the frame because the board moves them off-screen. Pens disappearing at the edge
  as the board exits is NORMAL, not relative movement.
Only judge the pens while they are fully on the visible board, and only flag a
genuine change in the spacing/arrangement AMONG the pens themselves.

Compare the frames step by step. Then output strictly as valid JSON:
{{
  "frame_observations": "how the pen spacing / relative arrangement evolves across the frames",
  "relative_movement_detected": true,
  "verdict": "NORMAL / ANOMALY",
  "anomaly_score": 0.0,
  "confidence": "LOW / MEDIUM / HIGH",
  "reason": "one or two sentences"
}}

Return JSON only. Do not include markdown.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/pens_small/pens_temporal_anomaly.mp4")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--start-frac", type=float, default=0.15)
    parser.add_argument("--end-frac", type=float, default=0.70)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    uris = extract_frames(args.video, args.frames, args.start_frac, args.end_frac)
    print(f"Video: {args.video}")
    print(f"Frames sent: {len(uris)} | Model: {args.model}\n")

    content = [{"image": u} for u in uris]
    content.append({"text": build_prompt(len(uris))})

    text, raw = call_qwen(model=args.model, content=content)
    print("=" * 80)
    print(text)
    print("=" * 80)
    try:
        parsed = extract_json_object(text)
        print(f"\nverdict={parsed.get('verdict')} | score={parsed.get('anomaly_score')} "
              f"| relative_movement={parsed.get('relative_movement_detected')} "
              f"| confidence={parsed.get('confidence')}")
    except Exception as e:
        print(f"\n[warn] could not parse JSON: {e}")

    usage = raw.get("usage", {})
    print(f"\nTokens: in={usage.get('input_tokens')} out={usage.get('output_tokens')} "
          f"image={usage.get('image_tokens', 'N/A')}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
