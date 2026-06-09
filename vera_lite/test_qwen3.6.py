import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vera_lite.dashscope_client import call_qwen


DEFAULT_VIDEO = "data/pens_small/pens_normal.mp4"
DEFAULT_MODEL = "qwen3.6-plus"


DESCRIBE_PROMPT = """
You are inspecting a short video from a conveyor-style production line.

Do NOT classify the video as normal or anomalous.
Do NOT assume what the objects "should" look like.
Your only task is to describe, objectively, what you visually observe.

Please answer:
1. What is the background / surface? What color is it?
2. What is the moving carrier (belt / board) and which direction does it move
   in the camera frame (left-to-right, right-to-left, etc.)?
3. How many distinct objects are carried on it? Count them.
4. What are these objects? Describe their shape, color, and parts.
5. For EACH object, which way does its tip / pointed end face
   (up, down, left, right, diagonal)? Are they all the same or different?
6. Does each object appear complete, or is any part (e.g. a cap) missing
   on any of them?
7. Are all objects the same color, or does any one differ?
8. How does the motion evolve over time? Is it continuous and smooth, or does
   it pause / stop / reverse at any point?
9. What details are uncertain due to blur, resolution, perspective, or occlusion?

Output format:

DETAILED SCENE DESCRIPTION:
[paragraph]

OBJECTS:
- Count:
- Description (shape/color/parts):
- Per-object tip direction:
- Any missing parts:
- Color consistency:

MOTION:
- Direction:
- Continuity (smooth / pauses / stops / reverses), with rough timing:

UNCERTAINTIES:
[list]
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask Qwen3.6 to describe a conveyor video without classifying it."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print("\nCalling model...\n")

    text, raw_response = call_qwen(
        model=args.model,
        content=[
            {"video": args.video},
            {"text": DESCRIBE_PROMPT},
        ],
    )

    print("=" * 80)
    print("MODEL DESCRIPTION")
    print("=" * 80)
    print(text)

    usage = raw_response.get("usage", {})
    if usage:
        print("\n" + "-" * 80)
        print("USAGE")
        print("-" * 80)
        print(f"Input tokens:  {usage.get('input_tokens')}")
        print(f"Output tokens: {usage.get('output_tokens')}")
        print(f"Total tokens:  {usage.get('total_tokens')}")
        print(f"Video tokens:  {usage.get('video_tokens')}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
