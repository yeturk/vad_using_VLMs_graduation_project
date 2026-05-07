import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vera_lite.dashscope_client import call_qwen


DEFAULT_VIDEO = "data/r01_clip09_normal.mp4"
DEFAULT_MODEL = "qwen3.6-plus"


DESCRIBE_PROMPT = """
You are inspecting a short video from an industrial conveyor belt.

Do NOT classify the video as normal or anomalous yet.
Do NOT use any predefined rule from previous prompts.

Your task is only to describe what you visually observe in detail.

Please focus on:
1. What objects are visible in the scene?
2. What is the conveyor belt direction in the camera frame?
3. What is the moving component?
4. What colors and parts are visible on the component?
5. Which side of the component has the red cap?
6. Which side of the component has the black pin side / metal pins?
7. How does the component move over time?
8. Does the component appear tilted, diagonal, upright, or aligned?
9. Does it stay on the belt?
10. What details are uncertain because of blur, resolution, perspective, or occlusion?

Output format:

DETAILED SCENE DESCRIPTION:
[paragraph]

FRAME/TIME OBSERVATIONS:
- Early:
- Middle:
- Late:

COMPONENT PARTS:
- Red cap location:
- Pin side location:
- Body posture/orientation:

MOTION:
[describe direction and smoothness]

UNCERTAINTIES:
[list any visual uncertainties]
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
