from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from vera_lite.dashscope_client import call_qwen

from .prompts import DESCRIPTION_PROMPT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "qwen3.6-plus"
DEFAULT_RUNS_DIR = ROOT / "three_pen_vad" / "runs"


def describe_video(video: str, model: str = DEFAULT_MODEL) -> str:
    text, _raw_response = call_qwen(
        model=model,
        content=[
            {"video": video},
            {"text": DESCRIPTION_PROMPT},
        ],
    )
    return text


def default_output_path(video: Path, runs_dir: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return runs_dir / run_id / f"{video.stem}_description.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask Qwen to describe a three-pen conveyor video without classification."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()

    output_path = args.out or default_output_path(Path(args.video), args.runs_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print("Calling model...")
    text = describe_video(video=args.video, model=args.model)
    output_path.write_text(text, encoding="utf-8")

    print(text)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

