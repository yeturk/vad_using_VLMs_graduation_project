from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from vera_lite.dashscope_client import call_qwen, extract_json_object

from .prompts import build_learner_prompt, get_guiding_questions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "qwen3.6-plus"
DEFAULT_RUNS_DIR = ROOT / "three_pen_vad" / "runs"


def run_learner(
    video: str,
    expected: str | None,
    model: str = DEFAULT_MODEL,
    prompt_version: str = "v1",
) -> dict[str, Any]:
    prompt = build_learner_prompt(expected=expected, prompt_version=prompt_version)
    questions = get_guiding_questions(prompt_version)
    start_time = time.perf_counter()
    text, raw_response = call_qwen(
        model=model,
        content=[
            {"video": video},
            {"text": prompt},
        ],
    )
    elapsed_seconds = time.perf_counter() - start_time
    parsed = extract_json_object(text)
    return {
        "video": video,
        "expected": expected,
        "model": model,
        "prompt_version": prompt_version,
        "questions": questions,
        "elapsed_seconds": elapsed_seconds,
        "raw_text": text,
        "parsed": parsed,
        "usage": raw_response.get("usage", {}),
        "request_id": raw_response.get("request_id"),
    }


def build_run_summary(result: dict[str, Any]) -> dict[str, Any]:
    usage = result.get("usage", {})
    parsed = result.get("parsed", {})
    return {
        "elapsed_seconds": round(float(result.get("elapsed_seconds", 0.0)), 2),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "video_tokens": usage.get("video_tokens"),
        "verdict": parsed.get("verdict") or parsed.get("primary_verdict"),
        "confidence": parsed.get("confidence"),
        "anomaly_score": parsed.get("anomaly_score"),
    }


def default_output_path(video: Path, runs_dir: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return runs_dir / run_id / f"{video.stem}_learner.json"


def print_summary(summary: dict[str, Any]) -> None:
    print("RUN SUMMARY")
    print(f"Elapsed seconds: {summary['elapsed_seconds']}")
    print(f"Input tokens:    {summary['input_tokens']}")
    print(f"Output tokens:   {summary['output_tokens']}")
    print(f"Total tokens:    {summary['total_tokens']}")
    print(f"Video tokens:    {summary['video_tokens']}")
    print(f"Verdict:         {summary['verdict']}")
    print(f"Confidence:      {summary['confidence']}")
    print(f"Anomaly score:   {summary['anomaly_score']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the three-pen anomaly learner on a conveyor video."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--expected",
        choices=["NORMAL", "MISSING_CAP", "WRONG_ORIENTATION", "COLOR_ANOMALY", "TEMPORAL_STUCK"],
        default=None,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()

    output_path = args.out or default_output_path(Path(args.video), args.runs_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Video: {args.video}")
    print(f"Expected: {args.expected}")
    print(f"Model: {args.model}")
    print(f"Prompt version: {args.prompt_version}")
    print("Calling model...")

    result = run_learner(
        video=args.video,
        expected=args.expected,
        model=args.model,
        prompt_version=args.prompt_version,
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(build_run_summary(result))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
