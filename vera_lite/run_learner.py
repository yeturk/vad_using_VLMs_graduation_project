import argparse
import json
from pathlib import Path
from typing import Any

from .dashscope_client import call_qwen, extract_json_object
from .prompts import build_learner_prompt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "vera_lite" / "guiding_questions.json"


def load_questions(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]


def run_learner(
    video: str,
    expected: str | None,
    model: str,
    questions_path: Path = DEFAULT_QUESTIONS,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    prompt = build_learner_prompt(questions, expected)
    text, raw_response = call_qwen(
        model=model,
        content=[
            {"video": video},
            {"text": prompt},
        ],
    )
    parsed = extract_json_object(text)
    return {
        "video": video,
        "expected": expected,
        "model": model,
        "questions": questions,
        "raw_text": text,
        "parsed": parsed,
        "usage": raw_response.get("usage", {}),
        "request_id": raw_response.get("request_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--expected", choices=["NORMAL", "ANOMALY"], default=None)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = run_learner(args.video, args.expected, args.model, args.questions)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
