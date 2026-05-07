import argparse
import json
from pathlib import Path
from typing import Any

from .dashscope_client import call_qwen, extract_json_object
from .prompts import build_optimizer_prompt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "vera_lite" / "guiding_questions.json"


def load_questions(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_optimizer(
    results_path: Path,
    model: str,
    questions_path: Path = DEFAULT_QUESTIONS,
) -> dict[str, Any]:
    question_data = load_questions(questions_path)
    learner_results = json.loads(results_path.read_text(encoding="utf-8"))
    prompt = build_optimizer_prompt(question_data["questions"], learner_results)
    text, raw_response = call_qwen(model=model, content=[{"text": prompt}])
    parsed = extract_json_object(text)
    return {
        "model": model,
        "previous_questions": question_data,
        "raw_text": text,
        "parsed": parsed,
        "usage": raw_response.get("usage", {}),
        "request_id": raw_response.get("request_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = run_optimizer(args.results, args.model, args.questions)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
