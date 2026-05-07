import argparse
import json
from datetime import datetime
from pathlib import Path

from .run_learner import run_learner
from .run_optimizer import run_optimizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "vera_lite" / "dataset_manifest.json"
DEFAULT_QUESTIONS = ROOT / "vera_lite" / "guiding_questions.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--optimizer-model", default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "vera_lite" / "runs")
    parser.add_argument("--skip-optimizer", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    learner_results = []
    for item in manifest:
        result = run_learner(
            video=item["video"],
            expected=item["expected"],
            model=args.model,
            questions_path=args.questions,
        )
        result["id"] = item["id"]
        result["anomaly_type"] = item.get("anomaly_type")
        result["description"] = item.get("description")
        learner_results.append(result)

        item_path = run_dir / f"{item['id']}_learner.json"
        item_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{item['id']}: expected={item['expected']} predicted={result['parsed'].get('verdict')}")

    results_path = run_dir / "learner_results.json"
    results_path.write_text(json.dumps(learner_results, ensure_ascii=False, indent=2), encoding="utf-8")

    optimizer_result = None
    if not args.skip_optimizer:
        optimizer_result = run_optimizer(
            results_path=results_path,
            model=args.optimizer_model or args.model,
            questions_path=args.questions,
        )
        (run_dir / "optimizer_result.json").write_text(
            json.dumps(optimizer_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        new_questions = {
            "version": "candidate",
            "source": f"optimizer_{args.optimizer_model or args.model}_{run_id}",
            "questions": optimizer_result["parsed"]["new_questions"],
            "reasoning": optimizer_result["parsed"].get("reasoning"),
        }
        (run_dir / "candidate_guiding_questions.json").write_text(
            json.dumps(new_questions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "run_id": run_id,
        "model": args.model,
        "optimizer_model": args.optimizer_model or args.model,
        "manifest": str(args.manifest),
        "questions": str(args.questions),
        "learner_results_path": str(results_path),
        "optimizer_result_written": optimizer_result is not None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
