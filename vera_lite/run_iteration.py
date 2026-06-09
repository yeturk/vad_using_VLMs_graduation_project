import argparse
import json
from datetime import datetime
from pathlib import Path

from .run_learner import run_learner
from .run_optimizer import run_optimizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "vera_lite" / "dataset_manifest.json"
DEFAULT_QUESTIONS = ROOT / "vera_lite" / "guiding_questions.json"


def compute_metrics(learner_results: list[dict]) -> dict:
    """Binary NORMAL/ANOMALY metrics, treating ANOMALY as the positive class."""
    tp = tn = fp = fn = 0
    for r in learner_results:
        expected = r["expected"]
        predicted = r["parsed"].get("verdict")
        if expected == "ANOMALY" and predicted == "ANOMALY":
            tp += 1
        elif expected == "NORMAL" and predicted == "NORMAL":
            tn += 1
        elif expected == "NORMAL" and predicted == "ANOMALY":
            fp += 1
        elif expected == "ANOMALY" and predicted == "NORMAL":
            fn += 1
        # predicted == "UNCLEAR" (or anything else) counts as an error below.

    total = len(learner_results)
    correct = tp + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--optimizer-model", default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "vera_lite" / "runs")
    parser.add_argument("--skip-optimizer", action="store_true")
    parser.add_argument(
        "--reveal-label",
        action="store_true",
        help="Leak the expected label into the learner prompt. NOT recommended; "
        "for ablation only. Default is blind: the learner never sees the label.",
    )
    parser.add_argument("--frames", type=int, default=0,
        help="If >0, feed the learner this many time-ordered frames instead of the video.")
    parser.add_argument("--start-frac", type=float, default=0.1)
    parser.add_argument("--end-frac", type=float, default=0.9)
    parser.add_argument("--fps", type=float, default=0.0,
        help="Video mode only: DashScope frame sampling rate.")
    parser.add_argument("--thinking-budget", type=int, default=None,
        help="Cap reasoning tokens so the model leaves room to emit the final JSON.")
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    learner_results = []
    for item in manifest:
        # Blind by default: the learner does not see the ground-truth label.
        prompt_expected = item["expected"] if args.reveal_label else None
        result = run_learner(
            video=item["video"],
            expected=prompt_expected,
            model=args.model,
            questions_path=args.questions,
            frames=args.frames,
            start_frac=args.start_frac,
            end_frac=args.end_frac,
            fps=args.fps,
            thinking_budget=args.thinking_budget,
            max_tokens=args.max_tokens,
        )
        result["id"] = item["id"]
        # Restore ground truth for evaluation/optimizer regardless of blind mode.
        result["expected"] = item["expected"]
        result["label_revealed_to_learner"] = args.reveal_label
        result["anomaly_type"] = item.get("anomaly_type")
        result["description"] = item.get("description")
        predicted = result["parsed"].get("verdict")
        result["correct"] = predicted == item["expected"]
        learner_results.append(result)

        item_path = run_dir / f"{item['id']}_learner.json"
        item_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        mark = "OK " if result["correct"] else "XX "
        print(f"{mark}{item['id']}: expected={item['expected']} predicted={predicted}")

    results_path = run_dir / "learner_results.json"
    results_path.write_text(json.dumps(learner_results, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = compute_metrics(learner_results)
    print(
        f"\nAccuracy: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.2f}  "
        f"(TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']})"
    )

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
        "label_revealed_to_learner": args.reveal_label,
        "frames": args.frames,
        "frame_window": [args.start_frac, args.end_frac] if args.frames else None,
        "manifest": str(args.manifest),
        "questions": str(args.questions),
        "learner_results_path": str(results_path),
        "optimizer_result_written": optimizer_result is not None,
        "metrics": metrics,
        "per_clip": [
            {"id": r["id"], "expected": r["expected"],
             "predicted": r["parsed"].get("verdict"), "correct": r["correct"]}
            for r in learner_results
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
