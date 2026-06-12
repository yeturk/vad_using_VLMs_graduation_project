"""Combine multiple learner runs into an OR-ensemble.

A clip is ANOMALY if ANY run flags it ANOMALY. This pays off when the individual
runs each have 0 false positives but disjoint false negatives (e.g. video mode
catches edge orientation / motion stop, dense-frame mode catches intra-object
movement): the union recovers all anomalies without adding false positives.
"""
import argparse
import json
from pathlib import Path


def load_verdicts(results_path: str) -> dict[str, dict]:
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    return {
        r["id"]: {"verdict": r["parsed"].get("verdict"), "expected": r["expected"]}
        for r in data
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Two or more learner_results.json paths to combine.")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Optional short names for each run, same order.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = [load_verdicts(p) for p in args.runs]
    labels = args.labels or [f"run{i+1}" for i in range(len(runs))]
    ids = sorted(set().union(*[set(r) for r in runs]))

    tp = tn = fp = fn = 0
    rows = []
    for cid in ids:
        expected = next(r[cid]["expected"] for r in runs if cid in r)
        verdicts = [r[cid]["verdict"] if cid in r else None for r in runs]
        ensemble = "ANOMALY" if any(v == "ANOMALY" for v in verdicts) else "NORMAL"
        correct = ensemble == expected
        if expected == "ANOMALY" and ensemble == "ANOMALY":
            tp += 1
        elif expected == "NORMAL" and ensemble == "NORMAL":
            tn += 1
        elif expected == "NORMAL" and ensemble == "ANOMALY":
            fp += 1
        else:
            fn += 1
        rows.append({"id": cid, "expected": expected,
                     "per_run": dict(zip(labels, verdicts)),
                     "ensemble": ensemble, "correct": correct})
        mark = "OK " if correct else "XX "
        per = "  ".join(f"{l}={v}" for l, v in zip(labels, verdicts))
        print(f"{mark}{cid}: expected={expected} ensemble={ensemble}  [{per}]")

    total = len(ids)
    correct = tp + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    metrics = {"total": total, "correct": correct,
               "accuracy": correct / total if total else 0.0,
               "tp": tp, "tn": tn, "fp": fp, "fn": fn,
               "precision": precision, "recall": recall, "f1": f1}
    print(f"\nEnsemble accuracy: {correct}/{total} = {metrics['accuracy']:.2f}  "
          f"(TP={tp} TN={tn} FP={fp} FN={fn}) | P={precision:.2f} R={recall:.2f} F1={f1:.2f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"runs": dict(zip(labels, args.runs)), "metrics": metrics, "per_clip": rows},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
