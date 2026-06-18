"""Iterative-optimizer ablation: does running the optimizer more times help, or overfit?

Starting from v3, we evaluate the questions (blind, video mode), then run the optimizer
to get v4, evaluate, run the optimizer again to get v5, evaluate. All under the canonical
setting (video, thinking_budget=2000) so v3/v4/v5 are directly comparable.

Run:  wsl -d Ubuntu-22.04 -- bash vera_lite/run_wsl.sh -u -m vera_lite.run_iterate_experiment
"""
import json
from datetime import datetime
from pathlib import Path

from .run_learner import run_learner
from .run_iteration import compute_metrics
from .dashscope_client import call_qwen, extract_json_object
from .prompts import build_optimizer_prompt

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "vera_lite" / "dataset_manifest.json"
V3 = ROOT / "vera_lite" / "guiding_questions_v3.json"
MODEL = "qwen3.6-plus"
TB, MAXTOK = 2000, 3000
MAX_Q = 6  # allow up to 6 so the optimizer is not forced to drop a question family

RUN = ROOT / "vera_lite" / "runs" / "iterate" / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN.mkdir(parents=True, exist_ok=True)


def evaluate(questions_path: Path, tag: str) -> tuple[list[dict], dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = []
    for item in manifest:
        r = run_learner(video=item["video"], expected=None, model=MODEL,
                        questions_path=questions_path, thinking_budget=TB, max_tokens=MAXTOK)
        r["id"] = item["id"]
        r["expected"] = item["expected"]
        r["anomaly_type"] = item.get("anomaly_type")
        predicted = r["parsed"].get("verdict")
        r["correct"] = predicted == item["expected"]
        results.append(r)
        print(f"[{tag}] {'OK ' if r['correct'] else 'XX '}{item['id']}: "
              f"expected={item['expected']} predicted={predicted}", flush=True)
    (RUN / f"{tag}_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    m = compute_metrics(results)
    print(f"[{tag}] Accuracy: {m['correct']}/{m['total']} = {m['accuracy']:.3f}  "
          f"(TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']})\n", flush=True)
    return results, m


def optimize(results: list[dict], questions_path: Path, out_path: Path, version: str) -> Path:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    prompt = build_optimizer_prompt(questions, results, max_questions=MAX_Q)
    text, _ = call_qwen(model=MODEL, content=[{"text": prompt}])
    parsed = extract_json_object(text)
    new_q = parsed["new_questions"]
    out_path.write_text(json.dumps({
        "version": version,
        "source": f"iterative_optimizer_{MODEL}",
        "questions": new_q,
        "reasoning": parsed.get("reasoning"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"--> optimizer produced {version} with {len(new_q)} questions", flush=True)
    return out_path


def main():
    summary = {}

    # v3
    res3, m3 = evaluate(V3, "v3"); summary["v3"] = m3
    # v3 -> v4
    v4 = optimize(res3, V3, RUN / "questions_v4.json", "v4-iter")
    res4, m4 = evaluate(v4, "v4"); summary["v4"] = m4
    # v4 -> v5
    v5 = optimize(res4, v4, RUN / "questions_v5.json", "v5-iter")
    res5, m5 = evaluate(v5, "v5"); summary["v5"] = m5

    best = max(summary, key=lambda k: summary[k]["accuracy"])
    print("==== ITERATION SUMMARY ====", flush=True)
    for v in ("v3", "v4", "v5"):
        mk = "  <== BEST" if v == best else ""
        print(f"  {v}: {summary[v]['correct']}/{summary[v]['total']} = "
              f"{summary[v]['accuracy']:.3f}{mk}", flush=True)
    best_path = {"v3": V3, "v4": v4, "v5": v5}[best]
    (RUN / "summary.json").write_text(json.dumps({
        "model": MODEL, "setting": f"video, blind, thinking_budget={TB}",
        "metrics": summary, "best_version": best, "best_questions": str(best_path),
        "per_version": {v: [{"id": r["id"], "expected": r["expected"],
                             "predicted": r["parsed"].get("verdict"), "correct": r["correct"]}
                            for r in {"v3": res3, "v4": res4, "v5": res5}[v]]
                        for v in ("v3", "v4", "v5")},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBest version: {best}  ->  {best_path}", flush=True)
    print(f"Run saved to: {RUN}", flush=True)


if __name__ == "__main__":
    main()
