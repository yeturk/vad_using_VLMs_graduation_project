"""Question-count ablation: does allowing MORE guiding questions help?

Starting from v3, we run the optimizer once with a high cap (max_questions=10),
then evaluate the resulting (larger) question set under the canonical setting
(video, blind, thinking_budget=2000) and compare it to v3.

To save time, we reuse v3's learner results from the most recent iterate run if
available; otherwise we evaluate v3 first.

Run:  wsl -d Ubuntu-22.04 -- bash vera_lite/run_wsl.sh -u -m vera_lite.run_manyq_experiment
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
MAXQ = 10  # the whole point: allow many more questions than the usual 5-6

RUN = ROOT / "vera_lite" / "runs" / "manyq" / datetime.now().strftime("%Y%m%d_%H%M%S")
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


def latest_v3_results() -> list[dict] | None:
    runs = sorted((ROOT / "vera_lite" / "runs" / "iterate").glob("*/v3_results.json"))
    if runs:
        print(f"Reusing v3 learner results from {runs[-1]}", flush=True)
        return json.loads(runs[-1].read_text(encoding="utf-8"))
    return None


def main():
    res3 = latest_v3_results()
    if res3 is None:
        print("No cached v3 results; evaluating v3 first.", flush=True)
        res3, m3 = evaluate(V3, "v3")
    else:
        m3 = compute_metrics(res3)
        print(f"[v3 cached] Accuracy: {m3['correct']}/{m3['total']} = {m3['accuracy']:.3f}\n", flush=True)

    # one optimizer step with a high question cap
    questions = json.loads(V3.read_text(encoding="utf-8"))["questions"]
    prompt = build_optimizer_prompt(questions, res3, max_questions=MAXQ)
    text, _ = call_qwen(model=MODEL, content=[{"text": prompt}])
    parsed = extract_json_object(text)
    new_q = parsed["new_questions"]
    manyq_path = RUN / "questions_manyq.json"
    manyq_path.write_text(json.dumps({
        "version": f"v3-manyq(max{MAXQ})",
        "source": f"optimizer_high_cap_{MODEL}",
        "questions": new_q,
        "reasoning": parsed.get("reasoning"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"--> optimizer produced {len(new_q)} questions (cap={MAXQ})\n", flush=True)

    resM, mM = evaluate(manyq_path, f"manyq_{len(new_q)}q")

    print("==== QUESTION-COUNT ABLATION ====", flush=True)
    print(f"  v3   ({len(questions)} questions): {m3['correct']}/{m3['total']} = {m3['accuracy']:.3f}", flush=True)
    print(f"  many ({len(new_q)} questions): {mM['correct']}/{mM['total']} = {mM['accuracy']:.3f}", flush=True)
    (RUN / "summary.json").write_text(json.dumps({
        "model": MODEL, "setting": f"video, blind, thinking_budget={TB}",
        "v3": {"n_questions": len(questions), "metrics": m3},
        "manyq": {"n_questions": len(new_q), "metrics": mM, "questions": new_q},
        "per_clip_manyq": [{"id": r["id"], "expected": r["expected"],
                            "predicted": r["parsed"].get("verdict"), "correct": r["correct"]}
                           for r in resM],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRun saved to: {RUN}", flush=True)


if __name__ == "__main__":
    main()
