"""Generate report figures from the real run outputs (matplotlib, no API).

Run with the global Python that has matplotlib (not grad2_env):
    python vera_lite/make_report_figures.py
Figures are written to the LaTeX template's Imgs/ folder.
"""
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "vera_lite" / "runs"
OUT = ROOT / "rapor_belgeleri" / "Graduation Project - Latex Template" / "Imgs"
OUT.mkdir(parents=True, exist_ok=True)

CLIPS = ["normal_01", "normal_02", "normal_03", "missing_cap_01", "missing_cap_02",
         "wrong_orientation_01", "wrong_orientation_02", "wrong_color_01",
         "missing_pen_01", "motion_stop_01", "motion_stop_02", "motion_temporal_01",
         "combined_01"]
SHORT = ["n1", "n2", "n3", "cap1", "cap2", "or1", "or2", "col", "pen", "st1", "st2",
         "temp", "comb"]


def load_summary(folder):
    f = list(glob.glob(str(folder / "*" / "summary.json")))
    if not f:
        f = [str(folder / "summary.json")]
    return json.loads(Path(sorted(f)[0]).read_text(encoding="utf-8"))


def blind_subdirs():
    # runs/blind has two timestamped dirs: v2 (older, ~0.77) and v3 (newer, ~0.92)
    dirs = sorted(glob.glob(str(RUNS / "blind" / "*" / "summary.json")))
    out = {}
    for d in dirs:
        m = json.loads(Path(d).read_text(encoding="utf-8"))["metrics"]
        out[round(m["accuracy"], 2)] = Path(d)
    return out


# ---------------------------------------------------------------------------
# Figure 1: method accuracy comparison
# ---------------------------------------------------------------------------
def fig_accuracy():
    bl = blind_subdirs()
    accs = {}
    accs["Q-set v2\n(video)"] = json.loads((bl[0.77]).read_text(encoding="utf-8"))["metrics"]["accuracy"]
    accs["Q-set v3\n(video)"] = json.loads((bl[0.92]).read_text(encoding="utf-8"))["metrics"]["accuracy"]
    accs["Dense\n[0.15-0.70]"] = load_summary(RUNS / "blind_frames")["metrics"]["accuracy"]
    accs["Dense wide\n[0.05-0.95]"] = load_summary(RUNS / "blind_frames_wide")["metrics"]["accuracy"]
    accs["Video\nfps=8"] = load_summary(RUNS / "blind_vidfps8")["metrics"]["accuracy"]
    accs["Ensemble"] = json.loads((RUNS / "ensemble" / "video_v3__dense16.json").read_text(encoding="utf-8"))["metrics"]["accuracy"]

    labels = list(accs.keys())
    vals = [accs[k] for k in labels]
    colors = ["#6c8ebf"] * (len(vals) - 1) + ["#2e7d32"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Blind accuracy (single run)")
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, ls="--", lw=0.7, color="grey")
    ax.set_title("Accuracy by input representation (13-clip pen dataset)")
    plt.xticks(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_accuracy.png", dpi=200)
    plt.close(fig)
    print("fig_accuracy.png")


# ---------------------------------------------------------------------------
# Figure 2: per-clip outcome matrix (video v3, dense, ensemble)
# ---------------------------------------------------------------------------
def fig_clip_matrix():
    bl = blind_subdirs()
    v3 = json.loads((bl[0.92]).read_text(encoding="utf-8"))
    dense = load_summary(RUNS / "blind_frames")
    ens = json.loads((RUNS / "ensemble" / "video_v3__dense16.json").read_text(encoding="utf-8"))

    def row_from_perclip(per):
        d = {c["id"]: c["correct"] for c in per}
        return [1 if d.get(cid) else 0 for cid in CLIPS]

    rows = [row_from_perclip(v3["per_clip"]),
            row_from_perclip(dense["per_clip"]),
            row_from_perclip(ens["per_clip"])]
    mat = np.array(rows)
    names = ["Video (v3)", "Dense frames", "Ensemble"]

    fig, ax = plt.subplots(figsize=(9, 2.6))
    cmap = matplotlib.colors.ListedColormap(["#c62828", "#2e7d32"])
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLIPS)))
    ax.set_xticklabels(SHORT, fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "OK" if mat[i, j] else "X", ha="center", va="center",
                    color="white", fontsize=7)
    ax.set_title("Per-clip correctness (green=correct, red=wrong)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_clip_matrix.png", dpi=200)
    plt.close(fig)
    print("fig_clip_matrix.png")


# ---------------------------------------------------------------------------
# Figure 3: latency by reasoning budget
# ---------------------------------------------------------------------------
def mean_speed(folder):
    secs = []
    for sub in glob.glob(str(folder / "*")):
        lf = sorted(glob.glob(os.path.join(sub, "*_learner.json")), key=os.path.getmtime)
        mt = [os.path.getmtime(x) for x in lf]
        if len(mt) > 1:
            difs = [mt[i] - mt[i - 1] for i in range(1, len(mt))]
            secs.append(sum(difs) / len(difs))
    return float(np.mean(secs)) if secs else float("nan")


def fig_latency():
    base = mean_speed(RUNS / "repeat" / "baseline")
    tb2k = mean_speed(RUNS / "repeat" / "tb2000")
    tb1k = mean_speed(RUNS / "blind_fast")
    labels = ["baseline\n(unbounded)", "thinking\nbudget=2000", "thinking\nbudget=1000"]
    vals = [base, tb2k, tb1k]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, vals, color=["#b0b0b0", "#2e7d32", "#66bb6a"],
                  edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}s", ha="center", fontsize=9)
    ax.set_ylabel("Seconds / clip (mean)")
    ax.set_title("Latency vs. reasoning budget (accuracy unchanged ~0.85)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency.png", dpi=200)
    plt.close(fig)
    print("fig_latency.png")


# ---------------------------------------------------------------------------
# Figure 4: reasoning tokens per clip (v3 baseline run)
# ---------------------------------------------------------------------------
def fig_reasoning():
    bl = blind_subdirs()
    folder = bl[0.92].parent
    vals = []
    for cid in CLIPS:
        p = folder / f"{cid}_learner.json"
        u = json.loads(p.read_text(encoding="utf-8")).get("usage", {})
        vals.append((u.get("output_tokens_details", {}) or {}).get("reasoning_tokens", 0) or 0)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(SHORT, vals, color="#8e24aa", edgecolor="black", linewidth=0.5)
    mx = int(np.argmax(vals))
    ax.text(mx, vals[mx] + 200, f"{vals[mx]}", ha="center", fontsize=9, color="#8e24aa")
    ax.set_ylabel("Reasoning tokens")
    ax.set_title("Reasoning tokens per clip (unbounded baseline)")
    plt.xticks(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_reasoning.png", dpi=200)
    plt.close(fig)
    print("fig_reasoning.png")


# ---------------------------------------------------------------------------
# Figure 5: per-clip miss frequency over the 6 repeat runs
# ---------------------------------------------------------------------------
def fig_repeat():
    miss = {c: 0 for c in CLIPS}
    n = 0
    for cfg in ["baseline", "tb2000"]:
        for s in glob.glob(str(RUNS / "repeat" / cfg / "*" / "summary.json")):
            n += 1
            for c in json.loads(Path(s).read_text(encoding="utf-8"))["per_clip"]:
                if not c["correct"]:
                    miss[c["id"]] = miss.get(c["id"], 0) + 1
    vals = [miss[c] for c in CLIPS]
    colors = ["#c62828" if v == n else ("#ef6c00" if v > 0 else "#bdbdbd") for v in vals]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(SHORT, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(f"Times missed (out of {n} runs)")
    ax.set_ylim(0, n + 0.5)
    ax.set_yticks(range(n + 1))
    ax.set_title(f"Per-clip miss frequency over {n} repeated blind runs")
    plt.xticks(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_repeat.png", dpi=200)
    plt.close(fig)
    print("fig_repeat.png")


if __name__ == "__main__":
    fig_accuracy()
    fig_clip_matrix()
    fig_latency()
    fig_reasoning()
    fig_repeat()
    print(f"\nAll figures written to: {OUT}")
