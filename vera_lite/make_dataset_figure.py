"""Dataset montage: one representative frame per anomaly type, with labels.

Run with the global Python that has cv2 + matplotlib:
    python vera_lite/make_dataset_figure.py
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "pens_small"
OUT = ROOT / "rapor_belgeleri" / "Graduation Project - Latex Template" / "Imgs"

# (file, label, fraction-of-clip to sample)
PANELS = [
    ("pens_normal.mp4",            "Normal (3 pens)",        0.45),
    ("pens_missing_cap.mp4",       "Missing cap",            0.45),
    ("pens_wrong_direction.mp4",   "Wrong orientation",      0.45),
    ("pens_wrong_color.mp4",       "Wrong colour",           0.45),
    ("pens_no_pen.mp4",            "Missing pen",            0.45),
    ("pens_conveyor_stops.mp4",    "Belt stop / freeze",     0.50),
    ("pens_temporal_anomaly.mp4",  "Unexpected pen movement",0.60),
]


def frame_at(path, frac):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
    ok, f = cap.read()
    cap.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None


def main():
    cols, rows = 4, 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 8))
    axes = axes.ravel()
    for ax, (fname, label, frac) in zip(axes, PANELS):
        img = frame_at(SRC / fname, frac)
        if img is not None:
            ax.imshow(img)
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    for ax in axes[len(PANELS):]:
        ax.axis("off")
    fig.suptitle("Pen-conveyor dataset: normal and anomaly examples", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig_dataset.jpg", dpi=110, bbox_inches="tight",
                pil_kwargs={"quality": 80})
    plt.close(fig)
    # remove the old heavy PNG if present
    old = OUT / "fig_dataset.png"
    if old.exists():
        old.unlink()
    print("fig_dataset.jpg ->", OUT)


if __name__ == "__main__":
    main()
