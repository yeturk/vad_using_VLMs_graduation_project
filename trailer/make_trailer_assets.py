"""Prepare images for trailer.html: extract real frames from the pen videos and copy
the generated figures into ./trailer_assets/ so the HTML can reference them.

This script lives in trailer/ ; the project root is its parent directory.
Run with the global Python (has cv2):  python trailer/make_trailer_assets.py
"""
import shutil
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent          # .../trailer
ROOT = HERE.parent                               # project root
SRC = ROOT / "data" / "pens_small"
IMGS = ROOT / "rapor_belgeleri" / "Graduation Project - Latex Template" / "Imgs"
OUT = HERE / "trailer_assets"
OUT.mkdir(exist_ok=True)

# Representative frames (file, output name, fraction of clip)
FRAMES = [
    ("pens_normal.mp4",          "normal.jpg",          0.45),
    ("pens_wrong_color.mp4",     "anomaly_color.jpg",   0.45),
    ("pens_missing_cap.mp4",     "anomaly_cap.jpg",     0.45),
    ("pens_wrong_direction.mp4", "anomaly_orient.jpg",  0.45),
    ("pens_no_pen.mp4",          "anomaly_pen.jpg",     0.45),
    ("pens_temporal_anomaly.mp4","anomaly_temporal.jpg",0.60),
]

for fname, out, frac in FRAMES:
    cap = cv2.VideoCapture(str(SRC / fname))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(OUT / out), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("frame:", out)

# Copy generated report figures
for fig in ["fig_dataset.jpg", "fig_clip_matrix.png", "fig_accuracy.png",
            "fig_latency.png", "fig_repeat.png"]:
    src = IMGS / fig
    if src.exists():
        shutil.copy(src, OUT / fig)
        print("copied:", fig)

# IPAD button frames (Phase-1 dataset on slide 3) — low-res 256x256 originals
for src_name, out in [("057.jpg", "ipad_button_1.jpg"), ("078.jpg", "ipad_button_2.jpg")]:
    src = ROOT / "data" / src_name
    if src.exists():
        shutil.copy(src, OUT / out)
        print("copied:", out)

# Sample pen clips embedded in the page (autoplay/loop on slides 3 & 8)
VID_OUT = OUT / "videos"
VID_OUT.mkdir(exist_ok=True)
for src_name, out in [
    ("pens_normal.mp4",          "normal.mp4"),
    ("pens_wrong_color.mp4",     "wrong_color.mp4"),
    ("pens_wrong_direction.mp4", "wrong_direction.mp4"),
    ("pens_temporal_anomaly.mp4","temporal_anomaly.mp4"),
]:
    src = SRC / src_name
    if src.exists():
        shutil.copy(src, VID_OUT / out)
        print("video:", out)

print("\nAll assets in:", OUT)
