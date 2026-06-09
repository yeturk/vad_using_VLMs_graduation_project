"""Downscale pens videos for VLM API use.

Keeps native FPS, reduces resolution (and re-encodes) to shrink file size.
Originals are left untouched; outputs go to data/pens_small/.
"""
import glob
import os
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "pens"
OUT_DIR = ROOT / "data" / "pens_small"
TARGET_HEIGHT = 1280  # portrait; width derived to keep aspect ratio


def compress(src: Path, dst: Path) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = TARGET_HEIGHT / h
    new_w = int(round(w * scale / 2) * 2)  # keep even
    new_h = int(round(h * scale / 2) * 2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (new_w, new_h))

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA))
        n += 1

    cap.release()
    writer.release()
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"{src.name}: {w}x{h} -> {new_w}x{new_h} | {n} frames | {size_mb:.1f} MB")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(glob.glob(str(SRC_DIR / "pens_*.mp4"))):
        src = Path(src)
        compress(src, OUT_DIR / src.name)
    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
