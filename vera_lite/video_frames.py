"""Shared helper: extract dense, time-ordered frames as JPEG base64 data URIs.

Used to feed the learner an image sequence instead of relying on the model's
internal (sparse) video sampling, which misses subtle temporal anomalies.
"""
import base64

import cv2


def extract_dense_frames(
    video: str,
    n: int,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
) -> list[str]:
    """Return n evenly spaced frames (within [start_frac, end_frac]) in time order.

    Trimming the edges excludes the board's entry/exit, where pens leaving the
    frame can otherwise look like relative pen movement.
    """
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo = int(total * start_frac)
    hi = max(lo + 1, int(total * end_frac))
    if n <= 1:
        indices = [(lo + hi) // 2]
    else:
        indices = [int(round(lo + i * (hi - lo - 1) / (n - 1))) for i in range(n)]

    uris: list[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            uris.append("data:image/jpeg;base64," + base64.b64encode(buf).decode())
    cap.release()
    return uris
