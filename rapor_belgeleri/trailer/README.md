# Trailer — Project Presentation Site

Single-page, scroll-snap presentation of the project (CSE 496 — Güler & Türk).

## Contents

| File | Purpose |
|------|---------|
| `trailer.html` | The presentation itself (open in any browser) |
| `trailer_assets/` | All images used by the page |
| `trailer_assets/videos/` | Sample pen clips embedded on slides 3 & 8 |
| `SPEECH.md` | ~3.5 min spoken script (B1 English), one block per slide |
| `make_trailer_assets.py` | Regenerates `trailer_assets/` from the source videos/figures |

## View it

Just open `trailer.html` in a browser (double-click, or drag into the window).
Navigate with the **arrow keys** or the dots on the right. No server needed.

## Slides

1. Title
2. Problem & Motivation
3. **Dataset Journey** — IPAD benchmark (Phase 1) → our custom pen-conveyor dataset (Phase 2)
4. Approach 1 — Multi-Persona Ensemble (on the IPAD button footage)
5. The VERA paper (CVPR 2025)
6. Approach 2 — VERA-lite pipeline (on our pen dataset)
7. Inside a verdict (structured JSON output)
8. Representation trade-off & OR-ensemble
9. Optimization & final results

## Regenerate the images

Run from the **project root** with a Python that has OpenCV (`cv2`):

```
python trailer/make_trailer_assets.py
```

It pulls:
- representative pen frames from `data/pens_small/*.mp4`,
- report figures from `rapor_belgeleri/.../Imgs/`,
- the low-res IPAD button frames (`data/057.jpg`, `data/078.jpg`).

into `trailer/trailer_assets/`.
