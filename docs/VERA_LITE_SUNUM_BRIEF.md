# Presentation Brief — VERA-lite for Video Anomaly Detection

> **Purpose:** This document contains everything needed to build a presentation about the
> **VERA-lite** part of the project only (the persona/voting work is intentionally excluded).
> Hand it to an LLM to generate slides. All numbers are real experimental results.
> Slide content is written in English (the team's previous slides are in English).

---

## 0. Metadata (title slide)
- **Title:** Video Anomaly Detection in Production/Assembly Lines — a VERA-lite approach
- **Course:** CSE 496 (Graduation Project)
- **Authors:** Alper Kaan Güler & Yunus Emre Türk
- **Advisor:** Asst. Prof. Habil Kalkan
- **University:** Gebze Technical University, Computer Engineering Department

---

## Suggested slide outline (≈14 slides)
1. Title
2. Problem & Motivation
3. The VERA paper (what it is)
4. From VERA to VERA-lite (our adaptation)
5. Dataset: why custom, what it contains
6. The VERA-lite pipeline
7. Guiding questions: v2 → v3 (optimizer)
8. Example model output (explainability)
9. The hard case: temporal anomaly
10. Input representations we tried (comparison)
11. The ensemble solution (13/13)
12. Speed & determinism (reasoning budget)
13. Repeatability (honest accuracy)
14. Findings, limitations, future work + references

---

## 1. Problem & Motivation
- **Goal:** detect anomalies on a production/assembly line from short video clips
  (decide NORMAL vs ANOMALY per clip).
- **Training-free approach:** instead of training a detector, *prompt* a frozen
  Vision-Language Model (VLM) with the scene description and anomaly rules.
- **Why:** no labelled training set, natural-language defect specs, adapts to new
  products without retraining, and produces **human-readable justifications**
  (explainability — important on a factory floor).
- **Model used:** `qwen3.6-plus` via the DashScope API (the only video-capable model we
  could reliably access; `qwen-vl-max-latest` → 403 Access Denied, `qwen3.7-max` → rejected
  video input).

---

## 2. The VERA Paper (explain the method)
**Reference:** Ye, Liu, He — *"VERA: Explainable Video Anomaly Detection via Verbalized
Learning of Vision-Language Models"*, CVPR 2025.

- **Key idea:** a frozen VLM prompted with an abstract question ("is there any anomaly?")
  performs poorly (only 53–65% AUC on UCF-Crime). VERA instead uses concrete, **learned
  guiding questions** that describe specific abnormal patterns.
- **Verbalized learning:** the guiding questions are treated as *learnable parameters*.
  In training, a **learner** VLM answers the questions to do a binary video-classification
  subtask; an **optimiser** VLM reads the learner's mistakes and rewrites the questions.
  Driven only by coarse (video-level) labels; **model weights stay frozen**.
- **Inference (coarse-to-fine):** segment-level anomaly scores → add scene context by
  ensembling → frame-level scores via temporal fusion (Gaussian smoothing + position
  weighting).
- **Result:** state-of-the-art explainable VAD on UCF-Crime and XD-Violence; transfers
  across models/datasets.

---

## 3. From VERA to VERA-lite (our adaptation)
We adopt VERA's **learner/optimiser loop over natural-language guiding questions on a
frozen VLM**, but simplify and re-target it for an industrial setting:
- **Clip-level binary verdict** (NORMAL/ANOMALY) instead of VERA's frame-level
  coarse-to-fine scoring.
- Small, purpose-built **industrial dataset** instead of UCF-Crime/XD-Violence.
- We **add**: a blind evaluation protocol, a study of input representations, a
  reasoning-budget control, and a two-view ensemble.

---

## 4. Dataset: why custom, what it contains
- **Why not IPAD (existing benchmark):** (1) low resolution (256×256) hides fine cues like
  a missing cap or subtle colour; (2) more decisively, no control over the specific anomaly
  types and no small cleanly-labelled set for a focused, per-anomaly quantitative study.
- **Our pen-conveyor dataset:** a white board slides left→right on a rail carrying **three
  identical pens**, fixed camera, controlled conditions, documented protocol.
- **13 clips** = 3 normal + 10 anomalous, **7 anomaly types:** missing cap, wrong
  orientation, wrong colour, missing pen, belt stop/freeze, unexpected (intra-object) pen
  movement, and a combined case.
- Recorded at 4K, downscaled to 720×1280 for the API.
- **Figure available:** `fig_dataset.jpg` (one frame per class).

---

## 5. The VERA-lite Pipeline
- **Learner VLM** answers the guiding questions for a clip → outputs structured JSON:
  per-question YES/NO/UNCLEAR + short visual evidence, then verdict, anomaly type, score
  (0–1), confidence.
- **Optimiser VLM** reviews the learner's mistakes → proposes improved questions.
- **Blind evaluation (our fix):** the original code leaked the expected label into the
  learner prompt (a first run scored a fake 13/13). We made **blind mode default**; the
  first honest score was **0.77**. All reported numbers are blind.
- We also added accuracy / precision / recall / F1 + confusion to the run summary and
  retry/back-off for transient API errors.

---

## 6. Describe Step → Writing the Questions
- Before writing any classifier prompt, we ran a **neutral "describe" prompt** (report what
  you see, no verdict).
- Findings: the model counts the 3 pens reliably, but perceives **colour weakly** ("greyish"
  not "blue") and is unsure whether the dark end is a cap.
- → This shaped the questions toward **relative comparisons** ("are all three the same
  colour?") rather than absolute attributes.

---

## 7. Guiding Questions: v2 → v3 (the optimizer at work)
- v2 = first hand-written set. One optimiser iteration rewrote them. Optimiser's verbatim
  rationale: v2 *"relies heavily on negative phrasing … VLMs frequently struggle with
  logical negation,"* so it produced *"direct, positive verification statements."*
- We additionally hardened orientation ("including a pen near the frame edge"), motion
  ("including a brief 1–3 s pause"), and added a 6th pen-movement question → **v3**.

**Table — questions by anomaly family:**

| Anomaly family | v2 (negated) | v3 (positive / hardened) |
|---|---|---|
| Count | no missing pen and no empty gap | exactly three pens in a single row |
| Orientation | none flipped or reversed | all pointing the same way, incl. a pen near the frame edge |
| Colour | none differing in colour | identical barrel and cap colours, no mismatch |
| Completeness | no bare tip, none shorter | fully capped, no exposed tip, all equal length |
| Belt motion | without stopping, pausing, reversing | advances at every moment, never halting even briefly (1–3 s) |
| Pen movement | (not asked) | pens stay fixed; no pen sliding relative to the board |

**Result:** accuracy **0.77 → 0.92** (single run). Only remaining miss: the temporal anomaly.

---

## 8. Example Model Output (explainability)
On the `wrong_colour` clip (blind v3 run), the model returned:
- Q1 YES (three pens), Q2 YES (same orientation), **Q3 NO** ("leftmost pen has a blue
  barrel, the other two are black/dark grey"), Q4 YES, Q5 YES, Q6 YES.
- **Verdict: ANOMALY, type wrong_colour, score 1.0, confidence HIGH.**
- Reason: "one is blue while the other two are black, violating the requirement that all
  pens be identical in colour."
- **Takeaway:** the per-question evidence is what makes the decision auditable.

---

## 9. The Hard Case: Temporal Anomaly
- `motion_temporal_01`: the pens shift on their own mid-clip while the belt moves normally.
- Missed by the v3 learner **and** by the open-ended describe prompt (which reported a
  perfectly normal scene).
- → The cause is the VLM's **sparse internal frame sampling**, not the wording of the
  questions. This motivated trying different input representations.

---

## 10. Input Representations We Tried (comparison)
We tried several ways to present the clip to the model:
1. **Native video** (default sampling).
2. **Dense frames (central window [0.15–0.70])** — 16 numbered frames + a relative-spacing
   prompt; central crop avoids entry/exit artefacts.
3. **Dense frames (wide window [0.05–0.95])** — backfired.
4. **Temporal grid (3×3 montage)** — single image of tiled frames.
5. **Higher video sampling (fps=8)** — quadrupled visual tokens (~10,100 → ~40,394).

**Table — representation comparison (blind, single run):**

| Method | Accuracy | FP | Catches temporal? | Note |
|---|---|---|---|---|
| Video (v3, default fps) | 0.92 (12/13) | 0 | no | misses temporal |
| Dense [0.15–0.70], 16f | 0.85 (11/13) | 0 | **yes** | misses orientation-2, stop-2 |
| Dense wide [0.05–0.95] | 0.62 (8/13) | 1 | yes | exit artefact → false positive |
| Temporal grid 3×3 | (probe) | yes | yes | shares dense blind spots + exit FP |
| Video + fps=8 + v3 | 0.85 (11/13) | 0 | no | higher fps degraded orientation |
| **Ensemble (video ∨ dense)** | **1.00 (13/13)** | **0** | **yes** | union of blind spots |

**Key insight:** No single representation reaches 13/13. There is an **intrinsic
continuous-video vs. discrete-frame trade-off**: continuous video catches edge-orientation
and the belt stop but misses intra-object motion; discrete frames catch the motion but lose
the other two. Also: **presentation matters more than frame count** — fps=8 alone did not
catch the temporal anomaly; a *dedicated relative-spacing prompt* did.

---

## 11. The Ensemble Solution (13/13)
- **OR-ensemble:** a clip is ANOMALY if **either** the video view or the dense-frame view
  flags it.
- Works because the two views have **disjoint blind spots** and **both have 0 false
  positives**, so the union adds no false positives.
  - temporal → video NORMAL, dense ANOMALY → ✓
  - orientation-2, stop-2 → video ANOMALY, dense NORMAL → ✓
- **Result: 13/13 = 1.00, FP=0, precision=recall=F1=1.00**, at zero extra inference cost
  (reuses the two existing runs).
- **Figure available:** `fig_clip_matrix.png` (per-clip correctness of video / dense /
  ensemble — visually shows the disjoint blind spots).

---

## 12. Speed & Determinism (reasoning budget)
- `qwen3.6-plus` is a *reasoning* model. On dense input it once spent **15,684 reasoning
  tokens** and returned an **empty answer** (crash). One clip used **13,394** reasoning
  tokens → dominated latency.
- **Fix:** a `thinking_budget` caps the reasoning and forces a final answer.

**Table — latency vs reasoning budget (mean over n=3):**

| Config | Seconds / clip | Accuracy |
|---|---|---|
| baseline (unbounded) | 85.2 | 0.846 |
| thinking_budget = 2000 | **49.9** (~41% faster) | 0.846 |
| thinking_budget = 1000 | 39.5 | ~0.85 |

- **Bonus finding:** capping the reasoning also made outputs **deterministic** — see next slide.
- **Note:** raising `fps` instead makes the call *slower* and *hurts* accuracy → wrong lever.
- **Figures available:** `fig_latency.png`, `fig_reasoning.png`.

---

## 13. Repeatability (honest accuracy)
- VLM outputs vary between runs → we ran the baseline and thinking_budget=2000 **three times each**.

**Table — repeatability (n=3 blind runs):**

| Config | n | Accuracy (each run) | Verdict flips* | Total FP |
|---|---|---|---|---|
| baseline (unbounded) | 3 | 11/13 (0.846) | 2 clips | 1 |
| thinking_budget = 2000 | 3 | 11/13 (0.846) | 0 clips | 0 |

\*Verdict flips = clips whose NORMAL/ANOMALY verdict was not unanimous across the 3 runs.

- **Honest reframing:** the single-run 0.92 was a *lucky draw*; the true value is **0.846
  (11/13) in every run**.
- The temporal clip is missed in all runs (deterministic limit); under the baseline two
  borderline clips flip between runs, but **thinking_budget=2000 produces identical per-clip
  verdicts (0 flips)** — i.e. capping reasoning improves **determinism** as well as speed.
- **Figure available:** `fig_repeat.png` (per-clip miss frequency over the 6 runs).

---

## 14. Findings, Limitations, Future Work
**Key findings:**
1. The VERA-style optimiser loop lifts accuracy (0.77 → 0.92 single run).
2. Intra-object temporal motion is a real VLM perception limit (missed even by open-ended
   description).
3. Continuous-vs-discrete representation trade-off is intrinsic; **presentation > frame count**.
4. **Ensemble** of complementary views reaches 13/13 with 0 FP.
5. Single-run accuracy is misleading → repeated runs give **0.846 ± 0.000**.
6. **Reasoning-budget control**: ~41% faster, same accuracy, fully deterministic.

**Limitations (be honest):**
- 13-clip pilot recorded under fixed, controlled conditions → reaching 13/13 by tuning on
  these exact clips risks overfitting; the controlled setup hides robustness to lighting,
  angle and resolution.
- Only one API model was reliably accessible. No frame-level evaluation yet.

**Future work:**
- Larger dataset with a proper train/test split + confidence intervals.
- Simple CV baselines (frame differencing, optical flow) to quantify what the VLM adds.
- Frame-level metrics (AUROC/AP). Calibration analysis (the model can be confidently wrong).
- A cascade (cheap detector first, VLM only on flagged segments) for real-time use.

---

## 15. References
- Ye, Liu, He, *VERA: Explainable Video Anomaly Detection via Verbalized Learning of VLMs*, CVPR 2025.
- Qwen Team, *Qwen2.5-VL Technical Report*, arXiv:2502.13923, 2025.
- (Optional) Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in LLMs*, NeurIPS 2022.

---

## Figures you can drop into the slides (in the report's Imgs/ folder)
- `fig_dataset.jpg` — dataset: one frame per anomaly type (slide 5)
- `fig_clip_matrix.png` — per-clip correctness video/dense/ensemble (slide 11)
- `fig_accuracy.png` — accuracy by representation (slide 10)
- `fig_latency.png` — latency by reasoning budget (slide 12)
- `fig_reasoning.png` — reasoning tokens per clip (slide 12)
- `fig_repeat.png` — per-clip miss frequency over 6 runs (slide 13)
- `grid_temporal.jpg` — the temporal-grid montage (slide 10, optional)
