# Poster Plan

## Goal

Prepare the final project poster by updating last semester's poster structure for this semester's work. The poster should explain the project quickly, show the custom dataset clearly, and make the final VERA-lite results easy to understand.

The poster should keep a similar visual identity to the previous GTU poster:

- GTU-style blue, red, and white color palette
- Clean academic layout
- Large title and strong section headers
- Visual-first explanation with short text blocks

Preferred size: A1 portrait if possible. A2 can work, but A1 is better because we need dataset examples, pipeline, guiding questions, and results on the same poster.

## Working Title

**Training-Free Video Anomaly Detection on Industrial Conveyor Lines using Vision-Language Models**

Optional subtitle:

**A VERA-lite Approach with Guiding Questions, Dense Frames, and Explainable Decisions**

## Core Message

The poster should tell one simple story:

> We built a training-free video anomaly detection pipeline for a controlled industrial conveyor-like scene. Instead of training a detector, we prompt a frozen Vision-Language Model with scene rules and guiding questions. A custom pen-conveyor dataset, optimized guiding questions, dense-frame representation, and an OR ensemble give strong results with explainable JSON outputs.

## Suggested Layout

Use a three-column poster layout.

```text
TOP
Title, authors, advisor, university logo

COLUMN 1
Problem and motivation
Dataset

COLUMN 2
VERA-lite method
Guiding questions
Explainability

COLUMN 3
Experiments
Final results
Limitations and future work

BOTTOM
Key takeaways, references, QR code or repository/demo link
```

## Section 1: Problem and Motivation

Purpose: Explain why this project matters.

Suggested text:

> Industrial production lines need continuous visual inspection. However, real anomaly data is rare, expensive to label, and often difficult to collect. Classical supervised models usually require many labeled defect examples and retraining when the product or layout changes.
>
> Our goal is to detect whether a short production-line video is normal or anomalous without training or fine-tuning a task-specific model.

Key bullets:

- Training-free anomaly detection
- Few examples and natural-language rules
- Explainable decisions instead of black-box alarms
- Adaptable to a new product by changing the prompt

Visual idea:

- Small comparison: "Classical VAD" vs "Our VLM Prompting Approach"
- Classical: labeled data -> training -> detector
- Our approach: video + prompt + guiding questions -> verdict + explanation

## Section 2: Dataset

Purpose: Show why we moved from public benchmark data to our own controlled setup.

Suggested text:

> We first examined the IPAD industrial anomaly dataset, but its low-resolution clips were too coarse for subtle visual differences such as pen color, cap presence, and orientation. Therefore, we recorded a custom pen-conveyor dataset with a fixed camera and controlled anomaly types.

Dataset facts:

- 13 total clips
- 3 normal clips
- 10 anomalous clips
- Original videos: 4K vertical, 30 FPS
- API videos: downscaled to 720x1280, FPS preserved
- Scene: three pens on a moving white board, fixed camera

Anomaly families:

- Wrong color
- Wrong orientation
- Missing cap
- Missing pen
- Board stop or freeze
- Unexpected pen movement
- Combined anomaly case

Important wording:

- Prefer "six primitive anomaly families plus one combined case" if space allows.
- Avoid saying the dataset is large. It is a controlled pilot dataset.

Visual idea:

- A 2-row grid of example frames:
  - Normal
  - Wrong color
  - Wrong orientation
  - Missing cap
  - Missing pen
  - Temporal / motion anomaly
- Use green border for normal and red/orange borders for anomalies.

## Section 3: VERA-lite Method

Purpose: Explain the final architecture in one visual pipeline.

Suggested text:

> Our method is inspired by VERA. We use a frozen Vision-Language Model as a learner. The learner receives the video, a scene description, anomaly rules, and guiding questions. It returns a structured JSON output containing a verdict, anomaly type, confidence, score, and evidence for each question.

Pipeline:

```text
Input video
  -> Preprocessing
  -> Scene description + anomaly rules
  -> Guiding questions
  -> Qwen3.6-plus learner
  -> JSON verdict and explanation
  -> Optimizer feedback for better questions
```

Important concepts:

- No model training
- No fine-tuning
- Prompt is the main adaptation mechanism
- Optimizer improves questions after mistakes
- Blind evaluation: expected labels are not shown to the learner

Visual idea:

- A horizontal flow diagram with 5 boxes:
  1. Video clip
  2. Prompt package
  3. Learner VLM
  4. JSON explanation
  5. Optimized questions

## Section 4: Guiding Questions

Purpose: Show how natural-language questions replace task-specific training.

Final six guiding question topics:

1. Pen count: exactly three pens on the board
2. Orientation: all pens point in the same direction
3. Color: all pens share identical barrel and cap colors
4. Completeness: every pen is fully capped and equal in length
5. Board motion: board moves steadily without stopping or freezing
6. Pen motion: pens stay fixed relative to the board

Suggested text:

> Guiding questions turn the anomaly definition into simple visual checks. The model answers each question with evidence, then combines the answers into a final verdict.

Optimization story:

- Initial questions detected most static anomalies.
- Temporal anomalies were harder because the model could miss short motion changes.
- The optimizer helped improve motion-related questions.
- The final v3 questions improved accuracy from 10/13 to 12/13 on native video evaluation.

Visual idea:

- Mini table:
  - Question topic
  - Detects
  - Example anomaly

## Section 5: Input Representation Experiments

Purpose: Explain why video alone was not enough and why dense frames helped.

Suggested text:

> We tested different ways of showing the same clip to the model. Native video works well for most static defects. Dense frames make temporal changes more explicit by giving the model ordered snapshots from the clip.

Representations:

- Native video
- Dense 16 frames from the central video window
- Wider dense frames
- Temporal grid
- Higher API sampling FPS

Key finding:

> Native video and dense frames make different mistakes, so they are complementary.

Short explanation:

- Native video: better for full-scene visual context
- Dense frames: better for visible mid-clip temporal changes
- Wide dense frames: can introduce entry/exit artifacts
- Higher FPS: increases tokens and latency, but did not solve the main temporal issue alone

Visual idea:

- Side-by-side:
  - Native video icon
  - Dense frame strip
  - OR ensemble box

## Section 6: Results

Purpose: Make the final result visible in a few seconds.

Main result:

| Method | Accuracy | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Video v2 | 10/13 | 7 | 3 | 0 | 3 |
| Video v3 | 12/13 | 9 | 3 | 0 | 1 |
| Dense 16 frames | 11/13 | 8 | 3 | 0 | 2 |
| OR ensemble | 13/13 | 10 | 3 | 0 | 0 |

Poster-friendly text:

> The final OR ensemble combines the video result and dense-frame result. If either representation detects an anomaly, the clip is marked anomalous. This solved all 13 pilot clips with no false positives.

Latency and stability:

- Baseline reasoning: about 85.2 seconds per clip
- `thinking_budget=2000`: about 49.9 seconds per clip
- Approximate speedup: 41 percent faster
- More stable JSON outputs and fewer verdict flips

Visual idea:

- Bar chart:
  - Video v2: 10/13
  - Video v3: 12/13
  - Dense 16: 11/13
  - OR ensemble: 13/13
- Optional small latency chart:
  - Baseline 85.2s
  - Thinking budget 49.9s

## Section 7: Explainability Example

Purpose: Show that the system does not only say "anomaly"; it explains why.

Suggested text:

> Each decision includes evidence for the guiding questions. This makes the output more useful for an operator than a single anomaly score.

Example JSON fields:

```json
{
  "verdict": "ANOMALOUS",
  "anomaly_type": "WRONG_ORIENTATION",
  "confidence": "HIGH",
  "anomaly_score": 0.95,
  "evidence": [
    {
      "question": "Are all pens aligned identically?",
      "answer": "NO",
      "reason": "One pen points in a different direction."
    }
  ]
}
```

Visual idea:

- Use one compact JSON card.
- Highlight only verdict, anomaly type, and one evidence line.
- Do not show a long raw JSON block on the poster.

## Section 8: Limitations and Future Work

Purpose: Be honest and academically careful.

Limitations:

- Small controlled pilot dataset
- Single camera and fixed scene
- Cloud API latency and cost
- No frame-level localization yet
- The prompt may overfit to this exact pen setup

Future work:

- Test on more products and camera angles
- Add more anomaly examples
- Evaluate frame-level localization
- Compare more VLMs
- Reduce latency with smaller models or local deployment
- Improve prompt optimization automatically

## Section 9: Key Takeaways

Suggested final box:

> Training-free VLM prompting can detect controlled production-line anomalies without task-specific model training.
>
> Guiding questions make the decision process explainable.
>
> Native video and dense-frame inputs are complementary.
>
> The final OR ensemble reached 13/13 correct decisions on the pilot dataset.

## References

Use short poster references:

1. Ye, Liu, He. "VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models." arXiv:2412.01095, 2024.
2. Liu et al. "IPAD: Industrial Process Anomaly Detection Dataset." arXiv:2404.15033, 2024.
3. Bai et al. "Qwen2.5-VL Technical Report." arXiv:2502.13923, 2025.
4. Yang et al. "Qwen3 Technical Report." arXiv:2505.09388, 2025.
5. Alibaba Cloud Model Studio / DashScope Qwen API Documentation.

## Visual Assets to Reuse or Create

Priority visuals:

1. Dataset example grid
2. VERA-lite pipeline diagram
3. Six guiding question table
4. Input representation comparison
5. Accuracy result chart
6. Optional latency chart
7. Optional QR code for repository, trailer, or demo

Candidate repo assets:

- `docs/Alper_Kaan_Güler-Yunus_Emre_Türk.pdf` or report PDF for final wording and figures
- `docs/VERA_LITE_SUNUM_BRIEF.md`
- `docs/SUNUM_PLANI.md`
- `vera_lite/PENS_VAD_EXPERIMENT_LOG.md`
- `vera_lite/make_dataset_figure.py`
- `vera_lite/make_report_figures.py`
- `vera_lite/runs/`

## Design Rules

- Keep text short. Poster readers should understand the main result in 30 seconds.
- Use visuals more than paragraphs.
- Keep section titles direct:
  - Problem
  - Dataset
  - Method
  - Guiding Questions
  - Results
  - Limitations
- Use green for normal and red/orange for anomalous examples.
- Avoid claiming general industrial deployment. Say "controlled pilot dataset" and "promising result."
- Avoid too much implementation detail. Put code names only where they support the story.

## Suggested Poster Build Order

1. Decide A1 or A2 size.
2. Duplicate last semester's poster and keep its style.
3. Replace title, authors, course info, and advisor if needed.
4. Remove old one-shot learning, LSTM, CLIP+CoOp sections.
5. Add the new three-column structure.
6. Add dataset images first.
7. Add pipeline diagram second.
8. Add results chart third.
9. Add short text blocks.
10. Check readability at full poster size.
11. Export PDF and review it visually.

## Open Decisions

- Poster size: A1 portrait or A2 portrait
- Whether to include QR code for demo/trailer/repository
- Whether to show latency chart or keep only accuracy chart
- Whether the final title should use "Production/Assembly Lines" or "Industrial Conveyor Lines"
- Whether to mention `thinking_budget=2000` in the main results or only in a small note
