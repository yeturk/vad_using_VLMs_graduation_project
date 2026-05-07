# VERA-lite for Industrial VAD

This folder contains a lightweight, VERA-inspired workflow for video anomaly
detection with DashScope Qwen VLMs.

The goal is not to train model weights. Instead, we iteratively improve the
guiding questions used in the prompt.

## Method

```text
video + guiding questions
-> learner VLM predicts NORMAL / ANOMALY and explains
-> optimizer VLM reviews expected labels and learner mistakes
-> optimizer proposes improved guiding questions
```

Current default model:

```text
qwen3.6-plus
```

Optional comparison model, not the current priority:

```text
qwen3.5-plus
```

## Files

```text
dataset_manifest.json  Small labeled experiment set
guiding_questions.json Current prompt questions
prompts.py             Learner and optimizer prompt templates
dashscope_client.py    Shared DashScope API helpers
run_learner.py         Run one video with current questions
run_optimizer.py       Improve questions from learner results
run_iteration.py       Run learner on all manifest items, then optimizer
```

## Usage

Make sure `.env` contains:

```text
DASHSCOPE_API_KEY=...
```

Run a single learner call:

```bash
python -m vera_lite.run_learner \
  --video data/r01_clip09_normal.mp4 \
  --expected NORMAL \
  --model qwen3.6-plus
```

Run one full VERA-lite iteration:

```bash
python -m vera_lite.run_iteration --model qwen3.6-plus
```

For now, we continue with `qwen3.6-plus` because the latest prompt refinement
gave correct and explainable results on the current normal/anomaly pair.

## Notes

- This first version uses direct video input.
- Temporal grid is intentionally postponed because the current videos have low
  resolution and grid composition may hide small details.
- Frame-level dense scoring and Gaussian smoothing are future extensions.
