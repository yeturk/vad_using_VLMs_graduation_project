# VERA-Lite Experiment Summary

Date prepared: 2026-06-03

## Purpose

This file summarizes the latest VERA-inspired work in this repository before the
next push. The goal of the work is industrial video anomaly detection (VAD) with
vision-language models (VLMs), focused on a small conveyor-belt push-button
component scenario.

The current implementation is not a model-training pipeline. Instead, it adapts
the main idea from the VERA paper into a lightweight prompt-learning workflow:
guiding questions are treated as improvable verbal parameters, and a learner VLM
uses those questions to inspect videos, explain evidence, and predict whether a
video is normal or anomalous.

Reference paper used:

- `firstpaper.pdf`: "VERA: Explainable Video Anomaly Detection via Verbalized
  Learning of Vision-Language Models"

Additional related paper currently present but not tracked by Git:

- `secondpaper.pdf`: "Video-Based Traffic Anomaly Detection with
  Vision-Language Models: A Survey"

## VERA Idea Used In This Project

The VERA paper proposes explainable VAD with frozen VLMs by learning/refining
guiding questions instead of changing model weights. The paper's high-level
loop is:

```text
video + guiding questions
-> learner VLM describes/reasons and predicts anomaly status
-> optimizer VLM reviews results and proposes better guiding questions
-> learned questions are reused during inference
```

Our repository implements a small "VERA-lite" version of this idea under
`vera_lite/`. The current scope is intentionally narrow:

- no model fine-tuning
- no parameter updates
- no frame-level dense anomaly smoothing yet
- direct video input to the VLM
- a small labeled manifest with one normal and one anomaly sample
- JSON outputs saved for reproducibility

## Implemented Files

Main VERA-lite files:

- `vera_lite/README.md`: usage notes and method overview
- `vera_lite/dataset_manifest.json`: labeled experiment manifest
- `vera_lite/guiding_questions.json`: current refined guiding questions
- `vera_lite/prompts.py`: learner and optimizer prompt templates
- `vera_lite/dashscope_client.py`: DashScope/Qwen API helper functions
- `vera_lite/run_learner.py`: runs the learner VLM on one video
- `vera_lite/run_optimizer.py`: runs the optimizer VLM on learner results
- `vera_lite/run_iteration.py`: runs learner over the manifest, then optimizer

Saved run artifacts:

- `vera_lite/runs/20260507_214236/learner_results.json`
- `vera_lite/runs/20260507_214236/optimizer_result.json`
- `vera_lite/runs/20260507_214236/candidate_guiding_questions.json`
- `vera_lite/runs/20260507_214236/summary.json`
- per-video learner outputs in the same run folder

## Model And API Details

Current main model:

```text
qwen3.6-plus
```

The same model was used for both roles:

- learner model: `qwen3.6-plus`
- optimizer model: `qwen3.6-plus`

API provider:

```text
DashScope MultiModalConversation
```

Environment variable required:

```text
DASHSCOPE_API_KEY=...
```

Default DashScope base URL used by the helper:

```text
https://dashscope-intl.aliyuncs.com/api/v1
```

Older/parallel work in the repository also explored local Hugging Face
inference with:

```text
Qwen/Qwen3-VL-8B-Instruct
```

That local Lightning workflow is documented in `WEEK12.md` and lives mainly
under `qwen_lighthing/`. The current VERA-lite results summarized here use the
DashScope `qwen3.6-plus` API path.

## Dataset Used

The current labeled manifest contains two videos:

| ID | Video | Expected label | Anomaly type | Description |
| --- | --- | --- | --- | --- |
| `normal_01` | `data/r01_clip09_normal.mp4` | `NORMAL` | `none` | Normal push-button component moving left to right on the conveyor belt. |
| `angle_anomaly_01` | `data/r01_clip06_angle_anomaly.mp4` | `ANOMALY` | `angle_misalignment` | Push-button component with angle/alignment anomaly while moving on the conveyor belt. |

Important note: the videos are referenced in the manifest, but they are not
listed as tracked files in the current Git file list.

## Prompt Design

The learner prompt defines the industrial scene:

- small push-button component on a green conveyor belt
- conveyor motion is left to right
- component has a red cap and a black pin side
- normal condition means upright/stable posture, smooth motion, and staying on
  the belt
- anomaly conditions include posture/alignment failure, reversed orientation,
  motion failure, or belt-position failure

The learner is required to return strict JSON with:

- per-question `YES / NO / UNCLEAR` answers
- short visual evidence for each guiding question
- final `NORMAL / ANOMALY / UNCLEAR` verdict
- anomaly type
- anomaly score from `0.0` to `1.0`
- confidence
- short explanation

The optimizer reads the learner results and proposes better guiding questions.
It is instructed to keep questions visually checkable and relevant to the same
production-line setup.

## Current Guiding Questions

The current manually refined questions in `vera_lite/guiding_questions.json`
are:

1. In fully visible frames, does the component maintain the normal
   upright/stable posture, with the red cap facing upward or visible on top
   rather than sideways?
2. Is the component free from side-lying posture, falling, tumbling, rolling, or
   strong unstable wobbling?
3. Are the red cap and pin side in a standard upright relationship, with no
   clear 90-degree sideways rotation or flipped posture compared with the normal
   reference?
4. Does the component translate smoothly from LEFT to RIGHT without stopping,
   moving backward, or drifting erratically?
5. Does the component remain fully on the belt surface, without overhanging,
   falling, or leaving the conveyor?

## Run Summary

Run ID:

```text
20260507_214236
```

Run command pattern:

```bash
python -m vera_lite.run_iteration --model qwen3.6-plus
```

Summary file:

```text
vera_lite/runs/20260507_214236/summary.json
```

The run used:

- manifest: `vera_lite/dataset_manifest.json`
- questions: `vera_lite/guiding_questions.json`
- learner results: `vera_lite/runs/20260507_214236/learner_results.json`
- optimizer output: written successfully

## Results

| Video ID | Expected | Predicted | Score | Confidence | Main reason |
| --- | --- | --- | --- | --- | --- |
| `normal_01` | `NORMAL` | `NORMAL` | `0.0` | `HIGH` | Component moved smoothly left to right, remained stable/upright, and stayed fully on the belt. |
| `angle_anomaly_01` | `ANOMALY` | `ANOMALY` | `0.9` | `HIGH` | Component was lying on its side with pins up and the red cap on the side face, violating the normal upright posture. |

On the current two-video manifest:

```text
correct predictions: 2 / 2
normal sample: correctly classified
angle/posture anomaly sample: correctly classified
```

This is a very small validation set, so the result should be treated as an
initial sanity check rather than a final performance measurement.

## Token Usage From Saved Results

Learner call for `normal_01`:

- input tokens: `1328`
- video tokens: `578`
- output tokens: `1314`
- total tokens: `2642`

Learner call for `angle_anomaly_01`:

- input tokens: `1331`
- video tokens: `578`
- output tokens: `2545`
- total tokens: `3876`

Optimizer call:

- input tokens: `2255`
- output tokens: `3747`
- total tokens: `6002`

## Optimizer Findings

The optimizer found that the earlier guiding questions had some ambiguity:

- a question could be answered `YES` just because the cap and pins were visible,
  even when the component was sideways
- dynamic instability checks such as tumbling/wobbling did not fully cover a
  static side-lying posture
- terms like "normal posture" and "smoothly" need more spatial grounding

The optimizer proposed candidate questions that explicitly check:

- upright posture with base flat on the belt
- no sideways, tilted, inverted, 90-degree rotated, or flipped orientation
- consistent left-to-right translation
- full footprint containment inside the belt boundaries

After the optimizer output, `guiding_questions.json` was manually refined into
the current five-question version that keeps the same idea but adds details from
the observed normal/anomaly pair.

## Current Git Status Before This Summary

Before this Markdown file was created:

- current branch: `yetb`
- upstream branch: `origin/yetb`
- local branch and remote branch pointed to the same commit
- only untracked file was `secondpaper.pdf`

Last local commit:

```text
aeef4bf86a99566acb9f325eda833abf1090a88d
2026-05-07 22:33:30 +0300
Add VERA-lite Qwen3.6 anomaly detection workflow
```

Last push visible in the local remote-tracking reflog:

```text
2026-05-07 22:33:39 +0300
origin/yetb updated by push to aeef4bf86a99566acb9f325eda833abf1090a88d
```

Git itself does not store a canonical push timestamp in commits. The timestamp
above comes from the local reflog for `origin/yetb`, so it is the best local
evidence available on this machine.

## Recommended Next Steps

1. Decide whether `secondpaper.pdf` should be included in Git.
2. Add this summary file and optionally `secondpaper.pdf`.
3. Commit the documentation update.
4. Push branch `yetb` to `origin`.
5. Continue by adding more normal/anomaly videos to the manifest before claiming
   robust accuracy.

