# Three Pen VAD

This folder contains the new VERA-lite style workflow for the real conveyor
videos with three pens.

Start with:

```text
three_pen_vad/START_HERE.md
```

The initial scope is intentionally small:

- preprocess the five pilot videos
- ask Qwen for observation-only video descriptions
- use those descriptions to design the first normality prompt
- compare V1 global-question prompting with V2 per-pen attribute extraction

The old `vera_lite/` folder remains the push-button/IPAD experiment archive.

## Pilot Data

Raw videos:

```text
data/grad2_pilot/videos/
```

Processed videos:

```text
data/grad2_pilot/processed/
```

Initial preprocessing target:

```text
keep original 4K resolution
convert to 20 FPS
trim 2 seconds from the start and 2 seconds from the end
```

## Commands

Preprocess all pilot videos:

```bash
conda activate grad2_env
python -m three_pen_vad.preprocess_video
```

Describe one processed video without classification:

```bash
conda activate grad2_env
python -m three_pen_vad.describe_video \
  --video data/grad2_pilot/processed/normal_01_4k_20fps_trim2s.mp4 \
  --out three_pen_vad/runs/manual_normal_01_description.txt
```

Run the V1 learner:

```bash
conda activate grad2_env
python -m three_pen_vad.run_learner \
  --prompt-version v1 \
  --video data/grad2_pilot/processed/normal_01_4k_20fps_trim2s.mp4 \
  --expected NORMAL \
  --out three_pen_vad/runs/manual_v1_normal_01_learner.json
```

Run the V2 learner with per-pen attribute extraction:

```bash
conda activate grad2_env
python -m three_pen_vad.run_learner \
  --prompt-version v2 \
  --video data/grad2_pilot/processed/combined_01_missing_cap_left_wrong_orientation_right_4k_20fps_trim2s.mp4 \
  --expected MISSING_CAP \
  --out three_pen_vad/runs/06_learner_v2_per_pen/manual_v2_combined_01_missing_cap_left_wrong_orientation_right_learner.json
```

V2 keeps one API call, but asks the model to first fill a per-pen observation
table and then report every detected anomaly in `detected_anomalies`.
