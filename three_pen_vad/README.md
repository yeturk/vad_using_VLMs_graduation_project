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
