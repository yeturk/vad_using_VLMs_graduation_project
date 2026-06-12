# Three Pen VAD Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first isolated workflow for preprocessing and observation-only Qwen descriptions on the three-pen conveyor pilot videos.

**Architecture:** Keep the new three-pen workflow separate from the older push-button `vera_lite` experiment. Use small Python modules for preprocessing, prompt text, and description-only model calls. Store large generated videos under ignored `data/grad2_pilot/processed/` paths.

**Tech Stack:** Python 3.10, OpenCV, DashScope Qwen via the existing `vera_lite.dashscope_client`, `unittest`.

---

### Task 1: Add Tests for Pilot Helpers

**Files:**
- Create: `tests/test_three_pen_vad.py`

- [x] **Step 1: Write tests for processed file naming, trim bounds, description prompt, and manifest IDs**

Run:

```bash
python -m unittest tests/test_three_pen_vad.py
```

Expected before implementation: fail because `three_pen_vad` does not exist.

### Task 2: Add Three-Pen Package

**Files:**
- Create: `three_pen_vad/__init__.py`
- Create: `three_pen_vad/README.md`
- Create: `three_pen_vad/dataset_manifest.json`
- Create: `three_pen_vad/prompts.py`
- Create: `three_pen_vad/preprocess_video.py`
- Create: `three_pen_vad/describe_video.py`

- [x] **Step 1: Implement the minimum package needed by the tests**

Run:

```bash
python -m unittest tests/test_three_pen_vad.py
```

Expected after implementation: all tests pass.

### Task 3: Verify Scripts

**Files:**
- Read: `three_pen_vad/preprocess_video.py`
- Read: `three_pen_vad/describe_video.py`

- [ ] **Step 1: Run unit tests**

```bash
python -m unittest tests/test_three_pen_vad.py
```

- [ ] **Step 2: Run module help for CLI sanity**

```bash
python -m three_pen_vad.preprocess_video --help
python -m three_pen_vad.describe_video --help
```

Expected: both commands print usage without importing errors.
