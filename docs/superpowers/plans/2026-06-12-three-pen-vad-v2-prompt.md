# Three-Pen VAD V2 Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a V2 learner prompt that uses per-pen attribute extraction so combined anomalies can be represented without increasing the guiding-question count.

**Architecture:** Keep the existing V1 prompt as the baseline. Add V2 prompt builders in `three_pen_vad/prompts.py`, then add a `--prompt-version v1/v2` option in `three_pen_vad/run_learner.py` so both prompts can be compared on the same videos and saved run files.

**Tech Stack:** Python 3.10, DashScope/Qwen API wrapper, `unittest`, JSON run artifacts.

---

### Task 1: Add V2 Prompt Tests

**Files:**
- Modify: `tests/test_three_pen_vad.py`

- [ ] **Step 1: Write failing tests**

Add tests that require:
- `build_learner_prompt(prompt_version="v1")` keeps the current V1 schema.
- `build_learner_prompt(prompt_version="v2")` asks for five guiding questions.
- V2 asks for per-pen fields: `cap_marker`, `cap_end`, `writing_tip_exposed`, `orientation`.
- V2 asks for `detected_anomalies` and `primary_verdict`.
- Expected labels are still absent from the prompt.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
python -m unittest tests.test_three_pen_vad.ThreePenVadTests.test_learner_prompt_v2_uses_per_pen_attribute_extraction
```

Expected: failure because `prompt_version` and the V2 prompt do not exist yet.

### Task 2: Implement Prompt Versioning

**Files:**
- Modify: `three_pen_vad/prompts.py`
- Modify: `three_pen_vad/run_learner.py`

- [ ] **Step 1: Add V2 constants**

Add `LEARNER_SYSTEM_CONTEXT_V2` and `GUIDING_QUESTIONS_V2` in `prompts.py`.

- [ ] **Step 2: Add prompt builder support**

Update `build_learner_prompt(expected=None, prompt_version="v1")` so V1 remains unchanged and V2 returns the per-pen extraction prompt.

- [ ] **Step 3: Add runner support**

Update `run_learner(..., prompt_version="v1")` and CLI `--prompt-version {v1,v2}`. Save `prompt_version` and the selected questions in the output JSON.

- [ ] **Step 4: Verify tests pass**

Run:

```bash
python -m unittest tests/test_three_pen_vad.py
```

Expected: all tests pass.

### Task 3: Update Documentation

**Files:**
- Modify: `three_pen_vad/START_HERE.md`
- Modify: `three_pen_vad/README.md`

- [ ] **Step 1: Document the V1 limitation**

Add the combined-video result: V1 correctly predicted `MISSING_CAP left_first` but missed `WRONG_ORIENTATION right_third`.

- [ ] **Step 2: Document the V2 idea**

Explain that V2 reduces the guiding-question count from 8 to 5 and changes the task from global yes/no questions to per-pen attribute extraction.

- [ ] **Step 3: Document comparison commands**

Add example WSL commands for running V2 on:
- `combined_01_missing_cap_left_wrong_orientation_right_4k_20fps_trim2s.mp4`
- `normal_01_4k_20fps_trim2s.mp4`
- `wrong_orientation_02_right_pen_4k_20fps_trim2s.mp4`

### Task 4: Manual Evaluation Handoff

**Files:**
- No code changes.

- [ ] **Step 1: Provide WSL commands**

Give the user commands that run V2 into `three_pen_vad/runs/06_learner_v2_per_pen/`.

- [ ] **Step 2: Compare outputs**

After the user runs the commands, compare:
- detected anomalies
- primary verdict
- elapsed seconds
- input/output/total/video tokens
- V1 versus V2 evidence quality
