# Week 12 Progress Notes

## Project Goal

The main goal is to detect and explain anomalies in industrial production and
assembly line videos using Vision-Language Models (VLMs).

Instead of training a task-specific computer vision model from scratch, we
describe the normal and abnormal production-line situations with text prompts.
Then we ask a VLM to inspect the video and explain whether the observed process
is normal or anomalous.

Current priority:

- Accuracy is more important than speed.
- We want interpretable model outputs, not only a final label.
- We want to compare different prompt strategies, input formats, and VLM models.

## Current Branch / Workflow Plan

Current working branch on Lightning:

```bash
alpers_test_api
```

Planned Git workflow from WSL:

```bash
git switch -c yetb
git add WEEK12.md
git commit -m "Add week 12 experiment notes"
git push -u origin yetb
```

Note: The push will be done later from WSL.

## Current Environment

Experiments are being run on Lightning AI.

Current machine:

- GPU: Tesla T4
- CUDA: Available
- Python: 3.12.11 in Lightning default conda environment
- Local model pipeline: Hugging Face Transformers

Important environment note:

Lightning Studio does not allow creating an extra virtual environment inside the
Studio. We must use the default conda environment.

Issue encountered:

```text
python -m venv grad2_env
Error: Venv creation is not allowed.
```

Resolution:

Use the existing Lightning environment and install required packages directly.

Another issue encountered:

`requirements.txt` installed `numpy==2.2.6`, which broke compatibility with
Lightning's existing SciPy / scikit-learn stack.

Error summary:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
ImportError: numpy.core.multiarray failed to import
```

Resolution:

```bash
pip install "numpy<2" --force-reinstall
```

After this, the environment imported the required packages correctly.

## Code Structure Under Investigation

The current local VLM experiment code is under:

```text
qwen_lighthing/
```

Files:

```text
qwen_lighthing/
├── personas_specialized.py
├── personas_complete.py
└── test_with_persons.py
```

### `personas_specialized.py`

Purpose:

Defines 20 different expert personas for production-line inspection. Each
persona uses a different professional perspective and vocabulary.

Examples:

- New line operator
- Mechanical engineer
- Computer vision engineer
- Toyota / Jidoka expert
- Hybrid checklist + intuition persona

The file selects 5 personas for voting:

```python
VOTING_SUBSET_IDS = ["P02", "P05", "P08", "P15", "P20"]
```

Interpretation:

This is a synthetic expert ensemble. The same video is shown to the same model
multiple times, but with different expert prompts. The final decision is made by
majority vote.

### `personas_complete.py`

Purpose:

Defines a more controlled and more explainable version of the persona prompts.

Main improvement over `personas_specialized.py`:

- Every persona uses the same full A/B/C checklist.
- Every persona must first describe what it sees in the frames.
- The model must reason before giving the final verdict.

Shared checklist:

```text
A. Orientation
B. Motion
C. Belt position
```

Important prompt idea:

```text
First describe entry/middle/exit observations.
Then decide orientation, motion, and belt position.
Then give the final verdict.
```

This is called an articulate-first prompt strategy.

Reason for this strategy:

Compact YES/NO prompts may cause false negatives because the model can jump to
the default answer without carefully inspecting the video.

### `test_with_persons.py`

Purpose:

Main inference orchestrator.

Pipeline:

```text
video input
-> choose persona mode: specialized or complete
-> load Qwen3-VL model
-> run one inference per voter/persona
-> parse each response into NORMAL / ANOMALY / UNCLEAR
-> majority vote
-> save JSON report
```

Default local model:

```python
Qwen/Qwen3-VL-8B-Instruct
```

This script is not a DashScope API script. It runs the model locally through
Hugging Face Transformers.

## Dataset Videos Currently Available

Known sample videos:

```text
data/r01_clip09_normal.mp4
data/r01_clip06_angle_anomaly.mp4
```

Other available sample videos:

```text
data/ornek_video.mp4
data/ornek_video_25fps.mp4
data/ornek_video_50fps.mp4
data/ornek_video_720p_25fps.mp4
```

Known image samples:

```text
data/036.jpg
data/057.jpg
data/078.jpg
data/145.jpg
```

## Experiment Axes

We plan to compare methods along four axes.

### 1. Input Format

Direct video input:

- Give the video directly to the VLM.
- Advantage: preserves motion information.
- Disadvantage: slower, more memory-heavy, metadata handling may be fragile.

Frame-grid input:

- Extract selected frames from the video.
- Arrange them in chronological order in a single grid image.
- Advantage: easier to inspect, easier to explain in presentation, often faster.
- Disadvantage: motion is only indirectly represented.

Frame-by-frame input:

- Analyze frames individually.
- Aggregate frame-level observations into a video-level decision.
- Advantage: most controllable and explainable.
- Disadvantage: may lose temporal continuity unless aggregation is carefully designed.

This is important because the advisor suggested testing frame-based prompting
instead of giving the whole video directly to the model.

### 2. Prompt Strategy

Prompt strategies to compare:

- Single direct prompt
- Specialized persona voting
- Complete checklist persona voting
- With consensus prompt
- Without consensus prompt

Current first baseline:

```text
specialized personas + no consensus
```

### 3. Model

Models to test:

- Local: `Qwen/Qwen3-VL-8B-Instruct`
- DashScope API: Qwen 3.5
- DashScope API: Qwen 3.6

Qwen 3.5 and Qwen 3.6 are especially important because the main goal is
accuracy, and API cost is acceptable for this stage.

### 4. Data

Minimum test set:

- One normal video
- One anomaly video

Next step:

- Add more process types
- Add more anomaly types if available
- Test whether prompts need to be process-specific

Important assumption:

Different production processes will need different prompt descriptions, but the
general method can stay the same.

## Completed Experiments

### Experiment 1: Local Qwen3-VL, Specialized Personas, Normal Video

Command:

```bash
python qwen_lighthing/test_with_persons.py \
  data/r01_clip09_normal.mp4 \
  --personas specialized \
  --no-consensus \
  --report reports/normal_specialized_no_consensus.json
```

Configuration:

```text
Model: Qwen/Qwen3-VL-8B-Instruct
Input type: Direct video
Prompt type: Specialized personas
Consensus: Disabled
Number of voters: 5
Expected label: NORMAL
```

Result:

```text
Final verdict: NORMAL
Confidence: HIGH (80%)
Vote breakdown: {'NORMAL': 4, 'ANOMALY': 1}
```

Voter details:

```text
P02 -> NORMAL
P05 -> NORMAL
P08 -> NORMAL
P15 -> NORMAL
P20 -> ANOMALY
```

Interpretation:

The final prediction is correct. The system classified the normal video as
normal.

Important observation:

P20 produced an inconsistent anomaly vote.

P20 response summary:

```text
CHECKLIST RESULT: N, Y, Y, Y, Y
INTUITION RESULT: Yes
AGREEMENT: NO
RESULT: ANOMALY DETECTED
```

Possible reason:

P20 may have interpreted partial visibility in entry/exit frames as a failed
check, even though partial visibility should not be considered an anomaly.

This is a motivation to test `personas_complete.py`, because complete prompts
explicitly state that partial visibility at entry/exit frames should not be
flagged as anomaly.

Runtime:

```text
P02: 278.3s
P05: 213.9s
P08: 122.1s
P15: 131.0s
P20: 107.7s
```

Approximate total inference time:

```text
14 minutes
```

Performance note:

The model partially offloaded to CPU:

```text
Some parameters are on the meta device because they were offloaded to the cpu.
```

This likely explains the slow runtime on T4.

### Experiment 10: Qwen3.6-Plus Description-Only Sanity Check

Status:

```text
Completed.
```

Script:

```bash
python vera_lite/test_qwen3.6.py \
  --video data/r01_clip09_normal.mp4 \
  --model qwen3.6-plus
```

and:

```bash
python vera_lite/test_qwen3.6.py \
  --video data/r01_clip06_angle_anomaly.mp4 \
  --model qwen3.6-plus
```

Purpose:

Before asking the model to classify normal/anomaly, we asked it to describe what
it visually observes. This was done to understand whether the model can see the
actual discriminative details in the videos.

Normal video description summary:

```text
- Component enters from the left and moves to the right.
- It is standing/upright.
- It moves smoothly.
- It may show mild rotation, diagonal appearance, or perspective tilt.
- It stays on the belt.
```

Anomaly video description summary:

```text
- Component enters from the left and moves to the right.
- It appears to be lying on its side.
- It shows slight tumbling/wobbling or unstable posture.
- It stays on the belt and continues moving.
```

Key finding:

```text
The key difference between the current normal and anomaly videos is not simply
"diagonal vs non-diagonal". The normal video can also look slightly diagonal or
rotated due to perspective. The more reliable difference is:

NORMAL  -> upright/stable posture
ANOMALY -> lying on side / unstable posture / tumbling-wobbling
```

This changed our prompt design.

### Experiment 11: VERA-lite Qwen3.6-Plus Learner After Prompt Refinement

Status:

```text
Completed.
```

Folder:

```text
vera_lite/
```

Model:

```text
qwen3.6-plus
```

Input type:

```text
Direct video
```

Prompt strategy:

```text
VERA-inspired guiding questions, manually refined after model description.
```

What changed in the prompt:

Old focus:

```text
- red cap right / pin side left
- component roughly parallel to belt direction
- diagonal/perpendicular posture as anomaly
```

Problem:

```text
The normal video also appears mildly diagonal/rotated due to camera perspective,
so the model could falsely flag normal motion as anomaly.
```

New focus:

```text
- component should remain upright/stable like the normal reference
- mild perspective tilt or mild rotation is not anomaly
- anomaly is lying on side, tumbling, strong wobbling, rolling, or unstable posture
- component should still move smoothly left-to-right and stay on the belt
```

Updated guiding questions:

```text
1. In fully visible frames, does the component keep the normal upright/stable posture instead of lying on its side?
2. Does the component avoid tumbling, strong wobbling, rolling, or unstable rotation while moving?
3. Are the red cap and black pin side visible as opposite ends of the same component, with no clear reversed-orientation evidence?
4. Does the component move smoothly from LEFT to RIGHT across the video?
5. Does the component stay fully on the belt surface without overhanging, falling, or leaving the belt?
```

Normal video command:

```bash
python -m vera_lite.run_learner \
  --video data/r01_clip09_normal.mp4 \
  --expected NORMAL \
  --model qwen3.6-plus
```

Normal video result:

```text
Expected: NORMAL
Prediction: NORMAL
Anomaly type: none
Anomaly score: 0.0
Confidence: HIGH
```

Reason summary:

```text
The component travels smoothly from left to right in a stable, upright position
with no signs of instability or orientation errors.
```

Anomaly video command:

```bash
python -m vera_lite.run_learner \
  --video data/r01_clip06_angle_anomaly.mp4 \
  --expected ANOMALY \
  --model qwen3.6-plus
```

Anomaly video result:

```text
Expected: ANOMALY
Prediction: ANOMALY
Anomaly type: posture_alignment_failure
Anomaly score: 0.9
Confidence: HIGH
```

Reason summary:

```text
The component is lying on its side with the red button facing sideways instead
of upwards, which is a clear posture/alignment anomaly compared with the normal
upright reference.
```

Interpretation:

```text
Prompt refinement based on model description fixed the classification behavior
on the current two videos:

normal video        -> NORMAL
angle anomaly video -> ANOMALY
```

This supports the VERA-inspired idea that better guiding questions can elicit
better reasoning from a frozen VLM without model training.

### Experiment 12: VERA-lite Full Iteration with Optimizer

Status:

```text
Completed.
```

Command:

```bash
python -m vera_lite.run_iteration --model qwen3.6-plus
```

Run folder:

```text
vera_lite/runs/20260507_214236/
```

Generated files:

```text
normal_01_learner.json
angle_anomaly_01_learner.json
learner_results.json
optimizer_result.json
candidate_guiding_questions.json
summary.json
```

Learner results:

```text
normal_01:
  Expected: NORMAL
  Prediction: NORMAL
  Type: none
  Score: 0.0
  Confidence: HIGH

angle_anomaly_01:
  Expected: ANOMALY
  Prediction: ANOMALY
  Type: posture_alignment_failure
  Score: 0.9
  Confidence: HIGH
```

Optimizer finding:

```text
The previous questions worked, but one question had a loophole:
"red cap and pin side are visible as opposite ends" can still be answered YES
even when the component is sideways, because both parts are technically visible.
```

Optimizer proposed:

```text
- Separate static posture/alignment from dynamic motion.
- Make sideways / 90-degree rotation / flipped posture explicit.
- Keep left-to-right translation and belt containment checks.
```

Decision:

We did not directly replace the questions with the optimizer output because one
phrase, "base flat against the belt", may be physically ambiguous for our
component. Instead, we manually created a safer v1 question set based on:

```text
1. Qwen3.6 description-only observations
2. successful learner outputs
3. optimizer critique
```

Current v1 guiding questions:

```text
1. In fully visible frames, does the component maintain the normal upright/stable posture, with the red cap facing upward or visible on top rather than sideways?
2. Is the component free from side-lying posture, falling, tumbling, rolling, or strong unstable wobbling?
3. Are the red cap and pin side in a standard upright relationship, with no clear 90-degree sideways rotation or flipped posture compared with the normal reference?
4. Does the component translate smoothly from LEFT to RIGHT without stopping, moving backward, or drifting erratically?
5. Does the component remain fully on the belt surface, without overhanging, falling, or leaving the conveyor?
```

Current model decision:

```text
Continue with qwen3.6-plus.
Qwen3.5 comparison is not necessary right now because qwen3.6-plus gives good
and explainable results after prompt refinement.
```

Next data plan:

The current pair of videos is not enough to finalize the prompt. Next, add more
videos from the same process:

```text
- more normal examples
- more posture/alignment anomaly examples
- possibly reversed orientation, stop/jam, and belt-edge examples if available
```

After adding more examples:

```text
1. Update vera_lite/dataset_manifest.json with each video and expected label.
2. Run qwen3.6-plus learner/iteration on the expanded set.
3. Inspect mistakes and weak explanations.
4. Use optimizer to propose improved guiding questions.
5. Manually accept or edit optimizer suggestions.
6. Repeat until the prompt handles all available examples consistently.
```

## Experiments In Progress

### Experiment 2: Local Qwen3-VL, Specialized Personas, Anomaly Video

Command:

```bash
python qwen_lighthing/test_with_persons.py \
  data/r01_clip06_angle_anomaly.mp4 \
  --personas specialized \
  --no-consensus \
  --report reports/anomaly_specialized_no_consensus.json
```

Configuration:

```text
Model: Qwen/Qwen3-VL-8B-Instruct
Input type: Direct video
Prompt type: Specialized personas
Consensus: Disabled
Number of voters: 5
Expected label: ANOMALY
```

Status:

```text
Completed.
```

Result:

```text
Final verdict: NORMAL
Confidence: MEDIUM (60%)
Vote breakdown: {'NORMAL': 3, 'ANOMALY': 2}
```

Voter details:

```text
P02 -> NORMAL
P05 -> ANOMALY
P08 -> NORMAL
P15 -> NORMAL
P20 -> ANOMALY
```

Interpretation:

```text
This is a false negative under the expected label ANOMALY.
```

However, after reviewing the video direction, we found that the initial prompt
was incorrect: the real belt direction is LEFT -> RIGHT, but the specialized
prompts described it as RIGHT -> LEFT. Therefore this result should be treated
as a preliminary baseline with a prompt-direction mismatch, not as a final model
failure.

Important observation:

P05 detected anomaly and specifically mentioned alignment/geometric problems.
This is useful because the anomaly clip name suggests an angle anomaly:

```text
r01_clip06_angle_anomaly.mp4
```

This motivated adding an explicit angle/alignment anomaly rule to the prompts.

P20 again produced an inconsistent output pattern, similar to E1.

Runtime:

```text
P02: 272.3s
P05: 254.6s
P08: 162.0s
P15: 134.9s
P20: 110.6s
```

Approximate total inference time:

```text
15.6 minutes
```

## Planned Experiments

### Experiment Track: VERA-lite Guiding Question Optimization

Status:

```text
Started.
```

Folder:

```text
vera_lite/
```

Reason for this track:

The local Qwen3-VL direct-video multi-persona pipeline was too slow on
Lightning T4 and caused CUDA out-of-memory during complete persona experiments.
After reading the VERA paper, we decided to keep the most relevant idea:
iteratively improving guiding questions instead of training model weights.

Current design:

```text
video + current guiding questions
-> learner VLM predicts NORMAL / ANOMALY and explains
-> optimizer VLM reviews expected labels and learner mistakes
-> optimizer proposes better guiding questions
```

Current default model:

```text
qwen3.6-plus
```

Comparison model:

```text
qwen3.5-plus
```

Deferred ideas:

```text
temporal grid
Gaussian smoothing
frame-level dense scoring
scene retrieval / similar segment ensemble
```

Reason for deferring temporal grid:

The current videos have low resolution, and putting frames into a grid would
make small details such as pin side, red cap, and angle/alignment even harder
to inspect. Temporal grid will be revisited after collecting higher-resolution
videos.

Initial guiding questions:

```text
1. In fully visible frames, is the red cap on the RIGHT side of the component?
2. In fully visible frames, is the black pin side on the LEFT side of the component?
3. Is the component's long axis roughly parallel to the conveyor belt movement direction, rather than clearly diagonal or perpendicular?
4. Does the component move smoothly from LEFT to RIGHT across the video?
5. Does the component stay fully on the belt surface without overhanging, falling, or leaving the belt?
```

### Experiment 3: Local Qwen3-VL, Complete Personas, Normal Video

Command:

```bash
python qwen_lighthing/test_with_persons.py \
  data/r01_clip09_normal.mp4 \
  --personas complete \
  --no-consensus \
  --report reports/normal_complete_no_consensus.json
```

Expected label:

```text
NORMAL
```

Purpose:

Check whether complete checklist prompts reduce inconsistent voter behavior.

### Experiment 4: Local Qwen3-VL, Complete Personas, Anomaly Video

Command:

```bash
python qwen_lighthing/test_with_persons.py \
  data/r01_clip06_angle_anomaly.mp4 \
  --personas complete \
  --no-consensus \
  --report reports/anomaly_complete_no_consensus.json
```

Expected label:

```text
ANOMALY
```

Purpose:

Evaluate the articulate-first complete checklist prompt strategy on anomaly
data.

### Experiment 5: Consensus Prompt Ablation

Command:

```bash
python qwen_lighthing/test_with_persons.py \
  data/r01_clip06_angle_anomaly.mp4 \
  --personas complete \
  --report reports/anomaly_complete_with_consensus.json
```

Purpose:

Compare:

```text
5 voters without consensus
vs
5 voters + 1 consensus voter
```

Question:

Does the consensus prompt improve final accuracy and reduce uncertainty?

### Experiment 6: DashScope Qwen 3.5

Status:

```text
Planned.
```

Purpose:

Test whether Qwen 3.5 API gives better anomaly detection and explanation
quality than local Qwen3-VL-8B.

Important:

Use the same videos and similar prompt structure to make the comparison fair.

### Experiment 7: DashScope Qwen 3.6

Status:

```text
Planned.
```

Purpose:

Test whether Qwen 3.6 improves accuracy over Qwen 3.5 and local Qwen3-VL-8B.

Accuracy is more important than cost and speed at this stage.

### Experiment 8: Frame-Grid Strategy

Status:

```text
Planned.
```

Idea:

Extract several frames from the video and combine them into a chronological
grid image.

Possible frame choices:

```text
4 frames: entry, early-middle, late-middle, exit
8 frames: denser temporal coverage
```

Prompt should explain:

```text
The grid is chronological.
Top-left is earliest.
Bottom-right is latest.
Analyze the object across frames before giving a verdict.
```

Purpose:

Test advisor's suggestion: instead of giving the whole video to the prompt,
give selected frames and ask the model to reason over them.

Expected advantages:

- More controlled input
- More explainable reasoning
- Possibly faster than direct video input
- Easier to show in presentation

Possible limitation:

- Motion anomalies may be harder to detect compared with direct video input.

### Experiment 9: Frame-by-Frame Strategy

Status:

```text
Planned.
```

Idea:

Analyze individual frames separately and then aggregate the observations.

Possible pipeline:

```text
video
-> extract frames
-> run VLM on each frame
-> collect observations
-> aggregate into final video-level verdict
```

Possible aggregation approaches:

- Rule-based aggregation
- Majority vote over frame-level labels
- Final VLM prompt that receives all frame observations and decides the video label

Question:

Does frame-by-frame reasoning improve accuracy and explainability compared with
direct video prompting?

## Result Table

| ID | Model | Input Type | Prompt Strategy | Video | Expected | Prediction | Confidence | Notes |
|---|---|---|---|---|---|---|---|---|
| E1 | Qwen3-VL-8B local | Direct video | Specialized, no consensus | `r01_clip09_normal.mp4` | NORMAL | NORMAL | 80% | Correct, but prompt had wrong belt direction. P20 voted anomaly. |
| E2 | Qwen3-VL-8B local | Direct video | Specialized, no consensus | `r01_clip06_angle_anomaly.mp4` | ANOMALY | NORMAL | 60% | False negative under old prompt. Direction mismatch found afterward. |
| E3 | Qwen3-VL-8B local | Direct video | Complete, no consensus | `r01_clip09_normal.mp4` | NORMAL | TODO | TODO | Planned. |
| E4 | Qwen3-VL-8B local | Direct video | Complete, no consensus | `r01_clip06_angle_anomaly.mp4` | ANOMALY | TODO | TODO | Planned. |
| E5 | Qwen3-VL-8B local | Direct video | Complete, with consensus | `r01_clip06_angle_anomaly.mp4` | ANOMALY | TODO | TODO | Planned. |
| E6 | Qwen 3.5 API | TBD | TBD | TBD | TBD | TODO | TODO | Planned. |
| E7 | Qwen 3.6 API | TBD | TBD | TBD | TBD | TODO | TODO | Planned. |
| E8 | TBD | Frame grid | TBD | TBD | TBD | TODO | TODO | Planned. |
| E9 | TBD | Frame-by-frame | TBD | TBD | TBD | TODO | TODO | Planned. |
| E10 | Qwen3.6-Plus API | Direct video | Description-only sanity check | `r01_clip09_normal.mp4` | N/A | N/A | N/A | Model described normal as upright/stable with mild rotation/perspective tilt. |
| E11 | Qwen3.6-Plus API | Direct video | Description-only sanity check | `r01_clip06_angle_anomaly.mp4` | N/A | N/A | N/A | Model described anomaly as lying on side / tumbling-wobbling. |
| E12 | Qwen3.6-Plus API | Direct video | VERA-lite refined guiding questions | `r01_clip09_normal.mp4` | NORMAL | NORMAL | HIGH | Correct after prompt refinement. Score 0.0. |
| E13 | Qwen3.6-Plus API | Direct video | VERA-lite refined guiding questions | `r01_clip06_angle_anomaly.mp4` | ANOMALY | ANOMALY | HIGH | Correct after prompt refinement. Type posture_alignment_failure, score 0.9. |
| E14 | Qwen3.6-Plus API | Direct video | VERA-lite full iteration + optimizer | manifest pair | NORMAL/ANOMALY | both correct | HIGH | Run saved under `vera_lite/runs/20260507_214236`; optimizer proposed sharper posture/motion questions. |

## Current Findings

1. The local Qwen3-VL pipeline runs successfully on Lightning T4.

2. The first normal-video experiment produced the correct final result.

3. Specialized persona voting can still contain inconsistent voters.

4. `personas_complete.py` is likely worth testing because it gives every voter a
   shared checklist and forces frame-level explanation before final decision.

5. Direct video input is slow on T4 because the 8B model appears to offload some
   parameters to CPU.

6. The advisor's frame-based suggestion is promising because it can improve
   controllability and explainability.

7. Qwen 3.5 and Qwen 3.6 through DashScope API should be tested because accuracy
   is the main priority and API cost is acceptable.

8. Important correction discovered after E1/E2:

   The real conveyor belt direction in the current videos is LEFT -> RIGHT.
   The initial specialized prompts incorrectly described the belt direction as
   RIGHT -> LEFT. Therefore E1 and E2 should be treated as preliminary baseline
   results with a prompt-direction mismatch.

9. The anomaly video is named `r01_clip06_angle_anomaly.mp4`, so the prompt must
   explicitly include angle/alignment anomaly, not only reversed orientation,
   stopped motion, and belt-edge failures.

Prompt correction applied after E2:

```text
Old prompt assumption:
  Belt direction = RIGHT -> LEFT

Correct prompt assumption:
  Belt direction = LEFT -> RIGHT
  Correct orientation = RED CAP on RIGHT, PIN SIDE on LEFT
  As the button travels right, RED CAP leads and PIN SIDE trails.

Added anomaly criterion:
  Angle/alignment anomaly = button is significantly diagonal or perpendicular
  instead of roughly parallel to the belt movement direction.
```

## Open Questions

1. Does the local Qwen3-VL model correctly detect the anomaly video after the
   corrected LEFT -> RIGHT prompt and angle/alignment criterion?

2. Does `complete` prompting improve consistency over `specialized` prompting?

3. Does adding the consensus prompt improve or hurt final decisions?

4. Does frame-grid input perform better than direct video input?

5. Does frame-by-frame analysis improve explainability?

6. How do Qwen 3.5 and Qwen 3.6 compare against local Qwen3-VL-8B?

7. Are errors caused more by model weakness, prompt weakness, or input format?

8. How process-specific do the prompts need to be?

## Near-Term Action Plan

1. Re-run specialized no-consensus experiments after prompt correction.

2. Run complete persona experiments on normal and anomaly videos after prompt
   correction.

3. Compare specialized vs complete prompting.

4. Decide whether consensus should be enabled in later experiments.

5. Design DashScope Qwen 3.5 / 3.6 experiments.

6. Design frame-grid input strategy.

7. Design frame-by-frame input strategy.

8. Review recent VLM-based video anomaly detection papers and extract method
   ideas that can be adapted to this project.

9. Update this file after every experiment.

## Literature Review Direction

The next literature review should focus on:

- VLMs for video anomaly detection
- Zero-shot anomaly detection with multimodal models
- Prompt engineering for industrial inspection
- Frame-based vs video-based VLM reasoning
- Multi-agent or multi-prompt VLM ensembles
- Chain-of-thought / reasoning-first prompting for visual inspection

Possible method ideas to look for:

- Temporal frame sampling
- Multi-frame grid prompting
- Caption-then-classify pipelines
- VLM-generated pseudo-labels
- Expert prompt ensembles
- Self-consistency voting
- Rule-based aggregation of VLM observations

## Notes For Presentation

Possible explanation:

```text
Our current method uses a multi-prompt VLM ensemble. The same production-line
video is given to the model multiple times with different expert personas.
Each persona produces a normal/anomaly decision and an explanation. The final
decision is obtained by majority voting.
```

Important talking point:

```text
Instead of training a new model for each product, we describe the inspection
rules in natural language. This makes the method more flexible for different
industrial processes.
```

Advisor-related talking point:

```text
We are also planning a frame-based strategy, where selected frames are passed to
the VLM instead of the full video. This may improve control and explainability,
especially for orientation-based anomalies.
```

## Commands Reference

Environment repair after NumPy conflict:

```bash
pip install "numpy<2" --force-reinstall
```

Normal video, specialized, no consensus:

```bash
python qwen_lighthing/test_with_persons.py \
  data/r01_clip09_normal.mp4 \
  --personas specialized \
  --no-consensus \
  --report reports/normal_specialized_no_consensus.json
```

Anomaly video, specialized, no consensus:

```bash
python qwen_lighthing/test_with_persons.py \
  data/r01_clip06_angle_anomaly.mp4 \
  --personas specialized \
  --no-consensus \
  --report reports/anomaly_specialized_no_consensus.json
```

Normal video, complete, no consensus:

```bash
python qwen_lighthing/test_with_persons.py \
  data/r01_clip09_normal.mp4 \
  --personas complete \
  --no-consensus \
  --report reports/normal_complete_no_consensus.json
```

Anomaly video, complete, no consensus:

```bash
python qwen_lighthing/test_with_persons.py \
  data/r01_clip06_angle_anomaly.mp4 \
  --personas complete \
  --no-consensus \
  --report reports/anomaly_complete_no_consensus.json
```

Anomaly video, complete, with consensus:

```bash
python qwen_lighthing/test_with_persons.py \
  data/r01_clip06_angle_anomaly.mp4 \
  --personas complete \
  --report reports/anomaly_complete_with_consensus.json
```
