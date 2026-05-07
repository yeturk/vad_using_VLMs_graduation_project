from __future__ import annotations


SYSTEM_CONTEXT = """
You are an industrial quality-control vision system.

Scene:
- A small push-button component moves on a green conveyor belt.
- The conveyor belt moves LEFT to RIGHT in the camera frame.
- The component has two distinct ends:
  - RED CAP: the red button cap.
  - PIN SIDE: black base with small metal pins.

Nominal condition:
- The component enters from the left and moves to the right.
- The component remains upright/stable on the belt, like the normal reference.
- In the current normal reference, the red cap appears upward/visible on top,
  while the black pin side remains part of the stable upright component posture.
- The red cap and black pin side are visible as distinct parts of the same component.
- Small perspective tilt, mild diagonal appearance, or slight rotation can happen
  in normal video and must not be flagged alone.
- The component moves smoothly and stays on the belt surface.

Anomaly conditions:
- Posture/alignment anomaly: component is lying on its side, fallen, tumbling,
  strongly wobbling, rolling, or visibly unstable compared with the normal
  upright/stable reference.
- Reversed orientation: red cap and pin side appear clearly swapped compared
  with the normal reference. Do not infer this from a partial or blurry frame.
- Motion anomaly: stopped, jammed, rolling/spinning, or moving backward.
- Position anomaly: overhanging, falling, or leaving the belt surface.

Do not flag these as anomalies:
- Partial visibility at entry or exit.
- Slight wobble.
- Small camera perspective angle.
- The black rectangular body appearing somewhat vertical due to perspective.
- Mild diagonal appearance in late frames if the component remains upright/stable.
- Mild rotation that does not look like tumbling or falling.
- Red cap or pins being difficult to localize in a blurry frame; use fully
  visible frames and posture over partial-frame guesses.
- Slight off-center position if the component is still fully on the belt.
""".strip()


def format_questions(questions: list[str]) -> str:
    return "\n".join(f"{idx}. {question}" for idx, question in enumerate(questions, 1))


def build_learner_prompt(questions: list[str], expected: str | None = None) -> str:
    expected_line = ""
    if expected:
        expected_line = (
            "\nThe expected label is provided only for experiment logging. "
            "Do not copy it blindly; inspect the video and reason from visual evidence."
            f"\nExpected label: {expected}\n"
        )

    return f"""
{SYSTEM_CONTEXT}

Current guiding questions:
{format_questions(questions)}
{expected_line}
Task:
1. Watch the entire video.
2. Answer each guiding question using visual evidence.
3. Decide whether the video is NORMAL or ANOMALY.
4. Give an anomaly score from 0.0 to 1.0.

Output strictly as valid JSON with this schema:
{{
  "question_answers": [
    {{
      "question_id": 1,
      "answer": "YES / NO / UNCLEAR",
      "evidence": "short visual evidence"
    }}
  ],
  "verdict": "NORMAL / ANOMALY / UNCLEAR",
  "anomaly_type": "none / posture_alignment_failure / reversed_orientation / motion_failure / belt_position_failure / unclear",
  "anomaly_score": 0.0,
  "confidence": "LOW / MEDIUM / HIGH",
  "reason": "one or two sentences explaining the verdict"
}}

Return JSON only. Do not include markdown.
""".strip()


def build_optimizer_prompt(
    questions: list[str],
    learner_results: list[dict],
    max_questions: int = 5,
) -> str:
    return f"""
You are the prompt optimizer for an industrial video anomaly detection system.

Goal:
Improve the guiding questions used by a learner VLM. The learner receives a video
and answers the guiding questions before predicting NORMAL or ANOMALY.

Important:
- Do not make the questions too vague.
- Do not make the questions overly specific to only one clip.
- Keep the questions visually checkable.
- Focus on the current production-line setup:
  belt LEFT to RIGHT, normal upright/stable reference posture, red cap visible
  upward/on top in normal examples, posture/alignment, motion, belt position.
- Limit the new guiding questions to at most {max_questions}.

Current guiding questions:
{format_questions(questions)}

Learner experiment results:
{learner_results}

Task:
Analyze the learner mistakes and weak explanations. Then propose improved
guiding questions that should reduce errors on both current and future similar
production-line videos.

Output strictly as valid JSON:
{{
  "reasoning": "explain what failed and why the questions should change",
  "new_questions": [
    "question 1",
    "question 2"
  ]
}}

Return JSON only. Do not include markdown.
""".strip()
