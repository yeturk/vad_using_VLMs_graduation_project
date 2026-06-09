from __future__ import annotations


SYSTEM_CONTEXT = """
You are an industrial quality-control vision system for a production line.

Scene:
- A white rectangular board slides on a linear rail (steel guide rails with a
  central lead screw) over a light wooden surface.
- The board moves LEFT to RIGHT in the camera frame and carries a row of pens.
- The pens are slender cylindrical writing instruments with a translucent barrel
  and a darker capped end. They are all meant to be identical to one another.

Nominal (NORMAL) condition:
- The board carries exactly THREE pens, evenly spaced in a row.
- All three pens are identical to EACH OTHER: same color, same length, same parts.
- All three pens point the SAME direction (same end up, same end down).
- Every pen is complete and capped: no pen shows a bare/exposed writing tip while
  the others are capped, and none looks clearly shorter than the others.
- The board moves smoothly and continuously from left to right, without stopping,
  freezing, pausing, reversing, or jamming.

Anomaly conditions (any ONE of these makes the clip ANOMALY):
- Missing cap: one pen lacks its cap / shows a bare writing tip while the others
  remain capped, or one pen looks clearly shorter or different at one end.
- Wrong orientation: one pen is flipped and points the opposite way relative to
  the other two, so the pens are NOT all aligned the same direction.
- Wrong color: one pen differs in color from the other two; the three pens are
  not all the same color.
- Missing pen / wrong count: when the board is fully in view there are not three
  pens (for example only two, or an empty gap where a pen should be).
- Motion failure: the board stops, freezes/pauses mid-travel, jams, or reverses
  instead of moving smoothly left to right.
- Combinations of the above also count as ANOMALY.

Judging guidance:
- Compare the three pens TO EACH OTHER. The line is NORMAL when the pens are
  mutually consistent (same color, same orientation, all capped and complete) and
  the motion is smooth; flag ANOMALY when one pen breaks the shared pattern or the
  motion is interrupted.
- Do NOT decide color in absolute terms (e.g. "blue"); decide whether all three
  pens share the SAME color.
- Count the pens only when the board is fully visible / centered, not during the
  partial entry or exit frames.
- Do not flag these: partial visibility at entry or exit, slight perspective tilt,
  minor spacing differences, mild blur, or a small camera angle.
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
  "anomaly_type": "none / missing_cap / wrong_orientation / wrong_color / missing_pen / motion_failure / unexpected_pen_movement / multiple / unclear",
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
- Focus on the current production-line setup: a white board moving LEFT to RIGHT
  carrying three pens; normal means the three pens are mutually consistent (same
  color, same orientation, all capped and complete, exactly three) and the motion
  is smooth. Anomalies: missing cap, wrong orientation, wrong color, missing pen,
  motion stop/freeze.
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
