from __future__ import annotations


DESCRIPTION_PROMPT = """
You are inspecting a short video from a real conveyor setup.

Do not classify the video as normal or anomalous.
Do not decide whether there is a defect yet.
Your task is only to describe what you visually observe.

Focus on:
1. What objects are visible in the scene?
2. What is the number of pens visible in the video?
3. Where are the pens located relative to each other?
4. What is the cap visibility for each pen?
5. Which pen, if any, appears to be missing a cap?
6. What is the orientation of each pen?
7. What surface or carrier are the pens placed on?
8. What conveyor parts or motion cues are visible?
9. Does the carrier or scene move over time? If yes, in which image direction?
10. What details are uncertain because of blur, occlusion, lighting, or perspective?

Output format:

DETAILED SCENE DESCRIPTION:
[one paragraph]

PEN OBSERVATIONS:
- Pen count:
- Left/first pen:
- Middle/second pen:
- Right/third pen:

CAP VISIBILITY:
- Left/first pen:
- Middle/second pen:
- Right/third pen:

ORIENTATION AND MOTION:
[describe pen orientation, carrier/conveyor motion, and any visible instability]

UNCERTAINTIES:
[list details that are not visually certain]
""".strip()


LEARNER_SYSTEM_CONTEXT = """
You are an industrial quality-control vision system inspecting a real conveyor setup.

Scene:
- A white rectangular carrier plate moves on a linear conveyor or actuator system.
- The camera view is top-down or near top-down.
- Three pens are placed on the white carrier.
- In normal videos, the carrier moves from LEFT to RIGHT in the image.
- The pens are arranged in a parallel row and remain stationary relative to the carrier.
- Each normal pen has a visible black cap-like part at the lower end.
- A missing-cap anomaly occurs when one pen lacks this black cap-like part and instead shows an exposed pointed writing tip.

Important visual rules:
- Analyze fully visible middle frames for pen count and cap visibility.
- Do not treat partial visibility during entry or exit as an anomaly.
- Sequential appearance or disappearance at the frame borders is expected because the carrier enters and exits the camera view.
- Do not classify based on blur, partial frames, or uncertain edge frames.
- If the cap-like part could be a grip but it is consistently visible on all three normal pens, treat it as the normal cap-like visual marker.
- For this experiment, only distinguish NORMAL from MISSING_CAP.
- Do not detect or report other anomaly types yet.
""".strip()


GUIDING_QUESTIONS = [
    "In fully visible middle frames, are exactly three pens visible on the white carrier?",
    "Does each pen show a visible black cap-like part at its lower end?",
    "Does any pen show an exposed pointed writing tip where the black cap-like part should be?",
    "Are the three pens parallel and similarly oriented on the carrier?",
    "Does the carrier move smoothly from LEFT to RIGHT while the pens remain stable relative to it?",
]


def format_questions(questions: list[str]) -> str:
    return "\n".join(f"{idx}. {question}" for idx, question in enumerate(questions, 1))


def build_learner_prompt(expected: str | None = None) -> str:
    _ = expected

    return f"""
{LEARNER_SYSTEM_CONTEXT}

Current guiding questions:
{format_questions(GUIDING_QUESTIONS)}

Task:
1. Watch the entire video.
2. Answer each guiding question using visual evidence.
3. Decide whether the video is NORMAL or MISSING_CAP.
4. Give an anomaly score from 0.0 to 1.0.

Decision rules:
- Predict NORMAL if all three pens are visible in the main middle frames and all three show the normal black cap-like part.
- Predict MISSING_CAP if one pen clearly lacks the black cap-like part and shows an exposed pointed writing tip.
- If the video is too unclear to decide, use UNCLEAR.

Output strictly as valid JSON with this schema:
{{
  "question_answers": [
    {{
      "question_id": 1,
      "answer": "YES / NO / UNCLEAR",
      "evidence": "short visual evidence"
    }}
  ],
  "verdict": "NORMAL / MISSING_CAP / UNCLEAR",
  "missing_cap_pen": "none / left_first / middle_second / right_third / unclear",
  "anomaly_score": 0.0,
  "confidence": "LOW / MEDIUM / HIGH",
  "reason": "one or two sentences explaining the verdict"
}}

Return JSON only. Do not include markdown.
""".strip()
