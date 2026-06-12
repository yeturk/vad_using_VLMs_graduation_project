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
- Each normal pen has a dark gray or black body appearance and a visible dark cap-like part at the lower end.
- A missing-cap anomaly occurs when one pen lacks this black cap-like part and instead shows an exposed pointed writing tip.
- A wrong-orientation anomaly occurs when one pen is reversed, upside down, tilted, or diagonal compared with the other pens.
- A color anomaly occurs when one pen has a clearly different color or material appearance from the normal dark pens, such as a blue body or a transparent/white cap-like part.
- A temporal stuck anomaly occurs when the carrier visibly stops, stalls, or fails to continue the expected LEFT-to-RIGHT motion while still in the inspection view.
- Compare early, middle, and late frames to judge carrier motion; a carrier that remains fixed for several seconds while fully visible should be treated as stuck.
- In normal videos, all three pens should have the cap-like part on the same end and should be parallel with similar orientation.

Important visual rules:
- Analyze fully visible middle frames for pen count and cap visibility.
- Do not treat partial visibility during entry or exit as an anomaly.
- Sequential appearance or disappearance at the frame borders is expected because the carrier enters and exits the camera view.
- Do not classify based on blur, partial frames, or uncertain edge frames.
- Do not treat "the pens become fully visible and then stay in place" as normal motion if the carrier stops progressing across later frames.
- If the cap-like part could be a grip but it is consistently visible on all three normal pens, treat it as the normal cap-like visual marker.
- For this experiment, distinguish NORMAL, MISSING_CAP, WRONG_ORIENTATION, COLOR_ANOMALY, and TEMPORAL_STUCK.
- Do not detect or report other anomaly types yet.
""".strip()


GUIDING_QUESTIONS = [
    "In fully visible middle frames, are exactly three pens visible on the white carrier?",
    "Do all pens match the expected normal appearance: dark body and dark cap-like part at the lower end?",
    "Does any pen show a clearly different color or material appearance, such as blue, transparent, or white parts?",
    "Does any pen lack the normal cap-like marker or show an exposed pointed writing tip?",
    "Are all pens similarly oriented, with no reversed, upside-down, tilted, or diagonal pen?",
    "Are the cap-like markers on the same end of all visible pens?",
    "Compare early, middle, and late frames: does the carrier keep changing position without stopping, stalling, or getting stuck?",
    "Do the pens remain stable relative to the carrier without sliding, falling, or unexpected disappearance?",
]


LEARNER_SYSTEM_CONTEXT_V2 = """
You are an industrial quality-control vision system inspecting a real conveyor setup.

Scene:
- A white rectangular carrier plate moves on a linear conveyor or actuator system.
- The camera view is top-down or near top-down.
- Three pens are placed on the white carrier.
- In normal videos, the carrier moves from LEFT to RIGHT in the image.
- In normal videos, all three pens have the same arrangement: the cap-like marker is present, it is on the same end for every pen, and the pens are parallel with similar orientation.
- A missing-cap anomaly occurs when a pen lacks the black cap-like marker and shows an exposed pointed writing tip where that marker should be.
- A wrong-orientation anomaly occurs when a pen still has its cap-like marker but is reversed, upside down, tilted, or diagonal compared with the other pens.

Important visual rules:
- Use fully visible middle frames for object inspection.
- Do not treat partial entry or exit at frame borders as an anomaly.
- Do not stop after finding the first anomaly. Inspect all three pens independently.
- First extract per-pen attributes, then decide anomalies from those attributes.
- For combined anomalies, report every detected anomaly in the detected_anomalies list.
- If a cap-like marker is absent, classify that pen as MISSING_CAP rather than WRONG_ORIENTATION.
- If a cap-like marker is present but on a different end or the pen angle differs from the others, classify that pen as WRONG_ORIENTATION.
- Do not detect or report anomaly types outside MISSING_CAP and WRONG_ORIENTATION.
""".strip()


GUIDING_QUESTIONS_V2 = [
    "In fully visible middle frames, are exactly three pens visible and stable on the white carrier?",
    "For each pen, report cap marker status, cap end, exposed writing tip, and orientation.",
    "Which pens, if any, have missing-cap evidence?",
    "Which pens, if any, have wrong-orientation evidence?",
    "Does the carrier motion look normal enough that the object inspection is reliable?",
]


def format_questions(questions: list[str]) -> str:
    return "\n".join(f"{idx}. {question}" for idx, question in enumerate(questions, 1))


def get_guiding_questions(prompt_version: str = "v1") -> list[str]:
    if prompt_version == "v1":
        return GUIDING_QUESTIONS
    if prompt_version == "v2":
        return GUIDING_QUESTIONS_V2
    raise ValueError(f"Unknown prompt_version: {prompt_version}")


def build_learner_prompt(expected: str | None = None, prompt_version: str = "v1") -> str:
    _ = expected

    if prompt_version == "v2":
        return f"""
{LEARNER_SYSTEM_CONTEXT_V2}

Current guiding questions:
{format_questions(GUIDING_QUESTIONS_V2)}

Task:
1. Watch the entire video.
2. Use per-pen attribute extraction before classification.
3. Fill the pen_observations object for left_first, middle_second, and right_third.
4. Add every detected MISSING_CAP or WRONG_ORIENTATION event to detected_anomalies.
5. Choose a primary_verdict:
   - NORMAL if no anomalies are detected.
   - MISSING_CAP if only missing-cap anomalies are detected.
   - WRONG_ORIENTATION if only wrong-orientation anomalies are detected.
   - MULTIPLE_ANOMALIES if more than one anomaly type or more than one anomalous pen is detected.
   - UNCLEAR if the video is too unclear to decide.
6. Give an anomaly score from 0.0 to 1.0.

Output strictly as valid JSON with this schema:
{{
  "question_answers": [
    {{
      "question_id": 1,
      "answer": "YES / NO / UNCLEAR",
      "evidence": "short visual evidence"
    }}
  ],
  "pen_observations": {{
    "left_first": {{
      "cap_marker": "present / absent / unclear",
      "cap_end": "top / bottom / none / unclear",
      "writing_tip_exposed": "yes / no / unclear",
      "orientation": "normal / reversed / tilted / unclear",
      "evidence": "short visual evidence"
    }},
    "middle_second": {{
      "cap_marker": "present / absent / unclear",
      "cap_end": "top / bottom / none / unclear",
      "writing_tip_exposed": "yes / no / unclear",
      "orientation": "normal / reversed / tilted / unclear",
      "evidence": "short visual evidence"
    }},
    "right_third": {{
      "cap_marker": "present / absent / unclear",
      "cap_end": "top / bottom / none / unclear",
      "writing_tip_exposed": "yes / no / unclear",
      "orientation": "normal / reversed / tilted / unclear",
      "evidence": "short visual evidence"
    }}
  }},
  "detected_anomalies": [
    {{
      "type": "MISSING_CAP / WRONG_ORIENTATION",
      "pen": "left_first / middle_second / right_third",
      "evidence": "short visual evidence"
    }}
  ],
  "primary_verdict": "NORMAL / MISSING_CAP / WRONG_ORIENTATION / MULTIPLE_ANOMALIES / UNCLEAR",
  "anomaly_score": 0.0,
  "confidence": "LOW / MEDIUM / HIGH",
  "reason": "one or two sentences explaining the detected anomalies"
}}

Return JSON only. Do not include markdown.
""".strip()

    if prompt_version != "v1":
        raise ValueError(f"Unknown prompt_version: {prompt_version}")

    return f"""
{LEARNER_SYSTEM_CONTEXT}

Current guiding questions:
{format_questions(GUIDING_QUESTIONS)}

Task:
1. Watch the entire video.
2. Answer each guiding question using visual evidence.
3. Decide whether the video is NORMAL, MISSING_CAP, WRONG_ORIENTATION, COLOR_ANOMALY, or TEMPORAL_STUCK.
4. Give an anomaly score from 0.0 to 1.0.

Decision rules:
- Predict NORMAL if all three pens are visible, match the expected dark-pen appearance, have the normal dark cap-like part, share the same orientation, remain stable, and the carrier moves normally.
- Predict MISSING_CAP if one pen clearly lacks the black cap-like part and shows an exposed pointed writing tip.
- Predict WRONG_ORIENTATION if one pen still has the cap-like part but is reversed, upside down, tilted, or diagonal compared with the other pens.
- Predict COLOR_ANOMALY if one pen clearly has a different color or material appearance from the normal dark pens, even if it is capped and correctly oriented.
- Predict TEMPORAL_STUCK if the carrier stops, stalls, gets stuck, or remains fixed for several seconds while fully visible instead of continuing LEFT-to-RIGHT motion.
- If multiple anomaly types appear possible, choose the clearest anomaly as the verdict and mention the other evidence in the reason.
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
  "verdict": "NORMAL / MISSING_CAP / WRONG_ORIENTATION / COLOR_ANOMALY / TEMPORAL_STUCK / UNCLEAR",
  "missing_cap_pen": "none / left_first / middle_second / right_third / unclear",
  "wrong_orientation_pen": "none / left_first / middle_second / right_third / unclear",
  "wrong_orientation_type": "none / reversed / tilted / unclear",
  "color_anomaly_pen": "none / left_first / middle_second / right_third / unclear",
  "temporal_anomaly": "none / carrier_stuck / unclear",
  "anomaly_score": 0.0,
  "confidence": "LOW / MEDIUM / HIGH",
  "reason": "one or two sentences explaining the verdict"
}}

Return JSON only. Do not include markdown.
""".strip()
