import json
import unittest
from pathlib import Path

from three_pen_vad.preprocess_video import (
    build_processed_name,
    compute_trim_window,
)
from three_pen_vad.prompts import (
    DESCRIPTION_PROMPT,
    GUIDING_QUESTIONS_V2,
    build_learner_prompt,
    get_guiding_questions,
)
from three_pen_vad.run_learner import build_run_summary


class ThreePenVadTests(unittest.TestCase):
    def test_build_processed_name_adds_fps_and_trim_suffix(self):
        source = Path("data/grad2_pilot/videos/normal_01.mp4")

        result = build_processed_name(source, target_fps=20, trim_start=2, trim_end=2)

        self.assertEqual(result, "normal_01_4k_20fps_trim2s.mp4")

    def test_build_processed_name_preserves_anomaly_variant(self):
        source = Path("data/grad2_pilot/videos/missing_cap_01_left_pen.mp4")

        result = build_processed_name(source, target_fps=20, trim_start=2, trim_end=2)

        self.assertEqual(result, "missing_cap_01_left_pen_4k_20fps_trim2s.mp4")

    def test_compute_trim_window_removes_start_and_end_seconds(self):
        start_frame, end_frame = compute_trim_window(
            frame_count=540,
            source_fps=30.0,
            trim_start=2.0,
            trim_end=2.0,
        )

        self.assertEqual(start_frame, 60)
        self.assertEqual(end_frame, 480)

    def test_compute_trim_window_rejects_over_trimmed_video(self):
        with self.assertRaises(ValueError):
            compute_trim_window(
                frame_count=90,
                source_fps=30.0,
                trim_start=2.0,
                trim_end=2.0,
            )

    def test_description_prompt_is_observation_only(self):
        prompt = DESCRIPTION_PROMPT.lower()

        self.assertIn("do not classify", prompt)
        self.assertIn("number of pens", prompt)
        self.assertIn("cap visibility", prompt)
        self.assertIn("conveyor", prompt)

    def test_dataset_manifest_contains_five_pilot_items(self):
        manifest_path = Path("three_pen_vad/dataset_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(len(manifest), 5)
        self.assertEqual(
            [item["id"] for item in manifest],
            [
                "normal_01",
                "normal_02",
                "normal_03",
                "missing_cap_01_left_pen",
                "missing_cap_02_middle_pen",
            ],
        )

    def test_learner_prompt_v1_targets_current_anomaly_set(self):
        prompt = build_learner_prompt(expected="NORMAL", prompt_version="v1")

        self.assertIn(
            "For this experiment, distinguish NORMAL, MISSING_CAP, WRONG_ORIENTATION, COLOR_ANOMALY, and TEMPORAL_STUCK.",
            prompt,
        )
        self.assertIn('"verdict": "NORMAL / MISSING_CAP / WRONG_ORIENTATION / COLOR_ANOMALY / TEMPORAL_STUCK / UNCLEAR"', prompt)
        self.assertIn('"missing_cap_pen": "none / left_first / middle_second / right_third / unclear"', prompt)
        self.assertIn('"wrong_orientation_pen": "none / left_first / middle_second / right_third / unclear"', prompt)
        self.assertIn('"color_anomaly_pen": "none / left_first / middle_second / right_third / unclear"', prompt)
        self.assertIn('"temporal_anomaly": "none / carrier_stuck / unclear"', prompt)
        self.assertIn('"wrong_orientation_type": "none / reversed / tilted / unclear"', prompt)
        self.assertIn("clearly different color", prompt)
        self.assertIn("stops, stalls", prompt)
        self.assertIn("Compare early, middle, and late frames", prompt)
        self.assertIn("remains fixed for several seconds while fully visible", prompt)
        self.assertLessEqual(len(get_guiding_questions("v1")), 8)

    def test_learner_prompt_does_not_include_expected_label(self):
        prompt = build_learner_prompt(expected="MISSING_CAP", prompt_version="v2")

        self.assertNotIn("Expected label", prompt)
        self.assertNotIn("provided only for experiment logging", prompt)

    def test_learner_prompt_includes_middle_frame_warning(self):
        prompt = build_learner_prompt(expected=None)

        self.assertIn("Analyze fully visible middle frames", prompt)
        self.assertIn("Do not treat partial visibility during entry or exit as an anomaly", prompt)
        self.assertIn("Return JSON only", prompt)

    def test_learner_prompt_v2_uses_per_pen_attribute_extraction(self):
        prompt = build_learner_prompt(expected="MISSING_CAP", prompt_version="v2")

        self.assertEqual(len(GUIDING_QUESTIONS_V2), 5)
        self.assertIn("per-pen attribute extraction", prompt)
        self.assertIn('"pen_observations"', prompt)
        self.assertIn('"left_first"', prompt)
        self.assertIn('"cap_marker": "present / absent / unclear"', prompt)
        self.assertIn('"cap_end": "top / bottom / none / unclear"', prompt)
        self.assertIn('"writing_tip_exposed": "yes / no / unclear"', prompt)
        self.assertIn('"orientation": "normal / reversed / tilted / unclear"', prompt)
        self.assertIn('"detected_anomalies"', prompt)
        self.assertIn('"primary_verdict": "NORMAL / MISSING_CAP / WRONG_ORIENTATION / MULTIPLE_ANOMALIES / UNCLEAR"', prompt)
        self.assertIn("Do not stop after finding the first anomaly", prompt)
        self.assertNotIn('"verdict": "NORMAL / MISSING_CAP / WRONG_ORIENTATION / UNCLEAR"', prompt)

    def test_get_guiding_questions_selects_prompt_version(self):
        v1_questions = get_guiding_questions("v1")
        v2_questions = get_guiding_questions("v2")

        self.assertEqual(len(v1_questions), 8)
        self.assertEqual(len(v2_questions), 5)
        self.assertTrue(any("color" in question.lower() for question in v1_questions))
        self.assertTrue(any("early, middle, and late" in question.lower() for question in v1_questions))
        self.assertIn("For each pen", v2_questions[1])
        with self.assertRaises(ValueError):
            get_guiding_questions("unknown")

    def test_build_run_summary_extracts_usage_and_elapsed_time(self):
        result = {
            "elapsed_seconds": 12.3456,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "video_tokens": 40,
            },
            "parsed": {
                "verdict": "NORMAL",
                "confidence": "HIGH",
                "anomaly_score": 0.0,
            },
        }

        summary = build_run_summary(result)

        self.assertEqual(summary["elapsed_seconds"], 12.35)
        self.assertEqual(summary["input_tokens"], 100)
        self.assertEqual(summary["output_tokens"], 50)
        self.assertEqual(summary["total_tokens"], 150)
        self.assertEqual(summary["video_tokens"], 40)
        self.assertEqual(summary["verdict"], "NORMAL")


if __name__ == "__main__":
    unittest.main()
