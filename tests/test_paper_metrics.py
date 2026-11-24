import unittest

import numpy as np

from evaluation.paper_metrics import (
    SRC_MANUAL,
    classification_metrics,
    evaluate_continuous_alarms,
    evaluate_dual_head_sisfall,
    hierarchical_metrics,
    perturb_raw_signal,
)


class ClassificationMetricsTests(unittest.TestCase):
    def test_generic_confusion_uses_fixed_class_order(self):
        report = classification_metrics([0, 0, 2], [0, 1, 2], n_classes=4)
        self.assertAlmostEqual(report["accuracy"], 2 / 3)
        self.assertEqual(len(report["confusion_matrix"]), 4)
        self.assertEqual(report["confusion_matrix"][3], [0, 0, 0, 0])
        self.assertAlmostEqual(report["macro_f1"], (2 / 3 + 0 + 1 + 0) / 4)

    def test_dual_head_excludes_autotagged_static_gold(self):
        report = evaluate_dual_head_sisfall(
            y_h1_true=[0, 1, 2, 2],
            y_h2_true=[1, 0, 0, 1],
            src_h2=[0, 0, SRC_MANUAL, 0],
            y_h1_pred=[0, 1, 2, 2],
            y_h2_pred=[0, 1, 0, 0],
        )
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["excluded_autotag_static"], 1)
        self.assertEqual(report["confusion_matrix"], [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ])
        self.assertEqual(report["classes"]["Transition"]["support"], 0)
        self.assertIsNone(report["major_subjects_SE01_SE15"])

    def test_major_subject_metrics_include_only_se01_through_se15(self):
        report = hierarchical_metrics(
            y_h1=[2, 2, 2],
            y_h2=[0, 1, 0],
            src_h2=[1, 1, 1],
            pred_h1=[2, 2, 2],
            pred_h2=[0, 0, 1],
            subjects=["se01", "SE15", "SE16"],
        )
        majors = report["major_subjects_SE01_SE15"]
        self.assertEqual(majors["count"], 2)
        self.assertEqual(majors["classes"]["Stable"]["support"], 1)
        self.assertEqual(majors["classes"]["Transition"]["support"], 1)
        self.assertEqual(majors["accuracy"], 0.5)

    def test_manual_source_does_not_make_non_static_head2_a_class(self):
        report = evaluate_dual_head_sisfall([0, 1], [1, 1], [1, 1], [0, 1], [1, 0])
        self.assertEqual(report["count"], 2)
        self.assertEqual(report["accuracy"], 1.0)

    def test_composition_uses_predicted_head1_for_predictions(self):
        report = evaluate_dual_head_sisfall(
            y_h1_true=[2, 0],
            y_h2_true=[0, 1],
            src_h2=[SRC_MANUAL, 0],
            y_h1_pred=[1, 2],
            y_h2_pred=[1, 1],
            subjects=["SE01", "SE15"],
        )
        expected = [
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ]
        self.assertEqual(report["confusion"], expected)
        self.assertEqual(report["major_subjects_SE01_SE15"]["confusion"], expected)

    def test_nonintegral_gold_and_invalid_head1_truth_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Head1"):
            evaluate_dual_head_sisfall([2.5], [0], [1], [2], [0])
        with self.assertRaisesRegex(ValueError, "manual Head2"):
            evaluate_dual_head_sisfall([2], [0.5], [1], [2], [0])

    def test_empty_major_subset_has_undefined_summary_metrics(self):
        report = evaluate_dual_head_sisfall([0], [0], [0], [0], [0], subjects=["SE16"])
        majors = report["major_subjects_SE01_SE15"]
        self.assertEqual(majors["count"], 0)
        self.assertIsNone(majors["accuracy"])
        self.assertIsNone(majors["macro_f1"])


class ContinuousAlarmTests(unittest.TestCase):
    def test_one_to_one_matching_by_recording_and_real_adl_duration(self):
        windows = [
            {"recording_id": "fall-a", "timestamp_seconds": 10, "alarm": True},
            {"recording_id": "fall-a", "timestamp_seconds": 11, "alarm": True},
            {"recording_id": "adl-b", "timestamp_seconds": 20, "alarm": True},
            {"recording_id": "adl-b", "timestamp_seconds": 20.5, "alarm": True},
            {"recording_id": "adl-b", "timestamp_seconds": 21, "alarm": True},
        ]
        result = evaluate_continuous_alarms(
            windows,
            [{"recording_id": "fall-a", "start_seconds": 9, "end_seconds": 12}],
            {
                "fall-a": {"duration_seconds": 60, "is_adl": False},
                "adl-b": {"duration_seconds": 3600, "is_adl": True},
            },
            refractory_seconds=1,
        )
        self.assertEqual(result["detected_events"], 1)
        self.assertEqual(result["event_sensitivity"], 1.0)
        self.assertEqual(result["refractory_suppressed_alarms"], 1)
        self.assertEqual(result["false_alarms_on_adl"], 2)
        self.assertEqual(result["false_alarms_per_hour"], 2.0)
        self.assertEqual(result["adl_duration_seconds"], 3600.0)

    def test_alarm_cannot_match_an_event_in_another_recording(self):
        result = evaluate_continuous_alarms(
            [{"recording_id": "adl", "timestamp_seconds": 5, "prediction": 1}],
            [{"recording_id": "fall", "start_seconds": 4, "end_seconds": 6}],
            {
                "adl": {"duration_seconds": 1800, "is_adl": True},
                "fall": {"duration_seconds": 30, "is_adl": False},
            },
        )
        self.assertEqual(result["detected_events"], 0)
        self.assertEqual(result["false_alarms_on_adl"], 1)
        self.assertEqual(result["false_alarms_per_hour"], 2.0)

    def test_overlapping_windows_do_not_inflate_exposure_hours(self):
        result = evaluate_continuous_alarms(
            [
                {"recording_id": "adl", "timestamp_seconds": 10, "alarm": 1},
                {"recording_id": "adl", "timestamp_seconds": 11, "alarm": 1},
            ],
            [],
            {"adl": {"duration_seconds": 7200, "is_adl": True}},
        )
        self.assertEqual(result["false_alarms_on_adl"], 2)
        self.assertEqual(result["false_alarms_per_hour"], 1.0)
        self.assertEqual(result["fall_events"], 0)
        self.assertIsNone(result["event_sensitivity"])

    def test_missing_official_events_timestamps_or_real_durations_are_rejected(self):
        info = {"r": {"duration_seconds": 10, "is_adl": True}}
        with self.assertRaisesRegex(ValueError, "fall_events"):
            evaluate_continuous_alarms([{"recording_id": "r", "timestamp_seconds": 1, "alarm": 0}], None, info)
        with self.assertRaisesRegex(ValueError, "timestamp_seconds"):
            evaluate_continuous_alarms([{"recording_id": "r", "alarm": 0}], [], info)
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            evaluate_continuous_alarms(
                [{"recording_id": "r", "timestamp_seconds": 1, "alarm": 0}], [], {"r": {"is_adl": True}}
            )

    def test_event_intervals_are_required_and_validated(self):
        windows = [{"recording_id": "r", "timestamp_seconds": 1, "alarm": 0}]
        info = {"r": {"duration_seconds": 10, "is_adl": False}}
        with self.assertRaisesRegex(ValueError, "start_seconds"):
            evaluate_continuous_alarms(windows, [{"recording_id": "r"}], info)
        with self.assertRaisesRegex(ValueError, "invalid"):
            evaluate_continuous_alarms(
                windows, [{"recording_id": "r", "start_seconds": 3, "end_seconds": 2}], info
            )

    def test_adl_exposure_requires_window_coverage_and_boolean_metadata(self):
        result = evaluate_continuous_alarms(
            [{"recording_id": "covered", "timestamp_seconds": 1, "alarm": False}],
            [],
            {
                "covered": {"duration_seconds": 3600, "is_adl": True},
                "uncovered": {"duration_seconds": 7200, "is_adl": True},
            },
        )
        self.assertEqual(result["adl_duration_seconds"], 3600.0)
        self.assertEqual(result["adl_recordings_without_window_coverage"], ["uncovered"])
        with self.assertRaisesRegex(ValueError, "boolean"):
            evaluate_continuous_alarms(
                [{"recording_id": "r", "timestamp_seconds": 1, "alarm": False}],
                [],
                {"r": {"duration_seconds": 10, "is_adl": "false"}},
            )

    def test_official_fall_events_cannot_be_marked_as_adl(self):
        with self.assertRaisesRegex(ValueError, "cannot be marked as ADL"):
            evaluate_continuous_alarms(
                [{"recording_id": "r", "timestamp_seconds": 1, "alarm": False}],
                [{"recording_id": "r", "start_seconds": 1, "end_seconds": 2, "is_adl": True}],
                {"r": {"duration_seconds": 10, "is_adl": False}},
            )


class RawPerturbationTests(unittest.TestCase):
    def test_noise_drift_and_channel_zero_are_reproducible_without_mutating_input(self):
        signal = np.ones((4, 3), dtype=np.float32)
        original = signal.copy()
        kwargs = dict(
            units="raw counts",
            sample_rate_hz=2.0,
            noise_std=0.1,
            noise_channels=[1],
            drift_per_second=2.0,
            zero_channels=[2],
            seed=41,
        )
        first = perturb_raw_signal(signal, **kwargs)
        second = perturb_raw_signal(signal, **kwargs)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(signal, original)
        np.testing.assert_array_equal(first[:, 2], np.zeros(4))
        np.testing.assert_allclose(first[:, 0], [1, 2, 3, 4])
        self.assertTrue(np.all(np.isfinite(first[:, 1])))
        self.assertFalse(np.allclose(first[:, 1], first[:, 0]))

    def test_units_and_channel_indices_are_explicit_and_valid(self):
        signal = np.ones((2, 2))
        with self.assertRaisesRegex(ValueError, "units"):
            perturb_raw_signal(signal, units="", sample_rate_hz=1)
        with self.assertRaisesRegex(ValueError, "out of range"):
            perturb_raw_signal(signal, units="g", sample_rate_hz=1, zero_channels=[2])


if __name__ == "__main__":
    unittest.main()
