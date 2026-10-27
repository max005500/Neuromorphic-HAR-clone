"""Check offline test reports preserve labels, cache indices and missing metrics."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.kfall_reports import save_evaluation_reports


class KFallEvaluationReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.destination = Path(self.temporary.name) / "test_best"
        self.split = {
            "y_h1": np.array([2, 0, 1, 2, 0, 1]),
            "file_idx": np.arange(6), "start": np.arange(6) * 102,
            "end_frame": np.arange(6) * 102 + 205,
            "subject": np.array(["SA07", "SA07", "SA21", "SA21", "SA38", "SA38"]),
            "lead_time_ms": np.array([np.nan, 50.0, np.nan, np.nan, 30.0, np.nan]),
        }
        self.metadata = {"recordings": [
            {"file": f"data/KFall/sensor_data/{subject}/record{index}.csv", "subject": subject,
             "code": f"T{index + 1:02d}", "trial": "R01"}
            for index, subject in enumerate(self.split["subject"])
        ]}

    def report(self, indices=(1, 2, 4, 3), probabilities=None, summary=None):
        indices = np.asarray(indices, dtype=np.int64)
        if probabilities is None:
            probabilities = [[0.8, 0.1, 0.1], [0.6, 0.3, 0.1],
                             [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]]
        return save_evaluation_reports(
            self.destination, y_true=self.split["y_h1"][indices],
            probabilities=np.asarray(probabilities), window_indices=indices,
            split=self.split, metadata=self.metadata,
            summary=summary or {"checkpoint": "best_model.pt", "best_epoch": 3, "loss": 0.5},
        )

    def test_outputs_metrics_predictions_and_actual_cache_indices(self):
        metrics = self.report()
        self.assertEqual(metrics["n"], 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["best_epoch"], 3)
        self.assertEqual(metrics["class_counts"], {"Fall": 2, "Dynamic": 1, "Static": 1})
        fall = metrics["fall_window_metrics"]
        self.assertEqual(fall["precision"], 0.5)
        self.assertEqual(fall["recall"], 0.5)
        self.assertEqual(fall["specificity"], 0.5)
        self.assertEqual(fall["false_fall_windows"], 1)
        self.assertEqual(fall["missed_fall_windows"], 1)
        self.assertEqual(set(metrics["per_subject"]), {"SA07", "SA21", "SA38"})
        self.assertEqual((self.destination / "confusion_matrix.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        np.testing.assert_array_equal(np.load(self.destination / "window_indices.npy"), [1, 2, 4, 3])
        np.testing.assert_array_equal(np.load(self.destination / "y_pred.npy"), [0, 0, 2, 2])
        np.testing.assert_array_equal(np.load(self.destination / "confusion_matrix.npy"), [[1, 0, 1], [1, 0, 0], [0, 0, 1]])
        with (self.destination / "predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["window_index"]) for row in rows], [1, 2, 4, 3])
        self.assertEqual([row["subject"] for row in rows], ["SA07", "SA21", "SA38", "SA21"])
        self.assertEqual(rows[2]["file"], self.metadata["recordings"][4]["file"])
        self.assertEqual(rows[2]["true_class"], "Fall")
        self.assertEqual(rows[2]["predicted_class"], "Static")
        self.assertEqual(float(rows[2]["confidence"]), 0.7)
        self.assertEqual(rows[0]["lead_time_ms"], "50.0")
        self.assertEqual(rows[1]["lead_time_ms"], "")
        self.assertEqual(json.loads((self.destination / "metrics.json").read_text()), metrics)

    def test_undefined_fall_and_absent_class_metrics_are_null(self):
        metrics = self.report(indices=(1, 4), probabilities=[[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
        self.assertIsNone(metrics["fall_window_metrics"]["specificity"])
        self.assertIsNone(metrics["fall_window_metrics"]["false_positive_rate"])
        self.assertIsNone(metrics["per_class"]["Dynamic"]["recall"])
        text = (self.destination / "metrics.json").read_text()
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertIn("null", text)

    def test_no_actual_or_predicted_falls_does_not_fabricate_recall(self):
        metrics = self.report(indices=(2, 3), probabilities=[[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        self.assertIsNone(metrics["fall_window_metrics"]["precision"])
        self.assertIsNone(metrics["fall_window_metrics"]["recall"])
        self.assertIsNone(metrics["fall_window_metrics"]["f1"])
        self.assertEqual(metrics["fall_window_metrics"]["specificity"], 1)

    def test_summary_nonfinite_values_are_saved_as_null(self):
        metrics = self.report(summary={"loss": float("nan"), "best_val_loss": float("inf")})
        self.assertIsNone(metrics["loss"])
        self.assertIsNone(metrics["best_val_loss"])
        json.loads((self.destination / "classification_report.json").read_text(),
                   parse_constant=lambda value: self.fail(f"Nonstandard JSON constant {value}"))

    def test_invalid_probabilities_fail_before_creating_outputs(self):
        for probabilities in ([[0.2, 0.2, 0.2]] * 4, [[np.nan, 0.5, 0.5]] * 4,
                              [[-0.1, 0.6, 0.5]] * 4, [[1, 0]] * 4):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                self.report(probabilities=probabilities)
        self.assertFalse(self.destination.exists())

    def test_wrong_ground_truth_or_repeated_indices_are_rejected(self):
        for indices, truth in (([1, 1], [0, 0]), ([1, 2], [1, 0])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                save_evaluation_reports(self.destination, y_true=np.asarray(truth),
                    probabilities=np.array([[0.8, 0.1, 0.1]] * 2), window_indices=np.asarray(indices),
                    split=self.split, metadata=self.metadata, summary={})
        self.assertFalse(self.destination.exists())

    def test_empty_test_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.report(indices=(), probabilities=np.empty((0, 3)))


if __name__ == "__main__":
    unittest.main()
