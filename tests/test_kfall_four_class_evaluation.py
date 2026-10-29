"""Four-class evaluation retains temporal labels and legacy checkpoint behavior."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from evaluation.kfall_reports import save_evaluation_reports
from preprocess.kfall_labels import class_names


class FourClassReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.destination = Path(self.temporary.name) / "reports"
        self.labels = np.array([0, 3, 1, 2, 3], dtype=np.int64)
        self.split = {
            "y_h1": self.labels,
            "subject": np.array(["SA07"] * 5), "file_idx": np.arange(5),
            "start": np.arange(5) * 10, "end_frame": np.arange(5) + 205,
            "lead_time_ms": np.array([100., 0., np.nan, np.nan, -900.]),
        }
        self.metadata = {"n_classes": 4, "recordings": [
            {"subject": "SA07", "file": f"record{index}.csv", "code": "T20", "trial": "R01"}
            for index in range(5)
        ]}
        self.probabilities = np.array([
            [.7, .1, .1, .1], [.1, .1, .1, .7], [.1, .7, .1, .1],
            [.1, .1, .7, .1], [.6, .1, .1, .2],
        ])

    def test_postimpact_outputs_have_four_probabilities_and_signed_timing(self):
        metrics = save_evaluation_reports(self.destination, y_true=self.labels,
            probabilities=self.probabilities, window_indices=np.arange(5), split=self.split,
            metadata=self.metadata, summary={"n_classes": 4})
        names = list(class_names(4).values())
        self.assertEqual(metrics["class_names"], names)
        self.assertEqual(metrics["n_classes"], 4)
        self.assertEqual(metrics["accuracy"], .8)
        self.assertEqual(metrics["class_counts"][names[3]], 2)
        self.assertEqual(metrics["per_class"][names[3]]["recall"], .5)
        self.assertEqual(np.load(self.destination / "confusion_matrix.npy").shape, (4, 4))
        self.assertEqual(np.load(self.destination / "probabilities.npy").shape, (5, 4))
        self.assertEqual((self.destination / "confusion_matrix.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        with (self.destination / "predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[1]["true_class"], names[3])
        self.assertAlmostEqual(float(rows[1]["prob_postimpact"]), .7)
        self.assertEqual(float(rows[1]["lead_time_ms"]), 0.)
        self.assertEqual(float(rows[4]["lead_time_ms"]), -900.)
        self.assertEqual(rows[2]["lead_time_ms"], "")
        self.assertEqual(metrics["fall_window_metrics"]["false_fall_windows"], 1)
        self.assertEqual(json.loads((self.destination / "metrics.json").read_text()), metrics)

    def test_probability_count_must_match_checkpoint_class_scheme(self):
        with self.assertRaisesRegex(ValueError, "class count"):
            save_evaluation_reports(self.destination, y_true=np.array([0]),
                probabilities=np.array([[.8, .1, .1]]), window_indices=np.array([0]),
                split=self.split, metadata=self.metadata, summary={"n_classes": 4})
        self.assertFalse(self.destination.exists())

    def test_four_probabilities_can_infer_scheme_without_optional_metadata(self):
        metadata = {"recordings": self.metadata["recordings"]}
        metrics = save_evaluation_reports(self.destination, y_true=self.labels,
            probabilities=self.probabilities, window_indices=np.arange(5), split=self.split,
            metadata=metadata, summary={})
        self.assertEqual(metrics["n_classes"], 4)


class FourClassCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_kfall_training import fixture_args, write_fixture
        from train_kfall import main as train_main

        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        write_fixture(cls.root)
        arguments = fixture_args(cls.root) + ["--post-impact-seconds", "1.2"]
        with patch("train_kfall.get_device", return_value=torch.device("cpu")), contextlib.redirect_stdout(io.StringIO()):
            train_main(arguments)
        cls.run_dir = next((cls.root / "runs").iterdir())
        checkpoint = torch.load(cls.run_dir / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        cls.cache = Path(config["cache_dir"]) / f"kfall_event_loader_{config['cache_name']}"

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_best_four_class_checkpoint_runs_on_saved_test(self):
        from evaluate_kfall import evaluate_run

        output = self.root / "evaluation"
        metrics = evaluate_run(self.run_dir, device="cpu", output_dir=output)
        self.assertEqual(metrics["n_classes"], 4)
        truth = np.load(output / "y_true.npy")
        self.assertIn(3, truth)
        probabilities = np.load(output / "probabilities.npy")
        self.assertEqual(probabilities.shape, (len(truth), 4))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(np.load(output / "y_pred.npy"),
                                      np.load(self.run_dir / "metrics" / "y_pred.npy"))
        self.assertEqual(metrics["best_epoch"], 1)
        self.assertEqual(metrics["post_impact_seconds"], 1.2)

    def test_postimpact_temporal_targets_are_checked_against_training_protocol(self):
        from evaluation.kfall_data import load_evaluation_data

        checkpoint = torch.load(self.run_dir / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=True)
        arrays, _, _, _ = load_evaluation_data(self.run_dir, checkpoint["config"])
        selected = np.flatnonzero(arrays["y_h1"] == 3)
        self.assertGreater(len(selected), 0)
        self.assertTrue(np.all(arrays["lead_time_ms"][selected] <= 0))
        altered = np.array(arrays["y_h1"])
        original = altered.copy()
        altered[selected[0]] = 0
        path = self.cache / "test" / "y_h1.npy"
        try:
            np.save(path, altered)
            with self.assertRaisesRegex(ValueError, "temporal labeling protocol"):
                load_evaluation_data(self.run_dir, checkpoint["config"])
        finally:
            np.save(path, original)


if __name__ == "__main__":
    unittest.main()
