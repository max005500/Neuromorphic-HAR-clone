"""Run saved best KFall checkpoints on their held-out windows without training."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from models.models import SingleHeadSCN
from test_kfall_training import fixture_args, write_fixture
from train_kfall import main as train_main


class KFallEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.fixtures = {}
        for mode in ("single", "dual"):
            root = cls.directory / mode
            root.mkdir()
            write_fixture(root)
            with patch("train_kfall.get_device", return_value=torch.device("cpu")), contextlib.redirect_stdout(io.StringIO()):
                train_main(fixture_args(root) + ["--head-mode", mode])
            run = next((root / "runs").iterdir())
            cache = root / "cache" / "kfall_event_loader_default"
            cls.fixtures[mode] = {
                "root": root, "run": run, "cache": cache,
                "truth": np.load(run / "metrics" / "y_true.npy"),
                "prediction": np.load(run / "metrics" / "y_pred.npy"),
                "metrics": json.loads((run / "metrics" / "test_metrics.json").read_text()),
                "metadata": json.loads((run / "dataset_metadata.json").read_text()),
            }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def output_dir(self, name):
        return self.directory / "evaluations" / self.id().rsplit(".", 1)[-1] / name

    def evaluate(self, mode="single", **kwargs):
        from evaluate_kfall import evaluate_run

        fixture = self.fixtures[mode]
        destination = self.output_dir(mode)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = evaluate_run(fixture["run"], device="cpu", output_dir=destination, **kwargs)
        return fixture, destination, metrics

    @staticmethod
    def fingerprints(paths):
        return {
            path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in paths
        }

    def test_saved_single_head_matches_original_test_and_preserves_inputs(self):
        fixture = self.fixtures["single"]
        inputs = [fixture["run"] / "checkpoints" / "best_model.pt", *fixture["cache"].rglob("*.npy"),
                  fixture["cache"] / "metadata.json", fixture["run"] / "config.json",
                  fixture["run"] / "dataset_metadata.json", fixture["run"] / "metrics" / "test_metrics.json"]
        before = self.fingerprints(inputs)
        fixture, destination, metrics = self.evaluate()
        np.testing.assert_array_equal(np.load(destination / "y_true.npy"), fixture["truth"])
        np.testing.assert_array_equal(np.load(destination / "y_pred.npy"), fixture["prediction"])
        self.assertEqual(metrics["n"], len(fixture["truth"]))
        self.assertEqual(metrics["best_epoch"], 1)
        self.assertAlmostEqual(metrics["accuracy"], fixture["metrics"]["acc"], places=7)
        self.assertAlmostEqual(metrics["loss"], fixture["metrics"]["loss"], places=6)
        self.assertEqual(before, self.fingerprints(inputs))
        self.assertEqual((destination / "confusion_matrix.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertTrue((destination / "metrics.json").is_file())
        probabilities = np.load(destination / "probabilities.npy")
        self.assertEqual(probabilities.shape, (len(fixture["truth"]), 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), fixture["prediction"])

    def test_evaluation_uses_inference_mode_without_training_or_raw_loading(self):
        observations = []
        original_forward = SingleHeadSCN.forward

        def record_forward(model, *args, **kwargs):
            observations.append((model.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
            return original_forward(model, *args, **kwargs)

        forbidden = AssertionError("Evaluation must not train, preprocess, or load source recordings.")
        with contextlib.ExitStack() as stack:
            for target in (
                "train_kfall.main", "train_kfall.build_data", "torch.optim.Adam",
                "loaders.kfall_event_loader.load_or_build_kfall_event_splits",
                "preprocess.preprocess_kfall.load_kfall_3class", "preprocess.preprocess_kfall.load_record",
            ):
                stack.enter_context(patch(target, side_effect=forbidden))
            stack.enter_context(patch.object(SingleHeadSCN, "forward", record_forward))
            _, _, metrics = self.evaluate()
        self.assertGreater(metrics["n"], 0)
        self.assertTrue(observations)
        self.assertTrue(all(state == (False, False, True) for state in observations))

    def test_cache_only_evaluation_uses_embedded_checkpoint_configuration(self):
        fixture = self.fixtures["single"]
        paths = [fixture["root"] / "sensor_data", fixture["root"] / "label_data", fixture["run"] / "config.json"]
        moved = []
        try:
            for path in paths:
                destination = path.with_name(path.name + ".offline")
                path.rename(destination)
                moved.append((path, destination))
            _, destination, metrics = self.evaluate()
            self.assertEqual(metrics["n"], len(fixture["truth"]))
            np.testing.assert_array_equal(np.load(destination / "y_pred.npy"), fixture["prediction"])
        finally:
            for original, destination in reversed(moved):
                destination.rename(original)

    def test_single_window_preserves_its_actual_test_index(self):
        fixture = self.fixtures["single"]
        index = len(fixture["truth"]) - 1
        _, destination, metrics = self.evaluate(window=index)
        self.assertEqual(metrics["n"], 1)
        np.testing.assert_array_equal(np.load(destination / "window_indices.npy"), [index])
        np.testing.assert_array_equal(np.load(destination / "y_true.npy"), fixture["truth"][[index]])
        self.assertEqual(np.load(destination / "probabilities.npy").shape, (1, 3))
        with (destination / "predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        index_column = next((name for name in ("window_index", "window_idx", "index") if name in rows[0]), None)
        self.assertIsNotNone(index_column)
        self.assertEqual(int(rows[0][index_column]), index)
        self.assertEqual(rows[0]["subject"], fixture["metadata"]["split_subjects"]["test"][0])

    def test_subject_filter_keeps_only_held_out_subjects(self):
        fixture = self.fixtures["single"]
        test_subject = fixture["metadata"]["split_subjects"]["test"][0]
        _, destination, metrics = self.evaluate(subject=test_subject, batch_size=1)
        self.assertEqual(metrics["n"], len(fixture["truth"]))
        np.testing.assert_array_equal(np.load(destination / "y_pred.npy"), fixture["prediction"])
        with self.assertRaises((ValueError, IndexError)):
            self.evaluate(subject=fixture["metadata"]["split_subjects"]["train"][0])

    def test_dual_checkpoint_evaluates_only_the_supervised_activity_head(self):
        fixture, destination, metrics = self.evaluate("dual")
        np.testing.assert_array_equal(np.load(destination / "y_true.npy"), fixture["truth"])
        np.testing.assert_array_equal(np.load(destination / "y_pred.npy"), fixture["prediction"])
        self.assertEqual(metrics["n"], len(fixture["truth"]))
        self.assertEqual(metrics["best_epoch"], 1)
        self.assertAlmostEqual(metrics["accuracy"], fixture["metrics"]["acc_h1"], places=7)
        self.assertAlmostEqual(metrics["loss"], fixture["metrics"]["loss"], places=6)
        self.assertNotIn("acc_h2", metrics)

    def test_cli_can_evaluate_an_explicit_checkpoint(self):
        from evaluate_kfall import main

        fixture = self.fixtures["single"]
        destination = self.output_dir("cli")
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--run-dir", str(fixture["run"]), "--checkpoint", str(fixture["run"] / "checkpoints" / "best_model.pt"),
                  "--output-dir", str(destination), "--device", "cpu", "--workers", "0"])
        metrics = json.loads((destination / "metrics.json").read_text())
        self.assertEqual(metrics["n"], len(fixture["truth"]))
        np.testing.assert_array_equal(np.load(destination / "y_pred.npy"), fixture["prediction"])

    def test_balanced_loss_requires_the_saved_training_class_weights(self):
        fixture = self.fixtures["single"]
        payload = torch.load(fixture["run"] / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(payload["config"]["class_weight"], "balanced")
        payload.pop("class_weight")
        checkpoint = self.directory / "missing_training_weights.pt"
        torch.save(payload, checkpoint)
        with self.assertRaisesRegex(ValueError, "(?i)(class.weight|training.*weight|weight.*training)"):
            self.evaluate(checkpoint=checkpoint)


if __name__ == "__main__":
    unittest.main()
