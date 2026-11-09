"""External evaluation uses KFall channels/units and never invents falls."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from evaluate_kfall_ucihar import UCIKFallDataset, evaluate_ucihar
from loaders.kfall_event_loader import KFallEventLoaderConfig
from test_kfall_training import fixture_args, write_fixture
from train_kfall import main


class KFallUCIHARTests(unittest.TestCase):
    def test_channels_units_and_temporal_context(self):
        signals = np.zeros((1, 128, 6), dtype=np.float32)
        signals[:, :, :3] = np.pi
        signals[:, :, 3:] = [1, 2, 3]
        cfg = KFallEventLoaderConfig(raw_win=120, out_win=1200)
        dataset = UCIKFallDataset(signals, np.array([0]), np.array([6]), cfg)
        np.testing.assert_allclose(dataset.signals[0, 0], [1, 2, 3, 180, 180, 180])
        self.assertEqual(dataset.windows, [(0, 0), (0, 61)])
        item = dataset[1]
        self.assertEqual(item["x"].shape, (1200, 6, 2))
        np.testing.assert_allclose(item["offset"], [1, 2, 3, 180, 180, 180])
        with self.assertRaisesRegex(ValueError, "No padding"):
            UCIKFallDataset(signals, np.array([0]), np.array([6]),
                            KFallEventLoaderConfig(raw_win=320, out_win=3200))
        signals[0, :, 3] = np.arange(128) / 50
        dataset = UCIKFallDataset(signals, np.array([0]), np.array([6]), cfg)
        captured = []
        def encode(signal, **kwargs):
            captured.append(signal)
            return np.zeros((cfg.out_win, 6, 2), dtype=np.uint8), None, None
        with patch("evaluate_kfall_ucihar.sigma_delta", side_effect=encode):
            dataset[0]
        np.testing.assert_allclose(captured[0][:, 0], np.linspace(0, 1.19, 1200), atol=1e-6)

    def test_saved_checkpoint_reports_negative_only_data_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            with patch("train_kfall.get_device", return_value=torch.device("cpu")), contextlib.redirect_stdout(io.StringIO()):
                main(fixture_args(root) + ["--post-impact-seconds", "1.2"])
            run = next((root / "runs").iterdir())
            checkpoint = run / "checkpoints" / "best_model.pt"
            original = checkpoint.read_bytes()
            signals = np.zeros((6, 128, 6), dtype=np.float32)
            signals[:, :, 5] = 1
            data = (signals, np.arange(6), np.arange(6))
            with patch("evaluate_kfall_ucihar.load_ucihar_split", return_value=data) as load:
                metrics = evaluate_ucihar(run, dataset_root=root / "uci", device="cpu", batch_size=3)
            load.assert_called_once_with(root / "uci", "test")
            self.assertEqual(checkpoint.read_bytes(), original)
            self.assertEqual(metrics["class_counts"]["Fall"], 0)
            self.assertEqual(metrics["class_counts"]["Dynamic"], 3)
            self.assertEqual(metrics["class_counts"]["Static"], 3)
            self.assertEqual(metrics["class_counts"]["PostImpact"], 0)
            self.assertIsNone(metrics["fall_window_metrics"]["recall"])
            predicted = np.load(Path(metrics["output_dir"]) / "y_pred.npy")
            self.assertEqual(metrics["fall_window_metrics"]["false_fall_windows"], int(np.sum(predicted == 0)))
            self.assertAlmostEqual(metrics["fall_window_metrics"]["false_positive_rate"],
                                   float(np.mean(predicted == 0)))
            self.assertEqual(len(metrics["per_subject"]), 6)
            self.assertIn("per_class", metrics["per_subject"]["UCI01"])
            self.assertEqual(len(metrics["per_source_activity"]), 6)
            saved = json.loads((Path(metrics["output_dir"]) / "metrics.json").read_text())
            self.assertEqual(saved["dataset"], "ucihar")
            self.assertTrue((Path(metrics["output_dir"]) / "predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
