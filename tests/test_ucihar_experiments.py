"""Six-class training ablations preserve subjects, labels and window durations."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from loaders.ucihar_event_loader import UCIHAREventLoaderConfig, prepare_windows
from run_ucihar_experiments import main, window_conditions
from train_ucihar import build_model, get_args, main as train_main


class UCIExperimentsTests(unittest.TestCase):
    def test_resampling_and_cropping_preserve_six_labels_and_subjects(self):
        x = np.zeros((6, 128, 6), dtype=np.float32)
        x[:, :, 0] = np.arange(128) / 50
        y, subjects = np.arange(6), np.arange(6) + 10
        for condition, count in zip(window_conditions(), (3, 1, 1)):
            raw = condition["raw_samples"]
            cfg = UCIHAREventLoaderConfig(raw_win=raw, sample_rate_hz=100,
                                          window_stride=raw // 2, out_win=raw * 10)
            windows, labels, domains = prepare_windows(x, y, subjects, cfg)
            self.assertEqual(windows.shape, (6 * count, raw, 6))
            np.testing.assert_array_equal(labels, np.repeat(y, count))
            np.testing.assert_array_equal(domains, np.repeat(subjects, count))
            np.testing.assert_allclose(windows[0, :min(raw, 255), 0],
                                       np.arange(min(raw, 255)) / 100, atol=1e-6)
            if raw == 256:
                self.assertAlmostEqual(float(windows[0, -1, 0]), 2.54, places=5)

    def test_no_offset_removes_the_branch_and_checkpoint_reconstructs(self):
        for enabled in (True, False):
            cfg = get_args([] if enabled else ["--no-offset"])
            model = build_model(cfg, torch.device("cpu"))
            self.assertEqual(hasattr(model, "offset_proj_spk"), enabled)
            self.assertEqual(model.head1.in_features, 72 if enabled else 64)
            restored = build_model(cfg, torch.device("cpu"))
            restored.load_state_dict(model.state_dict(), strict=True)

    def test_six_training_conditions_reuse_three_caches_and_save_metrics(self):
        rng = np.random.default_rng(42)
        raw = SimpleNamespace(meta={"split_subjects": {"train": [0], "val": [1], "test": [2]}})
        for name, subject in (("train", 0), ("val", 1), ("test", 2)):
            setattr(raw, f"X_{name}", rng.normal(size=(6, 128, 6)).astype(np.float32))
            setattr(raw, f"y_{name}", np.arange(6))
            setattr(raw, f"d_{name}", np.full(6, subject))
        commands = []
        def execute(command, **kwargs):
            commands.append(command)
            train_main(command[2:] + ["--conv-ch", "4", "4", "4", "--kernels", "3", "3", "3"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("run_ucihar_experiments.subprocess.run", side_effect=execute), \
                    patch("loaders.ucihar_event_loader._load_ucihar_preprocessed", return_value=raw) as load, \
                    patch("train_ucihar.get_device", return_value=torch.device("cpu")), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(["--output-dir", str(root / "results"), "--cache-dir", str(root / "cache"),
                      "--epochs", "1", "--workers", "0", "--batch-size", "6"])
            self.assertEqual(load.call_count, 3)
            self.assertEqual(len(commands), 6)
            rows = json.loads((root / "results" / "summary.json").read_text())
            self.assertEqual(len(rows), 6)
            for row in rows:
                run = Path(row["run_dir"])
                cfg = json.loads((run / "config.json").read_text())
                self.assertEqual(cfg["n_classes"], 6)
                self.assertEqual(cfg["use_offset"], row["use_offset"])
                metadata = json.loads((run / "dataset_metadata.json").read_text())
                self.assertEqual(metadata["split_subjects"], raw.meta["split_subjects"])
                truth = np.load(run / "metrics" / "y_true.npy")
                np.testing.assert_array_equal(np.unique(truth), np.arange(6))
                self.assertIn("test_macro_recall", row)
                checkpoint = torch.load(run / "checkpoints" / "best_model.pt", weights_only=True)
                model = build_model(SimpleNamespace(**checkpoint["config"]), torch.device("cpu"))
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.assertTrue((root / "results" / "comparison.png").is_file())
            self.assertTrue((root / "results" / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
