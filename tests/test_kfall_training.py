"""Exercise KFall training through a checkpoint and masked dual-head gradients."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from openpyxl import Workbook

from train_kfall import build_data, build_model, get_args, main
from train_sisfall import run_epoch_dual
from train.utils import make_grad_scaler


def write_fixture(root: Path) -> None:
    labels = root / "label_data"
    labels.mkdir()
    for subject in ("06", "07", "08"):
        sensor = root / "sensor_data" / f"SA{subject}"
        sensor.mkdir(parents=True)
        for task in (1, 4, 20):
            with (sensor / f"S{subject}T{task:02d}R01.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["TimeStamp(s)", "FrameCounter", "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"])
                for frame in range(1, 302):
                    phase = frame / 50.0
                    writer.writerow([(frame - 1) / 100.0, frame, np.sin(phase), 0.1, 1.0, np.cos(phase) * 20, 0.0, 1.0])
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["Task Code (Task ID)", "Trial ID", "Fall_onset_frame", "Fall_impact_frame"])
        sheet.append(["F01 (20)", 1, 150, 250])
        workbook.save(labels / f"SA{subject}_label.xlsx")
        workbook.close()


def fixture_args(root: Path) -> list[str]:
    return [
        "--dataset-root", str(root), "--cache-dir", str(root / "cache"),
        "--out-dir", str(root / "runs"), "--split", "1", "1", "1",
        "--epochs", "1", "--batch-size", "3", "--workers", "0",
        "--out-win", "128", "--conv-ch", "4", "4", "4",
        "--kernels", "3", "3", "3", "--quiet-loader",
    ]


class KFallTrainingTests(unittest.TestCase):
    def test_single_head_cli_saves_checkpoint_and_three_class_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                main(fixture_args(root))
            run = next((root / "runs").iterdir())
            checkpoint = torch.load(run / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["config"]["n_classes"], 3)
            self.assertFalse(any(key.startswith("head2") for key in checkpoint["model_state_dict"]))
            for chart in ("accuracy.png", "loss.png"):
                self.assertEqual((run / "metrics" / chart).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            true = np.load(run / "metrics" / "y_true.npy")
            pred = np.load(run / "metrics" / "y_pred.npy")
            np.testing.assert_array_equal(np.unique(true), [0, 1, 2])
            self.assertEqual(pred.shape, true.shape)
            self.assertTrue(np.isin(pred, [0, 1, 2]).all())
            summary = json.loads((run / "metrics" / "test_metrics.json").read_text())
            self.assertTrue(np.isfinite(summary["loss"]))
            self.assertFalse(summary["head2_supervised"])
            metadata = json.loads((run / "dataset_metadata.json").read_text())
            subjects = [set(value) for value in metadata["split_subjects"].values()]
            self.assertTrue(all(not left & right for i, left in enumerate(subjects) for right in subjects[i + 1:]))

    def test_dual_head_does_not_train_unannotated_posture_classifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            cfg = get_args(fixture_args(root) + ["--head-mode", "dual"])
            device = torch.device("cpu")
            data = build_data(cfg, device)
            model = build_model(cfg, device)
            previous = {key: value.detach().clone() for key, value in model.head2.state_dict().items()}
            optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=0.001)
            metrics = run_epoch_dual(model, data["loaders"]["train"], device, cfg, pos_weight_h2=1.0,
                                     optimizer=optimizer, scaler=make_grad_scaler(False))
            self.assertEqual(metrics["n_h2"], 0)
            self.assertEqual(metrics["loss_h2"], 0.0)
            self.assertTrue(np.isfinite(metrics["loss"]))
            for key, value in model.head2.state_dict().items():
                torch.testing.assert_close(value, previous[key])
            self.assertTrue(all(param.grad is None for param in model.head2.parameters()))


if __name__ == "__main__":
    unittest.main()
