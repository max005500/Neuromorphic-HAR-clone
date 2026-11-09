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
from evaluate_kfall import evaluate_run
from visualization.kfall_data import KFallWindowStore


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
        "--post-impact-seconds", "0", "--no-manual-labels",
    ]


class KFallTrainingTests(unittest.TestCase):
    def test_manual_normal_labels_and_postimpact_train_evaluate_and_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            annotations = root / "manual.json"
            annotations.write_text(json.dumps({
                "format": "kfall_manual_normal_labels", "version": 1,
                "dataset": "kfall", "sample_rate_hz": 100,
                "recordings": {
                    f"SA{subject}T01R01": {
                        "file": f"S{subject}T01R01.csv", "reviewed": True,
                        "intervals": [[1, 200, 1], [200, 302, 2]],
                    } for subject in ("06", "07", "08")
                },
            }))
            args = fixture_args(root)
            args.remove("--no-manual-labels")
            with contextlib.redirect_stdout(io.StringIO()):
                main(args + ["--post-impact-seconds", "1.2", "--raw-win", "120",
                             "--annotation-file", str(annotations)])
            run = next((root / "runs").iterdir())
            checkpoint = torch.load(run / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["config"]["n_classes"], 4)
            # T01's first windows override its native Static task label.
            truth = np.load(run / "metrics" / "y_true.npy")
            self.assertIn(3, truth)
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_run(run, device="cpu")
            self.assertIn("PostImpact", result["per_class"])
            self.assertEqual(np.load(run / "metrics" / "test_best" / "probabilities.npy").shape[1], 4)
            store = KFallWindowStore(root / "cache" / "kfall_event_loader_default")
            index = int(store.get_indices("test", class_id=3)[0])
            window = store.get_window("test", index)
            self.assertEqual(window.label, "PostImpact")
            self.assertIsNone(window.lead_time_ms)

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

    def test_no_offset_cli_trains_and_saves_reconstructible_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                main(fixture_args(root) + ["--no-offset"])
            run = next((root / "runs").iterdir())
            checkpoint = torch.load(run / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=False)
            self.assertFalse(checkpoint["config"]["use_offset"])
            self.assertFalse(any(key.startswith("offset_proj_spk.") for key in checkpoint["model_state_dict"]))
            cfg = get_args(fixture_args(root) + ["--no-offset"])
            model = build_model(cfg, torch.device("cpu"))
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            spiking_cfg = get_args(fixture_args(root) + ["--no-offset", "--spiking-heads"])
            spiking_model = build_model(spiking_cfg, torch.device("cpu"))
            logits, _ = spiking_model(torch.zeros(2, 128, 6, 2), offset=torch.ones(2, 6))
            self.assertEqual(tuple(logits.shape), (2, 3))
            summary = json.loads((run / "metrics" / "test_metrics.json").read_text())
            self.assertFalse(summary["use_offset"])
            self.assertGreater(summary["n"], 0)
            self.assertEqual(summary["n"], len(np.load(run / "metrics" / "y_true.npy")))

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
