"""Train-only theta sampling and validation-only ablation orchestration."""

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
from openpyxl import load_workbook

from calibrate_kfall_theta import reconstruction_score, select_windows, calibrate, build_parser
from run_kfall_ablations import build_command, execute_trial, export_report, trial_id, variants, main as run_ablations
from loaders.kfall_event_loader import DEFAULT_EXCLUDED_TASKS, _source_signature
from test_kfall_training import write_fixture
from train_kfall import main


class KFallAblationsTests(unittest.TestCase):
    def test_motion_sampling_is_balanced_and_training_only(self):
        rng = np.random.default_rng(42)
        signals = np.empty(24, dtype=object)
        records = []
        for i in range(24):
            signals[i] = rng.normal(size=(24, 6)).astype(np.float32) * (i + 1)
            records.append({"subject": f"SA{i % 4:02d}", "file": f"file{i}",
                            "code": "T01", "frames": np.arange(1, 25)})
        windows, manifest = select_windows(signals, records, np.arange(18), 12, 2, 0)
        self.assertEqual(windows.shape, (6, 12, 6))
        self.assertEqual([sum(row["regime"] == regime for row in manifest)
                          for regime in ("low", "medium", "high")], [2, 2, 2])
        self.assertTrue(all(int(row["file"][4:]) < 18 for row in manifest))
        _, repeated = select_windows(signals, records, np.arange(18), 12, 2, 0)
        self.assertEqual(manifest, repeated)
        errors, spikes = reconstruction_score(windows[:, :, :1], [0.1], 12, 24)
        self.assertTrue(np.isfinite(errors).all())
        self.assertTrue(((spikes >= 0) & (spikes <= 1)).all())

    def test_variants_change_one_factor_and_identity_includes_protocol(self):
        options = variants()
        baseline = options[0][1]
        for _, config in options[1:]:
            self.assertEqual(sum(value != baseline[key] for key, value in config.items()), 1)
        self.assertNotEqual(trial_id({"seed": 0}), trial_id({"seed": 1}))
        self.assertEqual(len(options), 12)

    def test_calibration_keeps_gyro_excludes_tasks_and_preserves_holdout(self):
        rng = np.random.default_rng(42)
        signals = np.empty(48, dtype=object)
        records = []
        for i in range(48):
            signals[i] = rng.normal(size=(24, 6)).astype(np.float32) * (i + 1)
            records.append({"subject": f"SA{i % 4:02d}", "file": f"file{i}",
                            "code": "T05" if i >= 44 else "T01", "frames": np.arange(1, 25)})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "theta.json"
            args = build_parser().parse_args([
                "--dataset-root", temporary, "--output", str(output), "--split", ".5", ".25", ".25",
                "--raw-win", "12", "--out-win", "24", "--per-regime", "2",
                "--candidates", ".04", ".1",
            ])
            with patch("calibrate_kfall_theta.load_kfall_3class", return_value=(signals, None, records, None)), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = calibrate(args)
            self.assertEqual(result["theta_best"][3:], result["theta_initial"][3:])
            self.assertTrue(all(row["subject"] in result["split_subjects"]["train"]
                                and row["code"] != "T05" for row in result["sampling"]))
            self.assertTrue(all(after <= before + 1e-6 for after, before in
                                zip(result["nrmse_after"][:3], result["nrmse_before"][:3])))
            self.assertEqual(len(result["history"]), 6)
            self.assertEqual(json.loads(output.read_text()), result)

    def test_trial_skips_test_reuses_success_and_writes_excel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            args = SimpleNamespace(
                dataset_root=str(root), cache_dir=str(root / "cache"), cache_name="trial",
                epochs=1, batch_size=3, workers=0, seed=0, split=[1, 1, 1],
                raw_win=120, out_win=128, stride=60, theta=[0.1] * 3 + [10] * 3,
                no_offset=False, annotation_file=None, no_manual_labels=True,
            )
            trial = {"id": "test_trial", "variant": "baseline", "config": variants()[0][1]}
            command = build_command(args, trial, 0, root)
            self.assertIn("--skip-test", command)
            def execute(command, **kwargs):
                main(command[2:] + ["--conv-ch", "4", "4", "4", "--kernels", "3", "3", "3"])
                return SimpleNamespace(returncode=0)
            with patch("run_kfall_ablations.subprocess.run", side_effect=execute), \
                    patch("train_kfall.get_device", return_value=torch.device("cpu")), \
                    contextlib.redirect_stdout(io.StringIO()):
                row = execute_trial(args, trial, 0, root)
            self.assertEqual(row["status"], "completed")
            run = Path(row["run_dir"])
            self.assertFalse((run / "metrics" / "test_metrics.json").exists())
            self.assertFalse((run / "metrics" / "y_true.npy").exists())
            with patch("run_kfall_ablations.subprocess.run", side_effect=AssertionError("rerun")):
                cached = execute_trial(args, trial, 0, root)
            self.assertEqual(cached, row)
            export_report(root, [row])
            workbook = load_workbook(root / "summary.xlsx")
            self.assertEqual(workbook.active.max_row, 2)
            workbook.close()
            self.assertEqual(json.loads((root / "summary.json").read_text())[0]["id"], trial["id"])

    def test_failed_trials_are_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(dataset_root="data/KFall", cache_dir=str(root),
                                   cache_name="x", epochs=1, batch_size=3, workers=0, seed=0,
                                   split=[.6, .3, .1], raw_win=120, out_win=1200, stride=60,
                                   theta=[.1]*6, no_offset=False, annotation_file=None, no_manual_labels=True)
            trial = {"id": "failure", "variant": "baseline", "config": variants()[0][1]}
            with patch("run_kfall_ablations.subprocess.run", return_value=SimpleNamespace(returncode=1)):
                row = execute_trial(args, trial, 0, root)
            self.assertEqual(row["status"], "failed")
            self.assertIn("error", row)

    def test_multi_gpu_orchestration_prepares_once_and_selects_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            theta = root / "theta.json"
            theta.write_text(json.dumps({
                "seed": 0, "split": [.6, .3, .1], "raw_win": 120, "out_win": 1200,
                "dead_zone": .5, "sd_init": "x0", "dataset_root": str(root),
                "theta_best": [.1]*3 + [10]*3, "excluded_tasks": list(DEFAULT_EXCLUDED_TASKS),
                "source_signature": _source_signature(root, root / "label_data"),
            }))
            def execute(args, trial, gpu, output):
                return {**trial, "gpu": gpu, "status": "completed", "val_loss": .25 if gpu == 1 else .5,
                        "val_accuracy": .8, "run_dir": str(output / trial["id"])}
            with patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.device_count", return_value=2), \
                    patch("train_kfall.main") as preprocess, \
                    patch("run_kfall_ablations.execute_trial", side_effect=execute) as run, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                run_ablations(["--dataset-root", str(root), "--theta-file", str(theta),
                               "--output-dir", str(root / "output"), "--gpus", "0", "1",
                               "--no-manual-labels"])
            preprocess.assert_called_once()
            self.assertIn("--preprocess-only", preprocess.call_args.args[0])
            self.assertEqual(run.call_count, 12)
            summary = json.loads((root / "output" / "summary.json").read_text())
            self.assertEqual(len(summary), 12)
            self.assertTrue(all(row["status"] == "completed" for row in summary))
            best = json.loads((root / "output" / "best_validation.json").read_text())
            self.assertEqual(best["val_loss"], .25)
            self.assertEqual(best["gpu"], 1)


if __name__ == "__main__":
    unittest.main()
