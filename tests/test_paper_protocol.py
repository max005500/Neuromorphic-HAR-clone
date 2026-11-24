"""Focused regression tests for the publication front-end protocol."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from calibrate_paper_theta import main as calibrate
from energy.measure_consumption import SCNEnergyProfiler
from evaluate_paper_run import evaluate
from evaluate_paper_continuous import evaluate_continuous
from loaders.ucihar_event_loader import UCIHAREventLoaderConfig, build_event_split
from models.models import SingleHeadSCN
from preprocess.frontends import encode_window
from preprocess.preprocess_ucihar import UCIHARRawData
from preprocess.preprocess_sisfall import load_sisfall_3class
from run_paper_experiments import main as prepare
from report_paper_results import main as report_results
from theta_optimization.paper_calibration import calibrate_training_theta
from train.paper_protocol import calibration_window_config, verify_calibration_subjects, verify_publication_data
from train_ucihar import main as train_ucihar
from train_sisfall import main as train_sisfall


class FrontendTests(unittest.TestCase):
    def test_raw_native_and_event_shapes_preserve_labels(self):
        x = np.random.default_rng(4).normal(size=(3, 128, 6)).astype(np.float32)
        cfg = UCIHAREventLoaderConfig(input_mode="raw", out_win=128)
        raw = build_event_split("train", x, np.arange(3), np.arange(3), cfg, verbose=False)
        np.testing.assert_array_equal(raw["x_ev"], x)
        self.assertEqual(raw["x_ev"].dtype, np.float32)
        events = build_event_split("train", x, np.arange(3), np.arange(3),
                                   replace(cfg, input_mode="sigma_delta", out_win=640), verbose=False)
        self.assertEqual(events["x_ev"].shape, (3, 640, 6, 2))
        np.testing.assert_array_equal(events["y"], raw["y"])
        np.testing.assert_array_equal(events["offset"], x[:, 0])
        with self.assertRaises(ValueError):
            encode_window(x[0], mode="invalid", out_win=128, theta=np.ones(6))

    def test_raw_ann_and_lif_charge_dense_first_convolution_even_for_zeros(self):
        for activation in ("relu", "lif"):
            with self.subTest(activation=activation):
                model = SingleHeadSCN(n_classes=6, input_mode="raw", activation=activation,
                                      conv_channels=(4, 4, 4), conv_kernel_sizes=(3, 3, 3),
                                      use_offset=False, p_drop=0).eval()
                profiler = SCNEnergyProfiler(model)
                profiler.register()
                try:
                    with torch.no_grad():
                        model(torch.zeros(2, 128, 6), offset=torch.zeros(2, 6))
                    profiler.add_batch(2)
                    summary, rows = profiler.summarize()
                finally:
                    profiler.remove()
                first = next(row for row in rows if row["layer"] == "conv1")
                self.assertEqual(first["dense_equiv_macs_total"], 2 * 32 * 4 * 6 * 3)
                self.assertEqual(first["effective_sops_total"], first["dense_equiv_macs_total"])
                self.assertGreater(first["energy_compute_norm_total"], 0)
                self.assertGreater(summary["energy_data_norm_total"], 0)
                if activation == "relu":
                    self.assertTrue(all(row["lif_ops_total"] == 0 for row in rows))
                else:
                    self.assertGreater(first["lif_ops_total"], 0)
                self.assertFalse(profiler.handles)

    def test_train_only_theta_is_reproducible_and_provenance_rejects_mismatch(self):
        x = np.random.default_rng(3).normal(size=(9, 12, 6)).astype(np.float32)
        labels = np.repeat(np.arange(3), 3)
        result = calibrate_training_theta(x, labels, out_win=20, per_class=2, seed=7)
        repeated = calibrate_training_theta(x, labels, out_win=20, per_class=2, seed=7)
        self.assertEqual(result, repeated)
        self.assertEqual(len(result["theta"]), 6)
        self.assertEqual(len(result["selected_training_window_indices"]), 6)
        for kwargs in ({"out_win": 1}, {"out_win": 20, "per_class": 0}):
            with self.assertRaises(ValueError):
                calibrate_training_theta(x, labels, **kwargs)
        cfg = SimpleNamespace(raw_win=12, n_ch=6, dataset_root="synthetic")
        cfg.theta_provenance = dict(training_subjects=["SA01"], dataset_root="synthetic",
                                    window_config=calibration_window_config(cfg), source_fingerprint=None)
        metadata = dict(split_subjects=dict(train=["SA01"], val=["SA02"], test=["SE01"]))
        verify_calibration_subjects(cfg, metadata)
        cfg.raw_win = 14
        with self.assertRaisesRegex(ValueError, "policy"):
            verify_calibration_subjects(cfg, metadata)
        cfg.raw_win = 12
        metadata["split_subjects"]["val"] = ["SA01"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            verify_calibration_subjects(cfg, metadata)

    def test_official_counts_not_a_window_count_proxy(self):
        cfg = SimpleNamespace(publication_protocol=True, case="official", sample_rate_hz=50,
                              raw_win=128, window_stride=None)
        metadata = dict(split_subjects=dict(train=list(range(17)), val=list(range(17, 21)), test=list(range(21, 30))))
        splits = {name: {"y": np.zeros(n)} for name, n in (("train", 6000), ("val", 1352), ("test", 2947))}
        verify_publication_data(cfg, splits, metadata, "ucihar")
        cfg.sample_rate_hz = 100
        with self.assertRaisesRegex(ValueError, "50 Hz"):
            verify_publication_data(cfg, splits, metadata, "ucihar")

    def test_manifest_has_real_baselines_and_requires_five_seeds(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temporary) / "matrix"
            manifest = prepare(["--output", str(output), "--datasets", "sisfall", "ucihar"])
            self.assertEqual(len(manifest["trials"]), 50)
            self.assertEqual(len(manifest["commands"]), 60)
            self.assertTrue(manifest["publication_eligible"])
            options = {row["variant"]: row["command"] for row in manifest["trials"] if row["dataset"] == "ucihar"}
            self.assertIn("raw", options["B1"])
            self.assertIn("relu", options["B1"])
            self.assertIn("--no-offset", options["B3"])
            self.assertNotIn("--no-offset", options["B4"])
            self.assertEqual(options["B5"][options["B5"].index("--out-win") + 1], "128")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                prepare(["--output", str(Path(temporary) / "bad"), "--seeds", "0"])
            pilot = prepare(["--output", str(Path(temporary) / "pilot"), "--seeds", "0", "--pilot"])
            self.assertFalse(pilot["publication_eligible"])

    def test_calibrate_train_checkpoint_and_profile_all_five_frontends(self):
        rng = np.random.default_rng(1)
        raw = UCIHARRawData(
            X_train=rng.normal(size=(24, 128, 6)).astype(np.float32),
            y_train=np.tile(np.arange(6), 4), d_train=np.repeat(np.arange(4), 6),
            X_test=rng.normal(size=(6, 128, 6)).astype(np.float32),
            y_test=np.arange(6), d_test=np.full(6, 4),
            channel_names=("gx", "gy", "gz", "ax", "ay", "az"), root=Path("synthetic"))
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            common = ["--data-root", str(root), "--cache-dir", str(root / "cache"),
                      "--out-dir", str(root / "runs"), "--workers", "0", "--batch-size", "6",
                      "--epochs", "1", "--val-ratio", ".25", "--quiet-loader",
                      "--conv-ch", "4", "4", "4", "--kernels", "3", "3", "3", "--skip-test"]
            theta = root / "theta.json"
            with patch("preprocess.preprocess_ucihar.load_ucihar_raw", return_value=raw):
                calibrate(["--dataset", "ucihar", "--output", str(theta),
                           "--event-out-win", "160", "--per-class", "1", "--", *common])
                for variant, mode, activation, offset, length in (
                    ("B1", "raw", "relu", False, 128), ("B2", "raw", "lif", False, 128),
                    ("B3", "sigma_delta", "lif", False, 160), ("B4", "sigma_delta", "lif", True, 160),
                    ("B5", "sigma_delta", "lif", True, 128),
                ):
                    with self.subTest(variant=variant):
                        args = [*common, "--run", variant, "--cache-name", variant, "--theta-file", str(theta),
                                "--input-mode", mode, "--activation", activation, "--out-win", str(length)]
                        if not offset:
                            args.append("--no-offset")
                        run = train_ucihar(args)
                        self.assertFalse((run / "metrics/test_metrics.json").exists())
                        report = evaluate(run, dataset="ucihar", split="test", gpu=-1)
                        self.assertEqual(report["energy"]["n_samples"], 6)
                        self.assertTrue(0 <= report["macro_f1"] <= 1)
                        saved = json.loads((run / "metrics/paper_test/result.json").read_text())
                        self.assertEqual(saved["energy_unit"], "normalized_to_one_dense_MAC_not_joules")
                        self.assertEqual(np.load(run / "metrics/paper_test/confusion_matrix.npy").shape, (6, 6))
                        if variant == "B4":
                            perturbed = evaluate(
                                run, dataset="ucihar", split="test", gpu=-1, tag="dropped_gyro_x",
                                perturbation={"units": "rad/s,g", "sample_rate_hz": 50, "zero_channels": [0]})
                            self.assertEqual(perturbed["energy"]["n_samples"], 6)
                            self.assertTrue((run / "metrics/paper_test_dropped_gyro_x/result.json").is_file())
                            self.assertTrue((run / "metrics/paper_test/result.json").is_file())

    def test_sisfall_manual_heads_and_continuous_full_recordings(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            manual = []
            for subject in range(1, 7):
                for code in ("F01", "D01", "D07"):
                    path = dataset / f"{code}_SA{subject:02d}_R01.txt"
                    rows = np.round(np.sin(np.arange(128)[:, None] / 10 + np.arange(9)[None, :]) * 50).astype(int)
                    np.savetxt(path, rows, fmt="%d", delimiter=",")
                    if code == "D07":
                        manual.append(dict(file=str(path), subject=f"SA{subject:02d}", code=code,
                                           trial="R01", starts=[0], y_win=[subject % 2]))
            common = ["--dataset-root", str(dataset), "--label-dir", str(root / "labels"),
                      "--cache-dir", str(root / "cache"), "--result-dir", str(root / "posture"),
                      "--out-dir", str(root / "runs"), "--raw-win", "128", "--out-win", "128",
                      "--stride-fall", "128", "--stride-static", "128", "--stride-dynamic", "128",
                      "--conv-ch", "4", "4", "4", "--kernels", "3", "3", "3", "--workers", "0",
                      "--epochs", "1", "--batch-size", "3", "--quiet-loader", "--skip-test", "--manual-eval-only"]
            with patch("loaders.sisfall_event_loader._load_or_build_posture_tags",
                       return_value={"manual_records": manual}):
                for mode in ("dual", "single"):
                    run = train_sisfall([*common, "--head-mode", mode, "--run", mode])
                    result = evaluate(run, dataset="sisfall", split="test", gpu=-1)
                    self.assertTrue(0 <= result["macro_f1"] <= 1)
                    self.assertEqual(result["energy"]["n_samples"], 3)
                    if mode == "dual":
                        metadata = json.loads((run / "dataset_metadata.json").read_text())
                        held_out = set(metadata["split_subjects"]["test"])
                        selected = [rec for rec in metadata["recordings"] if rec["subject"] in held_out]
                        truth = root / "events.json"
                        truth.write_text(json.dumps({
                            "recordings": {Path(rec["file"]).stem: {
                                "duration_seconds": 128 / 200, "is_adl": not Path(rec["file"]).name.startswith("F")}
                                for rec in selected},
                            "fall_events": [{"recording_id": Path(rec["file"]).stem,
                                             "start_seconds": .1, "end_seconds": .64}
                                            for rec in selected if Path(rec["file"]).name.startswith("F")],
                        }))
                        continuous = evaluate_continuous(run, truth, gpu=-1)
                        self.assertEqual(continuous["fall_events"], 1)
                        self.assertAlmostEqual(continuous["adl_duration_seconds"], 2 * .64)
            original = load_sisfall_3class(dataset, cache_dir=root / "cache")[0]
            modified = dataset / "D01_SA01_R01.txt"
            modified.write_text(modified.read_text() + "1,2,3,4,5,6,7,8,9\n")
            fresh, _, records, _ = load_sisfall_3class(dataset, cache_dir=root / "cache")
            index = next(i for i, record in enumerate(records) if record["file"] == str(modified))
            self.assertEqual(len(fresh[index]), len(original[index]) + 1)

    def test_runner_executes_manifest_and_exports_reportable_results(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset = root / "UCI HAR Dataset"
            for split, subject_ids in (("train", np.repeat(np.arange(1, 5), 6)),
                                       ("test", np.repeat([5], 6))):
                directory = dataset / split / "Inertial Signals"
                directory.mkdir(parents=True)
                labels = np.tile(np.arange(1, 7), len(subject_ids) // 6)
                np.savetxt(dataset / split / f"y_{split}.txt", labels, fmt="%d")
                np.savetxt(dataset / split / f"subject_{split}.txt", subject_ids, fmt="%d")
                for channel in ("body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
                                "total_acc_x_", "total_acc_y_", "total_acc_z_"):
                    samples = np.random.default_rng(len(channel)).normal(size=(len(labels), 128))
                    np.savetxt(directory / f"{channel}{split}.txt", samples, fmt="%.5f")
            profiles = root / "profiles.json"
            profiles.write_text(json.dumps({"tiny": {"conv_ch": [4, 4, 4], "kernels": [3, 3, 3]}}))
            args = ["--datasets", "ucihar", "--ucihar-root", str(dataset), "--output", str(root / "matrix"),
                    "--profiles", str(profiles), "--pilot", "--seeds", "0", "--epochs", "1",
                    "--calibration-per-class", "1", "--evaluate-test"]
            prepare(args)
            prepare([*args, "--execute", "--resume"])
            self.assertEqual(json.loads((root / "matrix/completed_commands.json").read_text()), list(range(6)))
            self.assertEqual(report_results([str(root / "matrix/results.csv"), "--output-dir", str(root / "report"),
                                            "--energy", "energy_pe_norm", "--descriptive",
                                            "--scn-variants", "B2", "B3", "B4", "B5"]), 0)
            with patch("run_paper_experiments.subprocess.run") as execute:
                prepare([*args, "--execute", "--resume"])
            self.assertEqual(execute.call_count, 10)
            self.assertTrue(all(Path(call.args[0][1]).name == "evaluate_paper_run.py"
                                for call in execute.call_args_list))


if __name__ == "__main__":
    unittest.main()
