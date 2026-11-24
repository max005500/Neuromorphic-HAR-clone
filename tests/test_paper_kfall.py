"""KFall publication protocol: real frontends, labels, windows and held-out metrics."""

import contextlib
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from openpyxl import load_workbook
import torch

from calibrate_paper_theta import main as calibrate
from evaluate_paper_continuous import evaluate_continuous
from evaluate_paper_run import evaluate
from evaluate_kfall import evaluate_run as evaluate_original
from loaders.kfall_event_loader import KFallEventLoaderConfig, signal_to_events, window_frame_arrays
from report_paper_results import main as report_results
from run_paper_experiments import main as prepare
from test_kfall_training import write_fixture
from train_kfall import build_data, get_args


def write_paper_fixture(root):
    write_fixture(root)
    annotations = {}
    for subject in ("06", "07", "08"):
        sensor = root / "sensor_data" / f"SA{subject}"
        for task in (1, 4, 20):
            path = sensor / f"S{subject}T{task:02d}R01.csv"
            with path.open("a", newline="") as stream:
                writer = csv.writer(stream)
                for frame in range(302, 1001):
                    writer.writerow([(frame - 1) / 100, frame, np.sin(frame / 50), .1, 1, 20 * np.cos(frame / 50), 0, 1])
        excluded = sensor / f"S{subject}T05R01.csv"
        excluded.write_text((sensor / f"S{subject}T01R01.csv").read_text())
        for task in (1, 4, 5, 20):
            annotations[f"SA{subject}T{task:02d}R01"] = {
                "file": f"S{subject}T{task:02d}R01.csv", "reviewed": True,
                "intervals": [[1, 1001, 2 if task in (1, 5) else 1]],
            }
        workbook = load_workbook(root / "label_data" / f"SA{subject}_label.xlsx")
        workbook.active["C2"] = 450
        workbook.active["D2"] = 600
        workbook.save(root / "label_data" / f"SA{subject}_label.xlsx")
        workbook.close()
    path = root / "manual.json"
    path.write_text(json.dumps({
        "format": "kfall_manual_normal_labels", "version": 1, "dataset": "kfall",
        "sample_rate_hz": 100, "recordings": annotations,
    }))
    return path


class KFallPaperTests(unittest.TestCase):
    def test_native_frontends_do_not_interpolate_missing_frames(self):
        signal = np.arange(18, dtype=np.float32).reshape(3, 6)
        frames = np.asarray([10, 11, 14])
        cfg = KFallEventLoaderConfig(raw_win=3, out_win=3, input_mode="raw")
        raw, offset = signal_to_events(signal, frames, cfg)
        np.testing.assert_array_equal(raw, signal)
        np.testing.assert_array_equal(offset, signal[0])
        events, _ = signal_to_events(signal, frames, replace(cfg, input_mode="sigma_delta"))
        self.assertEqual(events.shape, (3, 6, 2))
        recovered = window_frame_arrays({"file_idx": [0], "start": [1]},
                                        {"recordings": [{"frame_grid": [9, 10, 11, 14]}]}, 3)
        np.testing.assert_array_equal(recovered[0], frames)

    def test_manifest_includes_all_three_windows_without_mixing_them(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            manifest = prepare(["--output", str(Path(directory) / "matrix")])
            self.assertEqual(len(manifest["trials"]), 125)
            self.assertEqual(len(manifest["commands"]), 150)
            kfall = [trial for trial in manifest["trials"] if trial["dataset"] == "kfall"]
            self.assertEqual(len(kfall), 75)
            self.assertEqual({trial["raw_win"] for trial in kfall}, {120, 206, 320})
            for trial in kfall:
                command = trial["command"]
                width = trial["raw_win"]
                self.assertEqual(command[command.index("--stride") + 1], str(width // 2))
                length = width if trial["variant"] in {"B1", "B2", "B5"} else 10 * width
                self.assertEqual(command[command.index("--out-win") + 1], str(length))
                self.assertEqual(trial["head_mode"], "single")

    def test_train_only_calibration_and_raw_tensor_cache_for_each_window(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            annotation = write_paper_fixture(root)
            for width in (120, 206, 320):
                common = ["--dataset-root", str(root), "--annotation-file", str(annotation),
                          "--cache-dir", str(root / "cache"), "--cache-name", f"w{width}",
                          "--raw-win", str(width), "--out-win", str(width), "--stride", str(width // 2),
                          "--input-mode", "raw", "--cache-format", "pt", "--workers", "0", "--quiet-loader"]
                theta = root / f"theta_w{width}.json"
                calibrate(["--dataset", "kfall", "--output", str(theta), "--event-out-win", str(width * 10),
                           "--per-class", "1", "--", *common])
                calibration = json.loads(theta.read_text())
                data = build_data(get_args(common), torch.device("cpu"))
                self.assertEqual(calibration["training_subjects"], data["metadata"]["split_subjects"]["train"])
                self.assertEqual(calibration["window_config"]["exclude_tasks"],
                                 ["T05", "T13", "T14", "T16", "T18", "T19"])
                self.assertTrue(all(record["code"] != "T05" for record in data["metadata"]["recordings"]))
                self.assertTrue(all(dataset.dataset[0]["x"].dtype == torch.float32
                                    for dataset in data["loaders"].values()))
                self.assertEqual(data["splits"]["test"]["x_ev"].shape[1:], (width, 6))
                self.assertIn(3, data["splits"]["test"]["y_h1"])
                self.assertTrue(set(data["splits"]["test"]["y_h1"]) >= {0, 1, 2, 3})

    def test_full_kfall_matrix_cli_reports_and_continuous_official_truth(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            annotation = write_paper_fixture(root)
            profiles = root / "profiles.json"
            profiles.write_text(json.dumps({"tiny": {"conv_ch": [4, 4, 4], "kernels": [3, 3, 3]}}))
            output = root / "matrix"
            manifest = prepare([
                "--datasets", "kfall", "--kfall-root", str(root), "--kfall-annotation-file", str(annotation),
                "--kfall-windows", "120", "--output", str(output), "--profiles", str(profiles),
                "--pilot", "--seeds", "0", "--epochs", "1", "--calibration-per-class", "1",
                "--execute", "--evaluate-test",
            ])
            with (output / "results.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            self.assertEqual({row["window_raw_samples"] for row in rows}, {"120"})
            self.assertEqual(report_results([str(output / "results.csv"), "--output-dir", str(root / "report"),
                                            "--energy", "energy_pe_norm", "--descriptive"]), 0)
            b4 = next(trial for trial in manifest["trials"] if trial["variant"] == "B4")
            run = next(Path(b4["trial_dir"]).iterdir())
            clean = json.loads((run / "metrics/paper_test/result.json").read_text())
            self.assertTrue(clean["label_provenance"]["manual_annotation_signature"])
            self.assertTrue(list((run / "metrics/paper_test/by_subject").glob("SA*/confusion_matrix.png")))
            b1 = next(trial for trial in manifest["trials"] if trial["variant"] == "B1")
            original = evaluate_original(next(Path(b1["trial_dir"]).iterdir()), device="cpu")
            self.assertEqual(original["class_names"], ["Fall", "Dynamic", "Static", "PostImpact"])
            perturbation = {"units": "g,deg/s", "sample_rate_hz": 100, "noise_std": 0}
            evaluate(run, dataset="kfall", split="test", perturbation=perturbation, tag="identity")
            np.testing.assert_array_equal(np.load(run / "metrics/paper_test/y_pred.npy"),
                                          np.load(run / "metrics/paper_test_identity/y_pred.npy"))
            metrics = evaluate_continuous(run, dataset="kfall", stride=50)
            self.assertEqual(metrics["sample_rate_hz"], 100)
            self.assertEqual(metrics["fall_events"], 1)
            self.assertAlmostEqual(metrics["adl_duration_seconds"], 20.0)
            self.assertEqual(metrics["truth_source"], "official_KFall_XLSX_onset_to_impact")
            truth = json.loads((run / "metrics/paper_continuous/truth_used.json").read_text())
            self.assertAlmostEqual(truth["fall_events"][0]["start_seconds"], 4.49)
            self.assertAlmostEqual(truth["fall_events"][0]["end_seconds"], 5.99)


if __name__ == "__main__":
    unittest.main()
