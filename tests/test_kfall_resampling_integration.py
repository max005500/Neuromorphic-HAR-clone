"""Keep 200 Hz provenance consistent across training, evaluation and plots."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from compare_kfall_windows import (
    build_experiment_specs, build_parser, inspect_dataset, main as compare_main,
    training_command,
)
from evaluate_kfall import evaluate_run
from loaders.kfall_event_loader import signal_to_events
from test_kfall_training import fixture_args, write_fixture
from test_kfall_window_comparison import raw_fixture
from train_kfall import get_args as training_args, main as train_main
from visualization.kfall_data import KFallWindowStore
from visualization.kfall_plotting import draw_window


class ResampledCheckpointIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        write_fixture(cls.root)
        arguments = fixture_args(cls.root) + [
            "--post-impact-seconds", "1.2", "--resample-hz", "200", "--device", "cpu",
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            train_main(arguments)
        cls.run_dir = next((cls.root / "runs").iterdir())
        cls.checkpoint = torch.load(cls.run_dir / "checkpoints" / "best_model.pt",
                                    map_location="cpu", weights_only=True)
        config = cls.checkpoint["config"]
        cls.cache = Path(config["cache_dir"]) / f"kfall_event_loader_{config['cache_name']}"

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_best_resampled_checkpoint_preserves_test_predictions_and_frequency(self):
        destination = self.root / "evaluation_200hz"
        metrics = evaluate_run(self.run_dir, device="cpu", output_dir=destination)
        self.assertEqual(metrics["resample_hz"], 200)
        self.assertEqual(metrics["n_classes"], 4)
        self.assertIn(3, np.load(destination / "y_true.npy"))
        np.testing.assert_array_equal(np.load(destination / "y_true.npy"),
                                      np.load(self.run_dir / "metrics" / "y_true.npy"))
        np.testing.assert_array_equal(np.load(destination / "y_pred.npy"),
                                      np.load(self.run_dir / "metrics" / "y_pred.npy"))
        self.assertEqual(np.load(destination / "probabilities.npy").shape[1], 4)

    def test_checkpoint_cannot_silently_use_a_different_analog_rate(self):
        altered = deepcopy(self.checkpoint)
        altered["config"]["resample_hz"] = None
        checkpoint = self.root / "incorrect_rate.pt"
        torch.save(altered, checkpoint)
        destination = self.root / "incorrect_rate_output"
        with self.assertRaisesRegex(ValueError, "resample_hz differs"):
            evaluate_run(self.run_dir, checkpoint=checkpoint, device="cpu", output_dir=destination)
        self.assertFalse(destination.exists())

    def test_postimpact_viewer_uses_same_200_hz_signal_and_cached_events(self):
        store = KFallWindowStore(self.cache)
        index = int(store.get_indices("test", class_id=3)[0])
        window = store.get_window("test", index)
        self.assertEqual(window.class_id, 3)
        self.assertEqual(window.label, "Postimpact")
        self.assertEqual(window.resample_hz, 200)
        self.assertEqual(window.raw.shape, (205, 6))
        self.assertEqual(window.analog.shape, (409, 6))
        np.testing.assert_allclose(np.diff(window.time_analog), .005, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(window.analog[::2], window.raw)
        self.assertEqual(window.time_analog[-1], window.time_raw[-1])
        expected_events, expected_offset = signal_to_events(window.raw, window.frames, store.cfg)
        np.testing.assert_array_equal(window.events, expected_events)
        np.testing.assert_array_equal(window.offset, expected_offset)
        self.assertLessEqual(window.lead_time_ms, 0)
        self.assertLess(window.time_raw[-1], window.post_impact_end_time)

        figure = Figure(figsize=(10, 8))
        FigureCanvasAgg(figure)
        self.addCleanup(figure.clear)
        axes = figure.subplots(3, 2)
        draw_window(axes, window)
        figure.canvas.draw()
        for row in range(3):
            markers = [line for line in axes[row, 0].lines
                       if line.get_label() == "Interpolada (200 Hz)"]
            self.assertEqual(len(markers), 1)
            self.assertEqual(markers[0].get_marker(), ".")
            self.assertEqual(len(markers[0].get_xdata()), 409)
            np.testing.assert_array_equal(markers[0].get_ydata(), window.analog[:, row])


class ResampledComparisonIntegrationTests(unittest.TestCase):
    def test_training_auto_names_include_encoding_identity_and_honor_explicit_names(self):
        baseline = training_args(["--post-impact-seconds", "0"])
        self.assertEqual(baseline.cache_name, "default")
        first_values = [.05, .05, .05, 5., 5., 5.]
        second_values = [.15, .15, .15, 15., 15., 15.]
        first = training_args(["--resample-hz", "200", "--theta", *map(str, first_values)])
        second = training_args(["--resample-hz", "200", "--theta", *map(str, second_values)])
        self.assertNotEqual(first.cache_name, second.cache_name)
        self.assertIn("_enc", first.cache_name)
        self.assertNotEqual(first.cache_name, training_args(["--resample-hz", "200"]).cache_name)
        with tempfile.TemporaryDirectory() as temporary:
            theta_file = Path(temporary) / "theta.json"
            theta_file.write_text(json.dumps({"theta_best": first_values}))
            equivalent = training_args(["--resample-hz", "200", "--theta-file", str(theta_file)])
            self.assertEqual(first.cache_name, equivalent.cache_name)
            explicit = training_args(["--cache-name", "chosen", "--theta-file", str(theta_file)])
            self.assertEqual(explicit.cache_name, "chosen")
        for args in (["--dead-zone", ".25"], ["--sd-init", "zero"]):
            self.assertNotEqual(training_args(args).cache_name, training_args([]).cache_name)

    def test_200_hz_comparison_keeps_labels_and_passes_distinct_cache_to_training(self):
        args = build_parser().parse_args(["--split", "1", "1", "1", "--resample-hz", "200"])
        native = build_experiment_specs()
        resampled = build_experiment_specs(resample_hz=200)
        self.assertTrue(set(item.cache_name for item in native).isdisjoint(
            item.cache_name for item in resampled))
        with patch("preprocess.preprocess_kfall.load_kfall_3class", return_value=raw_fixture()):
            native_plan = inspect_dataset(args, native)
            new_plan = inspect_dataset(args, resampled)
        for old, new, experiment in zip(native_plan["experiments"], new_plan["experiments"], resampled):
            for split in ("train", "val", "test"):
                self.assertEqual(old["selected"][split]["endpoint_label_sha256"],
                                 new["selected"][split]["endpoint_label_sha256"])
                self.assertEqual(old["selected"][split]["class_counts"],
                                 new["selected"][split]["class_counts"])
                self.assertEqual(old["selected"][split]["subjects"],
                                 new["selected"][split]["subjects"])
            command = training_command(args, experiment)
            config = training_args(command[2:])
            self.assertEqual(config.resample_hz, 200)
            self.assertEqual(config.cache_name, experiment.cache_name)
            self.assertEqual(config.endpoint_context, 410)
            self.assertEqual(config.raw_win, experiment.raw_win)
            self.assertEqual(config.out_win, experiment.out_win)

    def test_theta_file_fingerprint_protects_existing_baseline_and_other_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            theta_file = root / "best_theta.json"

            def inspect(name, theta=None):
                arguments = ["--split", "1", "1", "1", "--resample-hz", "200",
                             "--results-dir", str(root / name)]
                if theta is not None:
                    theta_file.write_text(json.dumps({"theta_best": theta}))
                    arguments.extend(["--theta-file", str(theta_file)])
                with patch("preprocess.preprocess_kfall.load_kfall_3class", return_value=raw_fixture()), \
                        patch("compare_kfall_windows._coverage_plot"), \
                        patch("compare_kfall_windows.subprocess.run", side_effect=AssertionError("inspect trained")), \
                        contextlib.redirect_stdout(io.StringIO()):
                    compare_main(arguments)
                return json.loads((root / name / "window_plan.json").read_text())

            baseline = inspect("baseline")
            first_values = [.05, .05, .05, 5., 5., 5.]
            second_values = [.15, .15, .15, 15., 15., 15.]
            first = inspect("first", first_values)
            same = inspect("same", first_values)
            second = inspect("second", second_values)
            groups = [{experiment["cache_name"] for experiment in plan["experiments"]}
                      for plan in (baseline, first, same, second)]
            self.assertTrue(groups[0].isdisjoint(groups[1]))
            self.assertTrue(groups[0].isdisjoint(groups[3]))
            self.assertTrue(groups[1].isdisjoint(groups[3]))
            self.assertEqual(groups[1], groups[2])
            for command in first["commands"]:
                config = training_args(command[2:])
                self.assertEqual(config.theta, first_values)
                self.assertEqual(config.resample_hz, 200)
            baseline_again = json.loads((root / "baseline" / "window_plan.json").read_text())
            self.assertEqual(baseline_again, baseline)


if __name__ == "__main__":
    unittest.main()
