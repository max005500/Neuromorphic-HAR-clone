"""Observed-only 200 Hz interpolation and held-out threshold-search training."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from loaders.kfall_event_loader import (
    KFallEventLoaderConfig, _encode_event_batch, load_or_build_kfall_event_splits,
    signal_to_events,
)
from preprocess.kfall_resampling import resample_kfall_signal
from train_kfall import get_args, main, read_theta_file, training_subset_indices
from test_kfall_training import fixture_args, write_fixture


class AnalogResamplingTests(unittest.TestCase):
    def test_exact_200_hz_grid_keeps_observed_endpoints_and_native_values(self):
        for size in (205, 330, 410):
            frames = np.arange(501, 501 + size)
            signal = np.column_stack([np.arange(size)] * 6).astype(np.float32)
            values, times = resample_kfall_signal(signal, frames, resample_hz=200)
            self.assertEqual(len(values), 2 * size - 1)
            self.assertEqual(times[-1], (size - 1) / 100)
            np.testing.assert_allclose(np.diff(times), 1 / 200, rtol=0, atol=1e-15)
            np.testing.assert_array_equal(values[::2], signal)
            np.testing.assert_array_equal(values[-1], signal[-1])
            np.testing.assert_allclose(values[1], (signal[0] + signal[1]) / 2)

    def test_missing_native_frames_are_filled_within_observed_times(self):
        frames = np.array([101, 102, 105])
        signal = np.column_stack([frames - frames[0]] * 6).astype(np.float32)
        values, times = resample_kfall_signal(signal, frames, resample_hz=200)
        self.assertEqual(len(values), 9)
        self.assertEqual(times[-1], .04)
        np.testing.assert_allclose(values[:, 0], np.arange(9) / 2)
        self.assertTrue(np.all(times <= .04))

    def test_nondividing_rate_adds_observed_endpoint_without_extrapolation(self):
        signal = np.arange(18).reshape(3, 6)
        values, times = resample_kfall_signal(signal, np.array([1, 2, 3]), resample_hz=125)
        np.testing.assert_allclose(times, [0, .008, .016, .02])
        np.testing.assert_array_equal(values[-1], signal[-1])

    def test_invalid_rates_and_unsorted_frames_are_rejected(self):
        for rate in (0, 99, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                KFallEventLoaderConfig(resample_hz=rate)
        with self.assertRaises(ValueError):
            resample_kfall_signal(np.zeros((3, 6)), [1, 1, 2], resample_hz=200)

    def test_200_hz_batch_and_scalar_encoders_remain_bit_exact(self):
        random = np.random.default_rng(11)
        for size, steps in ((205, 2048), (330, 3297), (410, 4096)):
            for initialization in ("x0", "zero"):
                cfg = KFallEventLoaderConfig(raw_win=size, out_win=steps, resample_hz=200,
                                             sd_init=initialization)
                windows = [random.normal(size=(size, 6)).astype(np.float32) for _ in range(2)]
                frames = [np.cumsum(random.integers(1, 3, size=size)) for _ in windows]
                batch, offsets = _encode_event_batch(windows, frames, cfg)
                for index, (window, frame) in enumerate(zip(windows, frames)):
                    expected, offset = signal_to_events(window, frame, cfg)
                    np.testing.assert_array_equal(batch[index], expected)
                    np.testing.assert_array_equal(offsets[index], offset)

    def test_opt_in_rate_has_own_cache_identity_and_legacy_metadata_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            cfg = KFallEventLoaderConfig(raw_win=205, out_win=32, split=(1, 1, 1))
            _, legacy = load_or_build_kfall_event_splits(root, cache_dir=root / "cache", cfg=cfg,
                                                        verbose=False)
            self.assertEqual(legacy["version"], 1)
            self.assertNotIn("resample_hz", legacy["config"])
            with patch("loaders.kfall_event_loader.load_kfall_3class", side_effect=AssertionError("rebuilt")):
                _, reused = load_or_build_kfall_event_splits(root, cache_dir=root / "cache", cfg=cfg,
                                                            verbose=False)
            self.assertEqual(reused["source_signature"], legacy["source_signature"])
            _, new = load_or_build_kfall_event_splits(root, cache_dir=root / "cache", cfg=replace(cfg, resample_hz=200),
                                                     verbose=False)
            self.assertEqual(new["version"], 3)
            self.assertEqual(new["config"]["resample_hz"], 200)
            self.assertEqual(new["analog_resampling"]["event_steps"], 32)
            original = get_args([])
            upsampled = get_args(["--resample-hz", "200"])
            self.assertNotEqual(original.cache_name, upsampled.cache_name)


class ThresholdSearchTrainingTests(unittest.TestCase):
    def test_theta_file_rejects_other_datasets_and_accepts_kfall_or_unspecified(self):
        thresholds = [.05, .08, .09, 5., 7., 9.]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "theta.json"
            for dataset in ("sisfall", "ucihar"):
                path.write_text(json.dumps({"dataset": dataset, "theta_best": thresholds}))
                with self.subTest(dataset=dataset), self.assertRaisesRegex(ValueError, "must belong to KFall"):
                    read_theta_file(path)
            for document in ({"dataset": "kfall", "theta_best": thresholds}, {"theta_best": thresholds}):
                path.write_text(json.dumps(document))
                self.assertEqual(read_theta_file(path), thresholds)

    def test_theta_file_loads_optimizer_and_classifier_search_results(self):
        expected = [.05, .08, .09, 5, 7, 9]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "theta.json"
            for key in ("theta_best", "best_theta"):
                path.write_text(json.dumps({key: expected}))
                self.assertEqual(read_theta_file(path), expected)
                self.assertEqual(get_args(["--theta-file", str(path)]).theta, expected)
            path.write_text(json.dumps({"theta_best": [1, 2, 3]}))
            with self.assertRaises(ValueError):
                read_theta_file(path)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                get_args(["--theta-file", str(path), "--theta", "1", "1", "1", "1", "1", "1"])

    def test_training_sampling_is_stratified_reproducible_and_without_replacement(self):
        labels = np.repeat([0, 1, 2, 3], [20, 10, 5, 2])
        first = training_subset_indices(labels, 4, 8)
        np.testing.assert_array_equal(first, training_subset_indices(labels, 4, 8))
        self.assertEqual(len(np.unique(first)), len(first))
        np.testing.assert_array_equal(np.bincount(labels[first]), [4, 4, 4, 2])
        np.testing.assert_array_equal(training_subset_indices(labels, 0, 8), np.arange(len(labels)))

    def test_validation_only_uses_full_validation_and_never_runs_test_inference(self):
        import train_kfall

        real_runner = train_kfall.run_epoch_single
        observed_subjects = []

        def checked_runner(model, loader, *args, **kwargs):
            dataset = loader.dataset
            if hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            observed_subjects.append(set(np.asarray(dataset.split["subject"]).tolist()))
            return real_runner(model, loader, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            args = fixture_args(root) + ["--resample-hz", "200", "--validation-only", "--train-per-class", "1"]
            with contextlib.redirect_stdout(io.StringIO()), patch("train_kfall.run_epoch_single", checked_runner):
                main(args)
            run = next((root / "runs").iterdir())
            metadata = json.loads((run / "dataset_metadata.json").read_text())
            test_subjects = set(metadata["split_subjects"]["test"])
            self.assertTrue(all(not subjects & test_subjects for subjects in observed_subjects))
            self.assertFalse((run / "metrics" / "test_metrics.json").exists())
            self.assertFalse((run / "metrics" / "y_true.npy").exists())
            validation = json.loads((run / "metrics" / "validation_metrics.json").read_text())
            self.assertTrue(np.isfinite(validation["macro_f1"]))
            self.assertTrue(np.isfinite(validation["balanced_accuracy"]))
            actual_val = np.load(run / "metrics" / "y_true_val.npy")
            expected_val = np.load(root / "cache" / "kfall_event_loader_default" / "val" / "y_h1.npy")
            np.testing.assert_array_equal(actual_val, expected_val)
            subset = json.loads((run / "training_subset.json").read_text())
            self.assertEqual(subset["selected_training_windows"], 3)
            self.assertEqual(sum(subset["class_counts"].values()), 3)


if __name__ == "__main__":
    unittest.main()
