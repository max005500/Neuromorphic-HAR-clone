"""Keep KFall threshold calibration on the same analog grid as training."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from loaders.kfall_event_loader import KFallEventLoaderConfig, signal_to_events
from optimize_theta import build_parser
from preprocess.event_encoding import sigma_delta
from preprocess.kfall_resampling import resample_kfall_signal
from preprocess.interpolate_signal import interpolate_to_times
from theta_optimization.utils import RunConfig, ThetaOptimizer, build_split, prepare_kfall


class KFallThetaResamplingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.logger = logging.getLogger(self.id())
        self.logger.handlers = [logging.NullHandler()]
        self.logger.propagate = False

    def config(self, *extra, dataset="kfall"):
        args = build_parser().parse_args([
            "--dataset", dataset, "--ups", "10", "--dead-zone", ".5",
            "--data-root", str(self.root), "--results-root", str(self.root / "results"),
            "--split", "1", "1", "1", "--workers", "1", "--executor", "none", *extra,
        ])
        return RunConfig.from_args(args)

    def fixture(self):
        recordings, labels, metadata = [], [], []
        frames = np.arange(1, 701, dtype=np.int64)
        rng = np.random.default_rng(70)
        for subject in ("SA06", "SA07", "SA08"):
            for label in (0, 1, 2):
                recordings.append(rng.normal(size=(len(frames), 6)).astype(np.float32))
                labels.append(label)
                record = {"file": f"{subject}/S{subject[2:]}T{label:02d}R01.csv",
                          "subject": subject, "label_id": label, "T": len(frames), "frames": frames.copy()}
                if label == 0:
                    record.update(fall_onset_frame=400, fall_impact_frame=450,
                                  onset_idx=399, impact_idx=449)
                metadata.append(record)
        array = np.empty(len(recordings), dtype=object)
        array[:] = recordings
        return array, np.asarray(labels), metadata, 0

    def test_cli_preserves_legacy_default_and_scopes_resampling(self):
        self.assertIsNone(self.config().resample_hz)
        cfg = self.config("--resample-hz", "200")
        self.assertEqual(cfg.resample_hz, 200)
        self.assertEqual((cfg.raw_win, cfg.out_len), (205, 2048))
        for rate in ("0", "99", "nan", "inf"):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                self.config("--resample-hz", rate)
        for dataset in ("sisfall", "ucihar"):
            with self.subTest(dataset=dataset), self.assertRaisesRegex(ValueError, "KFall options"):
                self.config("--resample-hz", "200", dataset=dataset)

    def test_two_stage_interpolation_and_events_match_training_for_each_duration(self):
        rng = np.random.default_rng(813)
        for seconds in ("2.05", "3.30", "4.10"):
            with self.subTest(seconds=seconds):
                cfg = self.config("--window-seconds", seconds, "--resample-hz", "200")
                signal = rng.normal(size=(cfg.raw_win, 6)).astype(np.float32)
                frames = np.arange(42, 42 + cfg.raw_win)
                prepared = build_split(signal[None, ...], np.array([0]), cfg)
                analog, times = resample_kfall_signal(signal, frames, resample_hz=200)
                expected_input = interpolate_to_times(
                    analog, times, np.linspace(0, (cfg.raw_win - 1) / 100, cfg.out_len, dtype=np.float32),
                )
                np.testing.assert_array_equal(prepared.x_interp[0], expected_input)
                np.testing.assert_array_equal(prepared.x_native[0], signal)
                actual, _, _ = sigma_delta(prepared.x_interp[0], cfg.theta_init,
                                           dead_zone=cfg.dead_zone, return_reconstruction=False)
                expected, offsets = signal_to_events(signal, frames, KFallEventLoaderConfig(
                    raw_win=cfg.raw_win, out_win=cfg.out_len, resample_hz=200,
                    theta_sd=cfg.theta_init, dead_zone=cfg.dead_zone, sd_init=cfg.sd_init,
                ))
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(offsets, signal[0])
                self.assertEqual(len(analog), 2 * cfg.raw_win - 1)
                self.assertEqual(times[-1], (cfg.raw_win - 1) / 100)

    def test_resampling_changes_identity_but_keeps_subjects_endpoints_and_labels(self):
        original, interpolated = self.config(), self.config("--resample-hz", "200")
        fixture = self.fixture()
        with patch("preprocess.preprocess_kfall.load_kfall_3class", return_value=fixture):
            before = prepare_kfall(original, self.logger)
            after = prepare_kfall(interpolated, self.logger)
        self.assertNotEqual(original.result_dir, interpolated.result_dir)
        self.assertIn("resample200p0hz", str(interpolated.result_dir))
        self.assertNotIn("resample_hz", before.metadata)
        self.assertEqual(after.metadata["resample_hz"], 200)
        self.assertEqual(after.metadata["source_sample_rate_hz"], 100)
        self.assertEqual(before.metadata["split_subjects"], after.metadata["split_subjects"])
        self.assertEqual(before.metadata["window_endpoints"], after.metadata["window_endpoints"])
        np.testing.assert_array_equal(before.train.y, after.train.y)
        np.testing.assert_array_equal(after.t_native, np.arange(205, dtype=np.float64) / 100)
        self.assertNotEqual(ThetaOptimizer(before, original, self.logger).experiment_signature,
                            ThetaOptimizer(after, interpolated, self.logger).experiment_signature)

    def test_optimization_summary_records_interpolation_provenance(self):
        cfg = self.config("--resample-hz", "200", "--out-len", "32", "--max-epochs", "1",
                          "--train-per-class", "1", "--val-per-class", "1", "--test-per-class", "1",
                          "--sampled-test-only")
        with patch("preprocess.preprocess_kfall.load_kfall_3class", return_value=self.fixture()):
            data = prepare_kfall(cfg, self.logger)
        result = ThetaOptimizer(data, cfg, self.logger).optimize()
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["resample_hz"], 200)
        self.assertEqual(result["metadata"]["resample_hz"], 200)
        self.assertTrue(np.all(np.isfinite(result["theta_best"])))
        self.assertEqual(len(result["theta_best"]), 6)


if __name__ == "__main__":
    unittest.main()
