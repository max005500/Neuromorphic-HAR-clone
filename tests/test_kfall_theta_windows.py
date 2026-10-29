"""Keep theta optimization aligned with KFall durations and temporal targets."""

from __future__ import annotations

import logging
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from loaders.kfall_event_loader import KFallEventLoaderConfig, signal_to_events, window_targets_for_record
from optimize_theta import build_parser
from preprocess.event_encoding import sigma_delta
from preprocess.kfall_labels import class_names
from theta_optimization.utils import RunConfig, ThetaOptimizer, build_split, prepare_kfall


def raw_fixture():
    recordings, records, labels = [], [], []
    length = 1000
    frames = np.arange(1, length + 1)
    for subject_index, subject in enumerate(("SA06", "SA07", "SA08")):
        for task, label in ((20, 0), (4, 1), (1, 2)):
            samples = np.arange(length, dtype=np.float32)
            signal = np.column_stack([np.sin(samples / (30 + channel)) + channel + subject_index
                                       for channel in range(6)]).astype(np.float32)
            recordings.append(signal)
            labels.append(label)
            record = {"file": f"{subject}/S{subject[2:]}T{task:02d}R01.csv", "subject": subject,
                      "code": f"T{task:02d}", "trial": "R01", "label_id": label,
                      "T": length, "frames": frames.copy()}
            if label == 0:
                record.update(fall_onset_frame=600, fall_impact_frame=700,
                              onset_idx=599, impact_idx=699, post_impact_end_idx=819)
            records.append(record)
    array = np.empty(len(recordings), dtype=object)
    array[:] = recordings
    return array, np.asarray(labels, dtype=np.int64), records, 0


class KFallThetaWindowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.logger = logging.getLogger(self.id())
        self.logger.handlers = [logging.NullHandler()]
        self.logger.propagate = False

    def config(self, *extra, dataset="kfall", ups="10"):
        args = build_parser().parse_args([
            "--dataset", dataset, "--ups", ups, "--dead-zone", ".5",
            "--data-root", str(self.root), "--results-root", str(self.root / "results"),
            "--split", "1", "1", "1", "--train-per-class", "100", "--val-per-class", "100",
            "--test-per-class", "100", "--sampled-test-only", "--workers", "1", "--executor", "none",
            *extra,
        ])
        return RunConfig.from_args(args)

    def test_kfall_cli_defaults_and_scaled_durations(self):
        for seconds, raw, steps in (("2.05", 205, 2048), ("3.30", 330, 3297), ("4.10", 410, 4096)):
            with self.subTest(seconds=seconds):
                cfg = self.config("--window-seconds", seconds)
                self.assertEqual((cfg.raw_win, cfg.out_len), (raw, steps))
                self.assertEqual(cfg.post_impact_seconds, 1.2)
                self.assertEqual(cfg.label_names, class_names(4))
        default = self.config()
        self.assertEqual((default.raw_win, default.out_len), (205, 2048))
        self.assertEqual(self.config("--window-seconds", "4.1", "--out-len", "2048").out_len, 2048)
        self.assertEqual(self.config("--post-impact-seconds", "0").label_names, class_names(3))

    def test_invalid_context_duration_and_ambiguous_impact_are_rejected(self):
        for options in (("--window-seconds", "2.055"), ("--window-seconds", "0"),
                        ("--post-impact-seconds", "-1"),
                        ("--window-seconds", "4.1", "--endpoint-context", "330"),
                        ("--fall-policy", "annotated_interval")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.config(*options)
        legacy = self.config("--post-impact-seconds", "0", "--fall-policy", "annotated_interval")
        self.assertEqual(legacy.fall_policy, "annotated_interval")

    def test_non_kfall_defaults_are_unchanged_and_new_flags_are_scoped(self):
        sisfall = self.config(dataset="sisfall", ups="5")
        ucihar = self.config(dataset="ucihar", ups="5")
        self.assertEqual((sisfall.raw_win, sisfall.out_len, len(sisfall.label_names)), (410, 2048, 3))
        self.assertEqual((ucihar.raw_win, ucihar.out_len, len(ucihar.label_names)), (128, 640, 6))
        self.assertEqual(sisfall.post_impact_seconds, 0)
        self.assertEqual(ucihar.post_impact_seconds, 0)
        for dataset in ("sisfall", "ucihar"):
            with self.subTest(dataset=dataset), self.assertRaisesRegex(ValueError, "KFall options"):
                self.config("--window-seconds", "3.3", dataset=dataset, ups="5")

    def test_shared_duration_protocol_keeps_endpoints_labels_and_subjects(self):
        fixture = raw_fixture()
        prepared = []
        for duration in ("2.05", "3.30", "4.10"):
            cfg = self.config("--window-seconds", duration, "--endpoint-context", "410")
            with patch("preprocess.preprocess_kfall.load_kfall_3class", return_value=fixture):
                data = prepare_kfall(cfg, self.logger)
            self.assertEqual(data.train.x_native.shape[1:], (cfg.raw_win, 6))
            self.assertEqual(data.train.x_interp.shape[1:], (cfg.out_len, 6))
            self.assertEqual(set(data.train.y), {0, 1, 2, 3})
            self.assertEqual(data.label_names, class_names(4))
            for split, values in data.metadata["window_endpoints"].items():
                for file_index, endpoint, label in values:
                    record = fixture[2][file_index]
                    expected_cfg = KFallEventLoaderConfig(raw_win=cfg.raw_win, out_win=cfg.out_len,
                        stride=cfg.window_stride, endpoint_context=410, post_impact_seconds=1.2)
                    starts, targets = window_targets_for_record(fixture[0][file_index], record, expected_cfg)
                    expected = {(int(record["frames"][start + cfg.raw_win - 1]), int(target))
                                for start, target in zip(starts, targets)}
                    self.assertIn((endpoint, label), expected)
                    self.assertIn(record["subject"], data.metadata["split_subjects"][split])
                    if label == 3:
                        self.assertTrue(700 <= endpoint < 820)
            groups = [set(data.metadata["split_subjects"][name]) for name in ("train", "val", "test")]
            self.assertTrue(all(not left & right for index, left in enumerate(groups) for right in groups[index + 1:]))
            prepared.append(data)
        self.assertEqual(prepared[0].metadata["window_endpoints"], prepared[1].metadata["window_endpoints"])
        self.assertEqual(prepared[1].metadata["window_endpoints"], prepared[2].metadata["window_endpoints"])

    def test_legacy_preparation_omits_postimpact_targets(self):
        cfg = self.config("--post-impact-seconds", "0")
        with patch("preprocess.preprocess_kfall.load_kfall_3class", return_value=raw_fixture()):
            data = prepare_kfall(cfg, self.logger)
        self.assertEqual(set(data.train.y), {0, 1, 2})
        self.assertEqual(data.metadata["n_classes"], 3)
        self.assertEqual(data.metadata["post_impact_seconds"], 0)

    def test_interpolation_matches_the_training_encoder_grid(self):
        cfg = self.config("--window-seconds", "3.3")
        generator = np.random.default_rng(13)
        signal = generator.normal(size=(cfg.raw_win, 6)).astype(np.float32)
        data = build_split(signal[None, ...], np.array([3]), cfg)
        actual, _, _ = sigma_delta(data.x_interp[0], theta=np.asarray(cfg.theta_init),
                                   init=cfg.sd_init, dead_zone=cfg.dead_zone, return_reconstruction=False)
        expected_cfg = KFallEventLoaderConfig(raw_win=cfg.raw_win, out_win=cfg.out_len,
            theta_sd=cfg.theta_init, dead_zone=cfg.dead_zone, sd_init=cfg.sd_init)
        expected, _ = signal_to_events(signal, np.arange(50, 50 + cfg.raw_win), expected_cfg)
        np.testing.assert_array_equal(actual, expected)

    def test_experiment_paths_distinguish_duration_labels_and_encoder_length(self):
        paths = {self.config("--window-seconds", duration).result_dir for duration in ("2.05", "3.3", "4.1")}
        paths.add(self.config("--post-impact-seconds", "0").result_dir)
        paths.add(self.config("--out-len", "128").result_dir)
        self.assertEqual(len(paths), 5)

    def test_four_class_optimizer_smoke_and_checkpoint_signature(self):
        cfg = self.config("--out-len", "32", "--train-per-class", "1", "--val-per-class", "1",
                          "--test-per-class", "1", "--max-epochs", "1")
        with patch("preprocess.preprocess_kfall.load_kfall_3class", return_value=raw_fixture()):
            data = prepare_kfall(cfg, self.logger)
        optimizer = ThetaOptimizer(data, cfg, self.logger)
        result = optimizer.optimize()
        self.assertEqual(result["status"], "done")
        self.assertTrue(all(np.isfinite(score) for score in result["scores"].values()))
        self.assertEqual(len(result["n_windows"]["train"]), 4)
        self.assertEqual(result["metadata"]["n_classes"], 4)
        changed = ThetaOptimizer(data, replace(cfg, endpoint_context=410), self.logger)
        self.assertNotEqual(changed.experiment_signature, optimizer.experiment_signature)


if __name__ == "__main__":
    unittest.main()
