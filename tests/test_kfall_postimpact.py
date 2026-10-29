"""Temporal boundaries, shared context endpoints, and exact batched encoding."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from loaders.kfall_event_loader import (
    KFallEventLoaderConfig, _encode_event_batch, signal_to_events, window_targets_for_record,
)
from preprocess.kfall_labels import class_names, event_steps_for_samples, label_at_frame, window_samples
from train_kfall import get_args


class PostimpactBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.record = {"label_id": 0, "onset_idx": 8, "impact_idx": 14,
                       "fall_onset_frame": 109, "fall_impact_frame": 115,
                       "frames": np.arange(101, 141)}
        self.signal = np.zeros((40, 6), dtype=np.float32)
        self.cfg = KFallEventLoaderConfig(raw_win=5, out_win=16, stride=4, post_impact_seconds=.04)

    def test_impact_belongs_only_to_postimpact_and_end_is_exclusive(self):
        def target(frame):
            return label_at_frame(frame, self.record, post_impact_seconds=.04)
        self.assertIsNone(target(108))
        self.assertEqual(target(109), 0)
        self.assertEqual(target(114), 0)
        self.assertEqual(target(115), 3)
        self.assertEqual(target(118), 3)
        self.assertIsNone(target(119))

    def test_windows_use_endpoint_class_even_if_they_contain_both_phases(self):
        starts, labels = window_targets_for_record(self.signal, self.record, self.cfg)
        ends = starts + self.cfg.raw_win - 1
        np.testing.assert_array_equal(ends, [8, 12, 13, 16, 17])
        np.testing.assert_array_equal(labels, [0, 0, 0, 3, 3])
        for end, label in zip(ends, labels):
            self.assertEqual(label_at_frame(self.record["frames"][end], self.record,
                                            post_impact_seconds=.04), label)

    def test_shared_context_keeps_endpoints_and_labels_identical(self):
        pairs = []
        for length in (5, 7, 10):
            cfg = replace(self.cfg, raw_win=length, endpoint_context=10)
            starts, labels = window_targets_for_record(self.signal, self.record, cfg)
            self.assertTrue(np.all(starts >= 0))
            pairs.append(np.column_stack((starts + length - 1, labels)))
        for pair in pairs[1:]:
            np.testing.assert_array_equal(pair, pairs[0])

    def test_postimpact_uses_frame_time_with_missing_samples(self):
        frames = np.array([101, 103, 105, 107, 109, 111, 115])
        record = {"label_id": 0, "frames": frames, "onset_idx": 2, "impact_idx": 4,
                  "fall_onset_frame": 105, "fall_impact_frame": 109}
        cfg = replace(self.cfg, raw_win=2, stride=1)
        starts, labels = window_targets_for_record(np.zeros((7, 6)), record, cfg)
        np.testing.assert_array_equal(frames[starts + 1], [105, 107, 109, 111])
        np.testing.assert_array_equal(labels, [0, 0, 3, 3])
        saved = {key: value for key, value in record.items() if key != "frames"}
        saved["post_impact_end_idx"] = 6
        np.testing.assert_array_equal(window_targets_for_record(np.zeros((7, 0)), saved, cfg)[1], labels)

    def test_long_context_does_not_pad_with_future_samples(self):
        starts, labels = window_targets_for_record(self.signal, self.record,
                                                   replace(self.cfg, raw_win=20))
        self.assertEqual(len(starts), 0)
        self.assertEqual(len(labels), 0)

    def test_invalid_overlapping_label_policy_and_context_are_rejected(self):
        for changes in ({"fall_policy": "annotated_interval"}, {"post_impact_seconds": -.1},
                        {"post_impact_seconds": .015}, {"endpoint_context": 4}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.cfg, **changes)


class DurationAndBatchTests(unittest.TestCase):
    def test_scaled_time_resolution_and_four_class_cli_defaults(self):
        for duration, raw_win, out_win in ((2.05, 205, 2048), (3.30, 330, 3297), (4.10, 410, 4096)):
            cfg = get_args(["--window-seconds", str(duration)])
            self.assertEqual(window_samples(duration), raw_win)
            self.assertEqual(event_steps_for_samples(raw_win), out_win)
            self.assertEqual((cfg.raw_win, cfg.out_win, cfg.n_classes), (raw_win, out_win, 4))
            self.assertEqual(cfg.post_impact_seconds, 1.2)
            self.assertNotEqual(cfg.cache_name, "default")
        self.assertEqual(class_names(4)[3], "Postimpact")

    def test_legacy_cli_can_use_original_three_class_cache(self):
        cfg = get_args(["--post-impact-seconds", "0"])
        self.assertEqual(cfg.n_classes, 3)
        self.assertEqual(cfg.cache_name, "default")

    def test_batch_encoder_is_bit_exact_with_each_scalar_window(self):
        random = np.random.default_rng(47)
        for init in ("x0", "zero"):
            for raw_win, out_win in ((5, 16), (205, 2048), (330, 3297), (410, 4096)):
                cfg = KFallEventLoaderConfig(raw_win=raw_win, out_win=out_win, sd_init=init)
                windows = [random.normal(size=(raw_win, 6)).astype(np.float32) for _ in range(3)]
                frames = [np.cumsum(random.integers(1, 3, size=raw_win)).astype(np.int64) for _ in windows]
                batch, offsets = _encode_event_batch(windows, frames, cfg)
                for index, (window, grid) in enumerate(zip(windows, frames)):
                    expected, offset = signal_to_events(window, grid, cfg)
                    np.testing.assert_array_equal(batch[index], expected)
                    np.testing.assert_array_equal(offsets[index], offset)


if __name__ == "__main__":
    unittest.main()
