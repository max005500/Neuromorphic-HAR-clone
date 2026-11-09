"""Reproducible KFall duration and offset-ablation settings."""

from __future__ import annotations

import unittest

import numpy as np

from loaders.kfall_event_loader import KFallEventLoaderConfig
from preprocess.utils import subject_split_indices
from run_kfall_experiments import build_parser, window_conditions
from train_kfall import get_args


class KFallExperimentTests(unittest.TestCase):
    def test_kfall_protocol_keeps_complete_subjects_in_60_30_10_groups(self):
        split = KFallEventLoaderConfig().split
        self.assertEqual(split, (0.6, 0.3, 0.1))
        self.assertEqual(tuple(get_args([]).split), split)
        self.assertEqual(tuple(build_parser().parse_args([]).split), split)
        for n_subjects, expected in ((32, [19, 10, 3]), (31, [19, 9, 3])):
            with self.subTest(n_subjects=n_subjects):
                subjects = np.repeat(np.arange(n_subjects), np.arange(1, n_subjects + 1))
                indices = subject_split_indices(subjects, split=split, seed=0)
                groups = [set(subjects[index]) for index in indices]
                self.assertEqual([len(group) for group in groups], expected)
                self.assertTrue(all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:]))
                np.testing.assert_array_equal(np.sort(np.concatenate(indices)), np.arange(len(subjects)))

    def test_requested_durations_keep_approximately_one_khz_event_ticks(self):
        self.assertEqual(
            window_conditions(),
            [
                {"window_seconds": 1.20, "raw_samples": 120, "event_steps": 1200, "stride": 60},
                {"window_seconds": 2.06, "raw_samples": 206, "event_steps": 2060, "stride": 103},
                {"window_seconds": 3.20, "raw_samples": 320, "event_steps": 3200, "stride": 160},
            ],
        )


if __name__ == "__main__":
    unittest.main()
