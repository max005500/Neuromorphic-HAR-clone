"""Regression tests for subject isolation before window/event processing."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from preprocess.utils import subject_split_indices
from preprocess.preprocess_ucihar import UCIHARRawData, preprocess_ucihar
from loaders.sisfall_event_loader import EventLoaderConfig, load_or_build_sisfall_event_splits
from loaders.ucihar_event_loader import UCIHAREventLoaderConfig, load_or_build_ucihar_event_splits


class SubjectSplitTests(unittest.TestCase):
    def test_whole_subjects_and_complete_coverage(self):
        subjects = np.repeat([f"SA{i:02d}" for i in range(32)], np.arange(1, 33))
        indices = subject_split_indices(subjects, seed=7)
        groups = [set(subjects[idx]) for idx in indices]
        self.assertEqual([len(group) for group in groups], [26, 3, 3])
        for left, right in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(groups[left] & groups[right])
        np.testing.assert_array_equal(np.sort(np.concatenate(indices)), np.arange(len(subjects)))
        for original, repeated in zip(indices, subject_split_indices(subjects, seed=7)):
            np.testing.assert_array_equal(original, repeated)

    def test_assignment_independent_of_recording_order(self):
        subjects = np.repeat(np.arange(15), np.arange(1, 16))
        order = np.random.default_rng(3).permutation(len(subjects))
        first = subject_split_indices(subjects, seed=2)
        second = subject_split_indices(subjects[order], seed=2)
        self.assertEqual(
            [set(subjects[idx]) for idx in first],
            [set(subjects[order][idx]) for idx in second],
        )

    def test_small_dataset_keeps_requested_partitions_nonempty(self):
        indices = subject_split_indices(["a", "b", "c"], split=(0.98, 0.01, 0.01))
        self.assertEqual([len(idx) for idx in indices], [1, 1, 1])

    def test_zero_fraction_is_empty(self):
        indices = subject_split_indices([1, 1, 2, 2], split=(1, 0, 0))
        self.assertEqual([len(idx) for idx in indices], [4, 0, 0])

    def test_invalid_subjects_and_fractions_fail(self):
        cases = [
            ([1, 2], (0.8, 0.1, 0.1)),
            ([1, 2, 3], (float("nan"), 1, 1)),
            ([1, 2, 3], (float("inf"), 1, 1)),
            ([1, 2, 3], (-1, 1, 1)),
            ([1, 2, 3], (0, 0, 0)),
            (["a", "", "c"], (1, 1, 1)),
            (np.asarray([1, np.nan, 3]), (1, 1, 1)),
            (np.asarray([1, None, 3], dtype=object), (1, 1, 1)),
        ]
        for subjects, fractions in cases:
            with self.subTest(subjects=subjects, fractions=fractions):
                with self.assertRaises(ValueError):
                    subject_split_indices(subjects, split=fractions)


class UCIHARSubjectTests(unittest.TestCase):
    def setUp(self):
        train_subjects = np.repeat(np.arange(6), 3)
        test_subjects = np.repeat(np.arange(6, 8), 3)
        self.raw = UCIHARRawData(
            X_train=np.zeros((len(train_subjects), 128, 6), np.float32),
            y_train=np.tile(np.arange(3), 6), d_train=train_subjects,
            X_test=np.ones((len(test_subjects), 128, 6), np.float32),
            y_test=np.tile(np.arange(3), 2), d_test=test_subjects,
            channel_names=("gx", "gy", "gz", "ax", "ay", "az"),
            root=Path("synthetic/UCI HAR Dataset"),
        )

    def _prepare(self, **kwargs):
        with patch("preprocess.preprocess_ucihar.load_ucihar_raw", return_value=self.raw):
            return preprocess_ucihar(verbose=False, **kwargs)

    def _assert_disjoint(self, result):
        groups = [set(result.d_train), set(result.d_val), set(result.d_test)]
        self.assertTrue(all(groups))
        for left, right in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(groups[left] & groups[right])
        self.assertEqual(result.meta["split_strategy"], "subject")

    def test_official_test_retained_validation_holds_out_subjects(self):
        result = self._prepare(case="official", val_ratio=0.2)
        self._assert_disjoint(result)
        np.testing.assert_array_equal(result.d_test, self.raw.d_test)
        np.testing.assert_array_equal(result.X_test, self.raw.X_test)
        self.assertEqual(set(result.d_train) | set(result.d_val), set(self.raw.d_train))

    def test_random_and_subject_split_cases_assign_whole_subjects(self):
        for case in ("random", "subject_split"):
            with self.subTest(case=case):
                result = self._prepare(case=case)
                self._assert_disjoint(result)
                self.assertEqual(len(result.y_train) + len(result.y_val) + len(result.y_test), 24)

    def test_subject_domain_case_holds_out_validation_subjects(self):
        result = self._prepare(case="subject", target_domain=0)
        self._assert_disjoint(result)
        self.assertEqual(set(result.d_test), {0})
        self.assertEqual(set(result.d_train) | set(result.d_val), {1, 2, 3, 4})

    def test_corrupt_official_subject_overlap_fails(self):
        self.raw = replace(self.raw, d_test=np.zeros_like(self.raw.d_test))
        with self.assertRaisesRegex(ValueError, "Subject leakage"):
            self._prepare(case="official")

    def test_event_cache_retains_subject_metadata_and_rejects_legacy_version(self):
        raw = self._prepare(case="official")
        cfg = UCIHAREventLoaderConfig(out_win=20, num_workers=0)
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(data_root="synthetic", cache_dir=directory, cfg=cfg, verbose=False)
            with patch("loaders.ucihar_event_loader._load_ucihar_preprocessed", return_value=raw):
                splits, metadata = load_or_build_ucihar_event_splits(**kwargs)
            self.assertEqual(metadata["split_subjects"], raw.meta["split_subjects"])
            for name in ("train", "val", "test"):
                self.assertEqual(set(splits[name]["domain"]), set(raw.meta["split_subjects"][name]))
            with patch("loaders.ucihar_event_loader._load_ucihar_preprocessed") as raw_load:
                cached, cached_metadata = load_or_build_ucihar_event_splits(**kwargs)
                raw_load.assert_not_called()
            self.assertEqual(cached_metadata, metadata)
            np.testing.assert_array_equal(cached["test"]["domain"], splits["test"]["domain"])

            meta_path = Path(directory) / "ucihar_event_loader_default" / "metadata.json"
            meta_path.write_text(json.dumps(dict(metadata, version=1)), encoding="utf-8")
            with patch("loaders.ucihar_event_loader._load_ucihar_preprocessed", side_effect=RuntimeError("rebuild requested")):
                with self.assertRaisesRegex(RuntimeError, "rebuild requested"):
                    load_or_build_ucihar_event_splits(**kwargs)


class SisFallSubjectIntegrationTests(unittest.TestCase):
    def test_segmented_windows_and_cache_keep_subject_isolation(self):
        records = []
        xs = np.empty(12, dtype=object)
        labels = []
        for i in range(12):
            subject = f"SA{i // 2:02d}"
            xs[i] = np.ones((24, 6), dtype=np.float32) * (i + 1)
            labels.append(1)
            records.append({"file": f"D01_{subject}_R{i % 2:02d}.txt", "subject": subject, "code": "D01", "trial": f"R{i % 2:02d}"})
        cfg = EventLoaderConfig(raw_win=12, out_win=20, stride_dynamic=6, split=(0.5, 0.25, 0.25), num_workers=0)
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(dataset_root="synthetic", cache_dir=directory, result_dir=directory, cfg=cfg, verbose=False)
            with patch("loaders.sisfall_event_loader._load_sisfall", return_value=(xs, np.asarray(labels), records, 0)), patch("loaders.sisfall_event_loader._load_or_build_posture_tags", return_value={}):
                splits, metadata = load_or_build_sisfall_event_splits(**kwargs)
            groups = []
            for name in ("train", "val", "test"):
                actual = {records[int(idx)]["subject"] for idx in splits[name]["file_idx"]}
                groups.append(actual)
                self.assertEqual(actual, set(metadata["split_subjects"][name]))
                self.assertEqual(splits[name]["x_ev"].shape[1:], (20, 6, 2))
            for left, right in ((0, 1), (0, 2), (1, 2)):
                self.assertFalse(groups[left] & groups[right])
            with patch("loaders.sisfall_event_loader._load_sisfall") as raw_load:
                cached, cached_metadata = load_or_build_sisfall_event_splits(**kwargs)
                raw_load.assert_not_called()
            self.assertEqual(cached_metadata, metadata)
            np.testing.assert_array_equal(cached["test"]["file_idx"], splits["test"]["file_idx"])

            meta_path = Path(directory) / "sisfall_event_loader_default" / "metadata.json"
            old = dict(metadata, version=1)
            meta_path.write_text(json.dumps(old), encoding="utf-8")
            with patch("loaders.sisfall_event_loader._load_sisfall", side_effect=RuntimeError("rebuild requested")):
                with self.assertRaisesRegex(RuntimeError, "rebuild requested"):
                    load_or_build_sisfall_event_splits(**kwargs)


if __name__ == "__main__":
    unittest.main()
