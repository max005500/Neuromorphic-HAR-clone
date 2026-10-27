"""Ensure the viewer shows the exact cached window and its matching CSV samples."""

from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from openpyxl import Workbook

from loaders.kfall_event_loader import KFallEventLoaderConfig, load_or_build_kfall_event_splits
from preprocess.event_encoding import sigma_delta
from visualization.kfall_data import KFallWindowStore


class KFallViewerDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "KFall"
        labels = self.root / "label_data"
        labels.mkdir(parents=True)
        for subject in (6, 7, 8):
            folder = self.root / "sensor_data" / f"SA{subject:02d}"
            folder.mkdir(parents=True)
            for task in (1, 4, 20):
                with (folder / f"S{subject:02d}T{task:02d}R01.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["TimeStamp(s)", "FrameCounter", "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"])
                    for frame in range(101, 121):
                        value = frame - 101
                        writer.writerow([value / 100, frame, value / 10, np.sin(value),
                                         1.2, value, -value / 2, 3])
            workbook = Workbook()
            workbook.active.append(["Task Code (Task ID)", "Trial ID", "Fall_onset_frame", "Fall_impact_frame"])
            workbook.active.append(["F01 (20)", 1, 109, 114])
            workbook.save(labels / f"SA{subject:02d}_label.xlsx")
            workbook.close()
        self.config = KFallEventLoaderConfig(raw_win=5, out_win=16, stride=4, split=(1, 1, 1), verbose_every=0)
        self.cache_parent = self.directory / "cache"
        self.cache = self.cache_parent / "kfall_event_loader_default"
        load_or_build_kfall_event_splits(self.root, cache_dir=self.cache_parent, cfg=self.config, verbose=False)

    def test_lazy_read_only_cache_and_source_window_correspondence(self):
        store = KFallWindowStore(self.cache)
        self.assertEqual(store._splits, {})
        timestamps = {path: path.stat().st_mtime_ns for path in self.cache.rglob("*") if path.is_file()}
        with patch("loaders.kfall_event_loader.load_or_build_kfall_event_splits", side_effect=AssertionError("viewer rebuilt cache")):
            index = int(store.get_indices("train", class_id=0)[0])
            window = store.get_window("train", index)
        self.assertEqual(window.raw.shape, (5, 6))
        self.assertEqual(window.interpolated.shape, (16, 6))
        self.assertEqual(window.events.shape, (16, 6, 2))
        self.assertEqual(window.label, "Fall")
        self.assertEqual(window.stop, window.start + 5)
        self.assertFalse(window.events.flags.writeable)
        self.assertFalse(window.raw.flags.writeable)
        self.assertIsInstance(store._splits["train"]["x_ev"], np.memmap)
        expected = np.loadtxt(window.path, delimiter=",", skiprows=1)
        np.testing.assert_array_equal(window.raw, expected[window.start:window.stop, 2:8].astype(np.float32))
        np.testing.assert_array_equal(window.offset, window.raw[0])
        self.assertEqual(window.end_frame, int(window.frames[-1]))
        self.assertAlmostEqual(window.time_raw[-1], 0.04)
        self.assertAlmostEqual(window.time_events[-1], 0.04)
        self.assertLessEqual(window.fall_onset_time, window.time_raw[-1])
        self.assertGreater(window.fall_impact_time, window.time_raw[-1])
        self.assertGreater(window.lead_time_ms, 0)
        self.assertEqual(timestamps, {path: path.stat().st_mtime_ns for path in timestamps})

    def test_subject_and_class_filters_use_cached_split_indices(self):
        store = KFallWindowStore(self.cache)
        subjects = []
        for split in ("train", "val", "test"):
            subject = store.subjects(split)[0]
            subjects.append(subject)
            all_indices = store.get_indices(split)
            self.assertEqual(len(all_indices), store.split_size(split))
            for class_id in (0, 1, 2):
                indices = store.get_indices(split, class_id, subject=subject)
                self.assertGreater(len(indices), 0)
                self.assertTrue(np.all(store._splits[split]["y_h1"][indices] == class_id))
            np.testing.assert_array_equal(store.get_indices(split, subject=subject[2:]), all_indices)
            np.testing.assert_array_equal(store.get_indices(split, subject="Todos"), all_indices)
            self.assertEqual(len(store.get_indices(split, subject="SA99")), 0)
        self.assertEqual(len(set(subjects)), 3)

    def test_reconstruction_matches_encoder_with_initial_offset(self):
        store = KFallWindowStore(self.cache)
        index = int(store.get_indices("train", class_id=0)[0])
        window = store.get_window("train", index)
        events, _, reconstructed = sigma_delta(window.interpolated, window.theta,
                                               init="x0", dead_zone=self.config.dead_zone)
        np.testing.assert_array_equal(window.events, events)
        np.testing.assert_array_equal(window.reconstructed, reconstructed)

    def test_zero_initialization_includes_the_first_event(self):
        config = replace(self.config, sd_init="zero")
        load_or_build_kfall_event_splits(self.root, cache_dir=self.cache_parent, run_name="zero", cfg=config, verbose=False)
        store = KFallWindowStore(self.cache_parent / "kfall_event_loader_zero")
        window = store.get_window("train", 0)
        _, _, expected = sigma_delta(window.interpolated, window.theta, init="zero", dead_zone=config.dead_zone)
        np.testing.assert_array_equal(window.reconstructed, expected)
        first_step = window.events[0, :, 0].astype(np.float32) - window.events[0, :, 1].astype(np.float32)
        np.testing.assert_array_equal(window.reconstructed[0], first_step * window.theta)

    def test_moved_dataset_root_resolves_manifest_sources(self):
        moved = self.directory / "moved_dataset"
        shutil.move(self.root, moved)
        store = KFallWindowStore(self.cache, dataset_root=moved)
        window = store.get_window("train", 0)
        self.assertTrue(window.path.is_relative_to(moved))
        self.assertEqual(window.raw.shape, (5, 6))

    def test_frame_gaps_are_kept_on_the_physical_time_axis(self):
        for path in (self.root / "sensor_data").rglob("*.csv"):
            with path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            rows = [row for row in rows if row[1] != "103"]
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
        load_or_build_kfall_event_splits(self.root, cache_dir=self.cache_parent, run_name="gaps",
                                       cfg=self.config, verbose=False)
        store = KFallWindowStore(self.cache_parent / "kfall_event_loader_gaps")
        window = store.get_window("train", 0)
        np.testing.assert_array_equal(window.frames, [101, 102, 104, 105, 106])
        np.testing.assert_allclose(window.time_raw, [0, 0.01, 0.03, 0.04, 0.05])
        self.assertAlmostEqual(window.time_events[-1], 0.05)

    def test_changed_source_is_rejected_even_after_recording_was_cached(self):
        store = KFallWindowStore(self.cache)
        window = store.get_window("train", 0)
        with window.path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        row = window.start + 1
        rows[row][2] = str(float(rows[row][2]) + 0.25)
        with window.path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with self.assertRaisesRegex(ValueError, "do not match source CSV"):
            store.get_window("train", 0)

    def test_changed_cached_events_are_rejected(self):
        store = KFallWindowStore(self.cache)
        events = np.load(self.cache / "train" / "x_ev.npy", mmap_mode="r+")
        events[0, 0, 0, 0] = 1 - events[0, 0, 0, 0]
        events.flush()
        with self.assertRaisesRegex(ValueError, "do not match source CSV"):
            store.get_window("train", 0)

    def test_missing_cache_does_not_start_preprocessing(self):
        with patch("loaders.kfall_event_loader.load_or_build_kfall_event_splits", side_effect=AssertionError("viewer rebuilt cache")):
            with self.assertRaisesRegex(FileNotFoundError, "preprocess-only"):
                KFallWindowStore(self.directory / "missing")

    def test_invalid_selection_has_clear_errors(self):
        store = KFallWindowStore(self.cache)
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            store.get_indices("validation")
        with self.assertRaisesRegex(ValueError, "class_id"):
            store.get_indices("train", 8)
        with self.assertRaisesRegex(ValueError, "Subject"):
            store.get_indices("train", subject="invalid")
        with self.assertRaises(IndexError):
            store.get_window("train", -1)
        with self.assertRaises(IndexError):
            store.get_window("train", store.split_size("train"))


if __name__ == "__main__":
    unittest.main()
