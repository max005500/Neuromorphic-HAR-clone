"""Causal temporal windowing, subject isolation, and KFall cache integrity."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from openpyxl import Workbook

from loaders.kfall_event_loader import (
    DEFAULT_EXCLUDED_TASKS, KFallEventLoaderConfig, _source_signature, build_event_split, load_or_build_kfall_event_splits,
    signal_to_events, window_starts_for_record,
    window_activity_labels,
)
from preprocess.preprocess_kfall import CHANNELS
from labeling.kfall_manual_labels import load_manual_labels


class KFallEventTests(unittest.TestCase):
    def setUp(self):
        self.cfg = KFallEventLoaderConfig(raw_win=5, out_win=16, stride=4,
                                          split=(1, 1, 1), verbose_every=0)
        self.signal = np.arange(120, dtype=np.float32).reshape(20, 6) / 10
        self.info = {"label_id": 0, "onset_idx": 8, "impact_idx": 14,
                     "fall_onset_frame": 109, "fall_impact_frame": 115,
                     "subject": "SA06", "frames": np.arange(101, 121)}

    def test_endpoint_labels_and_tail_anchor(self):
        starts = window_starts_for_record(self.signal, self.info, self.cfg)
        np.testing.assert_array_equal(starts, [4, 8, 9])
        ends = starts + self.cfg.raw_win - 1
        self.assertTrue(np.all(ends >= self.info["onset_idx"]))
        self.assertTrue(np.all(ends < self.info["impact_idx"]))

    def test_impact_policy_is_explicit(self):
        starts = window_starts_for_record(self.signal, self.info,
                                          replace(self.cfg, fall_policy="annotated_interval"))
        self.assertEqual(starts[-1] + self.cfg.raw_win - 1, self.info["impact_idx"])

    def test_short_fall_phase_still_has_preimpact_window(self):
        info = {**self.info, "onset_idx": 10, "impact_idx": 11}
        starts = window_starts_for_record(self.signal, info, self.cfg)
        np.testing.assert_array_equal(starts, [6])

    def test_no_full_preimpact_context_returns_no_window(self):
        info = {**self.info, "onset_idx": 1, "impact_idx": 3}
        self.assertEqual(len(window_starts_for_record(self.signal, info, self.cfg)), 0)

    def test_changing_impact_and_future_cannot_change_preimpact_inputs(self):
        recordings = np.empty(1, dtype=object)
        recordings[0] = self.signal.copy()
        before = build_event_split(recordings, np.array([0]), [self.info], [0], self.cfg, verbose=False)
        recordings[0][self.info["impact_idx"]:] = 1e9
        after = build_event_split(recordings, np.array([0]), [self.info], [0], self.cfg, verbose=False)
        for key in ("x_ev", "offset", "end_frame", "lead_time_ms"):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertTrue(np.all(before["lead_time_ms"] > 0))
        np.testing.assert_array_equal(before["lead_time_ms"], [60, 20, 10])
        self.assertTrue(np.all(before["w_h2"] == 0))
        self.assertTrue(np.all(before["y_h2"] == 0.5))

    def test_actual_frame_spacing_controls_interpolation(self):
        # A temporal gap changes the interpolant even with identical sample rows.
        signal = np.zeros((5, 6), dtype=np.float32)
        signal[:, 0] = [0, 0, 1, 1, 1]
        regular, offset = signal_to_events(signal, np.arange(5), self.cfg)
        irregular, _ = signal_to_events(signal, np.array([0, 1, 10, 11, 12]), self.cfg)
        self.assertFalse(np.array_equal(regular, irregular))
        self.assertEqual(regular.shape, (16, 6, 2))
        self.assertTrue(np.isin(regular, [0, 1]).all())
        np.testing.assert_array_equal(offset, signal[0])

    def write_fixture(self, root):
        labels = root / "label_data"
        labels.mkdir()
        for subject in (6, 7, 8):
            sensor = root / "sensor_data" / f"SA{subject:02d}"
            sensor.mkdir(parents=True)
            for task in (1, 4, 20):
                with (sensor / f"S{subject:02d}T{task:02d}R01.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["TimeStamp(s)", "FrameCounter", *CHANNELS])
                    for frame in range(101, 121):
                        writer.writerow([(frame - 101) / 100, frame, *([frame / 100] * 6)])
            workbook = Workbook()
            workbook.active.append(["Task Code (Task ID)", "Trial ID", "Fall_onset_frame", "Fall_impact_frame"])
            workbook.active.append(["F01 (20)", 1, 109, 114])
            workbook.save(labels / f"SA{subject:02d}_label.xlsx")
            workbook.close()

    def test_cache_roundtrip_subject_manifest_and_source_invalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "KFall"
            root.mkdir()
            self.write_fixture(root)
            cache = Path(temporary) / "cache"
            splits, metadata = load_or_build_kfall_event_splits(root, cache_dir=cache, cfg=self.cfg,
                                                               mmap_mode="r", verbose=False)
            groups = [set(metadata["split_subjects"][name]) for name in ("train", "val", "test")]
            self.assertEqual(set.union(*groups), {"SA06", "SA07", "SA08"})
            self.assertTrue(all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:]))
            for name, arrays in splits.items():
                self.assertIsInstance(arrays["x_ev"], np.memmap)
                self.assertEqual(set(arrays["subject"]), set(metadata["split_subjects"][name]))
                for index in np.unique(arrays["file_idx"]):
                    self.assertIn(metadata["recordings"][int(index)]["subject"], arrays["subject"])
            with patch("loaders.kfall_event_loader.load_kfall_3class", side_effect=AssertionError("cache missed")):
                reloaded, _ = load_or_build_kfall_event_splits(root, cache_dir=cache, cfg=self.cfg, verbose=False)
            np.testing.assert_array_equal(reloaded["train"]["x_ev"], splits["train"]["x_ev"])
            with patch("loaders.kfall_event_loader.load_kfall_3class", side_effect=RuntimeError("config changed")):
                with self.assertRaisesRegex(RuntimeError, "config changed"):
                    load_or_build_kfall_event_splits(root, cache_dir=cache, cfg=replace(self.cfg, stride=2), verbose=False)
            # Editing an XLSX file must invalidate even an otherwise matching cache.
            target = root / "label_data" / "SA06_label.xlsx"
            target.touch()
            with patch("loaders.kfall_event_loader.load_kfall_3class", side_effect=RuntimeError("source changed")):
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    load_or_build_kfall_event_splits(root, cache_dir=cache, cfg=self.cfg, verbose=False)

    def test_source_signature_tracks_uppercase_extensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "labels"
            labels.mkdir()
            signature = _source_signature(root, labels)
            (root / "S06T20R01.CSV").write_text("sensor")
            (labels / "SA06_label.XLSX").write_text("annotation")
            changed = _source_signature(root, labels)
            self.assertEqual(changed["files"], 2)
            self.assertNotEqual(signature["sha256"], changed["sha256"])

    def test_task_exclusions_filter_all_subjects_and_trials_and_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            for subject in (6, 7, 8):
                sensor = root / "sensor_data" / f"SA{subject:02d}"
                for code in DEFAULT_EXCLUDED_TASKS:
                    for trial in (1, 2):
                        shutil.copyfile(sensor / f"S{subject:02d}T01R01.csv",
                                        sensor / f"S{subject:02d}{code}R{trial:02d}.csv")
            files = sorted((root / "sensor_data").rglob("*.csv"))
            annotation = root / "manual.json"
            annotation.write_text(json.dumps({
                "format": "kfall_manual_normal_labels", "version": 1,
                "dataset": "kfall", "sample_rate_hz": 100,
                "recordings": {
                    "SA" + path.stem[1:]: {
                        "file": path.name, "reviewed": True, "intervals": [[101, 121, 2]],
                    } for path in files
                },
            }))
            kwargs = dict(dataset_root=root, cache_dir=root / "cache",
                          annotation_file=annotation, verbose=False)
            original, before = load_or_build_kfall_event_splits(cfg=self.cfg, **kwargs)
            fall_counts = {name: int(np.sum(arrays["y_h1"] == 0)) for name, arrays in original.items()}
            cfg = replace(self.cfg, excluded_tasks=DEFAULT_EXCLUDED_TASKS)
            with contextlib.redirect_stdout(io.StringIO()):
                filtered, after = load_or_build_kfall_event_splits(cfg=cfg, **kwargs)
            self.assertEqual(after["split_subjects"], before["split_subjects"])
            self.assertEqual(after["raw_data"]["excluded_recordings"], 36)
            self.assertEqual(after["raw_data"]["excluded_samples"], 36 * 20)
            self.assertEqual(len(after["recordings"]), 9)
            self.assertEqual(after["manual_annotation_missing_recordings"], [])
            for name, arrays in filtered.items():
                self.assertEqual(int(np.sum(arrays["y_h1"] == 0)), fall_counts[name])
                self.assertLess(len(arrays["y_h1"]), len(original[name]["y_h1"]))
                for index in arrays["file_idx"]:
                    self.assertNotIn(after["recordings"][int(index)]["code"], DEFAULT_EXCLUDED_TASKS)
            output = io.StringIO()
            with patch("loaders.kfall_event_loader.load_kfall_3class", side_effect=AssertionError("cache missed")), \
                    contextlib.redirect_stdout(output):
                load_or_build_kfall_event_splits(cfg=cfg, **kwargs)
            self.assertIn("[tasks:cache]", output.getvalue())
            self.assertIn("removed=36 recordings | retained=9", output.getvalue())
            cache_metadata = root / "cache" / "kfall_event_loader_default" / "metadata.json"
            corrupted = json.loads(cache_metadata.read_text())
            corrupted["recordings"][0]["code"] = "T05"
            cache_metadata.write_text(json.dumps(corrupted))
            with self.assertRaisesRegex(ValueError, "contain excluded tasks"):
                load_or_build_kfall_event_splits(cfg=cfg, **kwargs)
            _, restored = load_or_build_kfall_event_splits(cfg=self.cfg, **kwargs)
            self.assertEqual(len(restored["recordings"]), 45)
            self.assertTrue(all(path.is_file() for path in files))
            with self.assertRaisesRegex(ValueError, "no KFall recordings"):
                load_or_build_kfall_event_splits(
                    cfg=replace(self.cfg, excluded_tasks=("T01", "T04", "T20", *DEFAULT_EXCLUDED_TASKS)),
                    **kwargs,
                )

    def test_invalid_encoder_settings_fail_early(self):
        for changes in ({"theta_sd": (np.nan,) * 6}, {"stride": 0}, {"n_ch": 9},
                        {"dead_zone": np.inf}, {"raw_win": 1}, {"fall_policy": "all"},
                        {"excluded_tasks": ("T00",)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.cfg, **changes)

    def test_manual_labels_and_exact_postimpact_endpoint_interval(self):
        record = np.zeros((700, 6), dtype=np.float32)
        frames = np.arange(1, 701)
        info = {
            "label_id": 0, "subject": "SA06", "frames": frames,
            "onset_idx": 399, "impact_idx": 449,
            "fall_onset_frame": 400, "fall_impact_frame": 450,
            "manual_intervals": [[1, 400, 1], [570, 701, 2]],
        }
        for raw_win in (120, 206, 320):
            cfg = replace(self.cfg, raw_win=raw_win, stride=raw_win // 2, post_impact_seconds=1.2)
            starts = window_starts_for_record(record, info, cfg)
            endpoints = frames[starts + raw_win - 1]
            labels = window_activity_labels(endpoints, info, cfg)
            self.assertTrue(np.any(labels == 0))
            self.assertTrue(np.any(labels == 3))
            self.assertIn(569, endpoints)
            self.assertTrue(np.all((endpoints[labels == 3] >= 450) & (endpoints[labels == 3] < 570)))
            np.testing.assert_array_equal(
                window_activity_labels(np.array([399, 400, 449, 450, 569, 570]), info, cfg),
                [1, 0, 0, 3, 3, 2],
            )
        gap_info = {**info, "frames": np.array([i for i in frames if i != 569])}
        gap_record = np.zeros((699, 6), dtype=np.float32)
        cfg = replace(self.cfg, raw_win=120, post_impact_seconds=1.2)
        ends = gap_info["frames"][window_starts_for_record(gap_record, gap_info, cfg) + 119]
        self.assertIn(568, ends)

    def test_postimpact_events_and_offsets_use_only_observed_window(self):
        cfg = replace(self.cfg, post_impact_seconds=0.03)
        records = np.empty(1, dtype=object)
        records[0] = self.signal.copy()
        before = build_event_split(records, np.array([0]), [self.info], [0], cfg, verbose=False)
        np.testing.assert_array_equal(np.unique(before["y_h1"]), [0, 3])
        post = before["y_h1"] == 3
        self.assertTrue(np.isnan(before["lead_time_ms"][post]).all())
        last = int(np.max(before["start"] + cfg.raw_win))
        records[0][last:] = 1e9
        after = build_event_split(records, np.array([0]), [self.info], [0], cfg, verbose=False)
        np.testing.assert_array_equal(before["x_ev"], after["x_ev"])
        np.testing.assert_array_equal(before["offset"], after["offset"])

    def test_manual_annotation_changes_invalidate_cached_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            path = root / "manual.json"
            data = {
                "format": "kfall_manual_normal_labels", "version": 1,
                "dataset": "kfall", "sample_rate_hz": 100,
                "recordings": {"SA06T01R01": {
                    "file": "S06T01R01.csv", "reviewed": True,
                    "intervals": [[101, 121, 1]],
                }},
            }
            path.write_text(json.dumps(data))
            with contextlib.redirect_stdout(io.StringIO()):
                splits, metadata = load_or_build_kfall_event_splits(
                    root, cache_dir=root / "cache", cfg=self.cfg,
                    annotation_file=path, verbose=False,
                )
            self.assertIsNotNone(metadata["manual_annotation_signature"])
            arrays = next(value for value in splits.values() if "SA06" in value["subject"])
            recording_index = next(i for i, value in enumerate(metadata["recordings"])
                                   if value["subject"] == "SA06" and value["task_id"] == 1)
            self.assertTrue(np.all(arrays["y_h1"][arrays["file_idx"] == recording_index] == 1))
            data["recordings"]["SA06T01R01"]["intervals"][0][2] = 2
            path.write_text(json.dumps(data))
            with patch("loaders.kfall_event_loader.load_kfall_3class", side_effect=RuntimeError("annotations changed")):
                with self.assertRaisesRegex(RuntimeError, "annotations changed"):
                    load_or_build_kfall_event_splits(
                        root, cache_dir=root / "cache", cfg=self.cfg, annotation_file=path, verbose=False,
                    )

    def test_invalid_manual_intervals_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manual.json"
            for intervals in ([[1, 5, 3]], [[1, 5, 1], [4, 8, 2]], [[5, 5, 1]]):
                path.write_text(json.dumps({
                    "format": "kfall_manual_normal_labels", "version": 1,
                    "dataset": "kfall", "sample_rate_hz": 100,
                    "recordings": {"SA06T01R01": {"reviewed": True, "intervals": intervals}},
                }))
                with self.assertRaises(ValueError):
                    load_manual_labels(path)


if __name__ == "__main__":
    unittest.main()
