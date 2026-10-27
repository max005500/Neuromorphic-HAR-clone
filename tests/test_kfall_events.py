"""Causal temporal windowing, subject isolation, and KFall cache integrity."""

from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from openpyxl import Workbook

from loaders.kfall_event_loader import (
    KFallEventLoaderConfig, _source_signature, build_event_split, load_or_build_kfall_event_splits,
    signal_to_events, window_starts_for_record,
)
from preprocess.preprocess_kfall import CHANNELS


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

    def test_invalid_encoder_settings_fail_early(self):
        for changes in ({"theta_sd": (np.nan,) * 6}, {"stride": 0}, {"n_ch": 9},
                        {"dead_zone": np.inf}, {"raw_win": 1}, {"fall_policy": "all"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.cfg, **changes)


if __name__ == "__main__":
    unittest.main()
