"""Native KFall format and temporal annotation integrity checks."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from openpyxl import Workbook

from preprocess.preprocess_kfall import CHANNELS, load_fall_labels, load_kfall_3class, load_record


class KFallRawTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "sensor_data" / "SA06").mkdir(parents=True)
        (self.root / "label_data").mkdir()

    def write_sensor(self, name="S06T20R01.csv", frames=(101, 102, 103, 104, 105)) -> Path:
        path = self.root / "sensor_data" / "SA06" / name
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["TimeStamp(s)", "FrameCounter", *CHANNELS, "EulerX", "EulerY", "EulerZ"])
            for frame in frames:
                writer.writerow([frame / 100, frame, 0.125, -0.75, 1.25, -15.5, 22.25, 0.0625,
                                 "unused", "unused", "unused"])
        return path

    def write_labels(self, rows=None, merge=False) -> Path:
        path = self.root / "label_data" / "SA06_label.xlsx"
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["Task Code (Task ID)", "Description", "Trial ID", "Fall_onset_frame", "Fall_impact_frame"])
        for row in rows if rows is not None else [("F01 (20)", "fall", 1, 103, 105)]:
            sheet.append(row)
        if merge:
            sheet.merge_cells("A2:A3")
        workbook.save(path)
        workbook.close()
        return path

    def test_channels_units_and_both_filename_styles(self):
        for name in ("S06T20R01.csv", "SA06T20R01.csv"):
            signals, frames, metadata = load_record(self.write_sensor(name))
            self.assertEqual(signals.shape, (5, 6))
            self.assertEqual(signals.dtype, np.float32)
            np.testing.assert_allclose(signals[0], [0.125, -0.75, 1.25, -15.5, 22.25, 0.0625])
            np.testing.assert_array_equal(frames, np.arange(101, 106))
            self.assertEqual(metadata["subject"], "SA06")
            self.assertEqual(metadata["acc_unit"], "g")
            self.assertEqual(metadata["gyr_unit"], "deg/s")

    def test_annotation_frames_are_resolved_by_counter_not_row_number(self):
        self.write_sensor()
        self.write_labels()
        signals, y, metadata, skipped = load_kfall_3class(self.root)
        self.assertEqual(signals.shape, (1,))
        self.assertEqual(signals[0].shape, (5, 6))
        np.testing.assert_array_equal(y, [0])
        self.assertEqual(skipped, 0)
        self.assertEqual(metadata[0]["onset_idx"], 2)
        self.assertEqual(metadata[0]["impact_idx"], 4)
        self.assertEqual(metadata[0]["fall_onset_frame"], 103)
        self.assertEqual(metadata[0]["fall_impact_frame"], 105)

    def test_merged_task_ids_are_forward_filled(self):
        self.write_labels([("F01 (20)", "fall", 1, 103, 105), (None, None, 2, 110, 125)], merge=True)
        labels = load_fall_labels(self.root / "label_data")
        self.assertEqual(labels[("SA06", 20, 1)], (103, 105))
        self.assertEqual(labels[("SA06", 20, 2)], (110, 125))

    def test_duplicate_labels_are_rejected(self):
        self.write_labels([("F01 (20)", "fall", 1, 103, 105), (None, None, 1, 103, 105)])
        with self.assertRaisesRegex(ValueError, "duplicate fall annotation"):
            load_fall_labels(self.root / "label_data")

    def test_missing_fall_annotation_is_rejected(self):
        self.write_sensor()
        self.write_labels([("F02 (21)", "fall", 1, 103, 105)])
        with self.assertRaisesRegex(ValueError, "missing fall annotation"):
            load_kfall_3class(self.root)

    def test_absent_annotation_frame_is_rejected(self):
        self.write_sensor(frames=(101, 102, 104, 105))
        self.write_labels()
        with self.assertRaisesRegex(ValueError, "Fall_onset_frame 103 is absent"):
            load_kfall_3class(self.root)

    def test_frame_gaps_are_retained_for_time_aware_interpolation(self):
        self.write_sensor(frames=(101, 103, 105))
        self.write_labels()
        _, _, metadata, _ = load_kfall_3class(self.root / "sensor_data")
        self.assertEqual(metadata[0]["onset_idx"], 1)
        self.assertEqual(metadata[0]["impact_idx"], 2)
        np.testing.assert_array_equal(metadata[0]["frames"], [101, 103, 105])

    def test_nonmonotonic_frames_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "FrameCounter must be strictly increasing"):
            load_record(self.write_sensor(frames=(101, 103, 102)))

    def test_duplicate_native_and_documented_filenames_are_rejected(self):
        self.write_sensor("S06T20R01.csv")
        self.write_sensor("SA06T20R01.csv")
        self.write_labels()
        with self.assertRaisesRegex(ValueError, "duplicate sensor recording"):
            load_kfall_3class(self.root)

    def test_static_family_includes_posture_changes_and_stairs_are_dynamic(self):
        self.assertEqual(load_record(self.write_sensor("S06T13R01.csv"))[2]["label_id"], 2)
        self.assertEqual(load_record(self.write_sensor("S06T35R01.csv"))[2]["label_id"], 1)


if __name__ == "__main__":
    unittest.main()
