"""Keep best-checkpoint evaluation tied to the original held-out cache."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from openpyxl import Workbook

from evaluation.kfall_data import CHECKPOINT_DATA_KEYS, load_evaluation_data
from loaders.kfall_event_loader import KFallEventLoaderConfig, load_or_build_kfall_event_splits


class KFallEvaluationDataTests(unittest.TestCase):
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
                        writer.writerow([(frame - 101) / 100, frame, frame / 100, 1, 0, frame, 0, 1])
            workbook = Workbook()
            workbook.active.append(["Task Code (Task ID)", "Trial ID", "Fall_onset_frame", "Fall_impact_frame"])
            workbook.active.append(["F01 (20)", 1, 109, 114])
            workbook.save(labels / f"SA{subject:02d}_label.xlsx")
            workbook.close()
        cfg = KFallEventLoaderConfig(raw_win=5, out_win=16, stride=4, split=(1, 1, 1), verbose_every=0)
        self.cache_parent = self.directory / "cache"
        self.cache = self.cache_parent / "kfall_event_loader_default"
        _, metadata = load_or_build_kfall_event_splits(self.root, cache_dir=self.cache_parent, cfg=cfg, verbose=False)
        self.run = self.directory / "run"
        self.run.mkdir()
        self.snapshot_path = self.run / "dataset_metadata.json"
        self.snapshot_path.write_text(json.dumps(metadata))
        config = asdict(cfg)
        self.checkpoint_config = {key: config[value] for key, value in CHECKPOINT_DATA_KEYS.items()}
        self.checkpoint_config.update(dataset="kfall", cache_dir=str(self.cache_parent), cache_name="default")

    def rewrite_cache_metadata(self, mutate):
        path = self.cache / "metadata.json"
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))

    def test_reads_only_test_arrays_without_raw_files_or_cache_rebuild(self):
        shutil.rmtree(self.root)
        stamps = {path: path.stat().st_mtime_ns for path in self.cache.rglob("*") if path.is_file()}
        original_load = np.load
        opened = []

        def tracked_load(path, *args, **kwargs):
            opened.append(Path(path))
            return original_load(path, *args, **kwargs)

        with patch("numpy.load", side_effect=tracked_load), patch(
            "loaders.kfall_event_loader.load_or_build_kfall_event_splits", side_effect=AssertionError("evaluation rebuilt data")
        ):
            arrays, current, snapshot, resolved = load_evaluation_data(self.run, self.checkpoint_config)
        self.assertEqual(resolved, self.cache)
        self.assertEqual(current["source_signature"], snapshot["source_signature"])
        self.assertEqual(len(opened), 11)
        self.assertTrue(all(path.parent.name == "test" for path in opened))
        self.assertTrue(all(isinstance(array, np.memmap) and not array.flags.writeable for array in arrays.values()))
        self.assertEqual(stamps, {path: path.stat().st_mtime_ns for path in stamps})

    def test_rejects_changed_encoder_and_checkpoint_settings(self):
        changed = {**self.checkpoint_config, "theta": [1] * 6}
        with self.assertRaisesRegex(ValueError, "Checkpoint theta"):
            load_evaluation_data(self.run, changed)
        self.rewrite_cache_metadata(lambda value: value["config"].update(stride=2))
        with self.assertRaisesRegex(ValueError, "preprocessing config"):
            load_evaluation_data(self.run, self.checkpoint_config)

    def test_runtime_settings_do_not_change_data_identity(self):
        self.rewrite_cache_metadata(lambda value: value["config"].update(batch_size=1, num_workers=0,
                                                                         pin_memory=False, verbose_every=9))
        arrays, _, _, _ = load_evaluation_data(self.run, self.checkpoint_config)
        self.assertGreater(len(arrays["y_h1"]), 0)

    def test_rejects_changed_source_signature(self):
        self.rewrite_cache_metadata(lambda value: value["source_signature"].update(sha256="different"))
        with self.assertRaisesRegex(ValueError, "source_signature"):
            load_evaluation_data(self.run, self.checkpoint_config)

    def test_rejects_subject_leakage_even_if_both_metadata_files_are_changed(self):
        value = json.loads(self.snapshot_path.read_text())
        value["split_subjects"]["test"] = value["split_subjects"]["train"]
        self.snapshot_path.write_text(json.dumps(value))
        (self.cache / "metadata.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "recording subjects|leakage"):
            load_evaluation_data(self.run, self.checkpoint_config)

    def test_rejects_duplicate_or_shifted_test_window_manifest(self):
        path = self.cache / "test" / "start.npy"
        starts = np.load(path)
        starts[1] = starts[0]
        np.save(path, starts)
        with self.assertRaisesRegex(ValueError, "file/start manifest"):
            load_evaluation_data(self.run, self.checkpoint_config)

    def test_rejects_changed_test_subject_or_label_arrays(self):
        path = self.cache / "test" / "subject.npy"
        subjects = np.load(path)
        subjects[0] = "SA99"
        np.save(path, subjects)
        with self.assertRaisesRegex(ValueError, "Test window subjects"):
            load_evaluation_data(self.run, self.checkpoint_config)

    def test_rejects_invalid_events(self):
        path = self.cache / "test" / "x_ev.npy"
        events = np.load(path)
        events[0, 0, 0] = [1, 1]
        np.save(path, events)
        with self.assertRaisesRegex(ValueError, "simultaneous ON/OFF"):
            load_evaluation_data(self.run, self.checkpoint_config)

    def test_rejects_missing_metadata_or_arrays_without_rebuilding(self):
        snapshot = self.snapshot_path.read_bytes()
        self.snapshot_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "metadata not found"):
            load_evaluation_data(self.run, self.checkpoint_config)
        self.snapshot_path.write_bytes(snapshot)
        (self.cache / "test" / "offset.npy").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "will not rebuild"):
            load_evaluation_data(self.run, self.checkpoint_config)

    def test_moved_cache_and_source_path_prefixes_are_supported(self):
        moved = self.directory / "moved_cache"
        shutil.copytree(self.cache, moved)
        metadata = json.loads((moved / "metadata.json").read_text())
        for record in metadata["recordings"]:
            record["file"] = f"/new/dataset/location/{Path(record['file']).name}"
        metadata["dataset_root"] = "/new/dataset/location"
        (moved / "metadata.json").write_text(json.dumps(metadata))
        arrays, _, _, resolved = load_evaluation_data(self.run, self.checkpoint_config, cache_dir=moved)
        self.assertEqual(resolved, moved)
        self.assertGreater(len(arrays["y_h1"]), 0)


if __name__ == "__main__":
    unittest.main()
