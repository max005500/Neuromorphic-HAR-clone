"""Compare matching recording endpoints rather than unrelated cache row IDs."""

from __future__ import annotations

import json
import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from loaders.kfall_event_loader import load_or_build_kfall_event_splits
from visualization.kfall_variants import KFallVariantStore


class KFallVariantDataTests(unittest.TestCase):
    def setUp(self):
        from test_kfall_viewer_data import KFallViewerDataTests

        self.fixture = KFallViewerDataTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.directory
        self.paths = []
        self.rows = []
        for size in (5, 8, 10):
            cfg = replace(self.fixture.config, raw_win=size, out_win=size * 2,
                          endpoint_context=10, post_impact_seconds=.12, resample_hz=200)
            _, metadata = load_or_build_kfall_event_splits(
                self.fixture.root, cache_dir=self.fixture.cache_parent,
                run_name=f"variant{size}", cfg=cfg, verbose=False,
            )
            path = self.fixture.cache_parent / f"kfall_event_loader_variant{size}"
            self.paths.append(path)
            self.rows.append({"cache_dir": str(path), "raw_win": size, "out_win": size * 2,
                              "resample_hz": 200, "post_impact_seconds": .12,
                              "endpoint_context": 10, "duration_seconds": size / 100,
                              "source_signature": metadata["source_signature"],
                              "split_subjects": metadata["split_subjects"]})
        self.manifest = self.directory / "comparison.json"
        self.manifest.write_text(json.dumps(self.rows))

    def test_join_is_lazy_read_only_sorted_and_returns_matching_postimpact_inputs(self):
        timestamps = {path: path.stat().st_mtime_ns for cache in self.paths for path in cache.rglob("*.npy")}
        with patch("visualization.kfall_data.load_record", side_effect=AssertionError("eager CSV read")):
            store = KFallVariantStore(cache_dirs=list(reversed(self.paths)))
            self.assertTrue(all(not item._splits for item in store.stores))
            self.assertEqual(store.durations, [.05, .08, .1])
            indices = store.get_indices("val", class_id=3)
            self.assertGreater(len(indices), 0)
            self.assertEqual(store.coverage["val"]["discarded_windows"], [0, 0, 0])
        windows = store.get_windows("val", int(indices[0]))
        self.assertEqual([len(window.raw) for window in windows], [5, 8, 10])
        self.assertEqual([len(window.analog) for window in windows], [9, 15, 19])
        self.assertEqual(len({(w.subject, w.code, w.trial, w.end_frame, w.class_id) for w in windows}), 1)
        self.assertTrue(all(w.class_id == 3 and w.label == "Postimpact" for w in windows))
        for window in windows:
            np.testing.assert_array_equal(window.raw[-1], windows[0].raw[-1])
            self.assertFalse(window.events.flags.writeable)
        self.assertEqual(timestamps, {path: path.stat().st_mtime_ns for path in timestamps})
        subject = windows[0].subject
        np.testing.assert_array_equal(store.get_indices("val", subject=subject[2:]),
                                      store.get_indices("val", subject=subject))
        self.assertEqual(store.subjects("val"), [subject])

    def test_shuffled_variant_rows_are_joined_by_endpoint_not_row_number(self):
        for path in (self.paths[1] / "val").glob("*.npy"):
            values = np.load(path, allow_pickle=False)
            np.save(path, values[::-1])
        store = KFallVariantStore(self.manifest)
        windows = store.get_windows("val", 0)
        self.assertNotEqual(windows[0].index, windows[1].index)
        self.assertEqual(len({(w.subject, w.code, w.trial, w.end_frame, w.class_id) for w in windows}), 1)

    def test_natural_coverage_reports_intersection_without_inventing_windows(self):
        natural = []
        for size in (5, 8, 10):
            cfg = replace(self.fixture.config, raw_win=size, out_win=size * 2,
                          endpoint_context=None, post_impact_seconds=.12, resample_hz=200)
            load_or_build_kfall_event_splits(self.fixture.root, cache_dir=self.fixture.cache_parent,
                                            run_name=f"natural{size}", cfg=cfg, verbose=False)
            natural.append(self.fixture.cache_parent / f"kfall_event_loader_natural{size}")
        store = KFallVariantStore(cache_dirs=natural)
        self.assertEqual(store.split_size("test"), 2)
        self.assertTrue(all(count > 2 for count in store.coverage["test"]["variant_windows"]))
        self.assertTrue(all(count > 0 for count in store.coverage["test"]["discarded_windows"]))
        for index in store.get_indices("test"):
            windows = store.get_windows("test", int(index))
            self.assertEqual(len({(w.end_frame, w.class_id) for w in windows}), 1)

    def test_incompatible_subjects_sources_thresholds_and_units_are_rejected(self):
        path = self.paths[1] / "metadata.json"
        original = path.read_text()
        for field in ("subjects", "source", "theta", "rate", "units"):
            metadata = json.loads(original)
            if field == "subjects":
                metadata["split_subjects"]["train"], metadata["split_subjects"]["test"] = (
                    metadata["split_subjects"]["test"], metadata["split_subjects"]["train"])
            elif field == "source":
                metadata["source_signature"]["sha256"] = "different dataset"
            elif field == "theta":
                metadata["config"]["theta_sd"][0] *= 2
            elif field == "rate":
                metadata["config"]["resample_hz"] = 100
            else:
                metadata["units"][0] = "counts"
            path.write_text(json.dumps(metadata))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "incompatible"):
                KFallVariantStore(cache_dirs=self.paths)
            path.write_text(original)

    def test_duplicate_windows_and_conflicting_endpoint_labels_are_rejected(self):
        for path in (self.paths[1] / "val").glob("*.npy"):
            values = np.load(path, allow_pickle=False)
            np.save(path, np.concatenate((values, values[:1])))
        store = KFallVariantStore(cache_dirs=self.paths)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            store.get_indices("val")

    def test_conflicting_labels_cannot_silently_disappear_from_the_intersection(self):
        path = self.paths[1] / "val" / "y_h1.npy"
        labels = np.load(path, allow_pickle=False)
        labels[0] = (labels[0] + 1) % 4
        np.save(path, labels)
        store = KFallVariantStore(cache_dirs=self.paths)
        with self.assertRaisesRegex(ValueError, "different labels"):
            store.get_indices("val")

    def test_relocated_manifest_uses_local_identity_checked_cache_copies(self):
        project = self.directory / "relocated_project"
        local = project / "data" / "kfall_cache"
        local.mkdir(parents=True)
        for row, path in zip(self.rows, self.paths):
            destination = local / path.name
            shutil.move(path, destination)
            old = self.directory / "old_project" / path.name
            old.mkdir(parents=True)
            wrong = json.loads((destination / "metadata.json").read_text())
            wrong["source_signature"]["sha256"] = "an unrelated stale copy"
            (old / "metadata.json").write_text(json.dumps(wrong))
            row["cache_dir"] = str(old)
        self.manifest.write_text(json.dumps(self.rows))
        with patch("visualization.kfall_variants.PROJECT_ROOT", project):
            store = KFallVariantStore(self.manifest, dataset_root=self.fixture.root)
        self.assertTrue(all(path.parent == local for path in store.cache_dirs))
        self.assertEqual(len(store.get_windows("test", 0)), 3)

    def test_legacy_three_class_caches_are_supported(self):
        paths = []
        for size in (5, 8):
            cfg = replace(self.fixture.config, raw_win=size, endpoint_context=10)
            load_or_build_kfall_event_splits(self.fixture.root, cache_dir=self.fixture.cache_parent,
                                            run_name=f"legacy{size}", cfg=cfg, verbose=False)
            paths.append(self.fixture.cache_parent / f"kfall_event_loader_legacy{size}")
        store = KFallVariantStore(cache_dirs=paths)
        self.assertEqual(store.n_classes, 3)
        self.assertGreater(len(store.get_indices("train", class_id=0)), 0)
        self.assertEqual(len(store.get_windows("train", 0)), 2)
        with self.assertRaises(ValueError):
            store.get_indices("train", class_id=3)

    def test_missing_manifest_and_invalid_variant_selection_give_clear_errors(self):
        with self.assertRaisesRegex(FileNotFoundError, "configs/kfall_theta_200hz.json"):
            KFallVariantStore(self.directory / "missing.json")
        with self.assertRaisesRegex(ValueError, "at least two"):
            KFallVariantStore(cache_dirs=self.paths[:1])
        with self.assertRaisesRegex(ValueError, "distinct"):
            KFallVariantStore(cache_dirs=[self.paths[0], self.paths[0]])
        with self.assertRaisesRegex(ValueError, "not both"):
            KFallVariantStore(self.manifest, cache_dirs=self.paths)


if __name__ == "__main__":
    unittest.main()
