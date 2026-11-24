"""Regression tests for SisFall TRAIN-only posture fitting and calibration."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import posture_tagging as posture
from loaders import sisfall_event_loader as loader


class PostureTrainOnlyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = np.empty(3, dtype=object)
        self.meta = []
        self.labels = {}
        for i, subject in enumerate(("SE01", "SA01", "SE15")):
            signal = rng.normal(size=(120, 6)).astype(np.float32)
            signal[60:] *= 10
            self.x[i] = signal
            filename = f"D01_{subject}_R01.txt"
            self.meta.append({
                "file": filename, "subject": subject, "code": "D01", "trial": "R01",
            })
            self.labels[filename] = {
                "stable_posture": [{"t0": 0.0, "t1": 0.5}],
                "postural_transition": [{"t0": 0.6, "t1": 1.2}],
                "ambiguous": [],
            }
        self.y = np.full(3, 2)
        self.cfg = posture.TaggingConfig(
            fs=100, win=20, hop=10, pad_hop=10, pad_rep=2,
            pca_k=2, ds_grav=2, steps=3, smooth_k=1,
        )

    def _calibrate(self, **kwargs):
        values = dict(x=self.x, y=self.y, meta=self.meta, labels=self.labels, cfg=self.cfg)
        values.update(kwargs)
        return posture.calibrate_and_tag_postures(**values, train_subjects=["SE01"])

    def test_held_out_signals_and_manuals_cannot_change_fitted_parameters(self):
        first_records, _, first_model, first_calibration = self._calibrate()
        changed_x = deepcopy(self.x)
        changed_labels = deepcopy(self.labels)
        for i in (1, 2):
            changed_x[i] = np.random.default_rng(i).normal(1000, 100, (180, 6)).astype(np.float32)
            changed_labels[self.meta[i]["file"]] = {
                "stable_posture": [], "postural_transition": [],
                "ambiguous": [{"t0": 0, "t1": 100}],
            }
        self.assertEqual(first_calibration["training_subjects"], ["SE01"])
        self.assertEqual(first_calibration["gold_files"], [self.meta[0]["file"]])
        for changes in (
            {"x": changed_x}, {"labels": changed_labels},
            {"x": changed_x, "labels": changed_labels},
        ):
            with self.subTest(changed=sorted(changes)):
                second_records, _, second_model, second_calibration = self._calibrate(**changes)
                self.assertEqual(first_calibration, second_calibration)
                for key in ("mu", "sd", "wp", "w_postural_transition", "b_postural_transition",
                            "w_stable_posture", "b_stable_posture", "training_windows", "training_files"):
                    np.testing.assert_array_equal(first_model[key], second_model[key])
                np.testing.assert_array_equal(first_records[0]["y_win"], second_records[0]["y_win"])
                self.assertEqual({record["subject"] for record in second_records}, {"SE01", "SA01", "SE15"})

    def test_thresholds_are_optimized_not_reported_defaults(self):
        cfg = replace(self.cfg, use_hyst=False, amb_ceil=0.4, th_pt=0.8, th_sp=0.8, margin=0.25)
        record = dict(
            self.meta[0], starts=np.asarray([0, 10]),
            p_postural_transition=np.asarray([0.2, 0.57], np.float32),
            p_stable_posture=np.asarray([0.57, 0.2], np.float32),
        )
        manual = dict(self.meta[0], starts=np.asarray([0, 10]), y_win=np.asarray([0., 1.]))
        with patch.object(posture.tagging, "train_and_tag_postures", return_value=([record], {})) as fit, patch.object(
            posture.tagging, "build_manual_posture_records", return_value=([manual], {})
        ):
            records, _, _, calibration = self._calibrate(cfg=cfg)
        self.assertEqual(set(fit.call_args.args[3]), {self.meta[0]["file"]})
        self.assertEqual(calibration["parameters"]["tau"], 0.5)
        self.assertEqual(calibration["parameters"]["m"], 0.05)
        self.assertEqual(calibration["training_score"], 1.0)
        np.testing.assert_array_equal(records[0]["y_win"], [0, 1])

    def test_missing_train_gold_and_single_class_fail_without_default_calibration(self):
        held_out_only = {self.meta[1]["file"]: self.labels[self.meta[1]["file"]]}
        with self.assertRaisesRegex(RuntimeError, "TRAIN manual gold"):
            self._calibrate(labels=held_out_only)
        stable_only = deepcopy(self.labels)
        stable_only[self.meta[0]["file"]]["postural_transition"] = []
        with self.assertRaisesRegex(RuntimeError, "stable and transition"):
            self._calibrate(labels=stable_only)
        with self.assertRaises(ValueError):
            self._calibrate_subjects([])
        with self.assertRaises(ValueError):
            self._calibrate_subjects(["SE99"])
        with self.assertRaises(TypeError):
            posture.calibrate_and_tag_postures(self.x, self.y, self.meta, self.labels, self.cfg)

    def _calibrate_subjects(self, subjects):
        return posture.calibrate_and_tag_postures(
            self.x, self.y, self.meta, self.labels, self.cfg, train_subjects=subjects,
        )

    def test_old_global_posture_cache_never_reused_and_artifacts_retain_fit(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            old = root / "posture_default"
            old.mkdir()
            np.save(old / "records.npy", np.asarray(["global-cache"], dtype=object))
            (old / "summary.json").write_text('{"cache_id": "legacy"}')
            kwargs = dict(
                dataset_root=root / "raw", label_dir=root / "labels",
                cache_dir=root, result_dir=root / "outputs", cfg=self.cfg, verbose=False,
            )
            with patch.object(posture.tagging, "_load_dataset", return_value=(self.x, self.y, self.meta, 0)), patch.object(
                posture.tagging, "_load_labels", return_value=self.labels
            ), patch.object(posture.tagging, "run_posture_tagging", side_effect=AssertionError("global API used")):
                first = posture.run_posture_tagging(**kwargs, train_subjects=["SE01"])
                second = posture.run_posture_tagging(**kwargs, train_subjects=["SA01"])
            self.assertNotEqual(first["paths"]["cache_dir"], second["paths"]["cache_dir"])
            self.assertIn("train_only_v3", first["paths"]["cache_dir"])
            saved = json.loads(Path(first["paths"]["summary"]).read_text())
            self.assertEqual(saved["calibration"], first["summary"]["calibration"])
            self.assertEqual(saved["training_subjects"], ["SE01"])
            with np.load(first["paths"]["model_cache"]) as model:
                for key in ("mu", "sd", "wp", "b_postural_transition", "b_stable_posture"):
                    np.testing.assert_array_equal(model[key], first["model"][key])

    def test_legacy_standalone_call_signature_remains_available(self):
        with patch.object(posture.tagging, "run_posture_tagging", return_value={"legacy": True}) as legacy:
            self.assertEqual(posture.run_posture_tagging(verbose=False), {"legacy": True})
        legacy.assert_called_once()

    def test_loader_supplies_exact_train_subjects_and_invalidates_old_cache(self):
        cfg = loader.EventLoaderConfig(
            raw_win=20, out_win=24, stride_static=10, num_workers=0,
            theta_sd=(1.,) * 6,
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            kwargs = dict(
                dataset_root=root / "raw", label_dir=root / "labels",
                cache_dir=root, result_dir=root / "results", cfg=cfg,
                posture_cfg=self.cfg, verbose=False,
            )
            with patch.object(loader, "_load_sisfall", return_value=(self.x, self.y, self.meta, 0)), patch.object(
                loader, "_load_or_build_posture_tags", return_value={}
            ) as build_posture:
                splits, metadata = loader.load_or_build_sisfall_event_splits(**kwargs)
            subjects = build_posture.call_args.kwargs["train_subjects"]
            self.assertEqual(subjects, metadata["split_subjects"]["train"])
            self.assertFalse(set(subjects) & set(metadata["split_subjects"]["val"]))
            self.assertFalse(set(subjects) & set(metadata["split_subjects"]["test"]))
            self.assertEqual(set().union(*map(set, metadata["split_subjects"].values())), {"SE01", "SA01", "SE15"})
            with patch.object(loader, "_load_sisfall", side_effect=AssertionError("valid cache rebuilt")):
                cached, _ = loader.load_or_build_sisfall_event_splits(**kwargs)
            np.testing.assert_array_equal(cached["train"]["y_h2"], splits["train"]["y_h2"])
            with patch.object(loader, "_load_sisfall", side_effect=RuntimeError("rebuild")):
                with self.assertRaisesRegex(RuntimeError, "rebuild"):
                    loader.load_or_build_sisfall_event_splits(
                        **dict(kwargs, posture_cfg=replace(self.cfg, margin=0.2))
                    )
                metadata["version"] = 2
                (root / "sisfall_event_loader_default" / "metadata.json").write_text(json.dumps(metadata))
                with self.assertRaisesRegex(RuntimeError, "rebuild"):
                    loader.load_or_build_sisfall_event_splits(**kwargs)

    def test_manual_changes_invalidate_input_fingerprint(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            labels = root / "labels"
            labels.mkdir()
            path = labels / "D01_SE01_R01.json"
            path.write_text('{"stable_posture": []}')
            first = posture.posture_input_fingerprint(root / "raw", labels)
            path.write_text('{"stable_posture": [{"t0": 0, "t1": 1}]}')
            self.assertNotEqual(first, posture.posture_input_fingerprint(root / "raw", labels))


if __name__ == "__main__":
    unittest.main()
