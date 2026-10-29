"""Validation-only threshold selection, independent scales, and run provenance."""

from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tune_kfall_thresholds import (
    build_candidates, build_parser, candidate_name, collect_candidate,
    export_results, main, search_settings, select_best, training_command,
    verify_comparable,
)


class ThresholdSearchTests(unittest.TestCase):
    def test_candidate_grid_changes_sensor_groups_independently(self):
        candidates = build_candidates()
        self.assertEqual(len(candidates), 9)
        self.assertEqual(candidates[0].theta, (.1, .1, .1, 10., 10., 10.))
        selected = next(candidate for candidate in candidates
                        if candidate.acc_scale == .5 and candidate.gyro_scale == 2.)
        self.assertEqual(selected.theta, (.05, .05, .05, 20., 20., 20.))
        self.assertEqual(len({candidate.theta for candidate in candidates}), 9)
        self.assertEqual(len({candidate.candidate_id for candidate in candidates}), 9)

    def test_invalid_vectors_and_scale_grids_are_rejected(self):
        for kwargs in ({"theta": [1] * 5}, {"theta": [0] * 6}, {"theta": [float("nan")] * 6},
                       {"acc_scales": []}, {"gyro_scales": [-1]}, {"acc_scales": [1, 1]},
                       {"gyro_scales": [float("inf")]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_candidates(**kwargs)

    def test_training_commands_disable_test_and_hold_experiment_settings(self):
        args = build_parser().parse_args(["--window-seconds", "3.30", "--endpoint-context", "410",
                                          "--train-per-class", "64", "--seed", "12"])
        candidates = build_candidates()
        baseline = training_command(args, candidates[0])
        changed = training_command(args, candidates[1])
        for command in (baseline, changed):
            self.assertIn("--validation-only", command)
            self.assertNotIn("--preprocess-only", command)
            for option, value in (("--raw-win", "330"), ("--out-win", "3297"),
                                  ("--endpoint-context", "410"), ("--resample-hz", "200.0"),
                                  ("--train-per-class", "64"), ("--seed", "12")):
                self.assertEqual(command[command.index(option) + 1], value)
        self.assertNotEqual(baseline[baseline.index("--cache-name") + 1], changed[changed.index("--cache-name") + 1])

    def test_sampling_resampling_and_duration_change_candidate_identity(self):
        candidate = build_candidates()[0]
        parser = build_parser()
        initial = candidate_name(candidate, search_settings(parser.parse_args([])))
        for arguments in (["--resample-hz", "100"], ["--train-per-class", "0"],
                          ["--window-seconds", "4.10"], ["--endpoint-context", "410"],
                          ["--epochs", "10"], ["--seed", "1"]):
            with self.subTest(arguments=arguments):
                self.assertNotEqual(initial, candidate_name(candidate, search_settings(parser.parse_args(arguments))))

    def test_theta_file_and_explicit_theta_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--theta-file", "winner.json", "--theta", "1", "1", "1", "1", "1", "1"])

    def test_theta_file_matches_numeric_candidate_identity_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            values = [.05, .05, .05, 5., 5., 5.]
            theta_file = directory / "winner.json"
            theta_file.write_text(json.dumps({"theta_best": values}))
            common = ["--dry-run", "--acc-scales", ".75", "1", "1.25", "--gyro-scales", ".75", "1", "1.25"]
            main([*common, "--results-dir", str(directory / "file"), "--theta-file", str(theta_file)])
            main([*common, "--results-dir", str(directory / "numeric"), "--theta", *map(str, values)])
            file_plan = json.loads((directory / "file/search_plan_dry_run.json").read_text())
            numeric_plan = json.loads((directory / "numeric/search_plan_dry_run.json").read_text())
            self.assertEqual(file_plan["candidates"], numeric_plan["candidates"])
            self.assertEqual(file_plan["settings"], numeric_plan["settings"])
            self.assertEqual(file_plan["candidates"][0]["theta"], values)
            for invalid in ({"theta_best": [0] * 6}, {"theta_best": [1] * 5}, {"theta_best": [float("nan")] * 6}):
                theta_file.write_text(json.dumps(invalid))
                with self.subTest(invalid=invalid), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main([*common, "--results-dir", str(directory / "invalid"), "--theta-file", str(theta_file)])

    def test_selection_uses_validation_and_loss_tiebreak_never_test(self):
        values = [
            {"index": 0, "macro_f1": .8, "balanced_accuracy": .9, "validation_loss": .5, "test_accuracy": 0.},
            {"index": 1, "macro_f1": .7, "balanced_accuracy": .95, "validation_loss": .1, "test_accuracy": 1.},
            {"index": 2, "macro_f1": .8, "balanced_accuracy": .8, "validation_loss": .4, "test_accuracy": 0.},
        ]
        self.assertEqual(select_best(values)["index"], 2)
        self.assertEqual(select_best(values, "balanced_accuracy")["index"], 1)
        values[0]["validation_loss"] = .4
        self.assertEqual(select_best(values)["index"], 0)
        with self.assertRaises(ValueError):
            select_best(values, "test_accuracy")

    def test_comparability_includes_subjects_endpoints_and_actual_train_subset(self):
        result = {"split_subjects": {"train": ["SA06"], "val": ["SA07"], "test": ["SA08"]},
                  "source_signature": {"sha256": "source"}, "endpoint_signatures": {"val": "samples"},
                  "training_indices_sha256": "subset"}
        verify_comparable([result, deepcopy(result)])
        for key in result:
            changed = deepcopy(result)
            changed[key] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_comparable([result, changed])

    @staticmethod
    def fixture(directory):
        import torch
        from train_kfall import get_args, training_subset_indices
        from loaders.kfall_event_loader import KFallEventLoaderConfig

        args = build_parser().parse_args([
            "--dataset-root", str(directory / "raw"), "--cache-dir", str(directory / "cache"),
            "--out-dir", str(directory / "runs"), "--results-dir", str(directory / "results"),
            "--epochs", "2", "--train-per-class", "1",
        ])
        candidate = build_candidates()[0]
        config = vars(get_args(training_command(args, candidate)[2:]))
        name = candidate_name(candidate, search_settings(args))
        run = args.out_dir / name / "created"
        (run / "metrics").mkdir(parents=True)
        (run / "checkpoints").mkdir()
        (run / "config.json").write_text(json.dumps(config))
        source = {"sha256": "source", "files": 2}
        metadata = {
            "source_signature": source,
            "split_subjects": {"train": ["SA06"], "val": ["SA07"], "test": ["SA08"]},
            "config": asdict(KFallEventLoaderConfig(resample_hz=200., post_impact_seconds=1.2)),
        }
        (run / "dataset_metadata.json").write_text(json.dumps(metadata))
        cache = args.cache_dir / f"kfall_event_loader_{name}"
        for split in ("train", "val"):
            split_dir = cache / split
            split_dir.mkdir(parents=True)
            for key, value in (("y_h1", [0, 1, 2, 3]), ("file_idx", [0, 1, 2, 3]), ("end_frame", [500] * 4)):
                np.save(split_dir / f"{key}.npy", np.asarray(value))
        (cache / "metadata.json").write_text(json.dumps(metadata))
        np.save(cache / "val/x_ev.npy", np.zeros((4, config["out_win"], 6, 2), dtype=np.uint8))
        truth = np.arange(4)
        np.save(run / "metrics/y_true_val.npy", truth)
        np.save(run / "metrics/y_pred_val.npy", truth)
        np.save(run / "training_indices.npy", training_subset_indices(truth, 1, args.seed))
        (run / "metrics/history.json").write_text(json.dumps([{"epoch": 1}, {"epoch": 2}]))
        (run / "metrics/validation_metrics.json").write_text(json.dumps({
            "loss": .5, "best_val_loss": .5, "best_epoch": 1,
        }))
        torch.save({"config": config, "epoch": 1, "best_val_loss": .5, "model_state_dict": {}},
                   run / "checkpoints/best_model.pt")
        return args, candidate, run, source, cache

    def test_completed_candidate_collects_only_validation_and_detects_test_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, candidate, run, source, cache = self.fixture(Path(temporary))
            result = collect_candidate(args, candidate, run, source)
            self.assertEqual(result["macro_f1"], 1.)
            self.assertEqual(result["balanced_accuracy"], 1.)
            self.assertEqual(result["validation_recall"]["Postimpact"], 1.)
            self.assertEqual(result["validation_event_rate"], 0.)
            self.assertNotIn("test_accuracy", result)
            (run / "metrics/test_metrics.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "without test inference"):
                collect_candidate(args, candidate, run, source)

    def test_collection_rejects_stale_source_incomplete_runs_and_changed_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, candidate, run, source, cache = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "source files"):
                collect_candidate(args, candidate, run, {"sha256": "changed", "files": 2})
            history_path = run / "metrics/history.json"
            history_path.write_text(json.dumps([{"epoch": 1}]))
            with self.assertRaisesRegex(ValueError, "not completed"):
                collect_candidate(args, candidate, run, source)
            history_path.write_text(json.dumps([{"epoch": 1}, {"epoch": 2}]))
            np.save(run / "metrics/y_true_val.npy", np.asarray([0, 1, 2, 2]))
            with self.assertRaisesRegex(ValueError, "full held-out validation"):
                collect_candidate(args, candidate, run, source)

    def test_dry_run_writes_nine_validation_only_commands_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch("tune_kfall_thresholds.subprocess.run", side_effect=AssertionError("dry-run trained a model")):
                main(["--results-dir", str(directory), "--dry-run"])
            plan = json.loads((directory / "search_plan_dry_run.json").read_text())
            self.assertEqual(len(plan["candidates"]), 9)
            self.assertFalse(plan["test_used_for_selection"])
            self.assertTrue(all("--validation-only" in row["command"] for row in plan["candidates"]))
            self.assertFalse((directory / "best_theta.json").exists())

    def test_dry_run_preserves_existing_training_outputs_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for filename in ("search_plan.json", "search_results.json", "best_theta.json", "threshold_validation.png"):
                (directory / filename).write_bytes(b"existing")
            main(["--results-dir", str(directory), "--dry-run", "--epochs", "10"])
            for filename in ("search_plan.json", "search_results.json", "best_theta.json", "threshold_validation.png"):
                self.assertEqual((directory / filename).read_bytes(), b"existing")

    def test_final_export_is_consumable_and_partial_export_has_no_winner(self):
        from train_kfall import read_theta_file

        with tempfile.TemporaryDirectory() as temporary:
            args, candidate, run, source, cache = self.fixture(Path(temporary))
            result = collect_candidate(args, candidate, run, source)
            args.results_dir.mkdir()
            plan = {"settings": search_settings(args), "source_signature": source}
            export_results(args, plan, [result])
            self.assertFalse((args.results_dir / "best_theta.json").exists())
            export_results(args, plan, [result], complete=True)
            winner = json.loads((args.results_dir / "best_theta.json").read_text())
            self.assertEqual(winner["selected_by"], "validation_macro_f1")
            self.assertEqual(read_theta_file(args.results_dir / "best_theta.json"), list(candidate.theta))
            self.assertFalse(winner["test_used_for_selection"])
            self.assertEqual((args.results_dir / "threshold_validation.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            export_results(args, plan, [])
            self.assertFalse((args.results_dir / "threshold_validation.png").exists())
            self.assertFalse((args.results_dir / "best_theta.json").exists())

    def test_collection_rejects_unknown_classes_even_if_cache_agrees(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, candidate, run, source, cache = self.fixture(Path(temporary))
            truth = np.asarray([0, 1, 2, 4])
            np.save(run / "metrics/y_true_val.npy", truth)
            np.save(cache / "val/y_h1.npy", truth)
            with self.assertRaisesRegex(ValueError, "full held-out validation"):
                collect_candidate(args, candidate, run, source)


if __name__ == "__main__":
    unittest.main()
