"""Check duration experiments preserve physical scale and comparable samples."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from compare_kfall_windows import (
    _completed_run_matches, _find_completed_run, _run_experiment, _verify_results,
    build_experiment_specs, build_parser, inspect_dataset, main,
    training_command,
)


def raw_fixture():
    recordings, labels, records = [], [], []
    for subject in ("SA06", "SA07", "SA08"):
        for task, label in ((20, 0), (4, 1), (1, 2)):
            length = 1200
            recordings.append(np.zeros((length, 6), dtype=np.float32))
            labels.append(label)
            record = {
                "file": f"{subject}/S{subject[2:]}T{task:02d}R01.csv",
                "subject": subject, "code": f"T{task:02d}", "trial": "R01",
                "task_id": task, "trial_id": 1, "label_id": label, "T": length,
                "frames": np.arange(1, length + 1),
            }
            if label == 0:
                record.update(fall_onset_frame=600, fall_impact_frame=700,
                              onset_idx=599, impact_idx=699, post_impact_end_idx=819)
            records.append(record)
    values = np.empty(len(recordings), dtype=object)
    values[:] = recordings
    return values, np.asarray(labels), records, 0


class WindowComparisonTests(unittest.TestCase):
    @staticmethod
    def completed_fixture(directory, *, name="kfall_example_20261027_120000", test_accuracy=.75):
        import torch
        from train_kfall import get_args

        args = build_parser().parse_args(["--mode", "train", "--epochs", "3", "--workers", "0",
                                          "--out-dir", str(directory / "runs"),
                                          "--cache-dir", str(directory / "cache"), "--reuse-completed"])
        experiment = build_experiment_specs()[0]
        expected = vars(get_args(training_command(args, experiment)[2:]))
        run = args.out_dir / experiment.cache_name / name
        (run / "metrics").mkdir(parents=True)
        (run / "checkpoints").mkdir()
        (run / "config.json").write_text(json.dumps({**expected, "run": name, "requested_run": experiment.run_name}))
        (run / "dataset_metadata.json").write_text("{}")
        (run / "metrics/history.json").write_text(json.dumps([
            {"epoch": epoch, "bad_epochs": 0} for epoch in (1, 2, 3)
        ]))
        (run / "metrics/test_metrics.json").write_text(json.dumps({
            "best_epoch": 2, "best_val_loss": .3, "acc": test_accuracy,
        }))
        for key in ("y_true", "y_pred"):
            np.save(run / "metrics" / f"{key}.npy", np.arange(4))
        torch.save({"config": expected, "model_state_dict": {"head1.weight": torch.ones(4, 2)},
                    "epoch": 2, "best_val_loss": .3}, run / "checkpoints/best_model.pt")
        return args, experiment, expected, run

    def test_default_durations_scale_encoder_steps_and_share_context(self):
        experiments = build_experiment_specs()
        self.assertEqual([experiment.raw_win for experiment in experiments], [205, 330, 410])
        self.assertEqual([experiment.out_win for experiment in experiments], [2048, 3297, 4096])
        self.assertEqual([experiment.endpoint_context for experiment in experiments], [410] * 3)
        self.assertEqual([experiment.stride for experiment in experiments], [102] * 3)
        self.assertEqual(len({experiment.cache_name for experiment in experiments}), 3)
        self.assertTrue(all(experiment.cache_name != "default" for experiment in experiments))
        self.assertTrue(all(experiment.post_impact_seconds == 1.2 for experiment in experiments))

    def test_fixed_event_length_and_natural_endpoints_are_explicit(self):
        experiments = build_experiment_specs(event_length_policy="fixed", endpoint_policy="natural")
        self.assertEqual([experiment.out_win for experiment in experiments], [2048] * 3)
        self.assertTrue(all(experiment.endpoint_context is None for experiment in experiments))

    def test_invalid_durations_and_duplicate_sample_counts_fail(self):
        for durations in ([], [0], [float("nan")], [2.055], [2.05, 2.05]):
            with self.subTest(durations=durations), self.assertRaises(ValueError):
                build_experiment_specs(durations)
        for settings in ({"stride": 0}, {"event_length_policy": "other"},
                         {"endpoint_policy": "other"}, {"post_impact_seconds": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                build_experiment_specs(**settings)

    def test_training_command_preserves_partition_and_endpoint_settings(self):
        args = build_parser().parse_args(["--mode", "train", "--seed", "13", "--split", ".6", ".2", ".2", "--device", "cpu"])
        experiment = build_experiment_specs()[1]
        command = training_command(args, experiment)
        def argument(name):
            return command[command.index(name) + 1]
        self.assertEqual(argument("--raw-win"), "330")
        self.assertEqual(argument("--out-win"), "3297")
        self.assertEqual(argument("--endpoint-context"), "410")
        self.assertEqual(argument("--post-impact-seconds"), "1.2")
        self.assertEqual(argument("--stride"), "102")
        self.assertEqual(argument("--seed"), "13")
        self.assertEqual(argument("--device"), "cpu")
        self.assertEqual(command[command.index("--split") + 1:command.index("--split") + 4], ["0.6", "0.2", "0.2"])
        self.assertNotIn("--preprocess-only", command)
        args.mode = "preprocess"
        self.assertIn("--preprocess-only", training_command(args, experiment))

    def test_inspection_uses_same_subjects_and_endpoint_label_pairs(self):
        args = build_parser().parse_args(["--split", "1", "1", "1"])
        with patch("preprocess.preprocess_kfall.load_kfall_3class", return_value=raw_fixture()):
            plan = inspect_dataset(args, build_experiment_specs())
        self.assertEqual(len(plan["classes"]), 4)
        for name in ("train", "val", "test"):
            selected = [experiment["selected"][name] for experiment in plan["experiments"]]
            self.assertEqual(len({tuple(value["subjects"]) for value in selected}), 1)
            self.assertEqual(len({value["endpoint_label_sha256"] for value in selected}), 1)
            self.assertTrue(all(value["class_counts"]["0"] > 0 for value in selected))
            self.assertTrue(all(value["class_counts"]["3"] > 0 for value in selected))
        short, _, long = plan["experiments"]
        self.assertGreater(short["natural"]["test"]["windows"], long["natural"]["test"]["windows"])

    def test_inspection_reports_falls_without_long_enough_preimpact_context(self):
        fixture = raw_fixture()
        for record in fixture[2]:
            if record["label_id"] == 0:
                record.update(fall_onset_frame=150, fall_impact_frame=300,
                              onset_idx=149, impact_idx=299, post_impact_end_idx=419)
        args = build_parser().parse_args(["--split", "1", "1", "1"])
        with patch("preprocess.preprocess_kfall.load_kfall_3class", return_value=fixture):
            plan = inspect_dataset(args, build_experiment_specs())
        baseline = plan["experiments"][0]
        self.assertGreater(baseline["natural"]["test"]["class_counts"]["0"], 0)
        self.assertEqual(baseline["selected"]["test"]["class_counts"]["0"], 0)
        self.assertEqual(baseline["selected"]["test"]["fall_recordings_without_preimpact_window"], 1)

    def test_result_validation_rejects_changed_samples_or_subjects(self):
        first = {"split_subjects": {"test": ["SA07"]}, "source_signature": {"sha256": "source"},
                 "endpoint_signatures": {"test": "manifest"}}
        for key, value in (("split_subjects", {"test": ["SA08"]}),
                           ("source_signature", {"sha256": "new"}),
                           ("endpoint_signatures", {"test": "changed"})):
            changed = deepcopy(first)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                _verify_results([first, changed], matched=True)
        changed = {**first, "endpoint_signatures": {"test": "changed"}}
        _verify_results([first, changed], matched=False)

    def test_inspect_cli_exports_plan_and_coverage_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch("preprocess.preprocess_kfall.load_kfall_3class", return_value=raw_fixture()), patch(
                "compare_kfall_windows.subprocess.run", side_effect=AssertionError("inspect launched training")
            ):
                main(["--results-dir", str(directory), "--split", "1", "1", "1"])
            plan = json.loads((directory / "window_plan.json").read_text())
            self.assertEqual(len(plan["experiments"]), 3)
            self.assertEqual(len(plan["commands"]), 3)
            self.assertEqual((directory / "window_counts.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual((directory / "natural_window_counts.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue((directory / "window_coverage.csv").is_file())

    def test_training_result_collects_created_run_and_four_class_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = build_parser().parse_args(["--mode", "train", "--out-dir", str(directory / "runs"),
                                              "--cache-dir", str(directory / "cache")])
            experiment = build_experiment_specs()[0]

            def train(command, **kwargs):
                cache = args.cache_dir / f"kfall_event_loader_{experiment.cache_name}"
                for split in ("train", "val", "test"):
                    target = cache / split
                    target.mkdir(parents=True)
                    for key, value in (("file_idx", [0, 0, 1, 2]), ("end_frame", [600, 700, 600, 600]),
                                       ("y_h1", [0, 3, 1, 2])):
                        np.save(target / f"{key}.npy", np.asarray(value))
                (cache / "metadata.json").write_text(json.dumps({
                    "split_subjects": {"train": ["SA06"], "val": ["SA07"], "test": ["SA08"]},
                    "source_signature": {"sha256": "source"}, "split_summary": {},
                }))
                run = args.out_dir / experiment.cache_name / "created_run"
                metrics = run / "metrics"
                metrics.mkdir(parents=True)
                (run / "config.json").write_text("{}")
                (metrics / "test_metrics.json").write_text(json.dumps({
                    "loss": .4, "acc": 1., "best_epoch": 1, "best_val_loss": .5, "total_time_sec": 2,
                }))
                (metrics / "history.json").write_text(json.dumps([
                    {"epoch": 1, "val": {"acc": .75, "loss": .5}},
                ]))
                for key in ("y_true", "y_pred"):
                    np.save(metrics / f"{key}.npy", np.arange(4))

            with patch("compare_kfall_windows.subprocess.run", side_effect=train):
                result = _run_experiment(args, experiment)
            self.assertEqual(result["test_accuracy"], 1.)
            self.assertEqual(result["test_macro_f1"], 1.)
            self.assertEqual(result["test_postimpact_recall"], 1.)
            self.assertEqual(result["test_per_class"]["Postimpact"]["support"], 1.)
            self.assertEqual(result["best_epoch"], 1)

    def test_reuse_requires_finished_matching_checkpoint_and_cache_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, experiment, expected, run = self.completed_fixture(Path(temporary))
            arrays = {"y_h1": np.arange(4)}
            with patch("evaluation.kfall_data.load_evaluation_data", return_value=(arrays, {}, {}, args.cache_dir)):
                self.assertTrue(_completed_run_matches(args, experiment, run, expected))
                changed = {**expected, "epochs": 100}
                self.assertFalse(_completed_run_matches(args, experiment, run, changed))
                changed = {**expected, "post_impact_seconds": 0., "n_classes": 3}
                self.assertFalse(_completed_run_matches(args, experiment, run, changed))
                (run / "metrics/test_metrics.json").unlink()
                self.assertFalse(_completed_run_matches(args, experiment, run, expected))
            args, experiment, expected, run = self.completed_fixture(Path(temporary), name="kfall_example_20261027_130000")
            with patch("evaluation.kfall_data.load_evaluation_data", side_effect=ValueError("cache mismatch")):
                self.assertFalse(_completed_run_matches(args, experiment, run, expected))

    def test_reuse_selects_latest_completed_timestamp_not_best_test_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args, experiment, _, older = self.completed_fixture(directory, name="kfall_example_20261027_120000", test_accuracy=.99)
            _, _, _, newest = self.completed_fixture(directory, name="kfall_example_20261027_130000", test_accuracy=.5)
            _, _, _, incomplete = self.completed_fixture(directory, name="kfall_example_20261027_140000", test_accuracy=1.)
            (incomplete / "metrics/test_metrics.json").unlink()
            with patch("evaluation.kfall_data.load_evaluation_data", return_value=({"y_h1": np.arange(4)}, {}, {}, args.cache_dir)):
                self.assertEqual(_find_completed_run(args, experiment), newest)
                with patch("compare_kfall_windows.subprocess.run", side_effect=AssertionError("reused run trained again")), patch(
                    "compare_kfall_windows._collect_run", return_value={"run_dir": str(newest), "reused_completed_run": True}
                ):
                    result = _run_experiment(args, experiment)
                self.assertTrue(result["reused_completed_run"])
                self.assertEqual(result["run_dir"], str(newest))
            self.assertNotEqual(older, newest)

    def test_parallel_runs_collect_complete_results_in_duration_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = build_parser().parse_args(["--split", "1", "1", "1"])
            with patch("preprocess.preprocess_kfall.load_kfall_3class", return_value=raw_fixture()):
                plan = inspect_dataset(args, build_experiment_specs())
            rows = {row["raw_win"]: row for row in plan["experiments"]}
            lock = threading.Lock()
            running = maximum = 0

            def prepare(args, experiment):
                nonlocal running, maximum
                with lock:
                    running += 1
                    maximum = max(maximum, running)
                time.sleep(.03)
                row = rows[experiment.raw_win]
                with lock:
                    running -= 1
                return {**asdict(experiment), "cache_dir": str(directory / experiment.cache_name),
                        "split_subjects": {name: row["selected"][name]["subjects"] for name in ("train", "val", "test")},
                        "source_signature": {"sha256": "same"},
                        "endpoint_signatures": {name: row["selected"][name]["endpoint_label_sha256"] for name in ("train", "val", "test")}}

            with patch("compare_kfall_windows.inspect_dataset", return_value=plan), patch(
                "compare_kfall_windows._run_experiment", side_effect=prepare
            ):
                main(["--mode", "preprocess", "--parallel-runs", "2", "--results-dir", str(directory)])
            results = json.loads((directory / "comparison.json").read_text())
            self.assertEqual(maximum, 2)
            self.assertEqual([row["raw_win"] for row in results], [205, 330, 410])

    def test_parallel_count_must_be_positive(self):
        with patch("compare_kfall_windows.inspect_dataset", side_effect=AssertionError("invalid arguments inspected data")):
            with self.assertRaises(SystemExit):
                main(["--parallel-runs", "0"])


if __name__ == "__main__":
    unittest.main()
