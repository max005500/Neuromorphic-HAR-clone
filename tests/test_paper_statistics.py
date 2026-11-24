"""Focused reporting regressions; fixtures stay under the project directory."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from evaluation.paper_statistics import (
    load_results,
    pareto_front,
    select_scn,
    summarize,
    system_energy,
    write_report,
)


class PaperStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / f".paper-statistics-tests-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)

    def csv(self, rows, name="results.csv"):
        path = self.root / name
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def runs(self, n=5, **overrides):
        return [
            dict(
                experiment="ablation", dataset="KFall", protocol="subject-split-v1",
                split="test", variant="B1", seed=seed, macro_f1=0.5 + seed / 20,
                accuracy=0.6 + seed / 20, params=120, energy_network_j=0.01,
            ) | overrides for seed in range(n)
        ]

    def candidates(self):
        rows = []
        for variant, score, energy in (("wide", 0.9, 0.03), ("small", 0.8, 0.01)):
            for split in ("val", "test"):
                for row in self.runs(variant=variant, split=split):
                    row.update(
                        macro_f1=score if split == "val" else 1 - score,
                        energy_network_j=energy if split == "val" else 0.1 - energy,
                        run_id=f"{variant}-{split}-{row['seed']}",
                    )
                    rows.append(row)
        return rows

    def test_sample_statistics_and_source_trace(self):
        path = self.csv(self.runs())
        rows, fields, metrics, manifest = load_results([path])
        summary = summarize(rows, fields, metrics)
        self.assertEqual(len(summary), 1)
        record = summary[0]
        self.assertEqual(record["status"], "PUBLISHABLE")
        self.assertEqual(record["n_seeds"], 5)
        self.assertAlmostEqual(record["macro_f1_mean"], 0.6)
        self.assertAlmostEqual(record["macro_f1_std"], statistics.stdev([0.5, 0.55, 0.6, 0.65, 0.7]))
        self.assertIn("+/-", record["macro_f1_mean_pm_std"])
        self.assertEqual(manifest[0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(record["origins"][0]["source_file"], str(path.resolve()))
        self.assertEqual(record["origins"][0]["source_row"], 2)

    def test_four_seeds_rejected_descriptive_singleton_sd_is_null(self):
        rows, fields, metrics, _ = load_results([self.csv(self.runs(4))])
        with self.assertRaisesRegex(ValueError, ">=5 distinct seeds"):
            summarize(rows, fields, metrics)
        self.assertEqual(summarize(rows, fields, metrics, descriptive=True)[0]["status"], "NON_PUBLISHABLE")
        rows, fields, metrics, _ = load_results([self.csv(self.runs(1), "singleton.csv")])
        record = summarize(rows, fields, metrics, descriptive=True)[0]
        self.assertIsNone(record["macro_f1_std"])
        self.assertIsNone(record["macro_f1_mean_pm_std"])

    def test_descriptive_is_never_labelled_publishable_even_with_five_seeds(self):
        rows, fields, metrics, _ = load_results([self.csv(self.runs())])
        self.assertEqual(summarize(rows, fields, metrics, descriptive=True)[0]["status"], "NON_PUBLISHABLE")

    def test_missing_values_are_not_zero_and_partial_metrics_fail(self):
        raw = self.runs()
        raw[0]["accuracy"] = ""
        for row in raw:
            row["energy_network_j"] = ""
        rows, fields, metrics, _ = load_results([self.csv(raw)])
        with self.assertRaisesRegex(ValueError, "accuracy: 4/5"):
            summarize(rows, fields, metrics)
        record = summarize(rows, fields, metrics, descriptive=True)[0]
        self.assertEqual(record["accuracy_n"], 4)
        self.assertAlmostEqual(record["accuracy_mean"], 0.725)
        self.assertEqual(record["energy_network_j_n"], 0)
        self.assertIsNone(record["energy_network_j_mean"])
        self.assertIsNone(record["energy_network_j_std"])
        for row in raw:
            row["macro_f1"] = ""
        rows, fields, metrics, _ = load_results([self.csv(raw, "no-primary.csv")])
        with self.assertRaisesRegex(ValueError, "macro_f1: 0/5"):
            summarize(rows, fields, metrics)

    def test_duplicate_seed_is_rejected_even_with_different_run_id_or_padding(self):
        raw = self.runs()
        raw.append(raw[1] | {"seed": "01", "run_id": "another-retry"})
        with self.assertRaisesRegex(ValueError, "Duplicate seed 1"):
            load_results([self.csv(raw)])
        path = self.csv(self.runs(), "same.csv")
        with self.assertRaisesRegex(ValueError, "Duplicate input file"):
            load_results([path, path])
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            load_results([path, self.csv(self.runs(), "copy.csv")])

    def test_protocol_split_and_extra_metadata_are_never_pooled(self):
        raw = self.runs(protocol="A", window_ms="200")
        raw += self.runs(protocol="B", window_ms="200")
        raw += self.runs(protocol="A", window_ms="400")
        raw += self.runs(protocol="A", window_ms="200", split="val")
        rows, fields, metrics, _ = load_results([self.csv(raw)])
        self.assertEqual(len(summarize(rows, fields, metrics)), 4)
        with self.assertRaisesRegex(ValueError, "protocols may not be pooled"):
            summarize(rows, tuple(field for field in fields if field != "protocol"), metrics)
        raw[0]["window_ms"] = ""
        with self.assertRaisesRegex(ValueError, "empty grouping field window_ms"):
            load_results([self.csv(raw, "missing-context.csv")])

    def test_invalid_csv_metrics_and_schema_fail_explicitly(self):
        for metric, value in (("macro_f1", "nan"), ("accuracy", "inf"), ("params", "-1"),
                              ("macro_f1", "87"), ("seed", "1.0")):
            with self.subTest(metric=metric, value=value):
                raw = self.runs()
                raw[0][metric] = value
                with self.assertRaises(ValueError):
                    load_results([self.csv(raw)])
        raw = self.runs()
        del raw[0]["protocol"]
        with self.assertRaises(ValueError):
            load_results([self.csv(raw[:1])])
        path = self.root / "malformed.csv"
        path.write_text(
            "experiment,dataset,protocol,split,variant,seed,macro_f1\nx,d,p,test,v,0,0.5,extra\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "malformed CSV"):
            load_results([path])

    def test_pareto_directions_ties_and_missing_for_both_scores(self):
        for score in ("macro_f1", "accuracy"):
            points = [{score: y, "energy_network_j": x} for x, y in (
                (1, 0.8), (2, 0.7), (2, 0.9), (1, 0.8), (0.5, 0.8), (None, 1),
            )]
            self.assertEqual(
                pareto_front(points, score=score),
                [False, False, True, False, True, None],
            )
        self.assertEqual(pareto_front([
            {"accuracy": 0.9, "energy_total_j": 1},
            {"accuracy": 0.9, "energy_total_j": 1},
        ], score="accuracy", energy="energy_total_j"), [True, True])

    def test_selection_only_uses_validation_and_preserves_audit_trail(self):
        rows, fields, _, _ = load_results([self.csv(self.candidates())])
        selected = select_scn(rows, fields, variants=["wide", "small"])
        self.assertEqual(len(selected), 10)
        for row in selected:
            self.assertEqual(row["selected_variant"], "wide" if row["variant"] == "SCNacc" else "small")
            self.assertEqual(row["split"], "test")
            self.assertIn("-val-", row["validation_run_id"])
            self.assertIn("-test-", row["run_id"])
            self.assertIn("validation_source_row", row)
        baseline = [(r["variant"], r["seed"], r["selected_variant"]) for r in selected]
        for row in rows:
            if row["split"] == "test":
                row["macro_f1"] = 1.0 if row["variant"] == "small" else 0
                row["energy_network_j"] = 0 if row["variant"] == "wide" else 100
        altered = select_scn(rows, fields, variants=["wide", "small"])
        self.assertEqual(baseline, [(r["variant"], r["seed"], r["selected_variant"]) for r in altered])
        self.assertNotIn("SCNsel", {row["variant"] for row in selected})

    def test_selection_ties_deterministic_and_missing_candidates_never_silently_drop(self):
        raw = self.candidates()
        for row in raw:
            if row["split"] == "val":
                row["macro_f1"], row["energy_network_j"] = 0.8, 0.01
        random.Random(7).shuffle(raw)
        rows, fields, _, _ = load_results([self.csv(raw)])
        selected = select_scn(rows, fields, variants=["wide", "small"])
        self.assertEqual({row["selected_variant"] for row in selected}, {"small"})
        for drop_pair in (False, True):
            incomplete = [
                row for row in rows if not (
                    row["variant"] == "small" and row["seed"] == 0
                    and (drop_pair or row["split"] == "test")
                )
            ]
            with self.assertRaisesRegex(ValueError, "Missing val/test|Incomplete seed"):
                select_scn(incomplete, fields, variants=["wide", "small"])
        with self.assertRaisesRegex(ValueError, "Unknown SCN"):
            select_scn(rows, fields, variants=["not-present"])
        with self.assertRaisesRegex(ValueError, "SCNsel"):
            select_scn(rows, fields, variants=["SCNsel"])

    def test_selected_configuration_can_vary_by_seed_without_splitting_summary(self):
        raw = self.candidates()
        for row in raw:
            row["model_id"] = row["variant"]
            row["variant"] = "SCN"
            if row["split"] == "val" and row["seed"] % 2:
                row["macro_f1"] = 1 - row["macro_f1"]
        rows, fields, metrics, _ = load_results([self.csv(raw)])
        selected = select_scn(rows, fields, variants=["SCN"])
        acc = [row for row in selected if row["variant"] == "SCNacc"]
        self.assertEqual({row["selected_model_id"] for row in acc}, {"wide", "small"})
        summary = summarize(selected, fields, metrics)
        self.assertEqual(len(summary), 2)
        self.assertTrue(all(row["n_seeds"] == 5 for row in summary))

    def test_system_energy_units_topology_sources_and_b1_raw_vs_analog(self):
        common = dict(
            network_source="run budget", duration_source="window definition",
            frontend_source="explicit topology assumption",
        )
        raw = system_energy(0.1, 2, input_mode="raw", frontend="adc",
                            adc_power_w=0.03, adc_source="user-supplied ADC measurement", **common)
        analog = system_energy(0.1, 2, input_mode="analog", frontend="none", **common)
        events = system_energy(0.1, 2, input_mode="events", frontend="adc+sigma_delta",
                               adc_power_w=0.03, adc_source="ADC measurement",
                               sigma_delta_power_w=0.02, sigma_delta_source="SD measurement", **common)
        self.assertAlmostEqual(raw["energy_frontend_j"], 0.06)
        self.assertAlmostEqual(raw["energy_total_j"], 0.16)
        self.assertAlmostEqual(analog["energy_total_j"], 0.1)
        self.assertAlmostEqual(events["energy_total_j"], 0.2)
        self.assertAlmostEqual(events["components"]["energy_sigma_delta_j"], 0.04)
        self.assertEqual(events["sources"]["adc"], "ADC measurement")
        for override in (
            {"adc_power_w": None}, {"adc_source": ""}, {"adc_power_w": -0.01},
            {"duration_s": 0}, {"duration_s": math.inf},
            {"input_mode": "raw", "frontend": "none", "adc_power_w": None, "adc_source": None},
            {"frontend": "sigma_delta", "input_mode": "events",
             "adc_power_w": None, "adc_source": None, "sigma_delta_power_w": 0.02},
        ):
            arguments = dict(
                network_energy_j=0.1, duration_s=2, input_mode="raw", frontend="adc",
                adc_power_w=0.03, adc_source="measurement", **common,
            ) | override
            with self.subTest(override=override), self.assertRaises(ValueError):
                system_energy(**arguments)

    def test_csv_energy_total_only_with_complete_components(self):
        raw = self.runs(
            input_mode="raw", frontend="adc", duration_s=2, duration_source="window spec",
            energy_network_source="network report", frontend_source="raw B1 needs ADC",
            adc_power_w=0.03, adc_source="user measurement",
        )
        raw += self.runs(
            input_mode="analog", frontend="none", duration_s=2, duration_source="window spec",
            energy_network_source="network report", frontend_source="analog B1 bypass assumption",
        )
        rows, fields, metrics, _ = load_results([self.csv(raw)])
        summary = summarize(rows, fields, metrics)
        self.assertEqual(len(summary), 2)
        self.assertEqual({row["input_mode"] for row in summary}, {"raw", "analog"})
        self.assertAlmostEqual(summary[0]["energy_total_j_mean"], 0.07)
        self.assertAlmostEqual(summary[1]["energy_total_j_mean"], 0.01)
        raw[0]["adc_source"] = ""
        with self.assertRaisesRegex(ValueError, "adc_source"):
            load_results([self.csv(raw, "incomplete-energy.csv")])
        rows, _, metrics, _ = load_results([self.csv(self.runs(), "compute-only.csv")])
        self.assertNotIn("energy_total_j", metrics)
        self.assertNotIn("energy_total_j", rows[0])

    def test_end_to_end_report_json_csv_figures_and_selection(self):
        path = self.csv(self.candidates())
        out = self.root / "report"
        report = write_report([path], out, scn_variants=["wide", "small"])
        self.assertEqual(report["status"], "PUBLISHABLE")
        self.assertEqual(len(report["summary"]), 6)
        saved = json.loads((out / "report.json").read_text())
        self.assertEqual(saved["sources"][0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(saved["statistics"]["std_ddof"], 1)
        self.assertTrue(saved["plots"])
        self.assertTrue(all(row["purpose"] == "DESCRIPTIVE_NOT_MODEL_SELECTION" for row in saved["pareto"]))
        for plot in saved["plots"]:
            self.assertEqual((out / plot["file"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        for name, count in (("summary.csv", 6), ("selected_results.csv", 10), ("results.csv", 20)):
            with (out / name).open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), count)

    def test_cli_fails_incomplete_without_writing_then_supports_explicit_descriptive(self):
        path = self.csv(self.runs(1))
        out = self.root / "cli-report"
        command = [
            sys.executable, str(Path(__file__).resolve().parents[1] / "report_paper_results.py"),
            str(path), "--output-dir", str(out),
        ]
        failed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(failed.returncode, 2)
        self.assertIn(">=5 distinct seeds", failed.stderr)
        self.assertFalse(out.exists())
        success = subprocess.run(command + ["--descriptive"], capture_output=True, text=True, check=False)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertIn("NON_PUBLISHABLE", success.stdout)
        self.assertEqual(json.loads((out / "report.json").read_text())["status"], "NON_PUBLISHABLE")

    def test_absent_energy_never_generates_fake_pareto_or_total(self):
        raw = self.runs()
        for row in raw:
            del row["energy_network_j"]
        path = self.csv(raw)
        report = write_report([path], self.root / "no-energy")
        self.assertEqual(report["plots"], [])
        self.assertTrue(all(row["energy_mean_j"] is None and row["pareto"] is None for row in report["pareto"]))
        with self.assertRaisesRegex(ValueError, "no explicit complete"):
            write_report([path], self.root / "no-total", energy="energy_total_j")

    def test_runner_schema_normalized_pe_is_not_joules_or_a_grouping_column(self):
        raw = self.runs()
        for row in raw:
            row["parameters"] = row.pop("params")
            row["energy_pe_norm"] = row.pop("energy_network_j") * 100
            row["source"] = f"runs/seed-{row['seed']}/summary.json"
        path = self.csv(raw)
        report = write_report([path], self.root / "normalized", energy="energy_pe_norm")
        self.assertEqual(len(report["summary"]), 1)
        self.assertEqual(report["summary"][0]["parameters_mean"], 120)
        self.assertNotIn("source", report["group_fields"])
        self.assertNotIn("parameters", report["group_fields"])
        self.assertEqual(report["pareto_axes"]["energy_unit"], "normalized_PE")
        self.assertEqual(report["pareto"][0]["energy_mean"], 1)
        self.assertIsNone(report["pareto"][0]["energy_mean_j"])
        self.assertNotIn("energy_total_j", report["metrics"])
        self.assertEqual(report["summary"][0]["origins"][0]["source"], raw[0]["source"])
        self.assertTrue(report["plots"])


if __name__ == "__main__":
    unittest.main()
