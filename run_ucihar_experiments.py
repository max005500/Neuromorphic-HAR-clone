#!/usr/bin/env python3
"""Train six-class UCI-HAR models for three windows, with/without offsets."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from run_kfall_experiments import _evaluate_run, _project_path
from train_ucihar import DEFAULT_THETA_UCIHAR

PROJECT_ROOT = Path(__file__).resolve().parent


def window_conditions():
    return [
        {"window_seconds": seconds, "raw_samples": raw, "event_steps": raw * 10,
         "stride": raw // 2}
        for seconds, raw in ((1.20, 120), (2.06, 206), (2.56, 256))
    ]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-dir", default="UCI HAR Dataset")
    parser.add_argument("--cache-dir", default="data/ucihar_window_cache")
    parser.add_argument("--output-dir", default="results/ucihar_window_experiments")
    parser.add_argument("--split", nargs=3, type=float, default=(0.6, 0.3, 0.1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--theta", nargs=6, type=float, default=DEFAULT_THETA_UCIHAR)
    parser.add_argument("--dead-zone", type=float, default=0.5)
    return parser


def save_summary(output, rows):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


def plot_summary(output, rows):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, key, title in zip(figure.subplots(1, 3),
                                ("test_accuracy", "test_macro_f1", "test_macro_recall"),
                                ("Accuracy", "Macro F1", "Macro recall")):
        for enabled, label in ((True, "With offset"), (False, "Without offset")):
            selected = [row for row in rows if row["use_offset"] == enabled]
            axis.plot([row["window_seconds"] for row in selected],
                      [row[key] for row in selected], marker="o", label=label)
        axis.set(title=title, xlabel="Window (s)", ylim=(0, 1))
        axis.grid(alpha=0.3)
    figure.axes[0].legend()
    figure.savefig(output / "comparison.png", dpi=160)
    figure.clear()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.patience < 1 or args.seed < 0:
        parser.error("epochs/batch-size/patience must be positive; workers/seed nonnegative.")
    if any(not np.isfinite(x) or x <= 0 for x in args.split) or not np.isclose(sum(args.split), 1):
        parser.error("split must contain three positive fractions summing to one.")
    if any(not np.isfinite(x) or x <= 0 for x in args.theta) or not np.isfinite(args.dead_zone) or args.dead_zone < 0:
        parser.error("theta must be finite and positive; dead-zone finite and nonnegative.")
    output = _project_path(args.output_dir)
    rows, manifest = [], None
    for condition in window_conditions():
        raw = condition["raw_samples"]
        for enabled in (True, False):
            name = f"ucihar_w{raw}_{'with_offset' if enabled else 'no_offset'}"
            parent = output / "runs" / name
            command = [
                sys.executable, str(PROJECT_ROOT / "train_ucihar.py"),
                "--data-root", str(_project_path(args.data_root)), "--dataset-dir", args.dataset_dir,
                "--cache-dir", str(_project_path(args.cache_dir)), "--cache-name", f"w{raw}",
                "--case", "subject_split", "--split", *map(str, args.split),
                "--sample-rate-hz", "100", "--raw-win", str(raw),
                "--window-stride", str(condition["stride"]),
                "--out-win", str(condition["event_steps"]),
                "--out-dir", str(parent), "--run", name,
                "--seed", str(args.seed), "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                "--gpu", str(args.gpu), "--patience", str(args.patience),
                "--theta", *map(str, args.theta), "--dead-zone", str(args.dead_zone),
                "--quiet-loader",
            ]
            if not enabled:
                command.append("--no-offset")
            if args.amp:
                command.append("--amp")
            print(f"[experiment] UCI-HAR {condition['window_seconds']:.2f}s | {name}", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            runs = sorted(parent.glob(f"{name}_*"), key=lambda path: path.stat().st_mtime_ns)
            if not runs:
                raise FileNotFoundError(f"No training output under {parent}.")
            run = runs[-1]
            metadata = json.loads((run / "dataset_metadata.json").read_text())
            current = metadata["split_subjects"]
            if manifest is not None and current != manifest:
                raise ValueError("UCI subject partitions changed between conditions.")
            manifest = current
            row = _evaluate_run(run, condition, enabled, args.seed)
            truth = np.load(run / "metrics" / "y_true.npy")
            predicted = np.load(run / "metrics" / "y_pred.npy")
            row["test_balanced_accuracy"] = float(balanced_accuracy_score(truth, predicted))
            row["test_macro_f1"] = float(f1_score(truth, predicted, labels=range(6), average="macro", zero_division=0))
            row["test_macro_recall"] = float(recall_score(truth, predicted, labels=range(6), average="macro", zero_division=0))
            rows.append(row)
            save_summary(output, rows)
    for row in rows:
        baseline = next(item for item in rows if item["window_seconds"] == row["window_seconds"] and item["use_offset"])
        row["accuracy_delta_vs_with_offset"] = row["test_accuracy"] - baseline["test_accuracy"]
        row["macro_f1_delta_vs_with_offset"] = row["test_macro_f1"] - baseline["test_macro_f1"]
    save_summary(output, rows)
    plot_summary(output, rows)
    print(f"[done] {output / 'summary.csv'}")


if __name__ == "__main__":
    main()
