#!/usr/bin/env python3
"""Compare KFall window durations and the auxiliary offset branch."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from loaders.kfall_event_loader import KFallEventLoaderConfig

PROJECT_ROOT = Path(__file__).resolve().parent
WINDOW_SECONDS = (1.20, 2.06, 3.20)
SAMPLE_RATE_HZ = 100
EVENT_RATE_HZ = 1000
CSV_FIELDS = (
    "window_seconds", "raw_samples", "event_steps", "stride", "use_offset",
    "seed", "best_epoch", "train_accuracy_at_best", "validation_accuracy_at_best",
    "test_accuracy", "test_balanced_accuracy", "test_macro_f1", "test_loss",
    "accuracy_delta_vs_with_offset", "macro_f1_delta_vs_with_offset",
    "training_time_sec", "n_test", "run_dir",
)


def window_conditions() -> list[dict[str, Any]]:
    """Keep the event time base near 1 kHz and overlap near 50%."""
    conditions = []
    for seconds in WINDOW_SECONDS:
        raw_samples = round(seconds * SAMPLE_RATE_HZ)
        conditions.append({
            "window_seconds": seconds,
            "raw_samples": raw_samples,
            "event_steps": round(raw_samples * EVENT_RATE_HZ / SAMPLE_RATE_HZ),
            "stride": round(raw_samples / 2),
        })
    return conditions


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    durations = [item["window_seconds"] for item in window_conditions()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "test_accuracy", "Test accuracy"),
        (axes[1], "test_macro_f1", "Test macro-F1"),
    ):
        for use_offset, label, marker in ((True, "With offset", "o"), (False, "Without offset", "s")):
            values = [
                next(row[metric] for row in rows
                     if row["window_seconds"] == seconds and row["use_offset"] is use_offset)
                for seconds in durations
            ]
            axis.plot(durations, values, marker=marker, label=label)
        axis.set(title=title, xlabel="Window duration (s)", ylabel="Score", ylim=(0, 1))
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _evaluate_run(run_dir: Path, condition: dict[str, Any], use_offset: bool, seed: int) -> dict[str, Any]:
    metrics_dir = run_dir / "metrics"
    metrics = json.loads((metrics_dir / "test_metrics.json").read_text(encoding="utf-8"))
    history = json.loads((metrics_dir / "history.json").read_text(encoding="utf-8"))
    y_true = np.load(metrics_dir / "y_true.npy")
    y_pred = np.load(metrics_dir / "y_pred.npy")
    best_epoch = int(metrics["best_epoch"])
    best_row = next((row for row in history if int(row["epoch"]) == best_epoch), None)
    if best_row is None:
        raise ValueError(f"{run_dir}: best epoch is missing from metrics/history.json.")
    if len(y_true) != int(metrics["n"]) or y_pred.shape != y_true.shape:
        raise ValueError(f"{run_dir}: test predictions do not match test_metrics.json.")

    return {
        **condition,
        "use_offset": use_offset,
        "seed": seed,
        "best_epoch": best_epoch,
        "train_accuracy_at_best": float(best_row["train"]["acc"]),
        "validation_accuracy_at_best": float(best_row["val"]["acc"]),
        "test_accuracy": float(metrics["acc"]),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_loss": float(metrics["loss"]),
        "training_time_sec": float(metrics["total_time_sec"]),
        "n_test": int(metrics["n"]),
        "run_dir": str(run_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/KFall")
    parser.add_argument("--label-dir", default=None)
    parser.add_argument("--annotation-file", default=None, help="Override train_kfall.py's default manual annotations.")
    parser.add_argument("--no-manual-labels", action="store_true")
    parser.add_argument("--post-impact-seconds", type=float, default=1.2)
    parser.add_argument("--cache-dir", default="data/kfall_cache")
    parser.add_argument("--output-dir", default="results/kfall_window_experiments")
    parser.add_argument("--split", type=float, nargs=3, default=KFallEventLoaderConfig().split,
                        help="Subject fractions for train/validation/test (default 60/30/10).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--theta", type=float, nargs=6, default=(0.1, 0.1, 0.1, 10.0, 10.0, 10.0))
    parser.add_argument("--dead-zone", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative.")
    if args.seed < 0 or args.patience < 1:
        parser.error("seed must be nonnegative and patience must be positive.")
    if len(args.split) != 3 or any(not np.isfinite(value) or value <= 0 for value in args.split) or abs(sum(args.split) - 1) > 1e-6:
        parser.error("split must contain three positive fractions that sum to 1.")

    dataset_root = _project_path(args.dataset_root)
    label_dir = None if args.label_dir is None else _project_path(args.label_dir)
    output_dir = _project_path(args.output_dir)
    runs_dir = output_dir / "runs"
    cache_dir = _project_path(args.cache_dir)
    train_script = PROJECT_ROOT / "train_kfall.py"
    rows: list[dict[str, Any]] = []
    subject_manifest: dict[str, list[str]] | None = None

    for condition in window_conditions():
        raw = condition["raw_samples"]
        window_tag = f"w{raw}"
        for use_offset in (True, False):
            mode = "with_offset" if use_offset else "no_offset"
            run_name = f"kfall_{window_tag}_{mode}"
            cache_name = f"kfall_{window_tag}"
            run_parent = runs_dir / run_name
            command = [
                sys.executable, str(train_script),
                "--dataset-root", str(dataset_root),
                "--cache-dir", str(cache_dir),
                "--cache-name", cache_name,
                "--out-dir", str(run_parent),
                "--run", run_name,
                "--split", *(str(value) for value in args.split),
                "--raw-win", str(raw),
                "--out-win", str(condition["event_steps"]),
                "--stride", str(condition["stride"]),
                "--theta", *(str(value) for value in args.theta),
                "--dead-zone", str(args.dead_zone),
                "--seed", str(args.seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--workers", str(args.workers),
                "--gpu", str(args.gpu),
                "--patience", str(args.patience),
                "--post-impact-seconds", str(args.post_impact_seconds),
                "--quiet-loader",
            ]
            if label_dir is not None:
                command.extend(("--label-dir", str(label_dir)))
            if args.annotation_file:
                command.extend(("--annotation-file", str(_project_path(args.annotation_file))))
            if args.no_manual_labels:
                command.append("--no-manual-labels")
            if not use_offset:
                command.append("--no-offset")

            print(
                f"[experiment] {condition['window_seconds']:.2f}s, "
                f"{'with' if use_offset else 'without'} offset",
                flush=True,
            )
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            run_dirs = sorted(run_parent.glob(f"{run_name}_*"), key=lambda path: path.stat().st_mtime_ns)
            if not run_dirs:
                raise FileNotFoundError(f"Training completed without creating a run under {run_parent}.")
            run_dir = run_dirs[-1]

            metadata = json.loads((run_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
            current_manifest = metadata["split_subjects"]
            if subject_manifest is None:
                subject_manifest = current_manifest
            elif current_manifest != subject_manifest:
                raise ValueError("Subject partitions differ between experiment conditions.")

            rows.append(_evaluate_run(run_dir, condition, use_offset, args.seed))
            _write_summary(output_dir / "summary.csv", rows)

    for row in rows:
        matching = next(
            candidate for candidate in rows
            if candidate["window_seconds"] == row["window_seconds"]
            and candidate["use_offset"] is not row["use_offset"]
        )
        row["accuracy_delta_vs_with_offset"] = (
            row["test_accuracy"] - matching["test_accuracy"] if not row["use_offset"] else 0.0
        )
        row["macro_f1_delta_vs_with_offset"] = (
            row["test_macro_f1"] - matching["test_macro_f1"] if not row["use_offset"] else 0.0
        )
    _write_summary(output_dir / "summary.csv", rows)
    _plot_summary(output_dir / "comparison.png", rows)
    print(f"[done] summary={output_dir / 'summary.csv'}")
    print(f"[done] plot={output_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
