#!/usr/bin/env python3
"""Compare KFall context durations with fixed subjects and explicit endpoint coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from preprocess.kfall_labels import class_names, event_steps_for_samples, window_samples
from preprocess.utils import save_json, subject_split_indices

PROJECT_ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
CLASS_NAMES = class_names(4)


@dataclass(frozen=True)
class WindowExperiment:
    duration_seconds: float
    raw_win: int
    out_win: int
    stride: int
    post_impact_seconds: float
    endpoint_context: int | None
    event_length_policy: str
    cache_name: str
    run_name: str
    resample_hz: float | None = None


def _tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def build_experiment_specs(
    durations: Sequence[float] = (2.05, 3.30, 4.10), *,
    event_length_policy: str = "scaled", fixed_out_win: int = 2048,
    stride: int = 102, post_impact_seconds: float = 1.2,
    endpoint_policy: str = "matched",
    resample_hz: float | None = None,
    encoding_tag: str | None = None,
) -> list[WindowExperiment]:
    """Resolve durations on KFall's 100 Hz grid without changing the baseline."""
    if not durations:
        raise ValueError("At least one duration is required.")
    if resample_hz is not None and (not math.isfinite(resample_hz) or resample_hz < 100):
        raise ValueError("resample_hz must be finite and >= 100 Hz.")
    if event_length_policy not in {"scaled", "fixed"}:
        raise ValueError("event_length_policy must be scaled or fixed.")
    if endpoint_policy not in {"matched", "natural"}:
        raise ValueError("endpoint_policy must be matched or natural.")
    if not isinstance(stride, (int, np.integer)) or not isinstance(fixed_out_win, (int, np.integer)) or stride < 1 or fixed_out_win < 2:
        raise ValueError("stride must be >= 1 and fixed_out_win must be >= 2.")
    if not math.isfinite(post_impact_seconds) or post_impact_seconds < 0:
        raise ValueError("post_impact_seconds must be finite and non-negative.")
    if not math.isclose(post_impact_seconds * 100, round(post_impact_seconds * 100), abs_tol=1e-8):
        raise ValueError("post_impact_seconds must represent whole samples at 100 Hz.")
    sizes = []
    for duration in durations:
        if not math.isfinite(float(duration)) or float(duration) <= 0:
            raise ValueError("Durations must be finite and positive.")
        samples = window_samples(float(duration))
        sizes.append(samples)
    if len(set(sizes)) != len(sizes):
        raise ValueError("Durations must not describe duplicate window lengths.")
    context = max(sizes) if endpoint_policy == "matched" else None
    experiments = []
    for raw_win in sorted(sizes):
        duration = raw_win / 100
        out_win = event_steps_for_samples(raw_win) if event_length_policy == "scaled" else int(fixed_out_win)
        stem = f"w{_tag(duration)}s_post{_tag(post_impact_seconds)}s_{endpoint_policy}_{event_length_policy}"
        if context is not None:
            stem += f"{context}"
        if resample_hz is not None:
            stem += f"_hz{resample_hz:g}".replace(".", "p")
        if encoding_tag is not None:
            stem += f"_enc{encoding_tag}"
        experiments.append(WindowExperiment(
            duration_seconds=duration, raw_win=raw_win, out_win=out_win, stride=int(stride),
            post_impact_seconds=float(post_impact_seconds), endpoint_context=context,
            event_length_policy=event_length_policy, cache_name=f"compare_{stem}", run_name=f"kfall_{stem}",
            resample_hz=resample_hz,
        ))
    return experiments


def _loader_config(experiment: WindowExperiment, args: argparse.Namespace):
    from loaders.kfall_event_loader import KFallEventLoaderConfig

    return KFallEventLoaderConfig(
        seed=args.seed, split=tuple(args.split), raw_win=experiment.raw_win,
        out_win=experiment.out_win, stride=experiment.stride,
        post_impact_seconds=experiment.post_impact_seconds,
        endpoint_context=experiment.endpoint_context,
        theta_sd=tuple(args.theta), dead_zone=args.dead_zone, sd_init=args.sd_init,
        fall_policy="pre_impact", verbose_every=0,
        resample_hz=experiment.resample_hz,
    )


def _endpoint_audit(recordings, records, partition, cfg) -> dict[str, Any]:
    from loaders.kfall_event_loader import window_targets_for_record

    result = {}
    for name, files in zip(SPLITS, partition):
        counts = {str(label): 0 for label in (range(4) if cfg.post_impact_seconds > 0 else range(3))}
        empty, missing_falls, fall_files = 0, 0, 0
        digest = hashlib.sha256()
        for file_index in files:
            file_index = int(file_index)
            record = records[file_index]
            starts, labels = window_targets_for_record(recordings[file_index], record, cfg)
            if not len(starts):
                empty += 1
            if record["label_id"] == 0:
                fall_files += 1
                missing_falls += int(not np.any(np.asarray(labels) == 0))
            for label in counts:
                counts[label] += int(np.sum(np.asarray(labels) == int(label)))
            endpoints = np.asarray(starts, dtype=np.int64) + cfg.raw_win - 1
            frames = np.asarray(record["frames"], dtype=np.int64)[endpoints]
            manifest = np.column_stack((np.full(len(starts), file_index, dtype=np.int64), frames,
                                         np.asarray(labels, dtype=np.int64)))
            digest.update(manifest.astype("<i8", copy=False).tobytes())
        result[name] = {
            "subjects": sorted({records[int(index)]["subject"] for index in files}),
            "recordings": len(files), "windows": sum(counts.values()), "class_counts": counts,
            "recordings_without_windows": empty, "fall_recordings": fall_files,
            "fall_recordings_without_preimpact_window": missing_falls,
            "endpoint_label_sha256": digest.hexdigest(),
        }
    return result


def inspect_dataset(args: argparse.Namespace, experiments: Sequence[WindowExperiment]) -> dict[str, Any]:
    """Inspect both independent-duration coverage and fair matched endpoints."""
    from preprocess.preprocess_kfall import load_kfall_3class

    recordings, _, records, skipped = load_kfall_3class(args.dataset_root, label_dir=args.label_dir)
    partition = subject_split_indices([record["subject"] for record in records], args.split, args.seed)
    rows = []
    for experiment in experiments:
        cfg = _loader_config(experiment, args)
        selected = _endpoint_audit(recordings, records, partition, cfg)
        natural_cfg = replace(cfg, endpoint_context=None)
        natural = selected if cfg.endpoint_context is None else _endpoint_audit(recordings, records, partition, natural_cfg)
        rows.append({**asdict(experiment), "selected": selected, "natural": natural})
    if args.endpoint_policy == "matched":
        for name in SPLITS:
            digests = {row["selected"][name]["endpoint_label_sha256"] for row in rows}
            if len(digests) != 1:
                raise ValueError(f"{name} comparison does not contain identical endpoint/label pairs across durations.")
    return {
        "dataset": "kfall", "seed": args.seed, "split": list(args.split),
        "dataset_root": str(Path(args.dataset_root).resolve()), "recordings": len(records), "skipped": skipped,
        "endpoint_policy": args.endpoint_policy, "event_length_policy": args.event_length_policy,
        "classes": {str(label): name for label, name in CLASS_NAMES.items()
                    if label < (4 if args.post_impact_seconds > 0 else 3)},
        "comparison_note": "All durations use the same subject partitions and class scheme. The original three-class checkpoint is not a four-class baseline.",
        "matched_note": "Matched endpoints may exclude falls that do not have enough earlier context for the longest window. Natural coverage is reported separately.",
        "experiments": rows,
    }


def training_command(args: argparse.Namespace, experiment: WindowExperiment) -> list[str]:
    command = [sys.executable, str(PROJECT_ROOT / "train_kfall.py"),
               "--dataset-root", str(Path(args.dataset_root).resolve()),
               "--cache-dir", str(Path(args.cache_dir).resolve()), "--cache-name", experiment.cache_name,
               "--out-dir", str(Path(args.out_dir).resolve() / experiment.cache_name),
               "--run", experiment.run_name, "--raw-win", str(experiment.raw_win),
               "--out-win", str(experiment.out_win), "--stride", str(experiment.stride),
               "--post-impact-seconds", str(experiment.post_impact_seconds),
               "--fall-policy", "pre_impact", "--seed", str(args.seed),
               "--split", *(str(value) for value in args.split),
               "--theta", *(str(value) for value in args.theta), "--dead-zone", str(args.dead_zone),
               "--sd-init", args.sd_init, "--epochs", str(args.epochs),
               "--batch-size", str(args.batch_size), "--workers", str(args.workers),
               "--gpu", str(args.gpu), "--device", args.device]
    if args.label_dir is not None:
        command.extend(("--label-dir", str(Path(args.label_dir).resolve())))
    if experiment.resample_hz is not None:
        command.extend(("--resample-hz", str(experiment.resample_hz)))
    if experiment.endpoint_context is not None:
        command.extend(("--endpoint-context", str(experiment.endpoint_context)))
    if args.mode == "preprocess":
        command.append("--preprocess-only")
    return command


def _cached_endpoints(cache: Path) -> dict[str, str]:
    digests = {}
    for name in SPLITS:
        arrays = [np.load(cache / name / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                  for key in ("file_idx", "end_frame", "y_h1")]
        manifest = np.column_stack(arrays).astype("<i8", copy=False)
        digests[name] = hashlib.sha256(manifest.tobytes()).hexdigest()
    return digests


def _collect_run(
    args: argparse.Namespace, experiment: WindowExperiment, run: Path | None = None,
    *, reused: bool = False,
) -> dict[str, Any]:
    """Collect one completed run or a prepared cache without modifying either."""
    cache = Path(args.cache_dir).resolve() / f"kfall_event_loader_{experiment.cache_name}"
    metadata = json.loads((cache / "metadata.json").read_text())
    result = {**asdict(experiment), "cache_dir": str(cache), "split_subjects": metadata["split_subjects"],
              "source_signature": metadata["source_signature"], "endpoint_signatures": _cached_endpoints(cache),
              "split_summary": metadata["split_summary"], "reused_completed_run": reused}
    if run is None:
        return result
    metrics = json.loads((run / "metrics" / "test_metrics.json").read_text())
    history = json.loads((run / "metrics" / "history.json").read_text())
    truth = np.load(run / "metrics" / "y_true.npy", allow_pickle=False)
    prediction = np.load(run / "metrics" / "y_pred.npy", allow_pickle=False)
    from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

    labels = list(range(4 if experiment.post_impact_seconds > 0 else 3))
    per_class = classification_report(truth, prediction, labels=labels, output_dict=True, zero_division=0)
    def recall(label):
        entry = per_class[str(label)]
        return float(entry["recall"]) if entry["support"] else None

    result.update(run_dir=str(run), history=history, test_loss=metrics["loss"],
                  test_accuracy=metrics.get("acc", metrics.get("acc_h1")),
                  test_balanced_accuracy=float(balanced_accuracy_score(truth, prediction)),
                  test_macro_f1=float(f1_score(truth, prediction, average="macro", zero_division=0)),
                  best_epoch=metrics["best_epoch"], best_val_loss=metrics["best_val_loss"],
                  training_time_sec=metrics.get("total_time_sec"),
                  test_fall_recall=recall(0),
                  test_postimpact_recall=recall(3) if experiment.post_impact_seconds > 0 else None,
                  test_per_class={CLASS_NAMES[label]: per_class[str(label)] for label in labels})
    return result


RUN_RUNTIME_FIELDS = {
    "run", "requested_run", "out_dir", "workers", "device", "gpu",
    "refresh_cache", "preprocess_only", "quiet_loader", "no_pin_memory",
}


def _config_identity(config: dict[str, Any]) -> str:
    normalized = dict(config)
    for key, default in {"resample_hz": None, "theta_file": None, "validation_only": False,
                         "train_per_class": 0}.items():
        normalized.setdefault(key, default)
    return json.dumps({key: value for key, value in normalized.items() if key not in RUN_RUNTIME_FIELDS},
                      sort_keys=True, separators=(",", ":"))


def _completed_run_matches(
    args: argparse.Namespace, experiment: WindowExperiment, run: Path,
    expected_config: dict[str, Any],
) -> bool:
    """Reuse only a finished, identically configured, provenance-checked run."""
    required = ("config.json", "dataset_metadata.json", "checkpoints/best_model.pt",
                "metrics/test_metrics.json", "metrics/history.json", "metrics/y_true.npy", "metrics/y_pred.npy")
    if any(not (run / filename).is_file() for filename in required):
        return False
    try:
        config = json.loads((run / "config.json").read_text())
        if _config_identity(config) != _config_identity(expected_config):
            return False
        history = json.loads((run / "metrics/history.json").read_text())
        metrics = json.loads((run / "metrics/test_metrics.json").read_text())
        if not history or not isinstance(history, list):
            return False
        last_epoch = int(history[-1]["epoch"])
        if last_epoch < int(config["epochs"]) and int(history[-1].get("bad_epochs", 0)) < int(config["patience"]):
            return False
        if not np.isfinite(float(metrics["best_val_loss"])):
            return False
        import torch
        from evaluation.kfall_data import load_evaluation_data

        saved = torch.load(run / "checkpoints/best_model.pt", map_location="cpu", weights_only=True)
        if not isinstance(saved.get("config"), dict) or not isinstance(saved.get("model_state_dict"), dict):
            return False
        if _config_identity(saved["config"]) != _config_identity(expected_config):
            return False
        if int(saved["epoch"]) != int(metrics["best_epoch"]) or not 1 <= int(saved["epoch"]) <= last_epoch:
            return False
        if not np.isclose(float(saved["best_val_loss"]), float(metrics["best_val_loss"]), rtol=0, atol=1e-12):
            return False
        cache = Path(args.cache_dir).resolve() / f"kfall_event_loader_{experiment.cache_name}"
        arrays, _, _, _ = load_evaluation_data(run, saved["config"], cache_dir=cache)
        truth = np.load(run / "metrics/y_true.npy", allow_pickle=False)
        predicted = np.load(run / "metrics/y_pred.npy", allow_pickle=False)
        if not np.array_equal(truth, arrays["y_h1"]) or predicted.shape != truth.shape:
            return False
        if not np.isin(predicted, np.arange(expected_config["n_classes"])).all():
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError):
        return False


def _run_time_key(run: Path) -> tuple[float, int, str]:
    """Order completed candidates by creation timestamp, never by test score."""
    import re

    match = re.search(r"_(\d{8})_(\d{6})(?:_(\d+))?$", run.name)
    if match:
        try:
            timestamp = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").timestamp()
            return timestamp, int(match.group(3) or 0), run.name
        except ValueError:
            pass
    return (run / "config.json").stat().st_mtime, 0, run.name


def _find_completed_run(args: argparse.Namespace, experiment: WindowExperiment) -> Path | None:
    from train_kfall import get_args

    expected = vars(get_args(training_command(args, experiment)[2:]))
    destination = Path(args.out_dir).resolve() / experiment.cache_name
    candidates = [path for path in destination.glob("*") if path.is_dir() and (path / "config.json").is_file()]
    for run in sorted(candidates, key=_run_time_key, reverse=True):
        if _completed_run_matches(args, experiment, run, expected):
            return run
    return None


def _run_experiment(args: argparse.Namespace, experiment: WindowExperiment) -> dict[str, Any]:
    if args.mode == "train" and getattr(args, "reuse_completed", False):
        previous = _find_completed_run(args, experiment)
        if previous is not None:
            print(f"[comparison:reuse] {experiment.duration_seconds:.2f} s: {previous}", flush=True)
            return _collect_run(args, experiment, previous, reused=True)
    destination = Path(args.out_dir).resolve() / experiment.cache_name
    before = set(destination.glob("*")) if destination.is_dir() else set()
    command = training_command(args, experiment)
    print(f"[comparison] {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if args.mode == "preprocess":
        return _collect_run(args, experiment)
    created = [path for path in destination.glob("*") if path not in before and (path / "config.json").is_file()]
    if len(created) != 1:
        raise RuntimeError("Cannot identify the training run created for this duration.")
    return _collect_run(args, experiment, created[0])


def _verify_results(results: Sequence[dict[str, Any]], matched: bool) -> None:
    first = results[0]
    for result in results[1:]:
        for key in ("split_subjects", "source_signature"):
            if result[key] != first[key]:
                raise ValueError(f"Window comparison has inconsistent {key}.")
        if matched and result["endpoint_signatures"] != first["endpoint_signatures"]:
            raise ValueError("Window comparison has different endpoint/label pairs across durations.")


def _coverage_plot(path: Path, inspection: dict[str, Any], coverage: str = "selected") -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(13, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    labels = inspection["classes"]
    rows = inspection["experiments"]
    try:
        for axis, split in zip(figure.subplots(1, 3), SPLITS):
            x = np.arange(len(rows))
            width = .8 / len(labels)
            for position, (label, name) in enumerate(labels.items()):
                values = [row[coverage][split]["class_counts"][label] for row in rows]
                axis.bar(x + (position - (len(labels) - 1) / 2) * width, values, width=width, label=name)
            axis.set(xticks=x, xticklabels=[f"{row['duration_seconds']:.2f} s" for row in rows],
                     ylabel="Ventanas", title=f"{split} · {'natural' if coverage == 'natural' else inspection['endpoint_policy']}",
                     xlabel="Duración nominal")
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
        figure.legend(*figure.axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=len(labels))
        figure.savefig(path, dpi=150)
    finally:
        figure.clear()


def _training_plot(path: Path, results: Sequence[dict[str, Any]]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(12, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    accuracy_axis, loss_axis = figure.subplots(1, 2)
    try:
        for result in results:
            history = result["history"]
            epochs = [row["epoch"] for row in history]
            accuracy = [row["val"].get("acc", row["val"].get("acc_h1", np.nan)) for row in history]
            label = f"{result['duration_seconds']:.2f} s"
            accuracy_axis.plot(epochs, accuracy, marker="o", markersize=3, label=label)
            loss_axis.plot(epochs, [row["val"]["loss"] for row in history], marker="o", markersize=3, label=label)
        accuracy_axis.set(title="Accuracy de validación", ylabel="Accuracy", ylim=(0, 1))
        accuracy_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        loss_axis.set(title="Loss de validación", ylabel="Loss")
        for axis in (accuracy_axis, loss_axis):
            axis.set_xlabel("Época")
            axis.grid(alpha=.25)
            axis.legend()
        figure.savefig(path, dpi=150)
    finally:
        figure.clear()


def _save_summary(path: Path, results: Sequence[dict[str, Any]]) -> None:
    fields = ("duration_seconds", "raw_win", "out_win", "stride", "post_impact_seconds",
              "endpoint_context", "resample_hz", "test_accuracy", "test_balanced_accuracy", "test_macro_f1",
              "test_fall_recall", "test_postimpact_recall", "test_loss", "best_val_loss", "best_epoch",
              "training_time_sec", "reused_completed_run", "run_dir", "cache_dir")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _save_coverage(path: Path, inspection: dict[str, Any]) -> None:
    rows = []
    for experiment in inspection["experiments"]:
        for coverage in ("selected", "natural"):
            for split in SPLITS:
                values = experiment[coverage][split]
                rows.append({
                    "duration_seconds": experiment["duration_seconds"], "raw_win": experiment["raw_win"],
                    "out_win": experiment["out_win"], "coverage": coverage, "split": split,
                    "windows": values["windows"], "subjects": len(values["subjects"]),
                    "recordings": values["recordings"],
                    "fall_recordings_without_preimpact_window": values["fall_recordings_without_preimpact_window"],
                    "fall_recordings": values["fall_recordings"],
                    **{name: values["class_counts"][label] for label, name in inspection["classes"].items()},
                })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    from loaders.kfall_event_loader import DEFAULT_THETA_KFALL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "preprocess", "train"), default="inspect",
                        help="inspect audits window coverage; preprocess builds separate caches; train runs all experiments.")
    parser.add_argument("--durations", type=float, nargs="+", default=[2.05, 3.30, 4.10])
    parser.add_argument("--dataset-root", type=Path, default=Path("data/KFall"))
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/kfall_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/kfall_window_comparison"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/kfall_window_comparison"))
    parser.add_argument("--event-length-policy", choices=("scaled", "fixed"), default="scaled")
    parser.add_argument("--fixed-out-win", type=int, default=2048)
    parser.add_argument("--endpoint-policy", choices=("matched", "natural"), default="matched",
                        help="matched uses the longest context's endpoints for all durations; natural retains each duration's own coverage.")
    parser.add_argument("--stride", type=int, default=102)
    parser.add_argument("--post-impact-seconds", type=float, default=1.2)
    parser.add_argument("--resample-hz", type=float, default=None,
                        help="Optional intermediate analog interpolation frequency, e.g. 200; event steps are unchanged.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=float, nargs=3, default=[.8, .1, .1])
    thresholds = parser.add_mutually_exclusive_group()
    thresholds.add_argument("--theta", type=float, nargs=6, default=list(DEFAULT_THETA_KFALL))
    thresholds.add_argument("--theta-file", type=Path, default=None,
                            help="Read theta_best from a calibration JSON file.")
    parser.add_argument("--dead-zone", type=float, default=.5)
    parser.add_argument("--sd-init", choices=("x0", "zero"), default="x0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--parallel-runs", type=int, default=1, help="Number of independent experiment subprocesses to run concurrently.")
    parser.add_argument("--reuse-completed", action="store_true", help="Reuse the latest fully completed, identically configured run for each duration in train mode.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from loaders.kfall_event_loader import DEFAULT_THETA_KFALL
        from train_kfall import read_theta_file

        if args.theta_file is not None:
            args.theta = read_theta_file(args.theta_file)
        encoding_tag = None
        if tuple(args.theta) != DEFAULT_THETA_KFALL or args.dead_zone != .5 or args.sd_init != "x0":
            encoding_tag = hashlib.sha256(json.dumps(
                [args.theta, args.dead_zone, args.sd_init], separators=(",", ":")
            ).encode()).hexdigest()[:12]
        if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.parallel_runs < 1:
            raise ValueError("epochs/batch-size/parallel-runs must be positive and workers must be nonnegative.")
        experiments = build_experiment_specs(args.durations, event_length_policy=args.event_length_policy,
                                             fixed_out_win=args.fixed_out_win, stride=args.stride,
                                             post_impact_seconds=args.post_impact_seconds,
                                             endpoint_policy=args.endpoint_policy,
                                             resample_hz=args.resample_hz, encoding_tag=encoding_tag)
        inspection = inspect_dataset(args, experiments)
        inspection["commands"] = [training_command(args, experiment) for experiment in experiments]
        args.results_dir.mkdir(parents=True, exist_ok=True)
        save_json(args.results_dir / "window_plan.json", inspection)
        _coverage_plot(args.results_dir / "window_counts.png", inspection)
        _coverage_plot(args.results_dir / "natural_window_counts.png", inspection, "natural")
        _save_coverage(args.results_dir / "window_coverage.csv", inspection)
        for row in inspection["experiments"]:
            test = row["selected"]["test"]
            print(f"[coverage] {row['duration_seconds']:.2f} s | raw={row['raw_win']} events={row['out_win']} | "
                  f"test={test['windows']} classes={test['class_counts']} | "
                  f"fall recordings without pre-impact context={test['fall_recordings_without_preimpact_window']}/{test['fall_recordings']}")
        if args.mode == "inspect":
            print(f"[saved] {args.results_dir / 'window_plan.json'}")
            return
        results = []
        planned_by_size = {row["raw_win"]: row for row in inspection["experiments"]}

        def collect_result(result):
            planned = planned_by_size[result["raw_win"]]
            expected = {name: planned["selected"][name]["endpoint_label_sha256"] for name in SPLITS}
            if result["endpoint_signatures"] != expected:
                raise ValueError("Encoded cache endpoint/label pairs differ from the inspected comparison plan.")
            results.append(result)
            results.sort(key=lambda value: value["duration_seconds"])
            _verify_results(results, args.endpoint_policy == "matched")
            save_json(args.results_dir / "comparison.json", results)
            _save_summary(args.results_dir / "comparison.csv", results)

        if args.parallel_runs == 1:
            for experiment in experiments:
                collect_result(_run_experiment(args, experiment))
        else:
            with ThreadPoolExecutor(max_workers=min(args.parallel_runs, len(experiments))) as pool:
                futures = [pool.submit(_run_experiment, args, experiment) for experiment in experiments]
                for future in as_completed(futures):
                    collect_result(future.result())
        if args.mode == "train":
            _training_plot(args.results_dir / "validation_comparison.png", results)
        print(f"[saved] {args.results_dir}")
    except (ValueError, FileNotFoundError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
