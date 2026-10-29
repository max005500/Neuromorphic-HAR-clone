#!/usr/bin/env python3
"""Compare Sigma-Delta thresholds using subject-independent validation scores.

Each candidate trains a new SNN from the same seed. Test inference is disabled
throughout this search. The selected vector is the best of a finite candidate
set, rather than a claim of a global optimum.
"""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from preprocess.kfall_labels import class_names, event_steps_for_samples, window_samples
from preprocess.utils import save_json

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_THETA = (0.1, 0.1, 0.1, 10.0, 10.0, 10.0)
RUNTIME_FIELDS = {"run", "requested_run", "out_dir", "cache_dir", "refresh_cache", "theta_file"}


@dataclass(frozen=True)
class ThresholdCandidate:
    index: int
    acc_scale: float
    gyro_scale: float
    theta: tuple[float, ...]
    candidate_id: str


def build_candidates(
    theta: Sequence[float] = DEFAULT_THETA,
    acc_scales: Sequence[float] = (0.5, 1.0, 2.0),
    gyro_scales: Sequence[float] = (0.5, 1.0, 2.0),
) -> list[ThresholdCandidate]:
    """Enumerate independent sensor-group scales, putting the baseline first."""
    initial = np.asarray(theta, dtype=np.float64)
    if initial.shape != (6,) or not np.isfinite(initial).all() or np.any(initial <= 0):
        raise ValueError("theta must contain six finite positive values.")
    for name, scales in (("acc-scales", acc_scales), ("gyro-scales", gyro_scales)):
        if not scales or any(not math.isfinite(float(value)) or value <= 0 for value in scales):
            raise ValueError(f"{name} must contain finite positive values.")
        if len(set(scales)) != len(scales):
            raise ValueError(f"{name} must not contain duplicates.")
    pairs = [(float(acc), float(gyro)) for acc in acc_scales for gyro in gyro_scales]
    pairs.sort(key=lambda pair: pair != (1.0, 1.0))
    result = []
    for index, (acc, gyro) in enumerate(pairs):
        values = tuple(float(value) for value in initial * np.asarray([acc] * 3 + [gyro] * 3))
        if not np.isfinite(values).all():
            raise ValueError("Scaled theta values must remain finite.")
        digest = hashlib.sha256(json.dumps(values).encode()).hexdigest()[:10]
        result.append(ThresholdCandidate(index, acc, gyro, values, f"theta_{digest}"))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/KFall"))
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/kfall_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/kfall_threshold_search"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/kfall_threshold_search"))
    parser.add_argument("--window-seconds", type=float, default=2.05)
    parser.add_argument("--out-win", type=int, default=None)
    parser.add_argument("--stride", type=int, default=102, help="Stride on the native 100 Hz frame grid.")
    parser.add_argument("--endpoint-context", type=int, default=None, help="Native samples required before all prediction endpoints.")
    parser.add_argument("--post-impact-seconds", type=float, default=1.2)
    parser.add_argument("--resample-hz", type=float, default=200.0, help="Intermediate analog interpolation rate; native annotations remain at 100 Hz.")
    thresholds = parser.add_mutually_exclusive_group()
    thresholds.add_argument("--theta", type=float, nargs=6, default=list(DEFAULT_THETA))
    thresholds.add_argument("--theta-file", type=Path, default=None,
                            help="Start from theta_best/best_theta in an optimizer or previous search JSON.")
    parser.add_argument("--acc-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--gyro-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--dead-zone", type=float, default=0.5)
    parser.add_argument("--sd-init", choices=["x0", "zero"], default="x0")
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3, help="Fixed exploratory training budget per candidate.")
    parser.add_argument("--train-per-class", type=int, default=128, help="Deterministic training subset per class; 0 uses all training windows.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--parallel-runs", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--metric", choices=["macro_f1", "balanced_accuracy"], default="macro_f1")
    parser.add_argument("--reuse-completed", action="store_true", help="Reuse only completed candidates with identical configuration and source provenance.")
    parser.add_argument("--dry-run", action="store_true", help="Write the candidate plan without encoding or training.")
    return parser


def _identity(config: dict[str, Any]) -> str:
    return json.dumps({key: value for key, value in config.items() if key not in RUNTIME_FIELDS},
                      sort_keys=True, separators=(",", ":"))


def search_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Include every training-affecting search option in the experiment identity."""
    return {
        "dataset_root": str(args.dataset_root.resolve()),
        "label_dir": str((args.label_dir or args.dataset_root / "label_data").resolve()),
        "raw_win": window_samples(args.window_seconds),
        "out_win": (args.out_win if args.out_win is not None
                    else event_steps_for_samples(window_samples(args.window_seconds))),
        "stride": args.stride, "endpoint_context": args.endpoint_context,
        "post_impact_seconds": args.post_impact_seconds, "resample_hz": args.resample_hz,
        "dead_zone": args.dead_zone, "sd_init": args.sd_init,
        "split": list(args.split), "seed": args.seed, "epochs": args.epochs,
        "train_per_class": args.train_per_class, "batch_size": args.batch_size,
        "workers": args.workers, "device": args.device, "gpu": args.gpu,
    }


def candidate_name(candidate: ThresholdCandidate, settings: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps({**settings, "theta": candidate.theta}, sort_keys=True).encode()).hexdigest()[:16]
    return f"tune_w{settings['raw_win']}_hz{settings['resample_hz']:g}_{digest}"


def training_command(args: argparse.Namespace, candidate: ThresholdCandidate) -> list[str]:
    settings = search_settings(args)
    name = candidate_name(candidate, settings)
    command = [sys.executable, str(PROJECT_ROOT / "train_kfall.py"),
               "--dataset-root", settings["dataset_root"], "--label-dir", settings["label_dir"],
               "--cache-dir", str(args.cache_dir.resolve()), "--cache-name", name,
               "--out-dir", str(args.out_dir.resolve() / name), "--run", f"kfall_{name}",
               "--raw-win", str(settings["raw_win"]), "--out-win", str(settings["out_win"]),
               "--stride", str(args.stride), "--post-impact-seconds", str(args.post_impact_seconds),
               "--resample-hz", str(args.resample_hz), "--fall-policy", "pre_impact",
               "--theta", *(str(value) for value in candidate.theta),
               "--dead-zone", str(args.dead_zone), "--sd-init", args.sd_init,
               "--split", *(str(value) for value in args.split), "--seed", str(args.seed),
               "--epochs", str(args.epochs), "--train-per-class", str(args.train_per_class),
               "--batch-size", str(args.batch_size), "--workers", str(args.workers),
               "--device", args.device, "--gpu", str(args.gpu), "--validation-only", "--quiet-loader"]
    if args.endpoint_context is not None:
        command.extend(("--endpoint-context", str(args.endpoint_context)))
    return command


def _endpoint_signature(cache: Path, split: str) -> str:
    columns = [np.load(cache / split / f"{key}.npy", mmap_mode="r", allow_pickle=False)
               for key in ("file_idx", "end_frame", "y_h1")]
    return hashlib.sha256(np.column_stack(columns).astype("<i8", copy=False).tobytes()).hexdigest()


def _event_rate(cache: Path) -> float:
    events = np.load(cache / "val/x_ev.npy", mmap_mode="r", allow_pickle=False)
    return sum(int(np.count_nonzero(events[start:start + 128])) for start in range(0, len(events), 128)) / events.size


def _check_subjects(metadata: dict[str, Any]) -> None:
    subjects = metadata["split_subjects"]
    groups = [set(subjects[split]) for split in ("train", "val", "test")]
    if any(not group for group in groups) or any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Threshold search requires three nonempty disjoint subject groups.")


def collect_candidate(
    args: argparse.Namespace, candidate: ThresholdCandidate, run: Path,
    source_signature: dict[str, Any], *, reused: bool = False,
) -> dict[str, Any]:
    """Validate a completed run using validation arrays and source metadata."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
    from train_kfall import get_args, training_subset_indices
    import torch

    expected = vars(get_args(training_command(args, candidate)[2:]))
    config = json.loads((run / "config.json").read_text())
    if _identity(config) != _identity(expected):
        raise ValueError("Candidate training configuration differs from this search.")
    if not config["validation_only"] or (run / "metrics/test_metrics.json").exists():
        raise ValueError("A threshold candidate must finish without test inference.")
    history = json.loads((run / "metrics/history.json").read_text())
    metrics = json.loads((run / "metrics/validation_metrics.json").read_text())
    if not history or (int(history[-1]["epoch"]) < args.epochs and
                       int(history[-1].get("bad_epochs", 0)) < int(config["patience"])):
        raise ValueError("Candidate training has not completed its requested budget.")
    checkpoint = torch.load(run / "checkpoints/best_model.pt", map_location="cpu", weights_only=True)
    if _identity(checkpoint["config"]) != _identity(expected):
        raise ValueError("Candidate checkpoint has a different configuration.")
    if int(checkpoint["epoch"]) != int(metrics["best_epoch"]):
        raise ValueError("Candidate validation metrics do not describe its best checkpoint.")
    if not np.isclose(float(checkpoint["best_val_loss"]), float(metrics["best_val_loss"]), rtol=0, atol=1e-12):
        raise ValueError("Candidate validation loss does not match its checkpoint.")
    metadata = json.loads((run / "dataset_metadata.json").read_text())
    _check_subjects(metadata)
    if metadata["source_signature"] != source_signature:
        raise ValueError("Candidate source files differ from this search.")
    name = candidate_name(candidate, search_settings(args))
    cache = args.cache_dir.resolve() / f"kfall_event_loader_{name}"
    cached_metadata = json.loads((cache / "metadata.json").read_text())
    for key in ("source_signature", "split_subjects", "config"):
        if cached_metadata[key] != metadata[key]:
            raise ValueError(f"Candidate cache provenance differs in {key}.")
    from loaders.kfall_event_loader import KFallEventLoaderConfig
    loader_cfg = KFallEventLoaderConfig(
        seed=args.seed, split=tuple(args.split), raw_win=config["raw_win"], out_win=config["out_win"],
        stride=args.stride, endpoint_context=args.endpoint_context, post_impact_seconds=args.post_impact_seconds,
        theta_sd=candidate.theta, dead_zone=args.dead_zone, sd_init=args.sd_init,
        fall_policy="pre_impact", resample_hz=args.resample_hz,
    )
    if metadata["config"] != json.loads(json.dumps(asdict(loader_cfg))):
        raise ValueError("Candidate cache has different encoder settings.")
    truth = np.load(run / "metrics/y_true_val.npy", allow_pickle=False)
    prediction = np.load(run / "metrics/y_pred_val.npy", allow_pickle=False)
    cached_truth = np.load(cache / "val/y_h1.npy", allow_pickle=False)
    labels = list(range(4 if args.post_impact_seconds > 0 else 3))
    if (truth.ndim != 1 or not len(truth) or not np.array_equal(truth, cached_truth)
            or prediction.shape != truth.shape or not np.isin(truth, labels).all()
            or not np.isin(prediction, labels).all()):
        raise ValueError("Candidate predictions do not match the full held-out validation split.")
    indices = np.load(run / "training_indices.npy", allow_pickle=False)
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("Candidate training subset indices are invalid.")
    train_labels = np.load(cache / "train/y_h1.npy", allow_pickle=False)
    if not np.array_equal(indices, training_subset_indices(train_labels, args.train_per_class, args.seed)):
        raise ValueError("Candidate training subset differs from this search's fixed sample selection.")
    score = {
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
    }
    recalls = recall_score(truth, prediction, labels=labels, average=None, zero_division=0)
    return {
        **asdict(candidate), "theta": list(candidate.theta), **score,
        "validation_accuracy": float(np.mean(truth == prediction)),
        "validation_loss": float(metrics["loss"]), "best_epoch": int(metrics["best_epoch"]),
        "best_val_loss": float(metrics["best_val_loss"]),
        "validation_recall": {class_names(len(labels))[label]: float(recalls[label]) for label in labels},
        "validation_windows": len(truth), "training_windows": len(indices),
        "training_indices_sha256": hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest(),
        "endpoint_signatures": {split: _endpoint_signature(cache, split) for split in ("train", "val")},
        "validation_event_rate": _event_rate(cache),
        "split_subjects": metadata["split_subjects"], "source_signature": source_signature,
        "run_dir": str(run.resolve()), "cache_dir": str(cache), "reused_completed_run": reused,
    }


def _run_candidate(args, candidate, source_signature) -> dict[str, Any]:
    name = candidate_name(candidate, search_settings(args))
    destination = args.out_dir.resolve() / name
    if args.reuse_completed and destination.exists():
        for run in sorted(destination.iterdir(), key=lambda path: path.name, reverse=True):
            if not run.is_dir():
                continue
            try:
                result = collect_candidate(args, candidate, run, source_signature, reused=True)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError):
                continue
            print(f"[threshold:reuse] {candidate.index + 1}: {run}", flush=True)
            return result
    before = set(destination.iterdir()) if destination.exists() else set()
    command = training_command(args, candidate)
    print(f"[threshold] {candidate.index + 1}: {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    created = [path for path in destination.iterdir()
               if path not in before and (path / "config.json").is_file()]
    if len(created) != 1:
        raise RuntimeError("Cannot identify the candidate training run just created.")
    return collect_candidate(args, candidate, created[0], source_signature)


def verify_comparable(results: Sequence[dict[str, Any]]) -> None:
    if not results:
        return
    for result in results[1:]:
        for key in ("split_subjects", "source_signature", "endpoint_signatures", "training_indices_sha256"):
            if result[key] != results[0][key]:
                raise ValueError(f"Threshold candidates have different {key}.")


def select_best(results: Sequence[dict[str, Any]], metric: str = "macro_f1") -> dict[str, Any]:
    if metric not in {"macro_f1", "balanced_accuracy"} or not results:
        raise ValueError("Selection requires completed candidates and a validation metric.")
    for result in results:
        if not np.isfinite([result[metric], result["validation_loss"]]).all():
            raise ValueError("Candidate validation scores must be finite.")
    return max(results, key=lambda row: (row[metric], -row["validation_loss"], -row["index"]))


def _plot(path: Path, results: Sequence[dict[str, Any]], metric: str) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(11, 7), layout="constrained")
    FigureCanvasAgg(figure)
    score_axis, events_axis = figure.subplots(2, 1, sharex=True)
    x = np.arange(len(results))
    selected = select_best(results, metric)
    colors = ["#dd8e26" if row["index"] == selected["index"] else "#3275aa" for row in results]
    score_axis.bar(x, [row[metric] for row in results], color=colors)
    score_axis.set(ylabel=f"Validation {metric.replace('_', ' ')}", ylim=(0, 1),
                   title="KFall · búsqueda finita de umbrales · selección con validation")
    events_axis.bar(x, [row["validation_event_rate"] for row in results], color=colors)
    events_axis.set(ylabel="Eventos / posiciones ON-OFF", xlabel="Multiplicadores del acelerómetro / giroscopio",
                    xticks=x, xticklabels=[f"{row['acc_scale']:g}× / {row['gyro_scale']:g}×" for row in results])
    for axis in (score_axis, events_axis):
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.savefig(path, dpi=150)
    figure.clear()


def export_results(args, plan, results, *, complete=False) -> None:
    ordered = sorted(results, key=lambda row: row["index"])
    verify_comparable(ordered)
    if not complete:
        (args.results_dir / "best_theta.json").unlink(missing_ok=True)
    save_json(args.results_dir / "search_results.json", {**plan, "complete": complete, "results": ordered})
    keys = ["index", "acc_scale", "gyro_scale", "theta", "macro_f1", "balanced_accuracy",
            "validation_accuracy", "validation_loss", "validation_event_rate", "best_epoch",
            "training_windows", "validation_windows", "run_dir", "reused_completed_run"]
    with (args.results_dir / "search_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in ordered)
    if not ordered:
        (args.results_dir / "threshold_validation.png").unlink(missing_ok=True)
        return
    _plot(args.results_dir / "threshold_validation.png", ordered, args.metric)
    if complete:
        winner = select_best(ordered, args.metric)
        save_json(args.results_dir / "best_theta.json", {
            "dataset": "kfall", "theta_best": winner["theta"], "selected_by": f"validation_{args.metric}",
            "selection_score": winner[args.metric], "candidate_count": len(ordered),
            "search_config": plan["settings"], "source_signature": plan["source_signature"],
            "split_subjects": winner["split_subjects"], "best_candidate": winner,
            "test_used_for_selection": False,
            "note": "Best of this finite candidate set and exploratory training budget; not a global optimum. Retrain on all training windows before final test evaluation.",
        })


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 1 or args.train_per_class < 0
            or args.parallel_runs < 1 or args.workers < 0):
        parser.error("epochs, batch-size and parallel-runs must be positive; train-per-class/workers nonnegative.")
    try:
        if args.theta_file is not None:
            from train_kfall import read_theta_file
            args.theta = read_theta_file(args.theta_file)
        candidates = build_candidates(args.theta, args.acc_scales, args.gyro_scales)
        settings = search_settings(args)
        from loaders.kfall_event_loader import KFallEventLoaderConfig, _source_signature
        KFallEventLoaderConfig(raw_win=settings["raw_win"], out_win=settings["out_win"], stride=args.stride,
                              endpoint_context=args.endpoint_context, post_impact_seconds=args.post_impact_seconds,
                              resample_hz=args.resample_hz, dead_zone=args.dead_zone)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    source_signature = _source_signature(args.dataset_root, Path(settings["label_dir"]))
    plan = {
        "dataset": "kfall", "settings": settings, "metric": args.metric,
        "source_signature": source_signature, "test_used_for_selection": False,
        "candidates": [{**asdict(candidate), "command": training_command(args, candidate)} for candidate in candidates],
        "note": "Each candidate starts from the same seed and uses identical subject partitions. Select using validation only; test inference is disabled.",
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        # A planning command must not replace the provenance of a live/completed
        # search that uses this same output directory.
        save_json(args.results_dir / "search_plan_dry_run.json", plan)
        print(f"[threshold:plan] {args.results_dir / 'search_plan_dry_run.json'}", flush=True)
        return
    save_json(args.results_dir / "search_plan.json", plan)
    (args.results_dir / "best_theta.json").unlink(missing_ok=True)
    results = []
    export_results(args, plan, results)
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        futures = {executor.submit(_run_candidate, args, candidate, source_signature): candidate for candidate in candidates}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            export_results(args, plan, results)
            print(f"[threshold:validation] acc={result['acc_scale']:g}× gyro={result['gyro_scale']:g}× | "
                  f"{args.metric}={result[args.metric]:.4f} | event_rate={result['validation_event_rate']:.4f}", flush=True)
    export_results(args, plan, results, complete=True)
    best = select_best(results, args.metric)
    print(f"[threshold:best] theta={best['theta']} | validation_{args.metric}={best[args.metric]:.4f}", flush=True)
    print(f"[threshold:done] {args.results_dir / 'best_theta.json'}", flush=True)


if __name__ == "__main__":
    main()
