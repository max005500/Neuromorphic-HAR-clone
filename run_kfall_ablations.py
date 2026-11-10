#!/usr/bin/env python3
"""Validation-only SCN ablations, one process per GPU, with resumable Excel reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openpyxl import Workbook
from tqdm.auto import tqdm

from train_kfall import get_args

ROOT = Path(__file__).resolve().parent


def variants():
    """One-factor-at-a-time screening, not an exhaustive Cartesian optimum."""
    baseline = {"tau": 0.75, "thresh": 0.5, "hard_reset": False,
                "p_drop": 0.35, "conv_ch": [32, 64, 64], "kernels": [32, 32, 8]}
    results = [("baseline", baseline)]
    for key, values in {
        "tau": [0.5, 0.9], "thresh": [0.3, 0.7], "hard_reset": [True],
        "p_drop": [0.0, 0.2, 0.5], "conv_ch": [[16, 32, 32], [32, 64, 128]],
        "kernels": [[16, 16, 8]],
    }.items():
        for i, value in enumerate(values):
            results.append((f"{key}_{i}", {**baseline, key: value}))
    return results


def export_report(output, rows):
    rows = sorted(rows, key=lambda row: (row["status"] != "completed", row.get("val_loss", float("inf"))))
    (output / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Validation"
    columns = ["id", "variant", "status", "val_loss", "val_accuracy",
               "best_epoch", "parameters", "duration_seconds", "gpu", "run_dir", "log", "error", "config"]
    sheet.append(columns)
    for row in rows:
        sheet.append([json.dumps(row[key]) if isinstance(row.get(key), (dict, list)) else row.get(key)
                      for key in columns])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    temporary = output / "summary.tmp.xlsx"
    workbook.save(temporary)
    temporary.replace(output / "summary.xlsx")


def trial_id(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def build_command(args, trial, gpu, directory):
    config = trial["config"]
    command = [
        sys.executable, str(ROOT / "train_kfall.py"),
        "--dataset-root", str(Path(args.dataset_root).resolve()),
        "--cache-dir", str(Path(args.cache_dir).resolve()), "--cache-name", args.cache_name,
        "--cache-format", "pt", "--out-dir", str(directory / "runs"),
        "--run", f"kfall_ablation_{trial['id']}", "--gpu", str(gpu),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--workers", str(args.workers), "--seed", str(args.seed),
        "--split", *map(str, args.split), "--raw-win", str(args.raw_win),
        "--out-win", str(args.out_win), "--stride", str(args.stride),
        "--post-impact-seconds", "1.2", "--theta", *map(str, args.theta),
        "--dead-zone", "0.5", "--skip-test", "--quiet-loader",
    ]
    for key in ("tau", "thresh", "p_drop"):
        command.extend(["--" + key.replace("_", "-"), str(config[key])])
    for key in ("conv_ch", "kernels"):
        command.extend(["--" + key.replace("_", "-"), *map(str, config[key])])
    if config["hard_reset"]:
        command.append("--hard-reset")
    if args.no_offset:
        command.append("--no-offset")
    if args.annotation_file:
        command.extend(["--annotation-file", str(Path(args.annotation_file).resolve())])
    if args.no_manual_labels:
        command.append("--no-manual-labels")
    return command


def execute_trial(args, trial, gpu, output):
    directory = output / trial["id"]
    directory.mkdir(exist_ok=True)
    record = directory / "result.json"
    if record.is_file():
        saved = json.loads(record.read_text())
        if saved["status"] == "completed":
            run = Path(saved["run_dir"])
            if not (run / "checkpoints" / "best_model.pt").is_file():
                raise FileNotFoundError(f"Completed trial checkpoint is missing: {run}")
            return saved
    row = {**trial, "gpu": gpu, "log": str(directory / "train.log")}
    command = build_command(args, trial, gpu, directory)
    (directory / "command.json").write_text(json.dumps(command, indent=2))
    previous_reports = set((directory / "runs").glob("*/metrics/validation_selection.json"))
    started = time.monotonic()
    with (directory / "train.log").open("w") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    row["duration_seconds"] = time.monotonic() - started
    if process.returncode:
        row.update(status="failed", error=f"Training exit code {process.returncode}; see log.")
    else:
        runs = list(set((directory / "runs").glob("*/metrics/validation_selection.json")) - previous_reports)
        if len(runs) != 1:
            row.update(status="failed", error="Expected exactly one completed validation report.")
        else:
            run = runs[0].parent.parent
            summary = json.loads(runs[0].read_text())
            history = json.loads((run / "metrics" / "history.json").read_text())
            best = next(item for item in history if item["epoch"] == summary["best_epoch"])
            row.update(status="completed", val_loss=summary["best_val_loss"],
                       val_accuracy=best["val"]["acc"], best_epoch=summary["best_epoch"],
                       parameters=summary["parameters"], run_dir=str(run))
    record.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/KFall")
    parser.add_argument("--cache-dir", default="data/kfall_ablation_cache")
    parser.add_argument("--cache-name", default="calibrated_w120")
    parser.add_argument("--output-dir", default="results/kfall_ablations")
    parser.add_argument("--theta-file", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", nargs=3, type=float, default=[0.6, 0.3, 0.1])
    parser.add_argument("--raw-win", type=int, default=120)
    parser.add_argument("--out-win", type=int, default=1200)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--no-offset", action="store_true")
    parser.add_argument("--annotation-file")
    parser.add_argument("--no-manual-labels", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write plan without preprocessing or training.")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs/batch-size positive; workers nonnegative.")
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error("Supply unique nonnegative GPU indices.")
    if args.seed < 0 or any(not math.isfinite(x) or x <= 0 for x in args.split) or not math.isclose(sum(args.split), 1):
        parser.error("Use a nonnegative seed and positive split fractions summing to one.")
    theta = json.loads(args.theta_file.read_text())
    for key in ("seed", "split", "raw_win", "out_win"):
        if theta[key] != getattr(args, key):
            parser.error(f"Calibration {key} must match the ablation settings.")
    if theta["dead_zone"] != 0.5 or theta["sd_init"] != "x0":
        parser.error("Calibration must use dead-zone=0.5 and sd-init=x0.")
    if Path(theta["dataset_root"]).resolve() != Path(args.dataset_root).resolve():
        parser.error("Calibration dataset root differs.")
    args.theta = theta["theta_best"]
    if len(args.theta) != 6 or any(not math.isfinite(x) or x <= 0 for x in args.theta):
        parser.error("Calibration must contain six finite positive theta values.")
    if not args.no_manual_labels:
        args.annotation_file = args.annotation_file or get_args([]).annotation_file
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = {key: value for key, value in vars(args).items()
                if key not in {"gpus", "dry_run", "output_dir", "theta_file"}}
    settings["theta_calibration"] = theta
    settings["annotation_sha256"] = (
        hashlib.sha256(Path(args.annotation_file).read_bytes()).hexdigest()
        if args.annotation_file and not args.no_manual_labels else None
    )
    trials = [{"id": trial_id({**settings, "config": config}), "variant": name, "config": config}
              for name, config in variants()]
    plan = {"settings": settings, "trials": trials}
    plan_path = output / "plan.json"
    if plan_path.is_file() and json.loads(plan_path.read_text()) != plan:
        parser.error("Output directory belongs to a different search protocol; choose a new --output-dir.")
    plan_path.write_text(json.dumps(plan, indent=2))
    if args.dry_run:
        print(f"[plan] {len(trials)} variants | {output / 'plan.json'}")
        return
    import torch
    if not torch.cuda.is_available() or any(gpu >= torch.cuda.device_count() for gpu in args.gpus):
        parser.error("Requested GPUs unavailable; use --dry-run to inspect the plan.")
    from loaders.kfall_event_loader import DEFAULT_EXCLUDED_TASKS, _source_signature
    if set(theta["excluded_tasks"]) != set(DEFAULT_EXCLUDED_TASKS):
        parser.error("Calibration task exclusions differ from training defaults.")
    if theta["source_signature"] != _source_signature(Path(args.dataset_root), Path(args.dataset_root) / "label_data"):
        parser.error("Dataset changed since calibration; recalibrate.")
    # Build shared NPY/PT files once before launching independent GPU processes.
    from train_kfall import main as train
    preparatory = build_command(args, trials[0], args.gpus[0], output)
    train(preparatory[2:] + ["--preprocess-only"])
    rows = [{**trial, "status": "pending"} for trial in trials]
    row_indices = {row["id"]: i for i, row in enumerate(rows)}
    export_report(output, rows)
    lock = Lock()
    # One worker owns a GPU and serializes its assigned trials.
    def run_worker(gpu, assigned):
        for trial in assigned:
            with lock:
                rows[row_indices[trial["id"]]].update(
                    status="running", gpu=gpu, log=str(output / trial["id"] / "train.log"),
                )
                export_report(output, rows)
            row = execute_trial(args, trial, gpu, output)
            with lock:
                rows[row_indices[trial["id"]]] = row
                export_report(output, rows)
                progress.update()
                tqdm.write(f"[{progress.n}/{len(trials)}] {row['variant']} {row['status']}")
    with tqdm(total=len(trials), desc="SCN ablations", unit="trial") as progress, \
            ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(run_worker, gpu, trials[i::len(args.gpus)])
                   for i, gpu in enumerate(args.gpus)]
        for future in as_completed(futures):
            future.result()
    completed = [row for row in rows if row["status"] == "completed"]
    if completed:
        best = min(completed, key=lambda row: (row["val_loss"], -row["val_accuracy"]))
        (output / "best_validation.json").write_text(json.dumps(best, indent=2) + "\n")
        print(f"[best validation loss] {best['run_dir']}")
    if len(completed) != len(trials):
        raise RuntimeError(f"{len(trials) - len(completed)} trials failed; inspect summary.xlsx and logs, then resume.")


if __name__ == "__main__":
    main()
