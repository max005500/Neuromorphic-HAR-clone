#!/usr/bin/env python3
"""Compare matched CNN/SCN baselines and compact KFall models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from evaluate_kfall import evaluate_run
from train_kfall import get_args, main as train


COMPACT = ("--conv-depth", "2", "--conv-ch", "8", "8", "8", "--kernels", "8", "5", "8")
VARIANTS = {
    "cnn": ("--activation", "relu", "--conv-depth", "3", "--gru-hidden", "0"),
    "scn": ("--activation", "lif", "--conv-depth", "3", "--gru-hidden", "0"),
    "compact": (*COMPACT, "--activation", "lif", "--gru-hidden", "0"),
    "compact_gru": (*COMPACT, "--activation", "lif", "--gru-hidden", "4"),
    "compact_srnn": (*COMPACT, "--activation", "lif", "--gru-hidden", "0", "--srnn-hidden", "4"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for all runs and summary.csv.")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate all frozen checkpoints after training.")
    parser.add_argument("training_args", nargs=argparse.REMAINDER, help="Shared train_kfall.py arguments after --.")
    args = parser.parse_args(argv)
    shared = args.training_args
    if shared[:1] == ["--"]:
        shared = shared[1:]
    cfg = get_args(shared)
    if cfg.preprocess_only or cfg.refresh_cache:
        parser.error("Prepare or refresh the shared cache separately before comparing architectures.")
    output = args.output_dir.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(str(exc))

    rows = []
    for variant, options in VARIANTS.items():
        run = train([*shared, "--srnn-hidden", "0", *options, "--head-mode", "single", "--no-spiking-heads",
                     "--skip-test", "--out-dir", str(output / "runs"), "--run", f"kfall_{variant}"])
        if run is None:
            raise RuntimeError(f"No training run saved for {variant}.")
        selection = json.loads((run / "metrics" / "validation_selection.json").read_text())
        history = json.loads((run / "metrics" / "history.json").read_text())
        best = next(row for row in history if row["epoch"] == selection["best_epoch"])
        rows.append({"variant": variant, "parameters": selection["parameters"], "best_epoch": best["epoch"],
                     "val_loss": best["val"]["loss"], "val_accuracy": best["val"]["acc"],
                     "run_dir": str(run)})

    if args.evaluate_test:
        device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
        for row in rows:
            result = evaluate_run(row["run_dir"], device=device, workers=cfg.workers)
            row.update(test_accuracy=result["accuracy"], test_balanced_accuracy=result["balanced_accuracy"],
                       test_macro_f1=result["macro_f1"], test_fall_recall=result["fall_window_metrics"]["recall"])

    summary = output / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[comparison] {summary}")


if __name__ == "__main__":
    main()
