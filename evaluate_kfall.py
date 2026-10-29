#!/usr/bin/env python3
"""Evaluate a saved KFall checkpoint on its original held-out test windows."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from evaluation.kfall_data import load_evaluation_data
from evaluation.kfall_reports import save_evaluation_reports
from preprocess.kfall_labels import class_names
from train.utils import extract_single_logits, make_autocast, set_seed
from train_kfall import KFallTrainingDataset, build_model


def _device(value: str | torch.device) -> torch.device:
    if str(value) == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda:<index>.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; select --device cpu.")
        if device.index is not None and not 0 <= device.index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {device}.")
    return device


def _subject(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.fullmatch(r"(?:SA|S)?(\d{1,2})", value.strip().upper())
    if match is None:
        raise ValueError("subject must be a KFall ID such as SA07.")
    return f"SA{int(match.group(1)):02d}"


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], SimpleNamespace]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("Checkpoint must contain model_state_dict and its training config.")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint has no model_state_dict.")
    cfg = SimpleNamespace(**checkpoint["config"])
    if getattr(cfg, "dataset", None) != "kfall" or getattr(cfg, "n_classes", None) not in {3, 4}:
        raise ValueError("Use a KFall checkpoint trained with three or four activity classes.")
    if getattr(cfg, "head_mode", None) not in {"single", "dual"}:
        raise ValueError("Checkpoint head_mode must be single or dual.")
    required = (
        "n_ch", "tau", "thresh", "hard_reset", "merge_polarities", "conv_ch",
        "kernels", "strides", "pool_kernels", "pool_strides", "pool_paddings",
        "offset_hidden", "spiking_heads", "head_rate_scale", "p_drop", "seed",
        "batch_size", "class_weight", "label_smoothing", "amp",
    )
    missing = [key for key in required if not hasattr(cfg, key)]
    if missing:
        raise ValueError(f"Checkpoint is missing training settings: {', '.join(missing)}.")
    if not isinstance(checkpoint.get("epoch"), int) or checkpoint["epoch"] < 1:
        raise ValueError("Checkpoint must identify its saved training epoch.")
    if not np.isfinite(float(checkpoint.get("best_val_loss", np.nan))):
        raise ValueError("Checkpoint must contain a finite best_val_loss.")
    return checkpoint, cfg


def evaluate_run(
    run_dir: str | Path, *, checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None, output_dir: str | Path | None = None,
    device: str | torch.device = "auto", batch_size: int | None = None,
    workers: int = 0, subject: str | None = None, window: int | None = None,
) -> dict[str, Any]:
    """Run inference only, preserving the saved test subjects and encoding.

    Loss uses the checkpoint's training class weights and the historical
    batch-mean aggregation. With the saved batch size it matches the training
    script's test loss; changing batch size can change weighted loss averaging.
    """
    started = time.perf_counter()
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training run not found: {run_dir}")
    checkpoint_path = (run_dir / "checkpoints" / "best_model.pt" if checkpoint is None
                       else Path(checkpoint).expanduser().resolve())
    saved, cfg = _load_checkpoint(checkpoint_path)
    names = class_names(cfg.n_classes)
    size = int(cfg.batch_size if batch_size is None else batch_size)
    if size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers must be non-negative.")
    dev = _device(device)
    subject = _subject(subject)
    split, metadata, run_metadata, resolved_cache = load_evaluation_data(
        run_dir, saved["config"], cache_dir=cache_dir,
    )
    indices = np.arange(len(split["y_h1"]), dtype=np.int64)
    if subject is not None:
        indices = indices[np.asarray(split["subject"])[indices] == subject]
        if not len(indices):
            raise ValueError(f"Subject {subject} has no windows in this run's held-out test partition.")
    if window is not None:
        if not isinstance(window, (int, np.integer)) or not 0 <= int(window) < len(split["y_h1"]):
            raise ValueError(f"window must be a test cache index in [0, {len(split['y_h1']) - 1}].")
        indices = indices[indices == int(window)]
        if not len(indices):
            raise ValueError(f"Test window {window} does not belong to subject {subject}.")
    if not len(indices):
        raise ValueError("The selected test partition has no windows.")

    destination = Path(output_dir).expanduser().resolve() if output_dir is not None else run_dir / "metrics" / "test_best"
    if output_dir is None and (subject is not None or window is not None):
        selection = "_".join(part for part in (f"subject_{subject}" if subject else "",
                                               f"window_{window}" if window is not None else "") if part)
        destination = destination / "subsets" / selection
    if destination.is_relative_to(resolved_cache):
        raise ValueError("Evaluation outputs must be stored outside the read-only event cache.")

    set_seed(cfg.seed)
    model = build_model(cfg, dev)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    class_weight = None
    if cfg.class_weight == "balanced":
        weight = saved.get("class_weight")
        if weight is None:
            raise ValueError("Checkpoint is missing its training class weights; they cannot be estimated from test.")
        class_weight = torch.as_tensor(weight, dtype=torch.float32, device=dev)
        if class_weight.shape != (cfg.n_classes,) or not torch.isfinite(class_weight).all() or torch.any(class_weight <= 0):
            raise ValueError(f"Checkpoint training class weights must contain {cfg.n_classes} finite positive values.")
    elif cfg.class_weight != "none":
        raise ValueError(f"Unknown checkpoint class weighting: {cfg.class_weight}.")

    dataset = KFallTrainingDataset(split, n_classes=cfg.n_classes)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=size, shuffle=False,
                        num_workers=int(workers), pin_memory=dev.type == "cuda")
    probabilities, targets = [], []
    loss_sum = 0.0
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            events = batch["x"].to(dev, non_blocking=True)
            offsets = batch["offset"].to(dev, non_blocking=True)
            target = batch["y"].to(dev, non_blocking=True)
            with make_autocast(bool(cfg.amp and dev.type == "cuda")):
                # extract_single_logits also selects Head 1 of the optional
                # dual architecture; KFall supplies no posture-head targets.
                logits = extract_single_logits(model(events, offset=offsets))
                if logits.shape != (len(target), cfg.n_classes) or not torch.isfinite(logits).all():
                    raise ValueError(f"The checkpoint produced invalid {cfg.n_classes}-class logits.")
                loss = F.cross_entropy(logits, target, weight=class_weight,
                                       label_smoothing=float(cfg.label_smoothing))
            loss_sum += float(loss.item()) * len(target)
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())
    inference_time = time.perf_counter() - inference_started
    y_true = np.concatenate(targets)
    probability = np.concatenate(probabilities)
    test_subjects = sorted(np.unique(np.asarray(split["subject"])[indices]).tolist())
    summary = {
        "dataset": "kfall", "split": "test", "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path), "best_epoch": saved["epoch"],
        "best_val_loss": float(saved["best_val_loss"]),
        "checkpoint_selection": "lowest_validation_loss" if checkpoint is None else "explicit_checkpoint",
        "cache_dir": str(resolved_cache), "device": str(dev), "batch_size": size,
        "workers": int(workers), "n": len(indices), "loss": loss_sum / len(indices),
        "loss_aggregation": "sample-count-weighted mean of batch cross-entropy means",
        "training_class_weights": None if class_weight is None else class_weight.detach().cpu().tolist(),
        "label_smoothing": float(cfg.label_smoothing), "head_mode": cfg.head_mode,
        "n_classes": cfg.n_classes, "class_names": list(names.values()),
        "head2_supervised": False, "fall_policy": cfg.fall_policy,
        "post_impact_seconds": float(getattr(cfg, "post_impact_seconds", 0.0)),
        "resample_hz": getattr(cfg, "resample_hz", None),
        "selection": "full_test" if subject is None and window is None else "test_subset",
        "test_subjects": test_subjects,
        "all_test_subjects": run_metadata["split_subjects"]["test"],
        "subject_filter": subject, "window_filter": window,
        "inference_time_sec": inference_time,
        "evaluation_time_sec": time.perf_counter() - started,
        "output_dir": str(destination),
    }
    if len(indices) == 1:
        prediction = int(probability[0].argmax())
        summary["selected_window"] = {
            "index": int(indices[0]), "subject": str(split["subject"][indices[0]]),
            "true": names[int(y_true[0])], "predicted": names[prediction],
            "probabilities": {name: float(probability[0, key]) for key, name in names.items()},
        }
    return save_evaluation_reports(destination, y_true=y_true, probabilities=probability,
                                   window_indices=indices, split=split, metadata=metadata, summary=summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run containing checkpoints/best_model.pt and dataset_metadata.json.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override the default best_model.pt checkpoint.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Override the saved event cache location; it must match the training snapshot.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to RUN/metrics/test_best, with separate subdirectories for subsets.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:<index>.")
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to the saved training batch size.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--subject", default=None, help="Evaluate only a held-out test subject, e.g. SA07.")
    parser.add_argument("--window", type=int, default=None, help="Evaluate a single actual test-cache window index (zero-based).")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metrics = evaluate_run(args.run_dir, checkpoint=args.checkpoint, cache_dir=args.cache_dir,
                               output_dir=args.output_dir, device=args.device, batch_size=args.batch_size,
                               workers=args.workers, subject=args.subject, window=args.window)
    except (ValueError, FileNotFoundError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    print(f"[checkpoint] {metrics['checkpoint']} | epoch={metrics['best_epoch']} | val_loss={metrics['best_val_loss']:.6f}")
    print(f"[test] n={metrics['n']} | subjects={metrics['test_subjects']} | accuracy={metrics['accuracy']:.2%} | loss={metrics['loss']:.6f}")
    if "selected_window" in metrics:
        print(json.dumps(metrics["selected_window"], indent=2, ensure_ascii=False))
    print(f"[saved] {metrics['output_dir']}")


if __name__ == "__main__":
    main()
