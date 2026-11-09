#!/usr/bin/env python3
"""Evaluate a frozen KFall model on UCI-HAR normal activities (no falls)."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluate_kfall import _device, _load_checkpoint
from evaluation.kfall_reports import save_evaluation_reports
from loaders.kfall_event_loader import CLASS_NAMES, KFallEventLoaderConfig
from preprocess.event_encoding import sigma_delta
from preprocess.interpolate_signal import interpolate_to_times
from preprocess.preprocess_ucihar import ACTIVITY_LABELS, load_ucihar_split
from train.utils import extract_single_logits, make_autocast, set_seed
from train_kfall import build_model


class UCIKFallDataset(Dataset):
    """Keep each 50 Hz UCI segment separate and use the saved KFall encoder."""

    def __init__(self, signals, activities, subjects, cfg: KFallEventLoaderConfig):
        signals = np.asarray(signals, dtype=np.float32)
        activities, subjects = np.asarray(activities), np.asarray(subjects)
        if signals.ndim != 3 or signals.shape[1:] != (128, 6) or not np.isfinite(signals).all():
            raise ValueError("UCI signals must have finite shape (N, 128, 6).")
        if (activities.shape != (len(signals),) or subjects.shape != activities.shape
                or activities.dtype.kind not in "iu" or subjects.dtype.kind not in "iu"
                or not np.isin(activities, range(6)).all() or not np.isin(subjects, range(30)).all()):
            raise ValueError("UCI activities/subjects must align and use zero-based IDs.")
        if not len(signals):
            raise ValueError("No UCI-HAR segments selected.")
        # Cover the original KFall endpoint without extrapolation at 50 Hz.
        self.native_length = int(np.ceil((cfg.raw_win - 1) / 2)) + 1
        if self.native_length > 128:
            raise ValueError(
                f"The checkpoint requires {cfg.raw_win / 100:.2f} s windows; UCI-HAR "
                "only supplies 128-sample (2.56 s) segments. No padding or concatenation is allowed."
            )
        self.cfg = cfg
        # UCI loader: gyro in rad/s first, total acceleration in g last.
        self.signals = signals[:, :, [3, 4, 5, 0, 1, 2]].copy()
        self.signals[:, :, 3:] *= 180.0 / np.pi
        self.activities = activities
        self.subjects = subjects
        self.windows = [(i, start) for i in range(len(signals))
                        for start in range(0, 129 - self.native_length, self.native_length)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        segment, start = self.windows[index]
        signal = self.signals[segment, start:start + self.native_length]
        raw = interpolate_to_times(
            signal, np.arange(self.native_length) / 50,
            np.arange(self.cfg.raw_win) / 100,
        )
        interpolated = interpolate_to_times(
            raw, np.arange(self.cfg.raw_win) / 100,
            np.linspace(0, (self.cfg.raw_win - 1) / 100, self.cfg.out_win),
        )
        events, _, _ = sigma_delta(
            interpolated, theta=np.asarray(self.cfg.theta_sd, dtype=np.float32),
            init=self.cfg.sd_init, dead_zone=self.cfg.dead_zone, return_reconstruction=False,
        )
        return {"x": torch.from_numpy(events), "offset": torch.from_numpy(raw[0].copy())}


def evaluate_ucihar(
    run_dir: str | Path, *, dataset_root: str | Path = "data/UCI HAR Dataset",
    split: str = "test", output_dir: str | Path | None = None,
    device: str = "auto", batch_size: int = 128, workers: int = 0,
) -> dict[str, Any]:
    if split not in {"test", "train", "all"}:
        raise ValueError("split must be test, train, or all.")
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative.")
    run_dir = Path(run_dir).expanduser().resolve()
    saved, model_cfg = _load_checkpoint(run_dir / "checkpoints" / "best_model.pt")
    if getattr(model_cfg, "architecture", "scn") == "compact_recurrent":
        raise ValueError("This external evaluator is window-independent; recurrent models require ordered recordings.")
    allowed = {field.name for field in fields(KFallEventLoaderConfig)}
    encoder_args = {key: value for key, value in saved["config"].items() if key in allowed}
    encoder_args["theta_sd"] = tuple(model_cfg.theta)
    cfg = KFallEventLoaderConfig(**encoder_args)
    # Fail before reading signals if the model's context cannot be supplied.
    if cfg.raw_win > 255:
        raise ValueError("UCI-HAR segments cannot supply this checkpoint's temporal context (maximum 2.56 s).")
    names = ("train", "test") if split == "all" else (split,)
    loaded = [load_ucihar_split(dataset_root, name) for name in names]
    signals, activities, subjects = (
        np.concatenate([part[i] for part in loaded]) for i in range(3)
    )
    dataset = UCIKFallDataset(signals, activities, subjects, cfg)
    destination = (run_dir / "metrics" / f"ucihar_{split}" if output_dir is None
                   else Path(output_dir).expanduser().resolve())
    root = Path(dataset_root).expanduser().resolve()
    if destination.is_relative_to(root):
        raise ValueError("Reports must be saved outside the source UCI dataset.")
    dev = _device(device)
    set_seed(model_cfg.seed)
    model = build_model(model_cfg, dev)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        shuffle=False, pin_memory=dev.type == "cuda")
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            with make_autocast(bool(model_cfg.amp and dev.type == "cuda")):
                logits = extract_single_logits(model(
                    batch["x"].to(dev, non_blocking=True).float(),
                    offset=batch["offset"].to(dev, non_blocking=True),
                ))
            if logits.shape != (len(batch["x"]), model_cfg.n_classes) or not torch.isfinite(logits).all():
                raise ValueError("Checkpoint produced invalid UCI-HAR logits.")
            chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probabilities = np.concatenate(chunks)
    segments = np.asarray([i for i, _ in dataset.windows])
    starts = np.asarray([start for _, start in dataset.windows])
    # WALKING/upstairs/downstairs -> Dynamic; sitting/standing/laying -> Static.
    truth = np.where(activities[segments] < 3, 1, 2).astype(np.int64)
    identities = np.asarray([f"UCI{subject + 1:02d}" for subject in subjects])
    records = []
    for name, part in zip(names, loaded):
        for row in range(len(part[1])):
            i = len(records)
            records.append({
                "file": str(root / name / f"y_{name}.txt"),
                "subject": identities[i], "code": ACTIVITY_LABELS[int(activities[i])],
                "trial": f"row_{row + 1:05d}",
            })
    arrays = {"y_h1": truth, "subject": identities[segments], "file_idx": segments,
              "start": starts, "end_frame": starts + dataset.native_length}
    predicted = probabilities.argmax(axis=1)
    per_activity = {}
    for activity, name in ACTIVITY_LABELS.items():
        selected = activities[segments] == activity
        count = int(selected.sum())
        per_activity[name] = {
            "n": count,
            "accuracy": float(np.mean(predicted[selected] == truth[selected])) if count else None,
            "false_fall_windows": int(np.sum(predicted[selected] == 0)),
            "false_fall_rate": float(np.mean(predicted[selected] == 0)) if count else None,
        }
    summary = {
        "dataset": "ucihar", "split": split, "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "checkpoints" / "best_model.pt"),
        "best_epoch": saved["epoch"], "device": str(dev),
        "output_dir": str(destination), "class_names": [CLASS_NAMES[i] for i in range(model_cfg.n_classes)],
        "n_source_segments": len(signals), "per_source_activity": per_activity,
        "test_subjects": sorted(set(identities.tolist())),
        "label_mapping": "UCI 1..3 -> Dynamic; UCI 4..6 -> Static; no Fall or PostImpact targets",
        "encoder_config": encoder_args, "source_sample_rate_hz": 50,
        "source_stride_samples": dataset.native_length,
        "unit_conversion": "total_acc in g; body_gyro rad/s -> degrees/s; acceleration before gyroscope",
        "evaluation_scope": "external_negative_only_window_classification",
        "timing_note": "start/end_frame refer to the 50 Hz UCI segment; no fall impact or warning time exists.",
        "negative_only_note": "Fall recall/sensitivity cannot be measured without falls. "
                              "False-positive rate is per window, not alerts/hour. "
                              "UCI source segments overlap; windows are not independent.",
        "domain_shift_note": "Different sensor, placement and preprocessing; external evaluation, not KFall test accuracy.",
    }
    return save_evaluation_reports(
        destination, y_true=truth, probabilities=probabilities,
        window_indices=np.arange(len(truth)), split=arrays,
        metadata={"recordings": records}, summary=summary,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/UCI HAR Dataset"))
    parser.add_argument("--split", choices=("test", "train", "all"), default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        metrics = evaluate_ucihar(**vars(args))
    except (ValueError, FileNotFoundError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: metrics[key] for key in (
        "accuracy", "balanced_accuracy", "macro_f1", "fall_window_metrics", "output_dir",
    )}, indent=2))


if __name__ == "__main__":
    main()
