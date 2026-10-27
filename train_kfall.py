#!/usr/bin/env python3
"""Train a subject-independent KFall SCN using temporal fall annotations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from train.utils import (
    count_parameters,
    fit_model,
    format_dual_epoch_line,
    format_single_epoch_line,
    get_device,
    json_ready,
    make_grad_scaler,
    print_kv_block,
    save_classification_outputs,
    save_json,
    scn_model_kwargs,
    set_seed,
)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from loaders.kfall_event_loader import KFallEventLoaderConfig, load_or_build_kfall_event_splits
from loaders.utils import balanced_class_weights
from models.models import DualHeadSCN, SingleHeadSCN
from preprocess.utils import named_class_counts
from train_sisfall import run_epoch_dual
from train_ucihar import run_epoch as run_epoch_single


CLASS_NAMES = {0: "Fall", 1: "Dynamic", 2: "Static"}


class KFallTrainingDataset(Dataset):
    """Read event windows on demand, retaining memory-mapped cache arrays."""

    def __init__(self, split: dict[str, np.ndarray]) -> None:
        self.split = split
        n = len(split["y_h1"])
        if split["x_ev"].shape[0] != n or split["offset"].shape != (n, 6):
            raise ValueError("KFall event windows, offsets, and labels must align.")
        if split["x_ev"].ndim != 4 or split["x_ev"].shape[2:] != (6, 2):
            raise ValueError("KFall events must have shape (N, Te, 6, 2).")
        if not np.isin(split["y_h1"], list(CLASS_NAMES)).all():
            raise ValueError("KFall activity labels must be Fall=0, Dynamic=1, or Static=2.")

    def __len__(self) -> int:
        return len(self.split["y_h1"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        split = self.split
        return {
            "x": torch.tensor(np.asarray(split["x_ev"][index]), dtype=torch.float32),
            "offset": torch.tensor(np.asarray(split["offset"][index]), dtype=torch.float32),
            "y": torch.tensor(int(split["y_h1"][index]), dtype=torch.long),
            "y_h1": torch.tensor(int(split["y_h1"][index]), dtype=torch.long),
            "y_h2": torch.tensor(float(split["y_h2"][index]), dtype=torch.float32),
            "src_h2": torch.tensor(int(split["src_h2"][index]), dtype=torch.uint8),
            "w_h2": torch.tensor(float(split["w_h2"][index]), dtype=torch.float32),
        }


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = KFallEventLoaderConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="data/KFall", help="KFall root containing sensor_data and label_data.")
    p.add_argument("--label-dir", default=None, help="Override the directory containing subject XLSX annotations.")
    p.add_argument("--cache-dir", default="data/kfall_cache")
    p.add_argument("--cache-name", default="default")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--preprocess-only", action="store_true", help="Build and inspect event splits without training.")
    p.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"), help="Fractions of subjects; every subject belongs to exactly one split.")
    p.add_argument("--raw-win", type=int, default=defaults.raw_win, help="Raw window samples at 100 Hz (default 205).")
    p.add_argument("--out-win", type=int, default=defaults.out_win, help="Event time steps after interpolation (default 2048).")
    p.add_argument("--stride", type=int, default=defaults.stride, help="Stride in raw samples.")
    p.add_argument("--fall-policy", choices=["pre_impact", "annotated_interval"], default=defaults.fall_policy, help="pre_impact excludes the impact sample; annotated_interval can include it.")
    p.add_argument("--theta", type=float, nargs=6, default=list(defaults.theta_sd), help="Initial encoder thresholds in g (first three) and deg/s (last three); tune using training subjects.")
    p.add_argument("--dead-zone", type=float, default=defaults.dead_zone)
    p.add_argument("--sd-init", choices=["x0", "zero"], default=defaults.sd_init)

    p.add_argument("--head-mode", choices=["single", "dual"], default="single", help="Three-class single head, or dual architecture with the unannotated posture head disabled.")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run", default="kfall_scn")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--quiet-loader", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--plateau-patience", type=int, default=5)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")

    p.add_argument("--tau", type=float, default=0.75)
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--hard-reset", action="store_true")
    p.add_argument("--conv-ch", type=int, nargs=3, default=[32, 64, 64])
    p.add_argument("--kernels", type=int, nargs=3, default=[32, 32, 8])
    p.add_argument("--strides", type=int, nargs=3, default=[4, 2, 1])
    p.add_argument("--pool-kernels", type=int, nargs=3, default=[2, 2, 2])
    p.add_argument("--pool-strides", type=int, nargs=3, default=[2, 2, 2])
    p.add_argument("--pool-paddings", type=int, nargs=3, default=[0, 0, 0])
    p.add_argument("--p-drop", type=float, default=0.35)
    p.add_argument("--merge-polarities", action="store_true")
    p.add_argument("--offset-hidden", type=int, default=8)
    p.add_argument("--head-rate-scale", type=float, default=8.0)
    p.add_argument("--spiking-heads", action=argparse.BooleanOptionalAction, default=False)
    cfg = p.parse_args(argv)
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.workers < 0:
        p.error("epochs and batch-size must be positive; workers must be nonnegative.")
    cfg.n_ch = 6
    cfg.dataset = "kfall"
    cfg.n_classes = 3
    cfg.lambda_h1 = 1.0
    cfg.lambda_h2 = 0.0
    cfg.use_ambiguous = False
    cfg.manual_weight = 1.0
    cfg.autotag_weight = 0.0
    return cfg


def build_data(cfg: argparse.Namespace, dev: torch.device) -> dict[str, Any]:
    loader_cfg = KFallEventLoaderConfig(
        seed=cfg.seed,
        split=tuple(cfg.split),
        raw_win=cfg.raw_win,
        out_win=cfg.out_win,
        stride=cfg.stride,
        theta_sd=tuple(cfg.theta),
        dead_zone=cfg.dead_zone,
        sd_init=cfg.sd_init,
        fall_policy=cfg.fall_policy,
    )
    splits, metadata = load_or_build_kfall_event_splits(
        dataset_root=cfg.dataset_root,
        label_dir=cfg.label_dir,
        cache_dir=cfg.cache_dir,
        run_name=cfg.cache_name,
        cfg=loader_cfg,
        refresh=cfg.refresh_cache,
        mmap_mode="r",
        verbose=not cfg.quiet_loader,
    )
    if any(len(split["y_h1"]) == 0 for split in splits.values()):
        raise ValueError("Each training, validation, and test split must contain windows.")
    datasets = {name: KFallTrainingDataset(split) for name, split in splits.items()}
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=name == "train",
            num_workers=cfg.workers,
            pin_memory=not cfg.no_pin_memory and dev.type == "cuda",
        )
        for name, dataset in datasets.items()
    }
    return {"splits": splits, "metadata": metadata, "loaders": loaders}


def build_model(cfg: argparse.Namespace, dev: torch.device) -> torch.nn.Module:
    kwargs = scn_model_kwargs(cfg)
    if cfg.head_mode == "single":
        return SingleHeadSCN(n_classes=3, **kwargs).to(dev)
    model = DualHeadSCN(n_classes_head1=3, **kwargs).to(dev)
    posture_head = model.head2_spk if cfg.spiking_heads else model.head2
    for param in posture_head.parameters():
        param.requires_grad_(False)
    return model


def main(argv: list[str] | None = None) -> None:
    cfg = get_args(argv)
    set_seed(cfg.seed)
    dev = get_device(cfg.gpu)
    data = build_data(cfg, dev)
    for name, split in data["splits"].items():
        subjects = data["metadata"].get("split_subjects", {}).get(name, [])
        print(f"[data:{name}] subjects={subjects} | N={len(split['y_h1'])} | counts={named_class_counts(split['y_h1'], CLASS_NAMES)}")
    if cfg.preprocess_only:
        cache_path = Path(cfg.cache_dir) / f"kfall_event_loader_{cfg.cache_name}"
        print(f"[done] cache={cache_path} | metadata={cache_path / 'metadata.json'}")
        return

    requested_run = cfg.run
    run_stem = requested_run if "kfall" in requested_run.lower() else f"{requested_run}_kfall"
    base_dir = Path(cfg.out_dir) / f"{run_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = base_dir
    suffix = 0
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
            run_dir = Path(f"{base_dir}_{suffix}")
    cfg.requested_run = requested_run
    cfg.run = run_dir.name
    metrics_dir = run_dir / "metrics"
    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    save_json(run_dir / "config.json", cfg)
    save_json(run_dir / "dataset_metadata.json", data["metadata"])

    class_weight = None
    if cfg.class_weight == "balanced":
        weights = balanced_class_weights(data["splits"]["train"]["y_h1"], n_classes=3)
        class_weight = torch.as_tensor(weights, dtype=torch.float32, device=dev)
    model = build_model(cfg, dev)
    total, trainable = count_parameters(model)
    print_kv_block("run", [("dataset", "KFall"), ("device", dev), ("run_dir", run_dir), ("head_mode", cfg.head_mode), ("fall_policy", cfg.fall_policy), ("params", total), ("trainable", trainable)])
    if cfg.head_mode == "dual":
        print("[head2] No temporal posture labels are available in KFall; posture loss and head parameters are disabled.")

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.plateau_patience, min_lr=cfg.min_lr)
    scaler = make_grad_scaler(enabled=bool(cfg.amp and dev.type == "cuda"))

    def epoch_runner(loader, optimizer=None, scaler=None, collect_predictions=False):
        kwargs = dict(optimizer=optimizer, scaler=scaler, class_weight=class_weight, collect_predictions=collect_predictions)
        if cfg.head_mode == "dual":
            return run_epoch_dual(model, loader, dev, cfg, pos_weight_h2=1.0, **kwargs)
        return run_epoch_single(model, loader, dev, cfg, **kwargs)

    fit_result = fit_model(
        model=model,
        train_loader=data["loaders"]["train"],
        val_loader=data["loaders"]["val"],
        cfg=cfg,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        metrics_dir=metrics_dir,
        ckpt_path=ckpt_path,
        run_epoch_fn=epoch_runner,
        format_epoch_fn=format_dual_epoch_line if cfg.head_mode == "dual" else format_single_epoch_line,
        checkpoint_extra_fn=lambda: {"class_weight": None if class_weight is None else class_weight.detach().cpu()},
    )
    if not ckpt_path.is_file():
        raise RuntimeError("No checkpoint was saved. Check training/validation loss values.")
    checkpoint = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = epoch_runner(data["loaders"]["test"], collect_predictions=True)
    if cfg.head_mode == "dual":
        y_true = test_metrics.pop("y_true_h1")
        y_pred = test_metrics.pop("y_pred_h1")
        test_metrics.pop("y_true_h2")
        test_metrics.pop("y_pred_h2")
        test_metrics["acc_h2"] = None
    else:
        y_true = test_metrics.pop("y_true")
        y_pred = test_metrics.pop("y_pred")
    save_classification_outputs(metrics_dir, y_true=y_true, y_pred=y_pred, labels=list(CLASS_NAMES), names=list(CLASS_NAMES.values()))
    test_metrics.update(
        best_val_loss=float(fit_result["best_val"]),
        best_epoch=int(checkpoint.get("epoch", -1)),
        total_time_sec=float(time.time() - fit_result["started_at"]),
        class_counts_test=named_class_counts(y_true, CLASS_NAMES),
        head2_supervised=False,
    )
    save_json(metrics_dir / "test_metrics.json", test_metrics)
    print(json.dumps(json_ready(test_metrics), indent=2, ensure_ascii=False))
    print(f"[done] saved_run={run_dir}")


if __name__ == "__main__":
    main()
