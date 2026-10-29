#!/usr/bin/env python3
"""Train a subject-independent KFall SCN using temporal fall annotations."""

from __future__ import annotations

import argparse
import hashlib
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
from torch.utils.data import DataLoader, Dataset, Subset

from loaders.kfall_event_loader import KFallEventLoaderConfig, load_or_build_kfall_event_splits
from loaders.utils import balanced_class_weights
from models.models import DualHeadSCN, SingleHeadSCN
from preprocess.utils import named_class_counts
from preprocess.kfall_labels import class_names, event_steps_for_samples, window_samples
from train_sisfall import run_epoch_dual
from train_ucihar import run_epoch as run_epoch_single


CLASS_NAMES = class_names(3)


def read_theta_file(path: str | Path) -> list[float]:
    """Read an optimizer/search result without silently replacing invalid values."""
    source = Path(path)
    try:
        document = json.loads(source.read_text())
    except FileNotFoundError as exc:
        raise ValueError(
            f"Threshold file not found: {source}. Current directory: {Path.cwd()}. "
            "Generated results/ files are not included in Git. From the repository root, "
            "use --theta-file configs/kfall_theta_200hz.json for the bundled calibration, "
            "or provide six values with --theta."
        ) from exc
    if not isinstance(document, dict):
        raise ValueError("theta-file must contain a JSON object with theta_best or best_theta.")
    if document.get("dataset", "kfall") != "kfall":
        raise ValueError("theta-file must belong to KFall; thresholds from other datasets can use different physical units.")
    values = document.get("theta_best", document.get("best_theta"))
    try:
        theta = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("theta-file must contain six positive numeric thresholds.") from exc
    if theta.shape != (6,) or not np.all(np.isfinite(theta)) or np.any(theta <= 0):
        raise ValueError("theta-file must contain six positive numeric thresholds.")
    return theta.tolist()


def training_subset_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Choose a deterministic class-stratified subset of training windows only."""
    labels = np.asarray(labels)
    if per_class < 0:
        raise ValueError("train-per-class must be nonnegative; 0 uses all training windows.")
    if per_class == 0:
        return np.arange(len(labels), dtype=np.int64)
    random = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        selected.extend(random.choice(candidates, min(per_class, len(candidates)), replace=False))
    return np.sort(np.asarray(selected, dtype=np.int64))


class KFallTrainingDataset(Dataset):
    """Read event windows on demand, retaining memory-mapped cache arrays."""

    def __init__(self, split: dict[str, np.ndarray], n_classes: int = 3) -> None:
        self.split = split
        n = len(split["y_h1"])
        if split["x_ev"].shape[0] != n or split["offset"].shape != (n, 6):
            raise ValueError("KFall event windows, offsets, and labels must align.")
        if split["x_ev"].ndim != 4 or split["x_ev"].shape[2:] != (6, 2):
            raise ValueError("KFall events must have shape (N, Te, 6, 2).")
        self.class_names = class_names(n_classes)
        if not np.isin(split["y_h1"], list(self.class_names)).all():
            raise ValueError(f"KFall labels must belong to {self.class_names}.")

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
    p.add_argument("--cache-name", default=None, help="Defaults to a name identifying duration, encoding, and label scheme.")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--preprocess-only", action="store_true", help="Build and inspect event splits without training.")
    p.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"), help="Fractions of subjects; every subject belongs to exactly one split.")
    duration = p.add_mutually_exclusive_group()
    duration.add_argument("--raw-win", type=int, default=None, help="Raw window samples at 100 Hz (default 205).")
    duration.add_argument("--window-seconds", type=float, default=None, help="Window duration, e.g. 2.05, 3.30, or 4.10 seconds.")
    p.add_argument("--out-win", type=int, default=None, help="Event steps; defaults to scaling the original 205-to-2048 ratio.")
    p.add_argument("--resample-hz", type=float, default=None, help="Optional analog interpolation rate, e.g. 200 Hz, before the event grid. Native annotations remain at 100 Hz.")
    p.add_argument("--stride", type=int, default=defaults.stride, help="Stride in raw samples.")
    p.add_argument("--endpoint-context", type=int, default=None, help="Minimum available context for shared comparison endpoints; must be >= raw-win.")
    p.add_argument("--post-impact-seconds", type=float, default=1.2, help="Postimpact label duration after impact; 0 preserves the legacy three-class task.")
    p.add_argument("--fall-policy", choices=["pre_impact", "annotated_interval"], default=defaults.fall_policy, help="pre_impact excludes the impact sample; annotated_interval can include it.")
    theta = p.add_mutually_exclusive_group()
    theta.add_argument("--theta", type=float, nargs=6, default=None, help="Encoder steps in g (first three) and deg/s (last three); tune using training subjects.")
    theta.add_argument("--theta-file", default=None, help="JSON result containing theta_best or best_theta from a threshold search.")
    p.add_argument("--dead-zone", type=float, default=defaults.dead_zone)
    p.add_argument("--sd-init", choices=["x0", "zero"], default=defaults.sd_init)

    p.add_argument("--head-mode", choices=["single", "dual"], default="single", help="Activity single head, or dual architecture with the unannotated posture head disabled.")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run", default="kfall_scn")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--validation-only", action="store_true", help="Train and evaluate the best checkpoint on validation; skip test inference during threshold selection.")
    p.add_argument("--train-per-class", type=int, default=0, help="Deterministically cap training windows per class for exploratory searches; 0 uses all train windows.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu"], default="auto", help="Force CPU or automatically use the selected GPU when available.")
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
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.workers < 0 or cfg.train_per_class < 0:
        p.error("epochs and batch-size must be positive; workers and train-per-class must be nonnegative.")
    try:
        cfg.theta = (read_theta_file(cfg.theta_file) if cfg.theta_file is not None else
                     list(defaults.theta_sd) if cfg.theta is None else cfg.theta)
        cfg.raw_win = (window_samples(cfg.window_seconds) if cfg.window_seconds is not None
                       else defaults.raw_win if cfg.raw_win is None else cfg.raw_win)
        cfg.window_seconds = cfg.raw_win / defaults.sample_rate_hz
        cfg.out_win = event_steps_for_samples(cfg.raw_win) if cfg.out_win is None else cfg.out_win
        KFallEventLoaderConfig(raw_win=cfg.raw_win, out_win=cfg.out_win, stride=cfg.stride,
                               post_impact_seconds=cfg.post_impact_seconds,
                               endpoint_context=cfg.endpoint_context, fall_policy=cfg.fall_policy,
                               theta_sd=tuple(cfg.theta), dead_zone=cfg.dead_zone,
                               resample_hz=cfg.resample_hz)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    if cfg.cache_name is None:
        post_tag = f"{cfg.post_impact_seconds:g}".replace(".", "p")
        cfg.cache_name = ("default" if cfg.post_impact_seconds == 0 and cfg.raw_win == 205 and cfg.out_win == 2048
                          and cfg.endpoint_context is None and cfg.resample_hz is None else
                          f"w{cfg.raw_win}_te{cfg.out_win}_post{post_tag}_ctx{cfg.endpoint_context or cfg.raw_win}_s{cfg.stride}")
        if cfg.resample_hz is not None:
            cfg.cache_name += f"_res{cfg.resample_hz:g}".replace(".", "p")
        if (tuple(cfg.theta) != defaults.theta_sd or cfg.dead_zone != defaults.dead_zone
                or cfg.sd_init != defaults.sd_init):
            encoding_tag = hashlib.sha256(json.dumps(
                [cfg.theta, cfg.dead_zone, cfg.sd_init], separators=(",", ":")
            ).encode()).hexdigest()[:12]
            cfg.cache_name += f"_enc{encoding_tag}"
    cfg.n_ch = 6
    cfg.dataset = "kfall"
    cfg.n_classes = 4 if cfg.post_impact_seconds > 0 else 3
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
        post_impact_seconds=getattr(cfg, "post_impact_seconds", 0.0),
        endpoint_context=getattr(cfg, "endpoint_context", None),
        resample_hz=getattr(cfg, "resample_hz", None),
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
    datasets = {name: KFallTrainingDataset(split, n_classes=cfg.n_classes) for name, split in splits.items()}
    train_indices = training_subset_indices(splits["train"]["y_h1"], getattr(cfg, "train_per_class", 0), cfg.seed)
    if getattr(cfg, "train_per_class", 0):
        datasets["train"] = Subset(datasets["train"], train_indices.tolist())
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
    return {"splits": splits, "metadata": metadata, "loaders": loaders,
            "training_indices": train_indices, "training_labels": splits["train"]["y_h1"][train_indices]}


def build_model(cfg: argparse.Namespace, dev: torch.device) -> torch.nn.Module:
    kwargs = scn_model_kwargs(cfg)
    if cfg.head_mode == "single":
        return SingleHeadSCN(n_classes=cfg.n_classes, **kwargs).to(dev)
    model = DualHeadSCN(n_classes_head1=cfg.n_classes, **kwargs).to(dev)
    posture_head = model.head2_spk if cfg.spiking_heads else model.head2
    for param in posture_head.parameters():
        param.requires_grad_(False)
    return model


def main(argv: list[str] | None = None) -> None:
    cfg = get_args(argv)
    set_seed(cfg.seed)
    dev = torch.device("cpu") if cfg.device == "cpu" else get_device(cfg.gpu)
    names = class_names(cfg.n_classes)
    data = build_data(cfg, dev)
    for name, split in data["splits"].items():
        subjects = data["metadata"].get("split_subjects", {}).get(name, [])
        print(f"[data:{name}] subjects={subjects} | N={len(split['y_h1'])} | counts={named_class_counts(split['y_h1'], names)}")
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
    np.save(run_dir / "training_indices.npy", data["training_indices"])
    save_json(run_dir / "training_subset.json", {
        "train_per_class": cfg.train_per_class, "seed": cfg.seed,
        "available_training_windows": len(data["splits"]["train"]["y_h1"]),
        "selected_training_windows": len(data["training_indices"]),
        "class_counts": named_class_counts(data["training_labels"], names),
        "train_subjects": data["metadata"]["split_subjects"]["train"],
    })

    class_weight = None
    if cfg.class_weight == "balanced":
        weights = balanced_class_weights(data["training_labels"], n_classes=cfg.n_classes)
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
    validation_metrics = epoch_runner(data["loaders"]["val"], collect_predictions=True)
    if cfg.head_mode == "dual":
        validation_true = validation_metrics.pop("y_true_h1")
        validation_pred = validation_metrics.pop("y_pred_h1")
        validation_metrics.pop("y_true_h2")
        validation_metrics.pop("y_pred_h2")
        validation_metrics["acc_h2"] = None
    else:
        validation_true = validation_metrics.pop("y_true")
        validation_pred = validation_metrics.pop("y_pred")
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    np.save(metrics_dir / "y_true_val.npy", validation_true)
    np.save(metrics_dir / "y_pred_val.npy", validation_pred)
    validation_metrics.update(
        acc=float(accuracy_score(validation_true, validation_pred)),
        macro_f1=float(f1_score(validation_true, validation_pred, labels=list(names), average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(validation_true, validation_pred)),
        best_val_loss=float(fit_result["best_val"]), best_epoch=int(checkpoint.get("epoch", -1)),
        class_counts_validation=named_class_counts(validation_true, names),
        class_counts_train=named_class_counts(data["training_labels"], names),
        total_time_sec=float(time.time() - fit_result["started_at"]),
    )
    save_json(metrics_dir / "validation_metrics.json", validation_metrics)
    if cfg.validation_only:
        print(json.dumps(json_ready(validation_metrics), indent=2, ensure_ascii=False))
        print(f"[done] saved_run={run_dir} | validation_only=True")
        return
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
    save_classification_outputs(metrics_dir, y_true=y_true, y_pred=y_pred, labels=list(names), names=list(names.values()))
    test_metrics.update(
        best_val_loss=float(fit_result["best_val"]),
        best_epoch=int(checkpoint.get("epoch", -1)),
        total_time_sec=float(time.time() - fit_result["started_at"]),
        class_counts_test=named_class_counts(y_true, names),
        head2_supervised=False,
    )
    save_json(metrics_dir / "test_metrics.json", test_metrics)
    print(json.dumps(json_ready(test_metrics), indent=2, ensure_ascii=False))
    print(f"[done] saved_run={run_dir}")


if __name__ == "__main__":
    main()
