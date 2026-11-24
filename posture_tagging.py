"""Command-line entry point for SisFall posture refinement tagging.

Examples:
    python posture_tagging.py
    python posture_tagging.py --train-subjects SA01 SA02 SE01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labeling import sisfall_posture_tagging as tagging
from labeling.sisfall_posture_tagging import TaggingConfig
from preprocess.utils import load_json

CALIBRATION_VERSION = 3


def posture_input_fingerprint(dataset_root: Path, label_dir: Path) -> str:
    """Invalidate artifacts when raw inputs or manual annotations change."""
    digest = hashlib.sha256()
    for root, suffix in ((dataset_root, ".txt"), (label_dir, ".json")):
        for path in sorted(path for path in root.rglob("*") if path.suffix.lower() == suffix):
            stat = path.stat()
            digest.update(str(path.relative_to(root)).encode())
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
            if suffix == ".json":
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _training_subjects(train_subjects: Iterable[str]) -> list[str]:
    if isinstance(train_subjects, (str, bytes)):
        raise ValueError("train_subjects must be a non-empty collection of subject IDs.")
    subjects = sorted({str(subject) for subject in train_subjects})
    if not subjects:
        raise ValueError("train_subjects must not be empty.")
    return subjects


def _fixed_predictions(
    record: dict[str, Any], cfg: TaggingConfig, tau: float, margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt = np.asarray(record["p_postural_transition"])
    sp = np.asarray(record["p_stable_posture"])
    if cfg.use_hyst:
        hp = tagging._hyst(pt, tau, 1.0 - tau)
        hs = tagging._hyst(sp, tau, 1.0 - tau)
    else:
        hp = (pt >= tau).astype(np.uint8)
        hs = (sp >= tau).astype(np.uint8)
    pred_pt = (pt >= tau) & ((pt - sp) >= margin)
    pred_sp = (sp >= tau) & ((sp - pt) >= margin)
    if cfg.use_hyst:
        pred_pt |= (hp > 0) & (hs == 0) & ((pt - sp) >= margin)
        pred_sp |= (hs > 0) & (hp == 0) & ((sp - pt) >= margin)
    yw = np.full(pt.shape, 0.5, np.float32)
    yw[pred_pt & ~pred_sp] = 1.0
    yw[pred_sp & ~pred_pt] = 0.0
    yw[(pt <= cfg.amb_ceil) & (sp <= cfg.amb_ceil)] = 0.5
    return yw, hp, hs


def calibrate_and_tag_postures(
    x: np.ndarray,
    y: np.ndarray,
    meta: list[dict[str, Any]],
    labels: dict[str, Any],
    cfg: TaggingConfig,
    *,
    train_subjects: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Fit normalization/PCA/classifiers and tau/m using TRAIN gold only.

    All recordings are then tagged using the frozen model and thresholds.
    No fallback to config defaults is considered a successful calibration.
    Subject IDs are preserved verbatim (in particular, SE is not mapped to SA).
    """
    subjects = _training_subjects(train_subjects)
    available = {str(record["subject"]) for record in meta}
    if not set(subjects) <= available:
        raise ValueError("train_subjects contains subjects absent from the recordings.")
    train_files = {
        tagging._meta_key(record)
        for record, target in zip(meta, y)
        if str(record["subject"]) in subjects and int(target) == 2
    }
    gold = {name: value for name, value in labels.items() if name in train_files}
    if not gold:
        raise RuntimeError("No TRAIN manual gold is available for posture calibration.")

    # The existing fitter learns scaling, PCA and both classifiers from labels.
    records, model = tagging.train_and_tag_postures(x, y, meta, gold, cfg)
    manual, _ = tagging.build_manual_posture_records(x, y, meta, labels, cfg)
    gold_by_file = {
        tagging._meta_key(record): {
            int(start): float(target)
            for start, target in zip(record["starts"], record["y_win"])
            if float(target) != 0.5
        }
        for record in manual
        if str(record["subject"]) in subjects
    }
    matched = []
    targets = []
    for record in records:
        if str(record["subject"]) not in subjects:
            continue
        by_start = gold_by_file.get(tagging._meta_key(record), {})
        indices = [i for i, start in enumerate(record["starts"]) if int(start) in by_start]
        if indices:
            matched.append((record, indices))
            targets.extend(by_start[int(record["starts"][i])] for i in indices)
    target = np.asarray(targets)
    if set(target.tolist()) != {0.0, 1.0}:
        raise RuntimeError("TRAIN gold must contain clear stable and transition windows.")

    best_score = -1.0
    best = None
    # Ambiguous predictions count as errors, preventing an all-abstaining optimum.
    for tau in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        for margin in (0.05, 0.10, 0.15, 0.20, 0.25):
            prediction = np.concatenate([
                _fixed_predictions(record, cfg, tau, margin)[0][indices]
                for record, indices in matched
            ])
            score = float(np.mean([
                np.mean(prediction[target == cls] == cls) for cls in (0.0, 1.0)
            ]))
            if score > best_score:
                best_score, best = score, (tau, margin)
    tau, margin = best
    for record in records:
        yw, hp, hs = _fixed_predictions(record, cfg, tau, margin)
        record.update(y_win=yw, y_hat_postural_transition=hp, y_hat_stable_posture=hs)
    calibration = {
        "version": CALIBRATION_VERSION,
        "strategy": "train_gold_balanced_accuracy_grid",
        "training_subjects": subjects,
        "gold_files": sorted(gold),
        "gold_windows": int(target.size),
        "selection_split": "train",
        "search_grid": {
            "tau": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
            "m": [0.05, 0.10, 0.15, 0.20, 0.25],
        },
        "parameters": {
            "tau": tau, "m": margin, "th_pt": tau, "th_sp": tau,
            "margin": margin, "amb_ceil": cfg.amb_ceil,
            "use_hyst": cfg.use_hyst, "hysteresis_on": tau,
            "hysteresis_off": 1.0 - tau,
        },
        "objective": "balanced_accuracy_with_abstentions_as_errors",
        "training_score": best_score,
    }
    return records, manual, model, calibration


def run_posture_tagging(
    dataset_root: str | Path = "data/SisFall_dataset",
    label_dir: str | Path = "data/labels_transitions",
    result_dir: str | Path = "data/sisfall_tagging_results",
    cache_dir: str | Path = "data/sisfall_cache",
    cfg: TaggingConfig | None = None,
    run_name: str = "default",
    refresh: bool = False,
    refresh_raw: bool = False,
    save_plot: bool = False,
    verbose: bool = True,
    *,
    train_subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Backward-compatible standalone API; event pipelines must pass TRAIN IDs.

    Train-only artifacts use a separate versioned namespace and are rebuilt
    rather than loading potentially globally calibrated legacy posture caches.
    """
    if train_subjects is None:
        return tagging.run_posture_tagging(
            dataset_root, label_dir, result_dir, cache_dir, cfg, run_name,
            refresh, refresh_raw, save_plot, verbose,
        )
    subjects = _training_subjects(train_subjects)
    cfg = TaggingConfig() if cfg is None else cfg
    x, y, meta, skipped = tagging._load_dataset(
        dataset_root, cfg.acc, cfg.dtype, cache_dir, refresh_raw, verbose
    )
    labels = tagging._load_labels(label_dir)
    records, manual, model, calibration = calibrate_and_tag_postures(
        x, y, meta, labels, cfg, train_subjects=subjects
    )
    identity = {
        "version": CALIBRATION_VERSION, "config": asdict(cfg),
        "training_subjects": subjects,
        "dataset_root": str(dataset_root), "label_dir": str(label_dir),
        "input_fingerprint": posture_input_fingerprint(Path(dataset_root), Path(label_dir)),
    }
    cache_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    stem = f"posture_train_only_v{CALIBRATION_VERSION}_{run_name}_{cache_id}"
    cache_root, output_root = Path(cache_dir) / stem, Path(result_dir) / stem
    summary = {
        **identity, "cache_id": cache_id, "run_name": run_name,
        "calibration": calibration, "training_subjects": subjects,
        "raw_data": {"recordings": len(x), "skipped": int(skipped)},
        "autotagged_records": tagging.summarize_records(records),
        "manual_records": tagging.summarize_records(manual),
        "model": {key: value for key, value in model.items() if not isinstance(value, np.ndarray)},
    }
    for root in (cache_root, output_root):
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "records.npy", np.asarray(records, dtype=object), allow_pickle=True)
        np.save(root / "manual_records.npy", np.asarray(manual, dtype=object), allow_pickle=True)
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        cache_root / "model.npz",
        **{key: np.asarray(value) for key, value in model.items()
           if isinstance(value, np.ndarray) or key.startswith("b_")},
    )
    if save_plot:
        tagging._save_probability_plot(
            records, x, meta, labels, cfg, output_root / "clear_window_probabilities.png"
        )
    return {
        "records": records, "manual_records": manual, "model": model, "summary": summary,
        "paths": {
            "records": str(output_root / "records.npy"),
            "manual_records": str(output_root / "manual_records.npy"),
            "model_cache": str(cache_root / "model.npz"),
            "summary": str(output_root / "summary.json"),
            "plot": str(output_root / "clear_window_probabilities.png") if save_plot else None,
            "output_dir": str(output_root), "cache_dir": str(cache_root),
        },
    }


def _path_value(cfg: dict[str, Any], key: str, default: str, override: str | None) -> str:
    if override is not None:
        return override
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    return str(cfg.get(key, paths.get(key, default)))


def _build_config(args: argparse.Namespace, cfg_json: dict[str, Any]) -> TaggingConfig:
    vals = asdict(TaggingConfig())
    vals.update({k: v for k, v in cfg_json.items() if k in vals})
    vals.update({k: v for k, v in cfg_json.get("posture_tagging", {}).items() if k in vals})

    for k in vals:
        v = getattr(args, k, None)
        if v is not None:
            vals[k] = v

    return TaggingConfig(**vals)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SisFall window-level stable-posture and postural-transition autotagging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Optional JSON config file. Values under 'posture_tagging' are supported.")
    p.add_argument("--dataset-root", default=None, help="Root directory containing SisFall TXT files.")
    p.add_argument("--label-dir", default=None, help="Directory containing manual posture-label JSON files.")
    p.add_argument("--result-dir", default=None, help="Directory where posture-tagging outputs are written.")
    p.add_argument("--cache-dir", default=None, help="Directory used for reusable posture-tagging artifacts.")
    p.add_argument("--run-name", default=None, help="Name used to group output and cache artifacts.")
    p.add_argument("--refresh", action="store_true", help="Rebuild posture-tagging artifacts even when a valid cache exists.")
    p.add_argument("--refresh-raw", action="store_true", help="Ask the raw SisFall loader to rebuild its own cache when supported.")
    p.add_argument("--quiet", action="store_true", help="Disable high-level progress messages.")
    p.add_argument("--save-plot", action="store_true", help="Save a probability scatter plot over clear manual windows.")
    p.add_argument("--train-subjects", nargs="+", default=None, help="Explicit TRAIN subject IDs for leakage-free fitting and tau/m calibration.")

    p.add_argument("--fs", type=float, default=None, help="Sensor sampling rate in Hz.")
    p.add_argument("--win", type=int, default=None, help="Window length in samples.")
    p.add_argument("--hop", type=int, default=None, help="Distance between consecutive windows in samples.")
    p.add_argument("--pad-hop", dest="pad_hop", type=int, default=None, help="Padding step used to provide context near signal boundaries.")
    p.add_argument("--pad-rep", dest="pad_rep", type=int, default=None, help="Number of repeated padding blocks at each boundary.")
    p.add_argument("--pca-k", dest="pca_k", type=int, default=None, help="Number of PCA components appended to normalized features; use 0 to disable PCA features.")
    p.add_argument("--ds-grav", dest="ds_grav", type=int, default=None, help="Downsampling factor for gravity direction variation features.")

    p.add_argument("--lr", type=float, default=None, help="Learning rate for the internal logistic-regression classifiers.")
    p.add_argument("--steps", type=int, default=None, help="Number of optimization steps for each binary classifier.")
    p.add_argument("--l2", type=float, default=None, help="L2 regularization strength for logistic-regression weights.")
    p.add_argument("--smooth-k", dest="smooth_k", type=int, default=None, help="Moving average size applied to output probabilities.")

    p.add_argument("--th-pt", dest="th_pt", type=float, default=None, help="Probability threshold for postural transition detection.")
    p.add_argument("--th-sp", dest="th_sp", type=float, default=None, help="Probability threshold for stable posture detection.")
    p.add_argument("--margin", type=float, default=None, help="Minimum probability gap between both posture classes.")
    p.add_argument("--amb-ceil", dest="amb_ceil", type=float, default=None, help="Maximum probability for both classes before forcing ambiguity.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-hyst", dest="use_hyst", action="store_true", default=None, help="Enable hysteresis over class probabilities.")
    g.add_argument("--no-hyst", dest="use_hyst", action="store_false", default=None, help="Disable hysteresis over class probabilities.")
    p.add_argument("--th-pt-on", dest="th_pt_on", type=float, default=None, help="Hysteresis activation threshold for postural transition.")
    p.add_argument("--th-pt-off", dest="th_pt_off", type=float, default=None, help="Hysteresis deactivation threshold for postural transition.")
    p.add_argument("--th-sp-on", dest="th_sp_on", type=float, default=None, help="Hysteresis activation threshold for stable posture.")
    p.add_argument("--th-sp-off", dest="th_sp_off", type=float, default=None, help="Hysteresis deactivation threshold for stable posture.")

    p.add_argument("--ywin-mode", dest="ywin_mode", choices=["mean", "center", "max", "q"], default=None, help="Manual-label aggregation mode for window targets.")
    p.add_argument("--ywin-q", dest="ywin_q", type=float, default=None, help="Quantile used when ywin-mode is q.")
    p.add_argument("--sigma-frac", dest="sigma_frac", type=float, default=None, help="Gaussian width factor for manual postural-transition intervals.")

    p.add_argument("--acc", choices=["ADXL345", "MMA8451Q"], default=None, help="Accelerometer source used from each SisFall sample.")
    p.add_argument("--dtype", default=None, help="Numeric dtype used when loading raw SisFall values.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_json = {} if args.config is None else load_json(args.config)
    cfg = _build_config(args, cfg_json)

    dataset_root = _path_value(cfg_json, "dataset_root", "data/SisFall_dataset", args.dataset_root)
    label_dir = _path_value(cfg_json, "label_dir", "data/labels_transitions", args.label_dir)
    result_dir = _path_value(cfg_json, "result_dir", "data/sisfall_tagging_results", args.result_dir)
    cache_dir = _path_value(cfg_json, "cache_dir", "data/sisfall_cache", args.cache_dir)
    run_name = args.run_name or str(cfg_json.get("run_name", "default"))

    run_posture_tagging(
        dataset_root=dataset_root,
        label_dir=label_dir,
        result_dir=result_dir,
        cache_dir=cache_dir,
        cfg=cfg,
        run_name=run_name,
        refresh=args.refresh,
        refresh_raw=args.refresh_raw,
        save_plot=args.save_plot,
        verbose=not args.quiet,
        train_subjects=args.train_subjects,
    )


if __name__ == "__main__":
    main()
