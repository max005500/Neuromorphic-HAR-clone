"""Strict calibration provenance and front-end ablation CLI options."""

from __future__ import annotations

import json
import math
from pathlib import Path


def add_paper_arguments(parser, *, sisfall=False):
    parser.add_argument("--input-mode", choices=("sigma_delta", "raw"), default="sigma_delta")
    parser.add_argument("--activation", choices=("lif", "relu"), default="lif")
    parser.add_argument("--theta-file", type=Path, help="Train-only calibration from calibrate_paper_theta.py.")
    parser.add_argument("--skip-test", action="store_true", help="Save validation checkpoint without using test.")
    parser.add_argument("--publication-protocol", action="store_true",
                        help="Require the complete official dataset and native sampling protocol.")
    if sisfall:
        parser.add_argument("--no-offset", dest="use_offset", action="store_false", default=True)
        parser.add_argument("--manual-eval-only", action="store_true",
                            help="Only manual static targets supervise evaluation and four-class metrics.")


def load_theta_config(cfg, dataset):
    if cfg.theta_file is None:
        return
    content = json.loads(Path(cfg.theta_file).read_text())
    if content.get("dataset") != dataset or content.get("calibration_scope") != "train_only":
        raise ValueError("Theta calibration must match the dataset and be train_only.")
    for key in ("seed", "raw_win", "dead_zone", "sd_init"):
        if content.get(key) != getattr(cfg, key):
            raise ValueError(f"Theta calibration mismatch: {key}.")
    theta = content.get("theta", [])
    if len(theta) != cfg.n_ch or any(not math.isfinite(value) or value <= 0 for value in theta):
        raise ValueError("Theta must contain one finite positive value per input channel.")
    cfg.theta = content["theta"]
    cfg.theta_provenance = content


def calibration_window_config(cfg):
    keys = ("raw_win", "n_ch", "acc", "dtype", "stride_fall", "stride_dynamic", "stride_static",
            "fall_policy", "sample_rate_hz", "window_stride", "interp_method", "case", "val_ratio",
            "stride", "exclude_tasks", "post_impact_seconds")
    return {key: getattr(cfg, key) for key in keys if hasattr(cfg, key)}


def calibration_source_fingerprint(metadata):
    if metadata.get("dataset") == "kfall":
        return {key: metadata[key] for key in ("source_signature", "manual_annotation_signature")}
    return metadata.get("posture_input_fingerprint")


def verify_calibration_subjects(cfg, metadata):
    subjects = metadata["split_subjects"]
    groups = [set(subjects[key]) for key in ("train", "val", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Subject leakage between training, validation or test.")
    provenance = getattr(cfg, "theta_provenance", None)
    if provenance is not None:
        if provenance.get("window_config") != calibration_window_config(cfg):
            raise ValueError("Theta calibration window/acquisition policy differs.")
        fingerprint = calibration_source_fingerprint(metadata)
        if provenance.get("source_fingerprint") != fingerprint:
            raise ValueError("Theta calibration input files or annotations changed.")
        if set(provenance["training_subjects"]) != groups[0]:
            raise ValueError("Theta was calibrated on different training subjects.")
        expected_root = getattr(cfg, "dataset_root", None)
        if expected_root is None:
            expected_root = Path(cfg.data_root) / cfg.dataset_dir
        if Path(provenance["dataset_root"]).resolve() != Path(expected_root).resolve():
            raise ValueError("Theta calibration dataset root differs.")


def verify_publication_data(cfg, splits, metadata, dataset):
    if not getattr(cfg, "publication_protocol", False):
        return
    subjects = metadata["split_subjects"]
    if dataset == "ucihar":
        if (cfg.case != "official" or cfg.sample_rate_hz != 50 or cfg.raw_win != 128
                or cfg.window_stride not in (None, 128)
                or len(splits["test"]["y"]) != 2947
                or len(splits["train"]["y"]) + len(splits["val"]["y"]) != 7352
                or len(set(subjects["train"]) | set(subjects["val"])) != 21
                or len(subjects["test"]) != 9):
            raise ValueError("Publication UCI requires official 21/9 subjects, 7352/2947 native 50 Hz segments.")
    elif dataset == "sisfall":
        all_subjects = set().union(*(set(values) for values in subjects.values()))
        expected = {f"SA{i:02d}" for i in range(1, 24)} | {f"SE{i:02d}" for i in range(1, 16)}
        if all_subjects != expected:
            raise ValueError("Publication SisFall requires all 38 subjects, including all 15 older SE01..SE15.")
        if not cfg.manual_eval_only or cfg.single_ambiguous != "drop":
            raise ValueError("Publication SisFall requires manual-only evaluation and ambiguous static dropping.")
    elif dataset == "kfall":
        all_subjects = set().union(*(set(values) for values in subjects.values()))
        if len(all_subjects) != 32:
            raise ValueError("Publication KFall requires all 32 subjects.")
        if cfg.annotation_file is None or metadata.get("manual_annotation_signature") is None:
            raise ValueError("Publication KFall requires the manual normal-activity annotation JSON.")
        if cfg.head_mode != "single":
            raise ValueError("Publication KFall uses one activity head; it has no manual posture-transition head gold.")
