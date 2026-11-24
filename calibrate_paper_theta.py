#!/usr/bin/env python3
"""Fit theta on the training partition only; trainer options follow --."""

import argparse
import importlib
from pathlib import Path

import torch

from theta_optimization.paper_calibration import calibrate_training_theta
from train.utils import save_json
from train.paper_protocol import calibration_source_fingerprint, calibration_window_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sisfall", "ucihar", "kfall"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-out-win", type=int, required=True)
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--rate-weight", type=float, default=0.1)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    shared = args.training_args[1:] if args.training_args[:1] == ["--"] else args.training_args
    module = importlib.import_module(f"train_{args.dataset}")
    base = module.get_args(shared)
    cfg = module.get_args([*shared, "--input-mode", "raw", "--out-win", str(base.raw_win)])
    if cfg.theta_file is not None:
        parser.error("Calibration must not consume another theta-file.")
    data = module.build_data(cfg, torch.device("cpu"))
    train = data["splits"]["train"]
    labels = train["y" if args.dataset == "ucihar" else "y_h1"]
    times = None
    if args.dataset == "kfall":
        from loaders.kfall_event_loader import window_frame_arrays
        frames = window_frame_arrays(train, data["metadata"], cfg.raw_win)
        times = (frames - frames[:, :1]) / 100.0
    result = calibrate_training_theta(train["x_ev"], labels, out_win=args.event_out_win, seed=cfg.seed,
                                      per_class=args.per_class, rate_weight=args.rate_weight,
                                      dead_zone=cfg.dead_zone, sd_init=cfg.sd_init, train_sample_times=times)
    root = Path(cfg.data_root) / cfg.dataset_dir if args.dataset == "ucihar" else cfg.dataset_root
    result.update(dataset=args.dataset, dataset_root=str(Path(root).resolve()), raw_win=cfg.raw_win,
                  training_subjects=data["metadata"]["split_subjects"]["train"],
                  split_subjects=data["metadata"]["split_subjects"],
                  label_source=getattr(cfg, "label_dir", None),
                  window_policy=getattr(cfg, "fall_policy", "official_segments"),
                  window_config=calibration_window_config(cfg),
                  source_fingerprint=calibration_source_fingerprint(data["metadata"]))
    save_json(args.output, result)
    print(f"[theta:train_only] {args.output} | {result['theta']}")


if __name__ == "__main__":
    main()
