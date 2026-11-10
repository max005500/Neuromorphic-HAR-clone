#!/usr/bin/env python3
"""Calibrate accelerometer Sigma-Delta steps on motion-balanced training subjects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from loaders.kfall_event_loader import DEFAULT_EXCLUDED_TASKS, TASK_CODES, _source_signature
from preprocess.event_encoding import sigma_delta
from preprocess.interpolate_signal import interpolate_to_times
from preprocess.preprocess_kfall import load_kfall_3class
from preprocess.utils import subject_split_indices


def select_windows(recordings, records, train_indices, raw_win, per_regime, seed):
    """Motion is an accelerometer variability proxy, not an activity annotation."""
    eligible = [int(i) for i in train_indices if len(recordings[int(i)]) >= raw_win]
    if len(eligible) < 3 * per_regime:
        raise ValueError(f"Need at least {3 * per_regime} eligible training recordings.")
    scores = {
        i: float(np.sqrt(np.mean(np.var(np.asarray(recordings[i])[:, :3], axis=0))))
        for i in eligible
    }
    ordered = sorted(eligible, key=lambda i: (scores[i], i))
    bins = np.array_split(ordered, 3)
    rng = np.random.default_rng(seed)
    selected, manifest = [], []
    for regime, pool in zip(("low", "medium", "high"), bins):
        # Round-robin across shuffled subjects prevents one participant dominating.
        by_subject = {}
        for index in rng.permutation(pool):
            by_subject.setdefault(records[int(index)]["subject"], []).append(int(index))
        subjects = list(by_subject)
        rng.shuffle(subjects)
        chosen = []
        while len(chosen) < per_regime:
            for subject in subjects:
                if by_subject[subject] and len(chosen) < per_regime:
                    chosen.append(by_subject[subject].pop())
        for index in chosen:
            signal = np.asarray(recordings[index], dtype=np.float32)
            starts = np.arange(0, len(signal) - raw_win + 1, raw_win)
            valid = [int(start) for start in starts
                     if np.all(np.diff(records[index]["frames"][start:start + raw_win]) == 1)]
            if not valid:
                raise ValueError(f"No consecutive-frame calibration crop in {records[index]['file']}.")
            start = int(rng.choice(valid))
            selected.append(signal[start:start + raw_win])
            manifest.append({
                "file": records[index]["file"], "subject": records[index]["subject"],
                "code": records[index]["code"], "regime": regime,
                "motion_score": scores[index], "start": start,
            })
    if len({item["subject"] for item in manifest}) < 2:
        raise ValueError("Calibration must sample at least two training subjects.")
    return np.stack(selected), manifest


def reconstruction_score(windows, theta, raw_win, out_win):
    """Mean per-window NRMSE: prevent high-motion recordings dominating."""
    native = np.arange(raw_win) / 100
    encoded_times = np.linspace(0, native[-1], out_win, dtype=np.float32)
    errors, rates = [], []
    for signal in windows:
        interpolated = interpolate_to_times(signal, native, encoded_times)
        events, _, reconstructed = sigma_delta(interpolated, theta=theta, dead_zone=0.5,
                                               init="x0", return_reconstruction=True)
        recovered = interpolate_to_times(reconstructed, encoded_times, native)
        rmse = np.sqrt(np.mean((signal - recovered) ** 2, axis=0))
        scale = np.maximum(np.std(signal, axis=0), 1e-3)
        errors.append(rmse / scale)
        rates.append(np.mean(events.sum(axis=-1), axis=0))
    return np.mean(errors, axis=0), np.mean(rates, axis=0)


def calibrate(args):
    if (args.seed < 0 or len(args.split) != 3 or any(not np.isfinite(x) or x <= 0 for x in args.split)
            or not np.isclose(sum(args.split), 1)):
        raise ValueError("Use a nonnegative seed and three positive subject fractions summing to one.")
    if args.raw_win < 2 or args.out_win < 2 or args.per_regime < 1:
        raise ValueError("Window sizes must be >=2; per-regime must be positive.")
    if len(args.candidates) < 2 or any(not np.isfinite(x) or x <= 0 for x in args.candidates):
        raise ValueError("Supply at least two finite positive theta candidates in g.")
    recordings, _, records, _ = load_kfall_3class(args.dataset_root)
    # Split the full subject population first, as in training.
    groups = subject_split_indices([record["subject"] for record in records], args.split, args.seed)
    train = [int(i) for i in groups[0] if records[int(i)]["code"] not in args.exclude_tasks]
    windows, manifest = select_windows(recordings, records, train, args.raw_win, args.per_regime, args.seed)
    theta = np.asarray(args.theta, dtype=np.float32)
    if theta.shape != (6,) or not np.isfinite(theta).all() or np.any(theta <= 0):
        raise ValueError("theta must contain six finite positive values.")
    initial = theta.copy()
    before, initial_rate = reconstruction_score(windows, theta, args.raw_win, args.out_win)
    history = []
    # Channels are independent in Sigma-Delta; optimize only accelerometer axes.
    for axis in range(3):
        best = (float("inf"), float(theta[axis]))
        candidates = np.unique(np.asarray([theta[axis], *args.candidates], dtype=np.float32))
        for candidate in tqdm(candidates, desc=f"Theta Acc{'XYZ'[axis]}", unit="candidate"):
            candidate = float(candidate)
            trial = theta.copy()
            trial[axis] = candidate
            errors, rates = reconstruction_score(windows[:, :, axis:axis + 1], trial[axis:axis + 1],
                                                  args.raw_win, args.out_win)
            value = float(errors[0])
            history.append({"axis": axis, "theta": candidate, "nrmse": value, "spike_fraction": float(rates[0])})
            tqdm.write(f"[theta] axis={axis} candidate={candidate:g} NRMSE={value:.6f}")
            if value < best[0]:
                best = (value, candidate)
        theta[axis] = best[1]
    after, rates = reconstruction_score(windows, theta, args.raw_win, args.out_win)
    result = {
        "dataset": "kfall", "theta_best": theta.tolist(), "theta_initial": initial.tolist(),
        "channels": ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"],
        "units": ["g", "g", "g", "deg/s", "deg/s", "deg/s"],
        "dead_zone": 0.5, "sd_init": "x0", "seed": args.seed, "split": args.split,
        "raw_win": args.raw_win, "out_win": args.out_win, "excluded_tasks": args.exclude_tasks,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "source_signature": _source_signature(Path(args.dataset_root), Path(args.dataset_root) / "label_data"),
        "split_subjects": {name: sorted({records[int(i)]["subject"] for i in indices})
                           for name, indices in zip(("train", "val", "test"), groups)},
        "sampling": manifest, "history": history, "nrmse_before": before.tolist(),
        "nrmse_after": after.tolist(), "spike_fraction_before": initial_rate.tolist(),
        "spike_fraction_after": rates.tolist(),
        "note": "Train-only reconstruction calibration; equal crops per motion tertile. "
                "Not a classification optimum, LOSO result, or hardware energy estimate.",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"[saved] {destination} | theta={theta.tolist()}")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/KFall")
    parser.add_argument("--output", default="results/kfall_calibration/theta.json")
    parser.add_argument("--split", nargs=3, type=float, default=[0.6, 0.3, 0.1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-win", type=int, default=120)
    parser.add_argument("--out-win", type=int, default=1200)
    parser.add_argument("--per-regime", type=int, default=10)
    parser.add_argument("--exclude-tasks", nargs="*", choices=TASK_CODES, default=list(DEFAULT_EXCLUDED_TASKS))
    parser.add_argument("--theta", nargs=6, type=float, default=[0.1, 0.1, 0.1, 10, 10, 10])
    parser.add_argument("--candidates", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.16, 0.32])
    return parser


if __name__ == "__main__":
    calibrate(build_parser().parse_args())
