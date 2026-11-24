"""Reproducible, class-balanced threshold selection from training windows only."""

from __future__ import annotations

import numpy as np

from preprocess.event_encoding import sigma_delta
from preprocess.interpolate_signal import interpolate_to_times, make_linear_resample_plan, resample_window_linear_exact


def calibrate_training_theta(train_windows, train_labels, *, out_win, seed=0, per_class=32,
                             rate_weight=0.1, scales=(0.05, 0.1, 0.25, 0.5, 1.0),
                             dead_zone=0.5, sd_init="x0", train_sample_times=None):
    """Minimize normalized MSE + rate_weight * spikes/(channel*time).

    Thresholds are chosen independently per channel; no validation/test arrays
    are accepted. Candidate steps scale the training signal's channel std.
    """
    x = np.asarray(train_windows)
    labels = np.asarray(train_labels)
    scales = np.asarray(scales, dtype=np.float64)
    if (x.ndim != 3 or len(x) == 0 or x.shape[1] < 2 or x.shape[2] == 0
            or labels.shape != (len(x),) or not np.isfinite(labels).all()
            or not isinstance(per_class, (int, np.integer)) or per_class < 1
            or not isinstance(out_win, (int, np.integer)) or out_win < 2
            or not np.isfinite(rate_weight) or rate_weight < 0
            or scales.ndim != 1 or len(scales) == 0 or not np.isfinite(scales).all()
            or np.any(scales <= 0)):
        raise ValueError("Invalid training windows, calibration sample count or objective.")
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(np.flatnonzero(labels == label),
                                       min(per_class, np.count_nonzero(labels == label)), replace=False)
                             for label in np.unique(labels)])
    signals = np.asarray(x[chosen], dtype=np.float32)
    if not np.isfinite(signals).all():
        raise ValueError("Nonfinite calibration signal.")
    times = None if train_sample_times is None else np.asarray(train_sample_times)
    if times is not None and (times.shape != x.shape[:2] or not np.isfinite(times).all()
                              or np.any(np.diff(times, axis=1) <= 0)):
        raise ValueError("Training sample times must be finite, increasing and aligned with training windows.")
    channel_std = np.maximum(signals.reshape(-1, x.shape[-1]).std(axis=0), 1e-6)
    plan = make_linear_resample_plan(in_len=x.shape[1], out_len=out_win)
    candidates = scales[:, None] * channel_std[None, :]
    scores = np.zeros_like(candidates)
    classes = np.unique(labels[chosen])
    for label in classes:
        selected = np.flatnonzero(labels[chosen] == label)
        for i in selected:
            if times is None:
                signal = resample_window_linear_exact(signals[i], plan=plan)
            else:
                source = times[chosen[i]] - times[chosen[i], 0]
                signal = interpolate_to_times(
                    signals[i], source, np.linspace(0.0, source[-1], out_win, dtype=np.float32),
                    method="linear")
            for j, theta in enumerate(candidates):
                events, _, reconstructed = sigma_delta(signal, theta=theta, dead_zone=dead_zone, init=sd_init)
                mse = np.mean((signal - reconstructed) ** 2, axis=0) / channel_std**2
                rate = events.sum(axis=2).mean(axis=0)
                scores[j] += (mse + rate_weight * rate) / (len(classes) * len(selected))
    best = scores.argmin(axis=0)
    return {
        "theta": candidates[best, np.arange(x.shape[-1])].tolist(),
        "calibration_scope": "train_only", "objective": "class_balanced_normalized_mse_plus_event_rate",
        "rate_weight": rate_weight, "scales": scales.tolist(), "scores_by_scale_channel": scores.tolist(),
        "selected_training_window_indices": chosen.tolist(), "seed": seed, "out_win": out_win,
        "dead_zone": dead_zone, "sd_init": sd_init,
    }
