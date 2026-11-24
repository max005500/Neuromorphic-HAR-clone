"""Matched raw and Sigma-Delta inputs for front-end ablations."""

from __future__ import annotations

import numpy as np

from preprocess.event_encoding import sigma_delta
from preprocess.interpolate_signal import make_linear_resample_plan, resample_window_linear_exact


def encode_window(signal, *, mode, out_win, theta, dead_zone=0.5, sd_init="x0"):
    signal = np.asarray(signal, dtype=np.float32)
    if mode not in {"raw", "sigma_delta"}:
        raise ValueError("input_mode must be raw or sigma_delta.")
    if signal.ndim != 2 or not np.isfinite(signal).all() or len(signal) < 2:
        raise ValueError("Expected finite (time, channels) signal with at least two samples.")
    resampled = resample_window_linear_exact(
        signal, plan=make_linear_resample_plan(in_len=len(signal), out_len=out_win))
    if mode == "raw":
        return resampled
    return sigma_delta(resampled, theta=theta, dead_zone=dead_zone, init=sd_init,
                       return_reconstruction=False)[0]
