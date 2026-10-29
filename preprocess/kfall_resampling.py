"""Optional analog resampling inside an already observed KFall window.

KFall's FrameCounter and annotations remain on the acquisition grid (100 Hz).
Resampling adds interpolated values, without moving annotations or requesting
samples after the last observed frame.
"""

from __future__ import annotations

import numpy as np

from preprocess.interpolate_signal import interpolate_to_times


def validate_resample_hz(resample_hz: float | None, sample_rate_hz: float = 100.0) -> None:
    """Accept native-rate interpolation or upsampling to a finite higher rate."""
    if resample_hz is not None and (
        not np.isfinite(resample_hz) or float(resample_hz) < sample_rate_hz
    ):
        raise ValueError("resample_hz must be finite and >= the native 100 Hz rate, or None.")


def resample_kfall_signal(
    signal: np.ndarray,
    frames: np.ndarray,
    *,
    sample_rate_hz: float = 100.0,
    resample_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return analog values and seconds relative to the first observed frame.

At 200 Hz, 205 contiguous native samples become 409 samples: both grids
include the same endpoints, separated by 2.04 seconds. For rates that do not
divide the observed duration exactly, the final observed timestamp is added
as a shorter final interval instead of extrapolating beyond the window.
"""
    validate_resample_hz(resample_hz, sample_rate_hz)
    values = np.asarray(signal, dtype=np.float32)
    frame_values = np.asarray(frames, dtype=np.int64)
    if values.ndim != 2 or len(values) < 2 or frame_values.shape != (len(values),):
        raise ValueError("A KFall window must contain >= 2 aligned signal rows and frame counters.")
    if np.any(np.diff(frame_values) <= 0):
        raise ValueError("KFall frame counters must be strictly increasing.")
    times = (frame_values - frame_values[0]).astype(np.float64) / float(sample_rate_hz)
    if resample_hz is None:
        return values, times
    duration = float(times[-1])
    count = int(np.floor(duration * float(resample_hz) + 1e-9))
    destination = np.arange(count + 1, dtype=np.float64) / float(resample_hz)
    if np.isclose(destination[-1], duration, rtol=0, atol=1e-10):
        destination[-1] = duration
    else:
        destination = np.append(destination, duration)
    interpolated = interpolate_to_times(values, times, destination, method="linear")
    # Exact endpoint copies retain the absolute offset and observed endpoint.
    interpolated[0] = values[0]
    interpolated[-1] = values[-1]
    return interpolated, destination
