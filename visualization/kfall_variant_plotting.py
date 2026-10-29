"""Compare KFall contexts on the same prediction-relative physical time axis."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Sequence

import numpy as np

from visualization.kfall_plotting import draw_window

if TYPE_CHECKING:
    from visualization.kfall_data import WindowData


def _relative_window(window: WindowData) -> SimpleNamespace:
    """Copy display coordinates while retaining the read-only cached arrays."""
    raw_times = np.asarray(window.time_raw, dtype=np.float64)
    if (raw_times.ndim != 1 or len(raw_times) < 2 or not np.isfinite(raw_times).all()
            or np.any(np.diff(raw_times) <= 0)):
        raise ValueError("Window time grids must be finite, strictly increasing and contain >= 2 samples.")
    origin = float(raw_times[-1])
    span = float(raw_times[-1] - raw_times[0])
    tolerance = 8 * np.finfo(np.float32).eps * max(1., abs(origin), span)

    def shift_grid(times) -> np.ndarray:
        values = np.asarray(times, dtype=np.float64)
        if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
                or np.any(np.diff(values) <= 0)
                or abs(float(values[0]) - float(raw_times[0])) > tolerance
                or abs(float(values[-1]) - origin) > tolerance):
            raise ValueError("Event/interpolation time grids must share the observed window endpoints.")
        # Event-grid endpoints originate as float32. Their sub-microsecond
        # rounding must not place an event after the prediction at zero.
        return np.clip(values - origin, -span, 0.)

    attributes = dict(vars(window))
    attributes["time_raw"] = raw_times - origin
    attributes["time_events"] = shift_grid(window.time_events)
    if getattr(window, "resample_hz", None) is not None:
        attributes["time_analog"] = shift_grid(window.time_analog)
    for name in ("fall_onset_time", "fall_impact_time", "post_impact_end_time"):
        moment = getattr(window, name, None)
        attributes[name] = (float(moment) - origin if moment is not None and np.isfinite(moment) else None)
    return SimpleNamespace(**attributes)


def draw_variant_comparison(
    axes: np.ndarray,
    windows: Sequence[WindowData],
    sensor: str = "acc",
    show_reconstruction: bool = True,
) -> None:
    """Plot one or more aligned cached windows with one signal/event pair each.

    ``axes`` has three rows (X/Y/Z) and ``2 * len(windows)`` columns. Every
    panel uses seconds relative to the common prediction endpoint at zero.
    The horizontal extent is the longest observed context; gray areas mark
    earlier samples unavailable to a shorter window. Each signal row also
    shares its vertical scale across variants.

    Cached values and source arrays are read without modification. When window
    identity metadata is present, differing recordings, endpoints or targets
    are rejected instead of displaying an unrelated comparison.
    """
    windows = list(windows)
    axes = np.asarray(axes, dtype=object)
    if not windows or axes.shape != (3, 2 * len(windows)):
        raise ValueError("axes must have shape (3, 2 * number_of_windows), with at least one window.")
    if sensor not in {"acc", "gyro"}:
        raise ValueError("sensor must be 'acc' or 'gyro'.")
    for name in ("subject", "code", "trial", "end_frame", "class_id"):
        values = [getattr(window, name) for window in windows if hasattr(window, name)]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(f"Comparison windows must share recording, endpoint and target ({name} differs).")
    relative = [_relative_window(window) for window in windows]
    start = min(float(window.time_raw[0]) for window in relative)
    prefix, units, channel_start = ("Acc", "g", 0) if sensor == "acc" else ("Gyr", "°/s", 3)

    for variant, (original, window) in enumerate(zip(windows, relative)):
        pair = axes[:, 2 * variant:2 * variant + 2]
        draw_window(pair, window, sensor=sensor, show_reconstruction=show_reconstruction)
        duration = len(original.raw) / 100.
        for row, suffix in enumerate("XYZ"):
            signal_axis, event_axis = pair[row]
            channel = row + channel_start
            channel_title = f"{prefix}{suffix} · θ={window.theta[channel]:g} {units}"
            if row == 0:
                signal_axis.set_title(
                    f"{duration:.2f} s · {len(original.raw)} muestras\n{channel_title}",
                    fontsize=9, loc="left",
                )
                counts = event_axis.get_title(loc="left")
                event_axis.set_title(f"{len(original.events)} pasos ΣΔ\n{counts}", fontsize=8, loc="left")
            else:
                signal_axis.set_title(channel_title, fontsize=9, loc="left")
                event_axis.set_title(event_axis.get_title(loc="left"), fontsize=8, loc="left")
            for axis in pair[row]:
                local_start = float(window.time_raw[0])
                if local_start > start + 1e-10:
                    axis.axvspan(start, local_start, color="#e2e8f0", alpha=.8,
                                 linewidth=0, zorder=-3, label="Sin contexto de entrada")
                    if row == 0:
                        axis.text((start + local_start) / 2, .035, "Sin contexto",
                                  transform=axis.get_xaxis_transform(), fontsize=7,
                                  color="#64748b", ha="center", va="bottom")
                axis.axvline(0., color="#475569", linestyle="--", linewidth=.8,
                             alpha=.8, label="Instante de predicción")
                axis.set_xlim(start, 0.)
                if row == 2:
                    axis.set_xlabel("Tiempo relativo a la predicción (s)", fontsize=8)
                axis.tick_params(labelsize=8)
            # A shared legend in the first pair keeps the other contexts clear.
            if variant and row == 0:
                for axis in pair[row]:
                    legend = axis.get_legend()
                    if legend is not None:
                        legend.remove()

    for row in range(3):
        signals = [axes[row, 2 * index] for index in range(len(windows))]
        lower = min(axis.get_ylim()[0] for axis in signals)
        upper = max(axis.get_ylim()[1] for axis in signals)
        for axis in signals:
            axis.set_ylim(lower, upper)
