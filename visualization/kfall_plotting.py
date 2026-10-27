"""Draw measured KFall signals beside their cached Sigma-Delta events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from visualization.kfall_data import WindowData


def _annotate_fall(axis, window: WindowData, start: float, stop: float) -> None:
    onset = window.fall_onset_time
    impact = window.fall_impact_time
    if onset is not None and np.isfinite(onset) and onset <= stop:
        end = min(stop, impact) if impact is not None and np.isfinite(impact) else stop
        if end > max(start, onset):
            axis.axvspan(max(start, onset), end, color="#dc2626", alpha=0.055, linewidth=0)
    for moment, label, color in ((onset, "Inicio caída", "#9333ea"),
                                 (impact, "Impacto", "#dc2626")):
        if moment is None or not np.isfinite(moment) or not start <= moment <= stop:
            continue
        axis.axvline(moment, color=color, linestyle=":", linewidth=1.3, label=label)
        at_right_edge = moment >= stop - (stop - start) * 0.05
        axis.text(moment, 0.98, label, transform=axis.get_xaxis_transform(),
                  color=color, fontsize=7, rotation=90, va="top",
                  ha="right" if at_right_edge else "left")


def draw_window(
    axes: np.ndarray,
    window: WindowData,
    sensor: str = "acc",
    show_reconstruction: bool = True,
) -> None:
    """Redraw three original/event pairs while retaining the figure's controls.

    ``axes`` must contain three rows and two columns. The right column displays
    every cached ON/OFF event; its signed positions are only a visualization of
    the two binary polarity channels supplied to the model.
    """
    axes = np.asarray(axes, dtype=object)
    if axes.shape != (3, 2):
        raise ValueError("axes must have shape (3, 2).")
    if sensor not in {"acc", "gyro"}:
        raise ValueError("sensor must be 'acc' or 'gyro'.")
    first_channel = 0 if sensor == "acc" else 3
    prefix, units = ("Acc", "g") if sensor == "acc" else ("Gyr", "°/s")
    time_raw = np.asarray(window.time_raw)
    time_events = np.asarray(window.time_events)
    if time_raw.ndim != 1 or time_events.ndim != 1 or min(time_raw.size, time_events.size) < 2:
        raise ValueError("Window time grids must be one-dimensional and contain at least two samples.")
    start, stop = float(time_events[0]), float(time_events[-1])
    if stop <= start:
        raise ValueError("Window time grids must be one-dimensional and span positive time.")

    for row, suffix in enumerate("XYZ"):
        channel = first_channel + row
        original_axis, event_axis = axes[row]
        original_axis.clear()
        event_axis.clear()

        original_axis.plot(time_events, window.interpolated[:, channel],
                           color="#2563eb", linewidth=1.3, alpha=0.85, label="Interpolada")
        original_axis.plot(time_raw, window.raw[:, channel], color="#334155",
                           linewidth=0.7, marker="o", markersize=2.2, alpha=0.8,
                           label="Original (100 Hz)")
        if show_reconstruction:
            original_axis.step(time_events, window.reconstructed[:, channel], where="post",
                               color="#059669", linewidth=1.0, alpha=0.85,
                               label="Reconstrucción ΣΔ")
        original_axis.set_title(f"{prefix}{suffix} · θ={window.theta[channel]:g} {units}",
                                fontsize=10, loc="left")
        original_axis.set_ylabel(f"{prefix}{suffix} ({units})")

        on_times = time_events[np.asarray(window.events[:, channel, 0], dtype=bool)]
        off_times = time_events[np.asarray(window.events[:, channel, 1], dtype=bool)]
        event_axis.axhline(0, color="#94a3b8", linewidth=0.6, zorder=0)
        event_axis.vlines(on_times, 0, 1, color="#2563eb", linewidth=0.8,
                          alpha=0.8, label="ON (+1)")
        event_axis.vlines(off_times, -1, 0, color="#ea580c", linewidth=0.8,
                          alpha=0.8, label="OFF (−1)")
        event_axis.set_title(f"{prefix}{suffix} · eventos ON: {len(on_times)} | OFF: {len(off_times)}",
                             fontsize=10, loc="left")
        event_axis.set_yticks([-1, 0, 1], ["OFF", "0", "ON"])
        event_axis.set_ylim(-1.25, 1.25)

        for axis in (original_axis, event_axis):
            _annotate_fall(axis, window, start, stop)
            # Future onset/impact annotations must not extend the visible input.
            axis.set_xlim(start, stop)
            axis.grid(True, color="#cbd5e1", alpha=0.45, linewidth=0.6)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="x", labelbottom=row == 2)
            if row == 2:
                axis.set_xlabel("Tiempo desde el inicio de la ventana (s)")
        if row == 0:
            handles, labels = original_axis.get_legend_handles_labels()
            signal_entries = [(handle, label) for handle, label in zip(handles, labels)
                              if label in {"Interpolada", "Original (100 Hz)", "Reconstrucción ΣΔ"}]
            original_axis.legend([entry[0] for entry in signal_entries],
                                 [entry[1] for entry in signal_entries],
                                 loc="best", fontsize=7, framealpha=0.8)
            # Annotation labels are shown next to their vertical lines; keep the
            # event legend focused on the two actual input polarity channels.
            handles, labels = event_axis.get_legend_handles_labels()
            event_axis.legend(handles[:2], labels[:2], loc="upper right", fontsize=7,
                              ncol=2, framealpha=0.8)
