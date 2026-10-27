"""Scrolling causal signal and prediction artists for a KFall recording."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


CLASS_NAMES = ("Fall", "Dynamic", "Static")
FALL_COLOR = "#dc2626"
NORMAL_COLOR = "#16a34a"
UNKNOWN_COLOR = "#94a3b8"
POST_IMPACT_COLOR = "#64748b"


class ReplayPlot:
    """Update five supplied axes without exposing observations after the cursor.

    Predictions begin at their window endpoints and are held until a newer
    result is available. The truth ribbon distinguishes the pre-impact falling
    interval from post-impact data, whose normal activity class is unannotated.
    """

    def __init__(
        self, axes: Sequence[Any], replay: Any, *, sensor: str = "acc", view_seconds: float = 3.0,
    ) -> None:
        from matplotlib.collections import PolyCollection
        from matplotlib.patches import Patch
        from matplotlib.ticker import PercentFormatter

        self.axes = np.asarray(axes, dtype=object).reshape(-1)
        if len(self.axes) != 5:
            raise ValueError("ReplayPlot requires five axes: XYZ, probability, and labels.")
        if sensor not in {"acc", "gyro"}:
            raise ValueError("sensor must be 'acc' or 'gyro'.")
        if not np.isfinite(view_seconds) or view_seconds <= 0:
            raise ValueError("view_seconds must be finite and positive.")
        self.replay = replay
        self.view_seconds = float(view_seconds)
        self.sensor = sensor
        self.times = np.asarray(replay.times, dtype=np.float64)
        self.endpoint_times = np.asarray(replay.endpoint_times, dtype=np.float64)
        if self.times.ndim != 1 or len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Recording times must contain at least two increasing samples.")
        if self.endpoint_times.ndim != 1 or np.any(np.diff(self.endpoint_times) <= 0):
            raise ValueError("Prediction endpoint times must be increasing.")
        if len(self.endpoint_times) != len(replay.predicted_classes) or replay.probabilities.shape != (len(self.endpoint_times), 3):
            raise ValueError("Prediction endpoints, classes, and probabilities must align.")

        self.raw_lines = []
        colors = ("#2563eb", "#ea580c", "#7c3aed")
        for axis, color in zip(self.axes[:3], colors):
            line, = axis.plot([], [], color=color, linewidth=1.35)
            self.raw_lines.append(line)
        self.probability_line, = self.axes[3].step([], [], where="post", color="#2563eb", linewidth=1.8)
        self.probability_marker, = self.axes[3].plot([], [], "o", color="#2563eb", markersize=4)
        self.axes[3].set_ylim(0, 1)
        self.axes[3].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        self.axes[3].set_ylabel("P(Fall)")
        self.axes[3].set_title("Probabilidad de caída · último resultado disponible", fontsize=9, loc="left")

        self.prediction_collection = PolyCollection([], edgecolors="none")
        self.truth_collection = PolyCollection([], edgecolors="none")
        self.axes[4].add_collection(self.prediction_collection)
        self.axes[4].add_collection(self.truth_collection)
        self.axes[4].set(ylim=(-0.12, 1.85), yticks=[1.3, 0.3],
                         yticklabels=["Predicción", "Etiqueta real"],
                         xlabel="Tiempo desde el inicio de la grabación (s)")
        self.axes[4].legend(handles=[
            Patch(facecolor=FALL_COLOR, label="Caída"),
            Patch(facecolor=NORMAL_COLOR, label="Sin caída"),
            Patch(facecolor=UNKNOWN_COLOR, label="Sin predicción"),
            Patch(facecolor=POST_IMPACT_COLOR, label="Posimpacto"),
        ], loc="upper right", bbox_to_anchor=(1, 1.30), fontsize=8, ncol=4, frameon=False)

        self.cursor_lines = []
        self.annotation_lines: list[tuple[float, Any]] = []
        for row, axis in enumerate(self.axes):
            cursor = axis.axvline(0, color="#0f172a", linewidth=1.1, alpha=0.8)
            self.cursor_lines.append(cursor)
            axis.grid(True, color="#cbd5e1", alpha=0.45, linewidth=0.65)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="x", labelbottom=row == 4)
            for moment, color in ((replay.fall_onset_time, "#9333ea"),
                                   (replay.fall_impact_time, "#dc2626")):
                if moment is not None:
                    line = axis.axvline(moment, color=color, linestyle=":", linewidth=1.2, visible=False)
                    self.annotation_lines.append((float(moment), line))
        self.annotation_texts = []
        for moment, text, color in ((replay.fall_onset_time, "Inicio real", "#9333ea"),
                                     (replay.fall_impact_time, "Impacto real", "#dc2626")):
            if moment is not None:
                label = self.axes[0].text(moment, 0.95, text, transform=self.axes[0].get_xaxis_transform(),
                                           rotation=90, va="top", ha="right", fontsize=8, color=color, visible=False)
                self.annotation_texts.append((float(moment), label))
        self.current_info: dict[str, Any] = {}
        self.update(float(self.times[0]))

    @staticmethod
    def _rectangles(intervals: list[tuple[float, float, str]], bottom: float):
        vertices, colors = [], []
        for left, right, color in intervals:
            if right > left:
                vertices.append([(left, bottom), (right, bottom), (right, bottom + 0.6), (left, bottom + 0.6)])
                colors.append(color)
        return vertices, colors

    def _prediction_intervals(self, now: float) -> list[tuple[float, float, str]]:
        start = float(self.times[0])
        available = int(np.searchsorted(self.endpoint_times, now, side="right"))
        if available == 0:
            return [(start, now, UNKNOWN_COLOR)]
        intervals = [(start, min(now, float(self.endpoint_times[0])), UNKNOWN_COLOR)]
        for index in range(available):
            left = float(self.endpoint_times[index])
            right = min(now, float(self.endpoint_times[index + 1])) if index + 1 < available else now
            color = FALL_COLOR if int(self.replay.predicted_classes[index]) == 0 else NORMAL_COLOR
            if intervals and intervals[-1][2] == color and intervals[-1][1] == left:
                intervals[-1] = (intervals[-1][0], right, color)
            else:
                intervals.append((left, right, color))
        return intervals

    def _truth_intervals(self, now: float) -> list[tuple[float, float, str]]:
        start = float(self.times[0])
        onset, impact = self.replay.fall_onset_time, self.replay.fall_impact_time
        if onset is None:
            return [(start, now, NORMAL_COLOR)]
        intervals = [(start, min(now, onset), NORMAL_COLOR)]
        if now > onset:
            intervals.append((max(start, onset), min(now, impact), FALL_COLOR))
        if now > impact:
            intervals.append((max(start, impact), now, POST_IMPACT_COLOR))
        return intervals

    def _real_phase(self, now: float) -> str:
        onset, impact = self.replay.fall_onset_time, self.replay.fall_impact_time
        if onset is None:
            return CLASS_NAMES[int(self.replay.record["label_id"])]
        if now < onset:
            return "Precaída / sin caída"
        if now < impact:
            return "Caída (preimpacto)"
        return "Posimpacto"

    def update(self, time_seconds: float, sensor: str | None = None) -> dict[str, Any]:
        """Reveal observations and decisions available at the selected cursor."""
        if not np.isfinite(time_seconds):
            raise ValueError("Replay time must be finite.")
        if sensor is not None:
            if sensor not in {"acc", "gyro"}:
                raise ValueError("sensor must be 'acc' or 'gyro'.")
            self.sensor = sensor
        now = float(np.clip(time_seconds, self.times[0], self.times[-1]))
        left = max(float(self.times[0]), now - self.view_seconds)
        right = max(float(self.times[0]) + self.view_seconds, now)
        observed = (self.times >= left) & (self.times <= now)
        first_channel = 0 if self.sensor == "acc" else 3
        prefix, unit = ("Acc", "g") if self.sensor == "acc" else ("Gyr", "°/s")
        for row, (axis, line) in enumerate(zip(self.axes[:3], self.raw_lines)):
            channel = first_channel + row
            line.set_data(self.times[observed], self.replay.raw[observed, channel])
            axis.set_ylabel(f"{prefix}{'XYZ'[row]} ({unit})")
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)

        available = int(np.searchsorted(self.endpoint_times, now, side="right"))
        prediction_index = available - 1 if available else None
        predicted_class = None
        p_fall = prediction_time = prediction_age = None
        predicted_label = "Sin predicción (ventana incompleta)"
        if available:
            prediction_time = float(self.endpoint_times[prediction_index])
            prediction_age = now - prediction_time
            predicted_class = int(self.replay.predicted_classes[prediction_index])
            p_fall = float(self.replay.probabilities[prediction_index, 0])
            predicted_label = "Caída" if predicted_class == 0 else f"Sin caída · {CLASS_NAMES[predicted_class]}"
            begin = max(0, int(np.searchsorted(self.endpoint_times[:available], left, side="right")) - 1)
            probability_times = self.endpoint_times[begin:available].copy()
            probability_times[0] = max(left, probability_times[0])
            probability_values = np.asarray(self.replay.probabilities[begin:available, 0])
            if probability_times[-1] < now:
                probability_times = np.append(probability_times, now)
                probability_values = np.append(probability_values, p_fall)
            self.probability_line.set_data(probability_times, probability_values)
            self.probability_marker.set_data([now], [p_fall])
        else:
            self.probability_line.set_data([], [])
            self.probability_marker.set_data([], [])

        for collection, intervals, bottom in (
            (self.prediction_collection, self._prediction_intervals(now), 1.0),
            (self.truth_collection, self._truth_intervals(now), 0.0),
        ):
            vertices, colors = self._rectangles(intervals, bottom)
            collection.set_verts(vertices)
            collection.set_facecolors(colors)
        for axis, cursor in zip(self.axes, self.cursor_lines):
            axis.set_xlim(left, right)
            cursor.set_xdata([now, now])
        for moment, line in self.annotation_lines:
            line.set_visible(moment <= now)
        for moment, label in self.annotation_texts:
            label.set_visible(left <= moment <= now)

        self.current_info = {
            "prediction_index": prediction_index, "predicted_class": predicted_class,
            "predicted_label": predicted_label, "p_fall": p_fall,
            "prediction_time": prediction_time, "prediction_age_sec": prediction_age,
            "real_phase": self._real_phase(now), "time": now,
        }
        return self.current_info
