#!/usr/bin/env python3
"""Replay a held-out KFall recording with checkpoint predictions and true fall timing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from visualization.kfall_replay_data import prepare_replay
from visualization.kfall_replay_plot import ReplayPlot
from preprocess.kfall_labels import class_names


class KFallReplayPlayer:
    """Animate causal predictions while the three IMU axes scroll in time."""

    def __init__(
        self, replay, *, sensor: str = "acc", view_seconds: float = 3.0,
        speed: float = 1.0, fps: float = 25, interactive: bool = True,
    ) -> None:
        from matplotlib import pyplot as plt

        if not np.isfinite(speed) or not 0.1 <= speed <= 4:
            raise ValueError("speed must be between 0.1 and 4.")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive.")
        self.replay = replay
        self.sensor, self.speed, self.fps = sensor, float(speed), float(fps)
        self.current_time = 0.0
        self.playing = bool(interactive)
        self.widgets: dict[str, Any] = {}
        self._updating = False
        self.animation = None
        self.figure, axes = plt.subplots(5, 1, figsize=(13.5, 9.5), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 1, 1, 0.85, 0.7]})
        self.axes = list(axes)
        self.figure.subplots_adjust(left=0.085, right=0.97, top=0.815,
                                    bottom=0.19 if interactive else 0.11, hspace=0.32)
        self.figure.canvas.manager.set_window_title("KFall · predicción frente a etiqueta real")
        self.figure.text(0.085, 0.966,
                         f"{replay.subject} · {replay.record['code']}/{replay.record['trial']} · "
                         f"TEST · mejor checkpoint: época {replay.best_epoch}", fontsize=15, weight="bold")
        self.prediction_text = self.figure.text(0.085, 0.925, "", fontsize=13, weight="bold")
        self.truth_text = self.figure.text(0.085, 0.892, "", fontsize=12)
        self.time_text = self.figure.text(0.085, 0.855, "", fontsize=9, color="#475569")
        self.figure.text(0.085, 0.008,
                         "Reproducción de inferencias por ventanas: cada decisión se muestra al terminar su entrada y se mantiene hasta la siguiente.",
                         fontsize=8, color="#475569")
        try:
            self.plot = ReplayPlot(self.axes, replay, sensor=sensor, view_seconds=view_seconds)
            if interactive:
                self._build_controls()
            self._render(draw=False)
            if interactive:
                from matplotlib.animation import FuncAnimation

                self.animation = FuncAnimation(self.figure, self._on_tick, init_func=self._init_animation,
                                               interval=1000 / self.fps, blit=False, cache_frame_data=False)
                self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        except Exception:
            plt.close(self.figure)
            raise

    def _build_controls(self) -> None:
        from matplotlib.widgets import Button, RadioButtons, Slider

        self.widgets["time"] = Slider(self.figure.add_axes([0.15, 0.095, 0.56, 0.023]),
                                       "Tiempo (s)", 0, float(self.replay.times[-1]), valinit=0)
        self.widgets["time"].on_changed(self._on_time)
        self.widgets["speed"] = Slider(self.figure.add_axes([0.80, 0.095, 0.16, 0.023]),
                                        "Velocidad", 0.1, 4.0, valinit=self.speed)
        self.widgets["speed"].on_changed(self._on_speed)
        for name, label, left, callback in (
            ("play", "Pausa", 0.085, self.toggle_play),
            ("restart", "Reiniciar", 0.20, self.restart),
            ("previous", "Anterior decisión", 0.315, lambda event: self.step_prediction(-1)),
            ("next", "Siguiente decisión", 0.465, lambda event: self.step_prediction(1)),
        ):
            button = Button(self.figure.add_axes([left, 0.036, 0.135 if name in {"previous", "next"} else 0.10, 0.037]), label)
            button.label.set_fontsize(9)
            button.on_clicked(callback)
            self.widgets[name] = button
        sensor = RadioButtons(self.figure.add_axes([0.77, 0.026, 0.20, 0.055]),
                               ["Acelerómetro", "Giroscopio"], active=0 if self.sensor == "acc" else 1)
        sensor.on_clicked(self._on_sensor)
        self.widgets["sensor"] = sensor

    def _render(self, *, draw: bool = True) -> None:
        info = self.plot.update(self.current_time, sensor=self.sensor)
        prediction = info.get("predicted_class")
        probability = info.get("p_fall")
        if prediction is None:
            text, color = "Modelo: sin predicción · esperando una ventana completa", "#64748b"
        else:
            activity = class_names(self.replay.probabilities.shape[1])[int(prediction)]
            state = "CAÍDA" if prediction == 0 else "POSTIMPACTO" if prediction == 3 else "SIN CAÍDA"
            text = f"Modelo: {state} · clase {activity} · P(Fall) = {probability:.1%}"
            color = "#dc2626" if prediction == 0 else "#d97706" if prediction == 3 else "#15803d"
            if self.replay.probabilities.shape[1] == 4:
                index = int(info["prediction_index"])
                text += f" · P(Postimpact) = {self.replay.probabilities[index, 3]:.1%}"
        self.prediction_text.set_text(text)
        self.prediction_text.set_color(color)
        self.truth_text.set_text(f"Etiqueta real: {info['real_phase']}")
        self.truth_text.set_color("#9333ea")
        index = info.get("prediction_index")
        endpoint = "—" if index is None else f"{self.replay.endpoint_times[index]:.3f} s"
        self.time_text.set_text(
            f"t = {self.current_time:.3f} / {self.replay.times[-1]:.3f} s · última decisión: {endpoint} · "
            f"entrada: {self.replay.raw_win} muestras → {self.replay.out_win} pasos ΣΔ · "
            f"avance de inferencia: {self.replay.prediction_stride} muestras · reproducción {self.speed:g}×"
        )
        self._sync_controls()
        if draw:
            self.figure.canvas.draw_idle()

    def _sync_controls(self) -> None:
        if not self.widgets:
            return
        self._updating = True
        try:
            self.widgets["time"].set_val(self.current_time)
            self.widgets["play"].label.set_text("Pausa" if self.playing else "Reproducir")
        finally:
            self._updating = False

    def _init_animation(self):
        self._render(draw=False)
        return []

    def seek(self, time_seconds: float) -> None:
        if not np.isfinite(time_seconds):
            raise ValueError("Replay time must be finite.")
        self.current_time = float(np.clip(time_seconds, 0.0, self.replay.times[-1]))
        if self.current_time >= self.replay.times[-1]:
            self.playing = False
        self._render()

    def _on_tick(self, frame=None):
        if self.playing:
            self.seek(self.current_time + self.speed / self.fps)
        return []

    def toggle_play(self, event=None) -> None:
        if self.current_time >= self.replay.times[-1]:
            self.current_time = 0.0
        self.playing = not self.playing
        self._render()

    def restart(self, event=None) -> None:
        self.seek(0.0)

    def step_prediction(self, direction: int) -> None:
        self.playing = False
        times = np.asarray(self.replay.endpoint_times)
        if not len(times):
            return
        index = (np.searchsorted(times, self.current_time, side="right") if direction > 0
                 else np.searchsorted(times, self.current_time, side="left") - 1)
        if index < 0:
            self.seek(0)
        else:
            self.seek(float(times[min(int(index), len(times) - 1)]))

    def _on_time(self, value) -> None:
        if not self._updating:
            self.seek(float(value))

    def _on_speed(self, value) -> None:
        if not self._updating:
            self.speed = float(value)
            self._render()

    def _on_sensor(self, label) -> None:
        self.sensor = "acc" if label == "Acelerómetro" else "gyro"
        self._render()

    def _on_key(self, event) -> None:
        if event.key == " ":
            self.toggle_play()
        elif event.key == "right":
            self.step_prediction(1)
        elif event.key == "left":
            self.step_prediction(-1)
        elif event.key == "home":
            self.restart()

    def save_frame(self, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        playing = self.playing
        self.playing = False
        try:
            self.figure.savefig(destination, dpi=140)
        finally:
            self.playing = playing
        return destination

    def save_gif(self, destination: str | Path) -> Path:
        from matplotlib.animation import PillowWriter

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        current_time, playing = self.current_time, self.playing
        frame_times = np.arange(0.0, float(self.replay.times[-1]), self.speed / self.fps)
        frame_times = np.append(frame_times, self.replay.times[-1])
        writer = PillowWriter(fps=self.fps)
        self.playing = False
        try:
            with writer.saving(self.figure, str(destination), dpi=80):
                for time_seconds in frame_times:
                    self.current_time = float(time_seconds)
                    self._render(draw=False)
                    writer.grab_frame()
        finally:
            self.current_time, self.playing = current_time, playing
            self._render(draw=False)
        return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Matching event cache containing metadata.json.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Local KFall root containing sensor_data.")
    parser.add_argument("--recording", default=None, help="Test CSV name, stem, or path; defaults to the first fall recording.")
    parser.add_argument("--window", type=int, default=None, help="Select the source recording of this actual test-cache window.")
    parser.add_argument("--subject", default=None, help="Restrict selection to a held-out subject such as SA07.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--prediction-stride", type=int, default=10, help="Raw samples between causal predictions, independent of annotations.")
    parser.add_argument("--sensor", choices=["acc", "gyro"], default="acc")
    parser.add_argument("--view-seconds", type=float, default=3.0, help="Width of the scrolling time view.")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed, between 0.1 and 4.")
    parser.add_argument("--fps", type=float, default=25.0, help="Animation frames per second; 10 is useful for smaller GIFs.")
    parser.add_argument("--paused", action="store_true", help="Open the player paused.")
    parser.add_argument("--time", type=float, default=0.0, help="Initial replay time, or the time of --save-frame.")
    parser.add_argument("--save-frame", type=Path, default=None, metavar="PNG")
    parser.add_argument("--save-gif", type=Path, default=None, metavar="GIF", help="Export the entire replay without a desktop window.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    import matplotlib

    exporting = args.save_gif is not None or args.save_frame is not None
    if exporting:
        matplotlib.use("Agg")
    elif matplotlib.get_backend().lower() in {"agg", "pdf", "svg", "ps", "cairo", "template"}:
        parser.error("La interfaz necesita un backend gráfico como TkAgg. Usa --save-gif o --save-frame para exportar sin escritorio.")
    from matplotlib import pyplot as plt

    try:
        replay = prepare_replay(args.run_dir, checkpoint=args.checkpoint, cache_dir=args.cache_dir,
                                dataset_root=args.dataset_root, recording=args.recording, window=args.window,
                                subject=args.subject, device=args.device, prediction_stride=args.prediction_stride,
                                batch_size=args.batch_size)
        player = KFallReplayPlayer(replay, sensor=args.sensor, view_seconds=args.view_seconds,
                                    speed=args.speed, fps=args.fps, interactive=not exporting)
        player.seek(args.time)
        if args.paused:
            player.playing = False
            player._render()
        print(f"[replay] {replay.path.name} | subject={replay.subject} | epoch={replay.best_epoch} | predictions={len(replay.endpoint_times)}")
        if args.save_frame is not None:
            print(f"[saved] {player.save_frame(args.save_frame)}")
        if args.save_gif is not None:
            print(f"[saved] {player.save_gif(args.save_gif)}")
        if exporting:
            plt.close(player.figure)
        else:
            plt.show()
    except (ValueError, FileNotFoundError, OSError, RuntimeError, ImportError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
