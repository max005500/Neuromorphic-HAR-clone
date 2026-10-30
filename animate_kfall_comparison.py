#!/usr/bin/env python3
"""Compare three trained KFall windows on one synchronized held-out recording."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from animate_kfall import KFallReplayPlayer
from evaluate_kfall import _load_checkpoint
from visualization.kfall_replay_data import prepare_replay
from visualization.kfall_replay_plot import ReplayPlot


def prepare_comparison(run_dirs, *, recording=None, **kwargs):
    if len(run_dirs) != 3:
        raise ValueError("Indica tres directorios de entrenamiento completos.")
    specs = []
    for run in run_dirs:
        run = Path(run).expanduser().resolve()
        _, cfg = _load_checkpoint(run / "checkpoints" / "best_model.pt")
        specs.append((cfg.raw_win, run))
    specs.sort()
    if [size for size, _ in specs] != [205, 330, 410]:
        raise ValueError("Se requieren modelos de 2.05, 3.30 y 4.10 segundos (205, 330 y 410 muestras).")
    # Select with the longest window first, then pin the identical source file.
    longest = prepare_replay(specs[-1][1], recording=recording, endpoint_context=410, **kwargs)
    replays = [prepare_replay(run, recording=longest.path.name, endpoint_context=410, **kwargs)
               for _, run in specs[:-1]] + [longest]
    validate_comparison(replays)
    partitions = [json.loads((replay.cache_dir / "metadata.json").read_text())["split_subjects"]
                  for replay in replays]
    if any({key: set(value) for key, value in partition.items()} !=
           {key: set(value) for key, value in partitions[0].items()} for partition in partitions[1:]):
        raise ValueError("Los modelos deben compartir las particiones de sujetos train/validation/test.")
    return replays


def validate_comparison(replays):
    reference = replays[0]
    for replay in replays[1:]:
        for key in ("subject", "code", "trial", "fall_onset_frame", "fall_impact_frame"):
            if replay.record.get(key) != reference.record.get(key):
                raise ValueError("Los modelos no comparten la misma grabación o anotaciones.")
        for key in ("split", "seed", "post_impact_seconds", "n_classes"):
            if replay.config.get(key) != reference.config.get(key):
                raise ValueError(f"Configuración incompatible entre modelos: {key}.")
        for name in ("times", "frames", "raw", "endpoint_rows", "endpoint_times"):
            if not np.array_equal(getattr(replay, name), getattr(reference, name)):
                raise ValueError(f"Las señales y decisiones deben estar sincronizadas: {name}.")


class KFallComparisonPlayer(KFallReplayPlayer):
    """Reuse playback, seek, sensor, export and keyboard controls of the single player."""
    def __init__(self, replays, *, sensor="acc", view_seconds=6, speed=1, fps=25, interactive=True):
        from matplotlib import pyplot as plt
        if len(replays) != 3:
            raise ValueError("Se requieren tres modelos.")
        validate_comparison(replays)
        if not np.isfinite(speed) or not 0.1 <= speed <= 4 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Velocidad válida: 0.1–4; fps debe ser positivo.")
        self.replays, self.replay = replays, replays[0]
        self.sensor, self.speed, self.fps = sensor, float(speed), float(fps)
        self.current_time, self.playing = 0.0, bool(interactive)
        self.widgets, self._updating, self.animation = {}, False, None
        self.figure, self.axes = plt.subplots(5, 3, figsize=(19, 10), sharex=True,
                                            gridspec_kw={"height_ratios": [1, 1, 1, .85, .7]})
        self.figure.subplots_adjust(left=.065, right=.98, top=.80, bottom=.19 if interactive else .09,
                                    hspace=.42, wspace=.32)
        self.figure.canvas.manager.set_window_title("KFall · comparativa de predicciones")
        self.figure.text(.065, .965, f"TEST · {self.replay.path.name} · comparación de mejores checkpoints",
                         fontsize=16, weight="bold")
        self.status = []
        self.plots = []
        for column, replay in enumerate(replays):
            x = .065 + column * .326
            self.figure.text(x, .925, f"Ventana {replay.raw_win / 100:.2f} s · época {replay.best_epoch}",
                             fontsize=13, weight="bold")
            self.status.append(self.figure.text(x, .885, "", fontsize=10))
            plot = ReplayPlot(self.axes[:, column], replay, sensor=sensor, view_seconds=view_seconds)
            plot.axes[4].get_legend().remove()
            self.plots.append(plot)
        self.truth_text = self.figure.text(.065, .845, "", fontsize=11, color="#9333ea")
        self.figure.text(.065, .008,
            "Rojo: caída · verde: sin caída · naranja: postimpacto · gris: sin decisión / sin etiqueta. "
            "Decisiones alineadas desde 4.09 s; no se usa información futura. Espacio: pausa; flechas: decisión.", fontsize=9)
        if interactive:
            self._build_controls()
            from matplotlib.animation import FuncAnimation
            self.animation = FuncAnimation(self.figure, self._on_tick, init_func=self._init_animation,
                interval=1000 / self.fps, blit=False, cache_frame_data=False)
            self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self._render(draw=False)

    def _render(self, *, draw=True):
        infos = [plot.update(self.current_time, sensor=self.sensor) for plot in self.plots]
        for replay, info, text in zip(self.replays, infos, self.status):
            index = info["prediction_index"]
            if index is None:
                text.set_text("Esperando contexto común de 410 muestras")
                text.set_color("#64748b")
            else:
                cls = int(info["predicted_class"])
                confidence = replay.probabilities[index, cls]
                text.set_text(f"{info['predicted_label']} ({confidence:.1%})\n"
                              f"P(Fall): {info['p_fall']:.1%} · decisión {info['prediction_time']:.2f} s")
                text.set_color("#dc2626" if cls == 0 else "#d97706" if cls == 3 else "#15803d")
        self.truth_text.set_text(f"t = {self.current_time:.2f} s · Etiqueta real: {infos[0]['real_phase']}")
        self._sync_controls()
        if draw:
            self.figure.canvas.draw_idle()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs=3, type=Path, required=True, metavar="RUN")
    parser.add_argument("--recording", help="CSV de test; por defecto se selecciona una caída.")
    parser.add_argument("--subject")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--prediction-stride", type=int, default=10)
    parser.add_argument("--sensor", choices=["acc", "gyro"], default="acc")
    parser.add_argument("--view-seconds", type=float, default=6)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--time", type=float, default=0)
    parser.add_argument("--save-frame", type=Path)
    parser.add_argument("--save-gif", type=Path)
    args = parser.parse_args(argv)
    import matplotlib
    exporting = bool(args.save_frame or args.save_gif)
    if exporting:
        matplotlib.use("Agg")
    elif matplotlib.get_backend().lower() in {"agg", "pdf", "svg", "ps", "cairo", "template"}:
        parser.error("Se necesita un backend gráfico como TkAgg; usa --save-frame para exportar sin escritorio.")
    from matplotlib import pyplot as plt
    try:
        print("[comparison] Cargando mejores pesos y calculando inferencias de los tres modelos…", flush=True)
        replays = prepare_comparison(args.run_dirs, recording=args.recording, subject=args.subject,
            dataset_root=args.dataset_root, device=args.device, batch_size=args.batch_size,
            prediction_stride=args.prediction_stride)
        player = KFallComparisonPlayer(replays, sensor=args.sensor, view_seconds=args.view_seconds,
            speed=args.speed, fps=args.fps, interactive=not exporting)
        player.seek(args.time)
        if args.paused:
            player.playing = False
            player._render()
        if args.save_frame:
            print(f"[saved] {player.save_frame(args.save_frame)}")
        if args.save_gif:
            print(f"[saved] {player.save_gif(args.save_gif)}")
        if exporting:
            plt.close(player.figure)
        else:
            plt.show()
    except (ValueError, FileNotFoundError, RuntimeError, OSError, KeyError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
