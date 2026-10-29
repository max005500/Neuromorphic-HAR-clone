#!/usr/bin/env python3
"""Inspect cached KFall windows and their Sigma-Delta events interactively."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from visualization.kfall_data import KFallWindowStore
from visualization.kfall_plotting import draw_window
from preprocess.kfall_labels import class_names

CLASS_FILTERS = {"Todas": None, "Fall": 0, "Dynamic": 1, "Static": 2}
SENSOR_FILTERS = {"Acelerómetro": "acc", "Giroscopio": "gyro"}


class KFallViewer:
    """Browse existing windows without changing their encoding or partitions."""

    def __init__(
        self, store: KFallWindowStore, *, split: str = "train", sensor: str = "acc",
        class_id: int | None = None, subject: str | None = None,
        index: int | None = None, show_reconstruction: bool = True,
        interactive: bool = True, export_dir: Path = Path("results/kfall_viewer"),
    ) -> None:
        from matplotlib import pyplot as plt

        self.store = store
        count = getattr(store, "n_classes", 4 if store.config.get("post_impact_seconds", 0) > 0 else 3)
        self.class_filters = {"Todas": None, **{name: index for index, name in class_names(count).items()}}
        self.sensor = sensor
        self.show_reconstruction = show_reconstruction
        self.export_dir = Path(export_dir)
        self.interactive = interactive
        self._updating = False
        self.widgets: dict[str, Any] = {}
        self.figure, self.axes = plt.subplots(3, 2, figsize=(15.5, 9.5), sharex=True)
        self.figure.canvas.manager.set_window_title("KFall · señales y eventos Sigma-Delta")
        self.figure.subplots_adjust(left=0.065, right=0.79 if interactive else 0.97,
                                    top=0.865, bottom=0.19 if interactive else 0.16,
                                    hspace=0.45, wspace=0.24)
        self.title = self.figure.text(0.065, 0.956, "", fontsize=15, weight="bold")
        self.description = self.figure.text(0.065, 0.918, "", fontsize=10)
        self.details = self.figure.text(0.065, 0.115 if interactive else 0.075, "", fontsize=9, va="top")
        self.status = self.figure.text(0.065, 0.01, "", fontsize=9, color="#475569")
        try:
            self.set_filters(split, class_id, self._normalize_subject(subject), index=index)
            if interactive:
                self._build_controls()
                self._sync_controls()
                self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        except Exception:
            plt.close(self.figure)
            raise

    @staticmethod
    def _normalize_subject(value: str | None) -> str | None:
        if value is None or value.strip().lower() in {"", "all", "todos", "todas"}:
            return None
        return value.strip().upper()

    @property
    def index(self) -> int:
        return int(self.indices[self.position])

    def set_filters(
        self, split: str, class_id: int | None, subject: str | None, *, index: int | None = None,
    ) -> None:
        indices = self.store.get_indices(split, class_id=class_id, subject=subject)
        if not len(indices):
            raise ValueError("No hay ventanas para esta combinación de partición, clase y sujeto.")
        position = 0
        if index is not None:
            positions = np.flatnonzero(indices == index)
            if not len(positions):
                raise ValueError(f"La ventana {index} no pertenece a los filtros seleccionados.")
            position = int(positions[0])
        window = self.store.get_window(split, int(indices[position]))
        self.split, self.class_id, self.subject = split, class_id, window.subject if subject is not None else None
        self.indices, self.position, self.window = indices, position, window
        self._sync_controls()
        self.redraw()

    def set_position(self, position: int) -> None:
        position = int(np.clip(position, 0, len(self.indices) - 1))
        window = self.store.get_window(self.split, int(self.indices[position]))
        self.position, self.window = position, window
        self._sync_controls()
        self.redraw()

    def redraw(self) -> None:
        window = self.window
        draw_window(self.axes, window, sensor=self.sensor, show_reconstruction=self.show_reconstruction)
        self.title.set_text(f"{window.subject} · {window.code}/{window.trial} · {window.label} · {self.split}")
        self.description.set_text(
            f"Ventana {self.index} de la caché · {self.position + 1}/{len(self.indices)} en el filtro · "
            f"{len(window.raw)} muestras originales → {len(window.events)} pasos de eventos · "
            f"duración observada: {window.time_raw[-1]:.3f} s"
        )
        channels = range(3) if self.sensor == "acc" else range(3, 6)
        unit = "g" if self.sensor == "acc" else "°/s"
        theta = ", ".join(f"{window.theta[channel]:g}" for channel in channels)
        offsets = ", ".join(f"{window.offset[channel]:.4g}" for channel in channels)
        lines = [f"CSV: {window.path.name} · filas {window.start}:{window.stop} (fin excluido) · "
                 f"frames {window.frames[0]}–{window.end_frame} · θ = [{theta}] {unit} · offsets = [{offsets}] {unit}"]
        if window.fall_onset_time is not None:
            margin = (f"tiempo restante al impacto: {window.lead_time_ms:.1f} ms" if window.lead_time_ms >= 0
                      else f"tiempo desde el impacto: {-window.lead_time_ms:.1f} ms")
            lines.append(f"Inicio de caída: {window.fall_onset_time:.3f} s · impacto anotado: {window.fall_impact_time:.3f} s · {margin}")
        else:
            lines.append("Actividad cotidiana: sin anotación temporal de caída.")
        self.details.set_text("\n".join(lines))
        self.status.set_text("Eventos leídos de la caché y verificados contra la ventana original.")
        self.status.set_color("#475569")
        self.figure.canvas.draw_idle()

    def _build_controls(self) -> None:
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

        def radio(name, bounds, labels, title, callback):
            axis = self.figure.add_axes(bounds)
            axis.set_title(title, fontsize=10, loc="left")
            widget = RadioButtons(axis, labels)
            widget.on_clicked(callback)
            self.widgets[name] = widget

        radio("split", [0.835, 0.745, 0.145, 0.12], ["train", "val", "test"], "Partición", self._on_split)
        radio("class", [0.835, 0.535, 0.145, 0.16], list(self.class_filters), "Clase", self._on_class)
        radio("sensor", [0.835, 0.38, 0.145, 0.10], list(SENSOR_FILTERS), "Sensor", self._on_sensor)

        subject_axis = self.figure.add_axes([0.845, 0.282, 0.125, 0.035])
        subject_axis.set_title("Sujeto: SA06 o Todos", fontsize=9, loc="left")
        self.widgets["subject"] = TextBox(subject_axis, "", initial=self.subject or "Todos")
        self.widgets["subject"].on_submit(self._on_subject)
        reconstruction = CheckButtons(self.figure.add_axes([0.825, 0.20, 0.16, 0.045]),
                                       ["Reconstrucción"], [self.show_reconstruction])
        reconstruction.on_clicked(self._on_reconstruction)
        self.widgets["reconstruction"] = reconstruction

        self.widgets["slider"] = Slider(self.figure.add_axes([0.12, 0.06, 0.46, 0.025]),
                                          "En filtro", 1, max(2, len(self.indices)), valinit=self.position + 1, valstep=1)
        self.widgets["slider"].on_changed(self._on_slider)
        index_axis = self.figure.add_axes([0.69, 0.052, 0.095, 0.038])
        self.widgets["index"] = TextBox(index_axis, "Índice real ", initial=str(self.index))
        self.widgets["index"].on_submit(self._on_index)

        for name, label, bounds, callback in (
            ("previous", "Anterior", [0.835, 0.115, 0.069, 0.045], lambda event: self._safely(lambda: self.set_position(self.position - 1))),
            ("next", "Siguiente", [0.913, 0.115, 0.069, 0.045], lambda event: self._safely(lambda: self.set_position(self.position + 1))),
            ("random", "Aleatoria", [0.835, 0.052, 0.069, 0.045], lambda event: self._safely(lambda: self.set_position(int(np.random.default_rng().integers(len(self.indices)))))),
            ("save", "Guardar PNG", [0.913, 0.052, 0.069, 0.045], lambda event: self._safely(self.save)),
        ):
            button = Button(self.figure.add_axes(bounds), label)
            button.label.set_fontsize(9)
            button.on_clicked(callback)
            self.widgets[name] = button

    def _sync_controls(self) -> None:
        if not self.widgets:
            return
        self._updating = True
        try:
            for key, labels, value in (
                ("split", ["train", "val", "test"], self.split),
                ("class", list(self.class_filters), next(label for label, value in self.class_filters.items() if value == self.class_id)),
                ("sensor", list(SENSOR_FILTERS), next(label for label, value in SENSOR_FILTERS.items() if value == self.sensor)),
            ):
                if self.widgets[key].value_selected != value:
                    self.widgets[key].set_active(labels.index(value))
            slider = self.widgets["slider"]
            slider.valmax = max(2, len(self.indices))
            slider.ax.set_xlim(1, slider.valmax)
            slider.set_val(self.position + 1)
            slider.set_active(len(self.indices) > 1)
            self.widgets["subject"].set_val(self.subject or "Todos")
            self.widgets["index"].set_val(str(self.index))
            self.widgets["previous"].set_active(self.position > 0)
            self.widgets["next"].set_active(self.position < len(self.indices) - 1)
        finally:
            self._updating = False

    def _safely(self, action) -> None:
        if self._updating:
            return
        try:
            action()
        except (ValueError, FileNotFoundError, OSError, IndexError) as exc:
            self._sync_controls()
            self.status.set_text(str(exc))
            self.status.set_color("#b91c1c")
            self.figure.canvas.draw_idle()

    def _on_split(self, label) -> None:
        def select():
            subject = self.subject if self.subject in self.store.subjects(label) else None
            self.set_filters(label, self.class_id, subject)
        self._safely(select)

    def _on_class(self, label) -> None:
        self._safely(lambda: self.set_filters(self.split, self.class_filters[label], self.subject))

    def _on_subject(self, text) -> None:
        self._safely(lambda: self.set_filters(self.split, self.class_id, self._normalize_subject(text)))

    def _on_sensor(self, label) -> None:
        if self._updating:
            return
        self.sensor = SENSOR_FILTERS[label]
        self._sync_controls()
        self.redraw()

    def _on_reconstruction(self, label) -> None:
        self.show_reconstruction = not self.show_reconstruction
        self.redraw()

    def _on_slider(self, value) -> None:
        self._safely(lambda: self.set_position(int(value) - 1))

    def _on_index(self, text) -> None:
        def select():
            try:
                index = int(text)
            except ValueError as exc:
                raise ValueError("El índice de ventana debe ser un número entero.") from exc
            positions = np.flatnonzero(self.indices == index)
            if not len(positions):
                raise ValueError(f"La ventana {index} no pertenece al filtro actual.")
            self.set_position(int(positions[0]))
        self._safely(select)

    def _on_key(self, event) -> None:
        if any(getattr(self.widgets[name], "capturekeystrokes", False) for name in ("subject", "index")):
            return
        actions = {"right": lambda: self.set_position(self.position + 1),
                   "left": lambda: self.set_position(self.position - 1),
                   "home": lambda: self.set_position(0),
                   "end": lambda: self.set_position(len(self.indices) - 1), "s": self.save}
        if event.key in actions:
            self._safely(actions[event.key])

    def save(self, destination: Path | None = None) -> Path:
        if destination is None:
            window = self.window
            destination = self.export_dir / f"{self.split}_{window.subject}_{window.code}_{window.trial}_w{self.index}_{self.sensor}.png"
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(destination, dpi=160)
        self.status.set_text(f"Guardado: {destination}")
        self.figure.canvas.draw_idle()
        print(f"[viewer] {destination}")
        return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/kfall_cache/kfall_event_loader_default"),
                        help="Existing event cache directory containing metadata.json and train/val/test.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Override the KFall source root if the dataset has moved.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--sensor", choices=["acc", "gyro"], default="acc")
    parser.add_argument("--class", dest="class_filter", choices=["all", "fall", "dynamic", "static", "postimpact", "postimpacto"], default="all")
    parser.add_argument("--subject", default=None, help="Filter by subject ID, e.g. SA06.")
    parser.add_argument("--window", type=int, default=None, help="Actual zero-based window index within the selected split.")
    parser.add_argument("--no-reconstruction", action="store_true", help="Hide the reconstructed signal overlay.")
    parser.add_argument("--export-dir", type=Path, default=Path("results/kfall_viewer"), help="Destination for the GUI's Guardar PNG button.")
    parser.add_argument("--save", type=Path, default=None, metavar="PNG", help="Save the selected window as a PNG without opening a window.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    import matplotlib

    if args.save is not None:
        matplotlib.use("Agg")
    elif matplotlib.get_backend().lower() in {"agg", "pdf", "svg", "ps", "cairo", "template"}:
        parser.error("Se necesita un backend gráfico de Matplotlib (por ejemplo TkAgg). "
                     "Puedes exportar sin interfaz con --save results/kfall_viewer/window.png.")
    from matplotlib import pyplot as plt

    try:
        store = KFallWindowStore(args.cache_dir, dataset_root=args.dataset_root)
        class_id = {"all": None, "fall": 0, "dynamic": 1, "static": 2, "postimpact": 3, "postimpacto": 3}[args.class_filter]
        viewer = KFallViewer(store, split=args.split, sensor=args.sensor, class_id=class_id,
                             subject=args.subject, index=args.window, show_reconstruction=not args.no_reconstruction,
                             interactive=args.save is None, export_dir=args.export_dir)
        if args.save is not None:
            viewer.save(args.save)
            plt.close(viewer.figure)
        else:
            plt.show()
    except (ValueError, FileNotFoundError, OSError, IndexError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
