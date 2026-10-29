#!/usr/bin/env python3
"""Browse matched KFall window durations, signals, and cached Sigma-Delta events."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from visualization.kfall_variants import KFallVariantStore
from visualization.kfall_variant_plotting import draw_variant_comparison


SENSORS = {"Acelerómetro": "acc", "Giroscopio": "gyro"}
CLASSES = {"all": None, "fall": 0, "dynamic": 1, "static": 2,
           "postimpact": 3, "postimpacto": 3}


class KFallVariantViewer:
    """Navigate the same recording and prediction endpoint in every variant."""

    def __init__(
        self, store: KFallVariantStore, *, split: str = "test", sensor: str = "acc",
        class_id: int | None = None, subject: str | None = None, index: int | None = None,
        duration: float | None = None, show_reconstruction: bool = True,
        interactive: bool = True, export_dir: Path = Path("results/kfall_variant_viewer"),
    ) -> None:
        from matplotlib import pyplot as plt

        if sensor not in SENSORS.values():
            raise ValueError("sensor must be acc or gyro.")
        self.store = store
        self.sensor = sensor
        self.show_reconstruction = show_reconstruction
        self.interactive = interactive
        self.export_dir = Path(export_dir)
        self.class_filters = {"Todas": None, **{name: key for key, name in store.class_names.items()}}
        self.view_filters = {"Todas": None, **{f"{value:.2f} s": index for index, value in enumerate(store.durations)}}
        self.selected_variant = None
        if duration is not None:
            positions = np.flatnonzero(np.isclose(store.durations, duration, rtol=0, atol=1e-8))
            if len(positions) != 1:
                raise ValueError(f"Duration {duration:g} s is unavailable; choose {store.durations}.")
            self.selected_variant = int(positions[0])
        self.widgets: dict[str, Any] = {}
        self._updating = False
        self.axes = None
        self.figure = plt.figure(figsize=(18, 9.5))
        self.figure.canvas.manager.set_window_title("KFall · comparación de ventanas y eventos")
        self.title = self.figure.text(.05, .975, "", fontsize=15, weight="bold", va="top")
        self.description = self.figure.text(.05, .944, "", fontsize=10, va="top")
        self.sampling = self.figure.text(.05, .912, "", fontsize=9, va="top")
        self.details = self.figure.text(.05, .145, "", fontsize=9, va="top")
        self.status = self.figure.text(.05, .012, "", fontsize=9, color="#475569")
        try:
            self.set_filters(split, class_id, self._subject(subject), index=index)
            if interactive:
                self._build_controls()
                self._sync_controls()
                self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        except Exception:
            plt.close(self.figure)
            raise

    @staticmethod
    def _subject(value: str | None) -> str | None:
        if value is None or value.strip().lower() in {"", "all", "todos", "todas"}:
            return None
        return value.strip().upper()

    @property
    def index(self) -> int:
        return int(self.indices[self.position])

    def set_filters(self, split, class_id, subject, *, index=None) -> None:
        indices = self.store.get_indices(split, class_id=class_id, subject=subject)
        if not len(indices):
            raise ValueError("No hay ventanas comunes para esa partición, clase y sujeto.")
        if index is None:
            index = (self.index if getattr(self, "split", None) == split and self.index in indices
                     else int(indices[0]))
        matches = np.flatnonzero(indices == index)
        if len(matches) != 1:
            raise ValueError(f"El índice comparativo {index} no pertenece al filtro seleccionado.")
        windows = self.store.get_windows(split, int(index))
        self.split, self.class_id = split, class_id
        self.subject = windows[0].subject if subject is not None else None
        self.indices, self.position, self.windows = indices, int(matches[0]), windows
        self._sync_controls()
        self.redraw()

    def set_position(self, position: int) -> None:
        position = int(np.clip(position, 0, len(self.indices) - 1))
        windows = self.store.get_windows(self.split, int(self.indices[position]))
        self.position, self.windows = position, windows
        self._sync_controls()
        self.redraw()

    def set_view(self, variant: int | None) -> None:
        if variant is not None and not 0 <= variant < len(self.store.durations):
            raise ValueError("La variante seleccionada no existe.")
        self.selected_variant = variant
        self._sync_controls()
        self.redraw()

    def redraw(self) -> None:
        selected = self.windows if self.selected_variant is None else [self.windows[self.selected_variant]]
        shape = (3, 2 * len(selected))
        if self.axes is None or self.axes.shape != shape:
            if self.axes is not None:
                for axis in self.axes.flat:
                    axis.remove()
            grid = self.figure.add_gridspec(
                *shape, left=.05, right=.81 if self.interactive else .98,
                top=.82, bottom=.235, hspace=.60, wspace=.34,
            )
            self.axes = grid.subplots(squeeze=False)
        draw_variant_comparison(self.axes, selected, sensor=self.sensor,
                                show_reconstruction=self.show_reconstruction)
        first = self.windows[0]
        self.title.set_text(f"{first.subject} · {first.code}/{first.trial} · {first.label} · {self.split}")
        self.description.set_text(
            f"Instante común: FrameCounter {first.end_frame} · índice comparativo {self.index} · "
            f"{self.position + 1}/{len(self.indices)} en el filtro · 0 s = final de la ventana"
        )
        entries = []
        for window in selected:
            intermediate = (f" → {len(window.analog)} a {window.resample_hz:g} Hz"
                            if getattr(window, "resample_hz", None) is not None else "")
            entries.append(f"{len(window.raw) / 100:.2f} s: {len(window.raw)} originales"
                           f"{intermediate} → {len(window.events)} pasos ΣΔ")
        self.sampling.set_text("   |   ".join(entries))
        channels = range(3) if self.sensor == "acc" else range(3, 6)
        units = "g" if self.sensor == "acc" else "°/s"
        theta = ", ".join(f"{first.theta[channel]:g}" for channel in channels)
        cache_indices = "; ".join(f"{duration:.2f} s: {window.index}"
                                  for duration, window in zip(self.store.durations, self.windows))
        coverage = self.store.coverage[self.split]
        self.details.set_text(
            f"CSV: {first.path.name} · θ = [{theta}] {units} · índices en las cachés: {cache_indices}\n"
            f"Ventanas comunes: {coverage['common_windows']} · por variante: {coverage['variant_windows']} · "
            f"fuera de la intersección: {coverage['discarded_windows']}"
        )
        self.status.set_text("Señales y eventos verificados contra el CSV. La zona gris indica contexto fuera de cada ventana.")
        self.status.set_color("#475569")
        self.figure.canvas.draw_idle()

    def _build_controls(self) -> None:
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

        def radio(name, bounds, mapping, title, callback):
            axis = self.figure.add_axes(bounds)
            axis.set_title(title, fontsize=10, loc="left")
            control = RadioButtons(axis, list(mapping))
            for label in control.labels:
                label.set_fontsize(9)
            control.on_clicked(callback)
            self.widgets[name] = control

        radio("view", [.845, .68, .145, .15], self.view_filters, "Ventana / comparación",
              lambda label: self._safely(lambda: self.set_view(self.view_filters[label])))
        radio("split", [.845, .52, .145, .11], {key: key for key in ("train", "val", "test")},
              "Partición", self._on_split)
        radio("class", [.845, .30, .145, .17], self.class_filters, "Clase",
              lambda label: self._safely(lambda: self.set_filters(self.split, self.class_filters[label], self.subject)))
        radio("sensor", [.845, .17, .145, .085], SENSORS, "Sensor", self._on_sensor)
        subject_axis = self.figure.add_axes([.85, .106, .135, .03])
        subject_axis.set_title("Sujeto: SA07 o Todos", fontsize=9, loc="left")
        self.widgets["subject"] = TextBox(subject_axis, "", initial=self.subject or "Todos")
        self.widgets["subject"].on_submit(
            lambda value: self._safely(lambda: self.set_filters(self.split, self.class_id, self._subject(value))))
        control = CheckButtons(self.figure.add_axes([.84, .035, .15, .04]),
                               ["Reconstrucción"], [self.show_reconstruction])
        control.labels[0].set_fontsize(9)
        control.on_clicked(self._on_reconstruction)
        self.widgets["reconstruction"] = control
        self.widgets["slider"] = Slider(self.figure.add_axes([.10, .072, .36, .022]),
                                          "En filtro", 1, max(2, len(self.indices)),
                                          valinit=self.position + 1, valstep=1)
        self.widgets["slider"].on_changed(lambda value: self._safely(lambda: self.set_position(int(value) - 1)))
        self.widgets["index"] = TextBox(self.figure.add_axes([.64, .063, .10, .034]),
                                          "Índice comparativo ", initial=str(self.index))
        self.widgets["index"].on_submit(self._on_index)
        for name, label, left, action in (
            ("previous", "Anterior", .10, lambda: self.set_position(self.position - 1)),
            ("next", "Siguiente", .24, lambda: self.set_position(self.position + 1)),
            ("random", "Aleatoria", .38, lambda: self.set_position(int(np.random.default_rng().integers(len(self.indices))))),
            ("save", "Guardar PNG", .62, self.save),
        ):
            control = Button(self.figure.add_axes([left, .025, .12, .027]), label)
            control.on_clicked(lambda event, action=action: self._safely(action))
            self.widgets[name] = control

    def _sync_controls(self) -> None:
        if not self.widgets:
            return
        self._updating = True
        try:
            for name, mapping, value in (
                ("view", self.view_filters, self.selected_variant),
                ("split", {key: key for key in ("train", "val", "test")}, self.split),
                ("class", self.class_filters, self.class_id), ("sensor", SENSORS, self.sensor),
            ):
                label = next(key for key, target in mapping.items() if target == value)
                if self.widgets[name].value_selected != label:
                    self.widgets[name].set_active(list(mapping).index(label))
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
        except (ValueError, FileNotFoundError, OSError, IndexError, RuntimeError) as exc:
            self._sync_controls()
            self.status.set_text(str(exc))
            self.status.set_color("#b91c1c")
            self.figure.canvas.draw_idle()

    def _on_split(self, label) -> None:
        self._safely(lambda: self.set_filters(
            label, self.class_id, self.subject if self.subject in self.store.subjects(label) else None))

    def _on_sensor(self, label) -> None:
        if self._updating:
            return
        self.sensor = SENSORS[label]
        self._sync_controls()
        self.redraw()

    def _on_reconstruction(self, label) -> None:
        if self._updating:
            return
        self.show_reconstruction = not self.show_reconstruction
        self.redraw()

    def _on_index(self, value) -> None:
        def select():
            try:
                index = int(value)
            except ValueError as exc:
                raise ValueError("El índice comparativo debe ser un número entero.") from exc
            positions = np.flatnonzero(self.indices == index)
            if len(positions) != 1:
                raise ValueError("El índice comparativo no pertenece al filtro actual.")
            self.set_position(int(positions[0]))
        self._safely(select)

    def _on_key(self, event) -> None:
        if any(getattr(self.widgets[key], "capturekeystrokes", False) for key in ("subject", "index")):
            return
        actions = {"right": lambda: self.set_position(self.position + 1),
                   "left": lambda: self.set_position(self.position - 1),
                   "home": lambda: self.set_position(0),
                   "end": lambda: self.set_position(len(self.indices) - 1), "s": self.save}
        if event.key in actions:
            self._safely(actions[event.key])

    def save(self, destination: str | Path | None = None) -> Path:
        window = self.windows[0]
        view = "all" if self.selected_variant is None else f"{self.store.durations[self.selected_variant]:.2f}s"
        destination = (self.export_dir / f"{self.split}_{window.subject}_{window.code}_{window.trial}_f{window.end_frame}_{view}_{self.sensor}.png"
                       if destination is None else Path(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(destination, dpi=160)
        self.status.set_text(f"Guardado: {destination}")
        self.figure.canvas.draw_idle()
        print(f"[variants] {destination}")
        return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--comparison", type=Path, default=None,
                         help="comparison.json from dataset preparation; defaults to the calibrated 200 Hz comparison.")
    sources.add_argument("--cache-dirs", type=Path, nargs="+", default=None,
                         help="Two or more existing event cache directories instead of a comparison manifest.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sensor", choices=["acc", "gyro"], default="acc")
    parser.add_argument("--class", dest="class_filter", choices=list(CLASSES), default="all")
    parser.add_argument("--subject", default=None)
    parser.add_argument("--window", type=int, default=None, help="Index in the common comparison, distinct from each cache's index.")
    parser.add_argument("--duration", type=float, default=None, help="Initially display one duration instead of all variants.")
    parser.add_argument("--no-reconstruction", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=Path("results/kfall_variant_viewer"))
    parser.add_argument("--save", type=Path, default=None, metavar="FILE", help="Export PNG/PDF/SVG without opening a desktop window.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    import matplotlib

    if args.save is not None:
        matplotlib.use("Agg")
    elif matplotlib.get_backend().lower() in {"agg", "pdf", "svg", "ps", "cairo", "template"}:
        parser.error("Se necesita un backend gráfico de Matplotlib (por ejemplo TkAgg). "
                     "Puedes exportar con --save results/kfall_variant_viewer/comparison.png.")
    from matplotlib import pyplot as plt

    try:
        store = KFallVariantStore(args.comparison, cache_dirs=args.cache_dirs, dataset_root=args.dataset_root)
        viewer = KFallVariantViewer(store, split=args.split, sensor=args.sensor,
                                    class_id=CLASSES[args.class_filter], subject=args.subject,
                                    index=args.window, duration=args.duration,
                                    show_reconstruction=not args.no_reconstruction,
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
