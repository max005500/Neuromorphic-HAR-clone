"""Matplotlib recording browser; GPU work stays off the GUI event thread."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import textwrap

from animate_kfall import KFallReplayPlayer
from visualization.kfall_replay_plot import ReplayPlot


class TemporalReplayBrowser(KFallReplayPlayer):
    PAGE_SIZE = 10
    SPLIT_LABELS = {'Test': 'test', 'Validación': 'val', 'Train': 'train'}

    def __init__(self, source, replay, *, sensor='acc', view_seconds=6., speed=1., subject=None, executor=None):
        super().__init__(replay, sensor=sensor, view_seconds=view_seconds, speed=speed, interactive=True)
        self.source, self.view_seconds = source, view_seconds
        self.split_name = replay.config.get('replay_split', 'test')
        self._browser_updating = False
        self._page, self._busy, self._closed = 0, False, False
        self._future = None
        self._executor = executor if executor is not None else ThreadPoolExecutor(max_workers=1,
                                                                                thread_name_prefix='kfall-replay')
        self._owns_executor = executor is None
        self.file_buttons = []
        try:
            self.catalog = source.records(self.split_name)
            # Reserve a sidebar without moving the plot's data-coordinate artists.
            self.figure.set_size_inches(17, 10, forward=True)
            self.figure.subplots_adjust(left=.34, right=.98, top=.815, bottom=.19)
            for text in self.figure.texts:
                text.set_x(.28 + .72 * text.get_position()[0])
            self.footer_text.set_fontsize(7)
            for widget in self.widgets.values():
                box = widget.ax.get_position()
                widget.ax.set_position([.28 + .72 * box.x0, box.y0, .72 * box.width, box.height])
            box = self.widgets['time'].ax.get_position()
            self.widgets['time'].ax.set_position([box.x0, box.y0, box.width - .035, box.height])
            self._build_browser(subject or '')
            self._load_timer = self.figure.canvas.new_timer(interval=100)
            self._load_timer.add_callback(self._poll_load)
            self.figure.canvas.mpl_connect('close_event', self.close_browser)
            self._refresh_files(follow_current=True)
        except Exception:
            self.close_browser()
            from matplotlib import pyplot as plt
            plt.close(self.figure)
            raise

    def _build_browser(self, subject):
        from matplotlib.widgets import Button, RadioButtons, TextBox
        self.figure.text(.02, .965, 'Grabaciones', fontsize=15, weight='bold')
        self.figure.text(.02, .938, 'Selecciona un archivo de la lista', fontsize=9)
        labels = list(self.SPLIT_LABELS)
        self.split_selector = RadioButtons(self.figure.add_axes([.02, .81, .235, .11]), labels,
            active=list(self.SPLIT_LABELS.values()).index(self.split_name))
        self.split_selector.on_clicked(self._change_split)
        self.figure.text(.02, .785, 'Filtrar sujeto (vacío = todos)', fontsize=9)
        self.subject_box = TextBox(self.figure.add_axes([.02, .738, .235, .034]), '', initial=subject)
        self.figure.text(.02, .711, 'Buscar archivo / tarea · Enter', fontsize=9)
        self.search_box = TextBox(self.figure.add_axes([.02, .665, .235, .034]), '')
        self.subject_box.on_submit(self._on_filter)
        self.search_box.on_submit(self._on_filter)
        self.page_text = self.figure.text(.02, .63, '', fontsize=9)
        self.page_previous = Button(self.figure.add_axes([.02, .16, .113, .036]), 'Página anterior')
        self.page_next = Button(self.figure.add_axes([.142, .16, .113, .036]), 'Página siguiente')
        for widget in (self.page_previous, self.page_next):
            widget.label.set_fontsize(8)
        self.page_previous.on_clicked(lambda event: self._move_page(-1))
        self.page_next.on_clicked(lambda event: self._move_page(1))
        self.status_text = self.figure.text(.02, .13, '', va='top', fontsize=9, color='#475569')

    def _status(self, message, *, error=False):
        self.status_text.set_text(textwrap.fill(str(message)[:300], width=35))
        self.status_text.set_color('#b91c1c' if error else '#475569')
        self.figure.canvas.draw_idle()

    def _refresh_files(self, *, follow_current=False):
        subject, query = self.subject_box.text.strip().upper(), self.search_box.text.strip().lower()
        self.filtered = [row for row in self.catalog
                         if subject in row['subject'].upper() and query in row['file'].lower()]
        if follow_current:
            index = next((i for i, row in enumerate(self.filtered)
                          if row['file'].lower() == self.replay.path.name.lower()), 0)
            self._page = index // self.PAGE_SIZE
        self._page = min(self._page, max(0, math.ceil(len(self.filtered) / self.PAGE_SIZE) - 1))
        self._show_page()

    def _show_page(self):
        from matplotlib.widgets import Button
        for _, button in self.file_buttons:
            button.disconnect_events()
            self.figure.delaxes(button.ax)
        self.file_buttons = []
        begin = self._page * self.PAGE_SIZE
        rows = self.filtered[begin:begin + self.PAGE_SIZE]
        pages = max(1, math.ceil(len(self.filtered) / self.PAGE_SIZE))
        self.page_text.set_text(f'{len(self.filtered)} archivos · página {self._page + 1}/{pages}')
        for number, row in enumerate(rows):
            current = (row['split'] == self.replay.config.get('replay_split', 'test')
                       and row['file'].lower() == self.replay.path.name.lower())
            button = Button(self.figure.add_axes([.02, .585 - number * .041, .235, .035]), row['file'],
                            color='#bfdbfe' if current else '#f1f5f9', hovercolor='#dbeafe')
            button.label.set_fontsize(9)
            button.on_clicked(lambda event, selected=row: self._select_file(selected))
            button.active = not self._busy
            self.file_buttons.append((row, button))
        if not self._busy:
            self._status('Selecciona un archivo.' if rows else 'Sin coincidencias. Revisa los filtros.')
        self.figure.canvas.draw_idle()

    def _on_filter(self, value):
        if not self._browser_updating and not self._busy:
            self._page = 0
            self._refresh_files()

    def _on_key(self, event):
        if any(getattr(getattr(self, name, None), 'capturekeystrokes', False)
               for name in ('subject_box', 'search_box')):
            return
        super()._on_key(event)

    def _move_page(self, direction):
        if self._busy:
            return
        last = max(0, math.ceil(len(self.filtered) / self.PAGE_SIZE) - 1)
        self._page = min(max(self._page + direction, 0), last)
        self._show_page()

    def _change_split(self, label):
        if self._browser_updating or self._busy:
            return
        split_name = self.SPLIT_LABELS[label]
        if split_name == self.split_name:
            return
        try:
            catalog = self.source.records(split_name)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            self._browser_updating = True
            try:
                self.split_selector.set_active(list(self.SPLIT_LABELS.values()).index(self.split_name))
            finally:
                self._browser_updating = False
            self._status(exc, error=True)
            return
        self.split_name, self.catalog, self._page = split_name, catalog, 0
        self._browser_updating = True
        try:
            self.subject_box.set_val('')
            self.search_box.set_val('')
        finally:
            self._browser_updating = False
        self._refresh_files()
        if self.filtered:
            self._select_file(self.filtered[0])

    def _set_busy(self, busy):
        self._busy = busy
        controls = [self.split_selector, self.subject_box, self.search_box,
                    self.page_previous, self.page_next, *(button for _, button in self.file_buttons)]
        for widget in controls:
            widget.active = not busy

    def _select_file(self, row):
        if self._busy or self._closed:
            return
        self.playing = False
        self._sync_controls()
        self._set_busy(True)
        self._status(f'Calculando predicciones: {row["file"]} · {row["split"].upper()}')
        try:
            self._future = self._executor.submit(self.source.replay, row['split'], recording=row['file'])
        except Exception as exc:
            self._set_busy(False)
            self._status(exc, error=True)
            return
        self._load_timer.start()

    def _poll_load(self):
        if self._closed or self._future is None:
            return True
        if not self._future.done():
            return True
        self._load_timer.stop()
        future, self._future = self._future, None
        try:
            replay = future.result()
            self._install_replay(replay)
            error = None
        except Exception as exc:
            error = str(exc)
        self._set_busy(False)
        self._show_page()
        if error is not None:
            self._status(error, error=True)
        else:
            self._status(f'{self.replay.path.name} · {self.replay.subject} · memoria reiniciada al inicio.')
        return True

    def _install_replay(self, replay):
        self.replay, self.current_time, self.playing = replay, 0., False
        for axis in self.axes:
            axis.clear()
        self.plot = ReplayPlot(self.axes, replay, sensor=self.sensor, view_seconds=self.view_seconds)
        self.recording_text.set_text(f'{replay.subject} · {replay.record["code"]}/{replay.record["trial"]} · '
            f'{replay.config.get("replay_split", "test").upper()} · mejor checkpoint: época {replay.best_epoch}')
        self.footer_text.set_text('Cada decisión se muestra al terminar su ventana. '
                                 f'Etiquetado guardado: {replay.config.get("label_policy", "legacy")}.')
        slider = self.widgets['time']
        slider.valmax = float(replay.times[-1])
        slider.ax.set_xlim(0, slider.valmax)
        self._updating = True
        try:
            slider.set_val(0.)
        finally:
            self._updating = False
        self._render()

    def close_browser(self, event=None):
        self._closed = True
        if hasattr(self, '_load_timer'):
            self._load_timer.stop()
        if self._future is not None:
            self._future.cancel()
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
