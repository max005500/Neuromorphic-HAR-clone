#!/usr/bin/env python3
"""Ajustar visualmente theta Sigma-Delta del giroscopio KFall, con acelerómetro fijo."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from preprocess.event_encoding import sigma_delta
from train_kfall import read_theta_file
from view_kfall import KFallViewer
from visualization.kfall_data import KFallWindowStore


def preview_gyro(window, gyro_theta, *, dead_zone=.5, sd_init='x0'):
    """Encode only gyro channels. Original offsets and accelerometer stay intact."""
    theta = np.asarray(gyro_theta, dtype=np.float64)
    if theta.shape != (3,) or not np.isfinite(theta).all() or np.any(theta <= 0):
        raise ValueError('Se requieren tres umbrales positivos y finitos en grados/s.')
    events, _, reconstruction = sigma_delta(window.interpolated[:, 3:], theta=theta,
        init=sd_init, dead_zone=dead_zone, return_reconstruction=True)
    candidate = replace(window,
        events=np.concatenate((window.events[:, :3], events), axis=1),
        reconstructed=np.concatenate((window.reconstructed[:, :3], reconstruction), axis=1),
        theta=np.r_[window.theta[:3], theta])
    error = candidate.reconstructed[:, 3:] - window.interpolated[:, 3:]
    metrics = dict(rmse=np.sqrt(np.mean(error.astype(np.float64) ** 2, axis=0)),
                   on=events[:, :, 0].sum(axis=0), off=events[:, :, 1].sum(axis=0))
    return candidate, metrics


class GyroThresholdTuner(KFallViewer):
    def __init__(self, store, *, output=Path('configs/kfall_theta_gyro_custom.json'),
                 gyro_theta=None, theta_min=.001, theta_max=1000., **kwargs):
        self.base_theta = np.asarray(store.cfg.theta_sd, dtype=np.float64).copy()
        self.gyro_theta = self.base_theta[3:].copy() if gyro_theta is None else np.asarray(gyro_theta, dtype=float)
        self.theta_min, self.theta_max = float(theta_min), float(theta_max)
        if (not np.isfinite([theta_min, theta_max]).all() or not 0 < theta_min < theta_max):
            raise ValueError('El rango de theta debe ser finito, positivo y creciente.')
        self._validate(self.gyro_theta)
        self._validate(self.base_theta[3:])
        self.output, self.linked, self._theta_updating = Path(output), False, False
        kwargs.setdefault('sensor', 'gyro')
        super().__init__(store, **kwargs)
        self.figure.canvas.manager.set_window_title('KFall · ajuste de umbrales del giroscopio')
        self.figure.subplots_adjust(bottom=.40)
        self.details.set_position((.065, .355))
        self.redraw()

    def _validate(self, values):
        values = np.asarray(values)
        if (values.shape != (3,) or not np.isfinite(values).all()
                or np.any(values < self.theta_min) or np.any(values > self.theta_max)):
            raise ValueError(f'Theta XYZ debe estar entre {self.theta_min:g} y {self.theta_max:g} grados/s.')

    def _build_controls(self):
        super()._build_controls()
        from matplotlib.widgets import Slider, TextBox, Button, CheckButtons
        for index, axis in enumerate('XYZ'):
            y = .285 - index * .043
            slider = Slider(self.figure.add_axes([.15, y, .40, .021]), f'log θ{axis}',
                            np.log10(self.theta_min), np.log10(self.theta_max),
                            valinit=np.log10(self.gyro_theta[index]))
            slider.on_changed(lambda value, i=index: self._change(i, 10 ** float(value)))
            box = TextBox(self.figure.add_axes([.665, y - .005, .115, .034]), f'θ{axis} °/s ',
                          initial=f'{self.gyro_theta[index]:g}')
            box.on_submit(lambda value, i=index: self._change(i, value))
            self.widgets[f'theta_slider_{index}'] = slider
            self.widgets[f'theta_text_{index}'] = box
        linked = CheckButtons(self.figure.add_axes([.82, .345, .17, .025]), ['Vincular XYZ'], [False])
        linked.on_clicked(lambda label: setattr(self, 'linked', not self.linked))
        self.widgets['linked'] = linked
        for key, title, x, callback in (
            ('reset_theta', 'Restaurar θ', .15, self.reset_theta),
            ('save_theta', 'Guardar theta JSON', .40, self.save_theta),
        ):
            button = Button(self.figure.add_axes([x, .125, .20, .039]), title)
            button.on_clicked(lambda event, action=callback: self._safely(action))
            self.widgets[key] = button
        self.figure.text(.065, .31, 'Umbrales del giroscopio · deslizador logarítmico o valor exacto en °/s', fontsize=10)
        self.figure.text(.065, .175,
            'θ = paso de reconstrucción; disparo = dead-zone × θ. Ajuste exploratorio sobre train; revisar también validation.',
            fontsize=9, color='#475569')

    def _change(self, index, value):
        if self._theta_updating or self._updating:
            return
        def update():
            values = self.gyro_theta.copy()
            if self.linked:
                values[:] = float(value)
            else:
                values[index] = float(value)
            self._validate(values)
            self.gyro_theta = values
            self._sync_theta_controls()
            self.redraw()
        self._safely(update)
        self._sync_theta_controls()

    def _sync_theta_controls(self):
        self._theta_updating = True
        try:
            for index, value in enumerate(self.gyro_theta):
                key = f'theta_slider_{index}'
                if key not in self.widgets:
                    continue
                self.widgets[key].set_val(np.log10(value))
                self.widgets[key].valtext.set_text(f'{value:.5g} °/s')
                self.widgets[f'theta_text_{index}'].set_val(f'{value:.8g}')
        finally:
            self._theta_updating = False

    def reset_theta(self):
        self.gyro_theta = self.base_theta[3:].copy()
        self._sync_theta_controls()
        self.redraw()

    def redraw(self):
        source = self.window
        candidate, self.metrics = preview_gyro(source, self.gyro_theta,
            dead_zone=self.store.cfg.dead_zone, sd_init=self.store.cfg.sd_init)
        try:
            self.window = candidate
            super().redraw()
        finally:
            self.window = source
        if self.sensor == 'gyro':
            for index in range(3):
                axis = self.axes[index, 0]
                if self.show_reconstruction:
                    axis.step(source.time_events, source.reconstructed[:, 3 + index], where='post',
                              color='#64748b', linestyle='--', linewidth=.9, alpha=.75, label='Reconstrucción base')
                base_rmse = np.sqrt(np.mean((source.reconstructed[:, 3 + index].astype(float)
                                            - source.interpolated[:, 3 + index]) ** 2))
                axis.set_title(axis.get_title() +
                    f" · RMSE {self.metrics['rmse'][index]:.3g} (base {base_rmse:.3g}) °/s", fontsize=9)
                self.axes[index, 1].set_title(self.axes[index, 1].get_title() +
                    f' · base: {int(source.events[:, 3 + index].sum())}', fontsize=9)
            self.axes[0, 0].legend(loc='best', fontsize=7)
        self.status.set_text(f'Acc fijo θ={self.base_theta[:3].tolist()} g · vista previa recalculada · caché y etiquetas originales conservadas.')
        self._sync_theta_controls()
        self.figure.canvas.draw_idle()

    def _on_key(self, event):
        if any(getattr(self.widgets.get(f'theta_text_{i}'), 'capturekeystrokes', False) for i in range(3)):
            return
        super()._on_key(event)

    def save_theta(self, destination=None):
        destination = self.output if destination is None else Path(destination)
        theta = np.r_[self.base_theta[:3], self.gyro_theta].tolist()
        document = {
            'dataset': 'kfall', 'theta_best': theta,
            'channels': ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ'],
            'units': ['g'] * 3 + ['degrees/s'] * 3,
            'calibration': {'method': 'manual_gyro_preview', 'cache_dir': str(self.store.cache_dir),
                'split': self.split, 'window_index': self.index, 'subject': self.window.subject,
                'recording': self.window.path.name, 'dead_zone': self.store.cfg.dead_zone,
                'sd_init': self.store.cfg.sd_init, 'raw_win': self.store.cfg.raw_win,
                'out_win': self.store.cfg.out_win, 'resample_hz': self.store.cfg.resample_hz,
                'accelerometer_fixed': True},
            'note': 'Ajuste visual, no óptimo validado. --theta-file carga solo los seis umbrales; conservar los demás parámetros del encoder.'}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        temporary.replace(destination)
        self.status.set_text(f'Umbrales guardados: {destination}')
        self.figure.canvas.draw_idle()
        print(f'[theta] {destination}', flush=True)
        return destination


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--cache-dir', type=Path, help='Caché con metadata.json y particiones.')
    source.add_argument('--run-dir', type=Path, help='Entrenamiento con config.json; usa su caché.')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--class', dest='class_filter', choices=['all', 'fall', 'dynamic', 'static', 'postimpact'], default='all')
    parser.add_argument('--subject')
    parser.add_argument('--window', type=int)
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument('--theta-file', type=Path, help='Reabrir un ajuste; los tres theta acc deben coincidir con la caché.')
    initial.add_argument('--gyro-theta', type=float, nargs=3)
    parser.add_argument('--theta-min', type=float, default=.001)
    parser.add_argument('--theta-max', type=float, default=1000.)
    parser.add_argument('--output', type=Path, default=Path('configs/kfall_theta_gyro_custom.json'))
    parser.add_argument('--save', type=Path, help='Exportar PNG sin escritorio.')
    parser.add_argument('--export-dir', type=Path, default=Path('results/kfall_gyro_tuner'))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    import matplotlib
    if args.save:
        matplotlib.use('Agg')
    elif matplotlib.get_backend().lower() in {'agg', 'pdf', 'svg', 'ps', 'cairo', 'template'}:
        parser.error('Se necesita un backend gráfico como TkAgg; usa --save para exportar sin escritorio.')
    from matplotlib import pyplot as plt
    try:
        cache = args.cache_dir
        if args.run_dir:
            from evaluation.kfall_data import _resolve_cache
            cfg = json.loads((args.run_dir / 'config.json').read_text())
            cache = _resolve_cache(cfg, None)
        store = KFallWindowStore(cache, dataset_root=args.dataset_root)
        gyro = args.gyro_theta
        if args.theta_file:
            theta = read_theta_file(args.theta_file)
            if not np.array_equal(theta[:3], np.asarray(store.cfg.theta_sd)[:3]):
                raise ValueError('El theta del acelerómetro debe coincidir con la caché; solo se ajusta el giroscopio.')
            gyro = theta[3:]
        viewer = GyroThresholdTuner(store, output=args.output, gyro_theta=gyro,
            theta_min=args.theta_min, theta_max=args.theta_max, split=args.split,
            class_id={'all': None, 'fall': 0, 'dynamic': 1, 'static': 2, 'postimpact': 3}[args.class_filter],
            subject=args.subject, index=args.window, interactive=args.save is None, export_dir=args.export_dir)
        if args.save:
            viewer.save(args.save)
            viewer.save_theta()
            plt.close(viewer.figure)
        else:
            plt.show()
    except (ValueError, OSError, KeyError, RuntimeError, ImportError, IndexError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
