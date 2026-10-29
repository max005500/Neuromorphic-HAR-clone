"""Keep multi-duration KFall views aligned to the same observed endpoint."""

from __future__ import annotations

import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from visualization.kfall_variant_plotting import draw_variant_comparison


def variant_window(size, *, end_frame=1000, class_id=0):
    frames = np.arange(end_frame - size + 1, end_frame + 1)
    raw_times = np.arange(size, dtype=np.float64) / 100.
    times = np.linspace(0., raw_times[-1], 4 * size - 3)
    raw = (frames - end_frame)[:, None] / 10. + np.arange(6)[None, :]
    interpolated = np.column_stack([np.interp(times, raw_times, raw[:, ch]) for ch in range(6)])
    analog_times = np.arange(2 * size - 1, dtype=np.float64) / 200.
    analog = np.column_stack([np.interp(analog_times, raw_times, raw[:, ch]) for ch in range(6)])
    events = np.zeros((len(times), 6, 2), dtype=np.uint8)
    events[[0, -1], 0, 0] = 1
    events[len(times) // 2, 0, 1] = 1
    events[-1, 3, 0] = 1
    events[1, 3, 1] = 1
    return SimpleNamespace(
        subject="SA07", code="T20", trial="R01", end_frame=end_frame, class_id=class_id,
        raw=raw, interpolated=interpolated, reconstructed=interpolated + .01,
        time_raw=raw_times, time_events=times, events=events, frames=frames,
        offset=raw[0], theta=np.array([.025] * 3 + [1.25] * 3),
        resample_hz=200., analog=analog, time_analog=analog_times,
        fall_onset_time=raw_times[-1] - .025,
        fall_impact_time=raw_times[-1] + .01,
        post_impact_end_time=raw_times[-1] + .03,
    )


class KFallVariantPlottingTests(unittest.TestCase):
    def axes(self, count):
        figure = Figure(figsize=(6 * count, 8))
        FigureCanvasAgg(figure)
        self.addCleanup(figure.clear)
        return figure, figure.subplots(3, 2 * count, squeeze=False)

    def test_three_contexts_have_common_prediction_origin_and_gray_missing_history(self):
        figure, axes = self.axes(3)
        windows = [variant_window(size) for size in (5, 8, 10)]
        before = deepcopy(windows)
        draw_variant_comparison(axes, windows)
        for variant, window in enumerate(windows):
            signal = axes[0, variant * 2]
            lines = {line.get_label(): line for line in signal.lines}
            native = lines["Original (100 Hz)"]
            np.testing.assert_allclose(native.get_xdata(), window.time_raw - window.time_raw[-1])
            self.assertEqual(native.get_xdata()[-1], 0)
            np.testing.assert_array_equal(native.get_ydata(), window.raw[:, 0])
            for axis in axes[:, variant * 2:variant * 2 + 2].flat:
                np.testing.assert_allclose(axis.get_xlim(), [-.09, 0])
                gray = [patch for patch in axis.patches if patch.get_label() == "Sin contexto de entrada"]
                self.assertEqual(len(gray), int(variant < 2))
            for name, value in vars(window).items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, getattr(before[variant], name))
        figure.canvas.draw()

    def test_events_keep_all_positions_and_never_extend_beyond_prediction(self):
        _, axes = self.axes(2)
        windows = [variant_window(5), variant_window(10)]
        draw_variant_comparison(axes, windows)
        for variant, window in enumerate(windows):
            collections = {collection.get_label(): collection for collection in axes[0, 2 * variant + 1].collections}
            on = collections["ON (+1)"].get_segments()
            off = collections["OFF (−1)"].get_segments()
            expected = window.time_events - window.time_raw[-1]
            np.testing.assert_allclose([segment[0, 0] for segment in on], expected[[0, -1]])
            np.testing.assert_allclose([segment[0, 0] for segment in off], expected[[len(expected) // 2]])
            self.assertTrue(all(segment[:, 0].max() <= 0 for segment in on + off))

    def test_float32_encoder_endpoint_rounding_is_clamped_to_zero(self):
        _, axes = self.axes(1)
        window = variant_window(10)
        window.time_events = window.time_events.astype(np.float32)
        draw_variant_comparison(axes, [window])
        collection = next(collection for collection in axes[0, 1].collections if collection.get_label() == "ON (+1)")
        self.assertEqual(collection.get_segments()[-1][0, 0], 0.)

    def test_200_hz_overlay_and_shared_sensor_scales_survive_gyroscope_toggle(self):
        _, axes = self.axes(2)
        windows = [variant_window(5), variant_window(10)]
        draw_variant_comparison(axes, windows, sensor="gyro", show_reconstruction=False)
        for row, suffix in enumerate("XYZ"):
            for variant, window in enumerate(windows):
                axis = axes[row, 2 * variant]
                lines = {line.get_label(): line for line in axis.lines}
                self.assertNotIn("Reconstrucción ΣΔ", lines)
                self.assertIn("Interpolada (200 Hz)", lines)
                np.testing.assert_array_equal(lines["Original (100 Hz)"].get_ydata(), window.raw[:, row + 3])
                np.testing.assert_allclose(lines["Interpolada (200 Hz)"].get_xdata(), window.time_analog - window.time_raw[-1])
                self.assertEqual(axis.get_ylabel(), f"Gyr{suffix} (°/s)")
            self.assertEqual(axes[row, 0].get_ylim(), axes[row, 2].get_ylim())
        self.assertIsNotNone(axes[0, 0].get_legend())
        self.assertIsNone(axes[0, 2].get_legend())

    def test_annotations_stay_inside_available_context_and_hide_future_phases(self):
        _, axes = self.axes(2)
        windows = [variant_window(5, class_id=3), variant_window(10, class_id=3)]
        for window in windows:
            end = window.time_raw[-1]
            window.fall_onset_time = end - .07
            window.fall_impact_time = end - .03
            window.post_impact_end_time = end + .02
        draw_variant_comparison(axes, windows)
        for variant in range(2):
            lines = {line.get_label(): line for line in axes[0, variant * 2].lines}
            self.assertEqual("Inicio caída" in lines, variant == 1)
            self.assertIn("Impacto", lines)
            self.assertNotIn("Fin postimpacto", lines)
            np.testing.assert_allclose(lines["Impacto"].get_xdata(), [-.03, -.03])
        for window in windows:
            window.post_impact_end_time = window.time_raw[-1]
        draw_variant_comparison(axes, windows)
        end_line = next(line for line in axes[0, 0].lines if line.get_label() == "Fin postimpacto")
        np.testing.assert_allclose(end_line.get_xdata(), [0, 0])

    def test_mismatched_recording_endpoint_or_target_is_rejected(self):
        _, axes = self.axes(2)
        for field, value in (("end_frame", 1001), ("subject", "SA08"), ("code", "T21"),
                             ("trial", "R02"), ("class_id", 3)):
            windows = [variant_window(5), variant_window(10)]
            setattr(windows[1], field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "must share"):
                draw_variant_comparison(axes, windows)

    def test_invalid_layout_sensor_and_unobserved_time_grid_fail_clearly(self):
        _, axes = self.axes(1)
        window = variant_window(5)
        with self.assertRaisesRegex(ValueError, "shape"):
            draw_variant_comparison(axes, [])
        with self.assertRaisesRegex(ValueError, "shape"):
            draw_variant_comparison(axes[0], [window])
        with self.assertRaisesRegex(ValueError, "sensor"):
            draw_variant_comparison(axes, [window], sensor="Euler")
        window.time_events[-1] += .01
        with self.assertRaisesRegex(ValueError, "observed window endpoints"):
            draw_variant_comparison(axes, [window])
        window.time_raw[1] = window.time_raw[0]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            draw_variant_comparison(axes, [window])

    def test_redraw_reuses_axes_without_accumulating_graphics(self):
        figure, axes = self.axes(2)
        control = figure.add_axes([.01, .01, .02, .02])
        windows = [variant_window(5), variant_window(10)]
        draw_variant_comparison(axes, windows)
        counts = [(len(axis.lines), len(axis.collections), len(axis.patches)) for axis in axes.flat]
        draw_variant_comparison(axes, windows)
        self.assertIn(control, figure.axes)
        self.assertEqual(len(figure.axes), 13)
        self.assertEqual(counts, [(len(axis.lines), len(axis.collections), len(axis.patches)) for axis in axes.flat])


if __name__ == "__main__":
    unittest.main()
