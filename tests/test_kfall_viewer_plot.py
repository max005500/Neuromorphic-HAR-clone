"""Verify the viewer plots cached events and the matching physical time axes."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from visualization.kfall_plotting import draw_window


class KFallViewerPlotTests(unittest.TestCase):
    def setUp(self):
        self.figure = Figure(figsize=(12, 8))
        FigureCanvasAgg(self.figure)
        self.axes = self.figure.subplots(3, 2)
        raw = np.arange(5, dtype=np.float32)[:, None] + np.arange(6)[None, :] * 10
        interpolated = np.linspace(0, 4, 9)[:, None] + np.arange(6)[None, :] * 10
        events = np.zeros((9, 6, 2), dtype=np.uint8)
        events[[1, 3], 0, 0] = 1
        events[6, 0, 1] = 1
        events[0, 1, 0] = 1
        events[8, 1, 1] = 1
        events[7, 3, 0] = 1
        events[2, 3, 1] = 1
        self.window = SimpleNamespace(
            raw=raw, interpolated=interpolated, events=events,
            reconstructed=interpolated + 0.1, time_raw=np.linspace(0, 0.04, 5),
            time_events=np.linspace(0, 0.04, 9), offset=raw[0],
            fall_onset_time=0.025, fall_impact_time=0.08, theta=np.array([0.1] * 3 + [10.] * 3),
        )
        self.addCleanup(self.figure.clear)

    def test_accelerometer_axes_use_exact_signal_channels_and_time_grids(self):
        draw_window(self.axes, self.window)
        for row, name in enumerate(("AccX", "AccY", "AccZ")):
            axis = self.axes[row, 0]
            by_label = {line.get_label(): line for line in axis.lines}
            np.testing.assert_array_equal(by_label["Original (100 Hz)"].get_xdata(), self.window.time_raw)
            np.testing.assert_array_equal(by_label["Original (100 Hz)"].get_ydata(), self.window.raw[:, row])
            np.testing.assert_array_equal(by_label["Interpolada"].get_ydata(), self.window.interpolated[:, row])
            np.testing.assert_array_equal(by_label["Reconstrucción ΣΔ"].get_ydata(), self.window.reconstructed[:, row])
            self.assertEqual(axis.get_ylabel(), f"{name} (g)")
        self.figure.canvas.draw()

    def test_polarity_marks_preserve_all_cached_event_positions(self):
        draw_window(self.axes, self.window)
        collections = {collection.get_label(): collection for collection in self.axes[0, 1].collections}
        on_segments = collections["ON (+1)"].get_segments()
        off_segments = collections["OFF (−1)"].get_segments()
        np.testing.assert_allclose([segment[0, 0] for segment in on_segments], self.window.time_events[[1, 3]])
        np.testing.assert_allclose([segment[0, 0] for segment in off_segments], self.window.time_events[[6]])
        self.assertEqual(len(on_segments), 2)
        self.assertEqual(len(off_segments), 1)
        for segment in on_segments:
            np.testing.assert_array_equal(segment[:, 1], [0, 1])
        for segment in off_segments:
            np.testing.assert_array_equal(segment[:, 1], [-1, 0])

    def test_gyroscope_switch_and_reconstruction_toggle(self):
        draw_window(self.axes, self.window, sensor="gyro", show_reconstruction=False)
        for row, name in enumerate(("GyrX", "GyrY", "GyrZ")):
            axis = self.axes[row, 0]
            by_label = {line.get_label(): line for line in axis.lines}
            self.assertNotIn("Reconstrucción ΣΔ", by_label)
            np.testing.assert_array_equal(by_label["Original (100 Hz)"].get_ydata(), self.window.raw[:, row + 3])
            self.assertEqual(axis.get_ylabel(), f"{name} (°/s)")
        collections = {collection.get_label(): collection for collection in self.axes[0, 1].collections}
        np.testing.assert_allclose(collections["ON (+1)"].get_segments()[0][:, 0], self.window.time_events[7])

    def test_annotation_positions_do_not_expose_future_impact(self):
        draw_window(self.axes, self.window)
        for axis in self.axes.flat:
            self.assertEqual(axis.get_xlim(), (0.0, 0.04))
            lines = {line.get_label(): line for line in axis.lines}
            self.assertIn("Inicio caída", lines)
            self.assertNotIn("Impacto", lines)
            np.testing.assert_allclose(lines["Inicio caída"].get_xdata(), [0.025, 0.025])
        self.window.fall_impact_time = 0.04
        draw_window(self.axes, self.window)
        self.assertIn("Impacto", {line.get_label() for line in self.axes[0, 0].lines})
        self.assertEqual(self.axes[0, 0].get_xlim(), (0.0, 0.04))

    def test_redraw_preserves_controls_and_does_not_accumulate_data(self):
        control = self.figure.add_axes((0.01, 0.01, 0.05, 0.02))
        control.set_label("viewer control")
        draw_window(self.axes, self.window)
        counts = [(len(axis.lines), len(axis.collections)) for axis in self.axes.flat]
        draw_window(self.axes, self.window)
        self.assertEqual(len(self.figure.axes), 7)
        self.assertIn(control, self.figure.axes)
        self.assertEqual(counts, [(len(axis.lines), len(axis.collections)) for axis in self.axes.flat])

    def test_invalid_sensor_or_axis_layout_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "sensor"):
            draw_window(self.axes, self.window, sensor="Euler")
        with self.assertRaisesRegex(ValueError, "shape"):
            draw_window(self.axes[0], self.window)


if __name__ == "__main__":
    unittest.main()
