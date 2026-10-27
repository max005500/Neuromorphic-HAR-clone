"""Ensure replay plots reveal only observations and decisions available in time."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from visualization.kfall_replay_plot import (
    FALL_COLOR, POST_IMPACT_COLOR, UNKNOWN_COLOR, ReplayPlot,
)


def replay_fixture(label=0):
    times = np.arange(51, dtype=float) / 10
    raw = np.column_stack([np.sin(times * (channel + 1)) + channel * 10 for channel in range(6)])
    endpoints = np.array([2.0, 2.5, 3.0, 4.0, 5.0])
    probabilities = np.array([[.1, .8, .1], [.8, .1, .1], [.2, .1, .7], [.9, .05, .05], [.1, .8, .1]])
    return SimpleNamespace(
        raw=raw, times=times, endpoint_rows=(endpoints * 10).astype(int),
        endpoint_times=endpoints, probabilities=probabilities,
        predicted_classes=probabilities.argmax(axis=1), record={"label_id": label},
        fall_onset_time=1.5 if label == 0 else None,
        fall_impact_time=3.5 if label == 0 else None,
    )


class KFallReplayPlotTests(unittest.TestCase):
    def setUp(self):
        self.replay = replay_fixture()
        self.figure = Figure(figsize=(10, 9))
        FigureCanvasAgg(self.figure)
        self.axes = self.figure.subplots(5, 1, sharex=True)
        self.plot = ReplayPlot(self.axes, self.replay, view_seconds=3)
        self.addCleanup(self.figure.clear)

    def test_future_signal_predictions_and_annotations_are_hidden(self):
        info = self.plot.update(1.0)
        self.assertIsNone(info["prediction_index"])
        self.assertIsNone(info["predicted_class"])
        self.assertIsNone(info["p_fall"])
        self.assertIn("ventana incompleta", info["predicted_label"])
        for line in self.plot.raw_lines:
            self.assertLessEqual(np.max(line.get_xdata()), 1.0)
        self.assertEqual(len(self.plot.probability_line.get_xdata()), 0)
        self.assertTrue(all(not line.get_visible() for _, line in self.plot.annotation_lines))
        for collection in (self.plot.prediction_collection, self.plot.truth_collection):
            for path in collection.get_paths():
                self.assertLessEqual(path.vertices[:, 0].max(), 1.0)

    def test_prediction_begins_exactly_at_its_endpoint_and_is_held(self):
        self.assertIsNone(self.plot.update(1.999)["prediction_index"])
        first = self.plot.update(2.0)
        self.assertEqual(first["prediction_index"], 0)
        self.assertEqual(first["predicted_class"], 1)
        self.assertAlmostEqual(first["p_fall"], .1)
        held = self.plot.update(2.4)
        self.assertEqual(held["prediction_index"], 0)
        self.assertAlmostEqual(held["prediction_age_sec"], .4)
        np.testing.assert_allclose(self.plot.probability_line.get_xdata(), [2, 2.4])
        np.testing.assert_allclose(self.plot.probability_line.get_ydata(), [.1, .1])
        following = self.plot.update(2.5)
        self.assertEqual(following["prediction_index"], 1)
        self.assertEqual(following["predicted_class"], 0)
        self.assertEqual(following["predicted_label"], "Caída")

    def test_truth_onset_is_inclusive_and_impact_is_exclusive(self):
        self.assertEqual(self.plot.update(1.499)["real_phase"], "Precaída / sin caída")
        self.assertEqual(self.plot.update(1.5)["real_phase"], "Caída (preimpacto)")
        self.assertEqual(self.plot.update(3.499)["real_phase"], "Caída (preimpacto)")
        self.assertEqual(self.plot.update(3.5)["real_phase"], "Posimpacto")
        self.plot.update(4.0)
        truth_colors = self.plot.truth_collection.get_facecolors()
        from matplotlib.colors import to_rgba
        self.assertTrue(any(np.allclose(color, to_rgba(FALL_COLOR)) for color in truth_colors))
        self.assertTrue(any(np.allclose(color, to_rgba(POST_IMPACT_COLOR)) for color in truth_colors))

    def test_scrolling_window_and_sensor_switch_use_observed_values(self):
        self.plot.update(4.25, sensor="gyro")
        for channel, (axis, line) in enumerate(zip(self.axes[:3], self.plot.raw_lines), start=3):
            x, y = line.get_data()
            self.assertGreaterEqual(x.min(), 1.25)
            self.assertLessEqual(x.max(), 4.25)
            mask = (self.replay.times >= 1.25) & (self.replay.times <= 4.25)
            np.testing.assert_array_equal(y, self.replay.raw[mask, channel])
            self.assertIn("Gyr", axis.get_ylabel())
            np.testing.assert_allclose(axis.get_xlim(), [1.25, 4.25])
        self.assertTrue(all(line.get_visible() for _, line in self.plot.annotation_lines))

    def test_seeking_backward_hides_future_results_without_accumulating_artists(self):
        counts = [(len(axis.lines), len(axis.collections), len(axis.texts)) for axis in self.axes]
        self.plot.update(5.0)
        self.plot.update(.5)
        self.assertIsNone(self.plot.current_info["prediction_index"])
        self.assertTrue(all(not line.get_visible() for _, line in self.plot.annotation_lines))
        self.assertEqual(counts, [(len(axis.lines), len(axis.collections), len(axis.texts)) for axis in self.axes])
        from matplotlib.colors import to_rgba
        np.testing.assert_allclose(self.plot.prediction_collection.get_facecolors(), [to_rgba(UNKNOWN_COLOR)])

    def test_adl_truth_uses_activity_and_has_no_fall_annotations(self):
        for label, name in ((1, "Dynamic"), (2, "Static")):
            with self.subTest(label=label):
                figure = Figure()
                FigureCanvasAgg(figure)
                plot = ReplayPlot(figure.subplots(5, 1), replay_fixture(label))
                self.assertEqual(plot.update(3)["real_phase"], name)
                self.assertEqual(plot.annotation_lines, [])
                figure.clear()

    def test_invalid_parameters_fail_and_snapshot_renders(self):
        with self.assertRaises(ValueError):
            ReplayPlot(self.axes, self.replay, view_seconds=0)
        with self.assertRaises(ValueError):
            self.plot.update(float("nan"))
        with self.assertRaises(ValueError):
            self.plot.update(2, sensor="euler")
        self.plot.update(3)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "replay.png"
            self.figure.savefig(destination)
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
