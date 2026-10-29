"""Exercise variant selection and matched navigation without a desktop."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from test_kfall_viewer_gui import FakeWindowStore
from view_kfall_variants import KFallVariantViewer, main


class FakeVariantStore:
    durations = [.05, .08, .10]
    class_names = {0: "Fall", 1: "Dynamic", 2: "Static"}

    def __init__(self):
        self.base = FakeWindowStore()
        self.coverage = {split: {"common_windows": len(rows),
                                 "variant_windows": [len(rows)] * 3,
                                 "discarded_windows": [0] * 3}
                         for split, rows in self.base.rows.items()}

    def subjects(self, split):
        return self.base.subjects(split)

    def get_indices(self, split, class_id=None, subject=None):
        return self.base.get_indices(split, class_id, subject)

    def get_windows(self, split, index):
        source = self.base.get_window(split, index)
        result = []
        for size in (5, 8, 10):
            frames = np.arange(206 - size, 206)
            times = np.arange(size) / 100
            event_times = np.linspace(0, times[-1], 2 * size - 1)
            raw = np.column_stack([np.sin(frames / 100 + channel) for channel in range(6)])
            analog = np.column_stack([np.interp(event_times, times, raw[:, channel]) for channel in range(6)])
            data = vars(source).copy()
            data.update(raw=raw, interpolated=analog, reconstructed=analog.copy(),
                        events=np.zeros((len(event_times), 6, 2), dtype=np.uint8),
                        frames=frames, time_raw=times, time_events=event_times,
                        offset=raw[0].copy(), start=205 - size, stop=205,
                        resample_hz=200., analog=analog, time_analog=event_times,
                        post_impact_end_time=None)
            if source.class_id == 0:
                data.update(fall_onset_time=(203 - frames[0]) / 100,
                            fall_impact_time=(209 - frames[0]) / 100)
            result.append(SimpleNamespace(**data))
        return result


class VariantViewerTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("matplotlib.backends.backend_agg.FigureCanvasAgg.draw_idle")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = FakeVariantStore()

    def viewer(self, **kwargs):
        viewer = KFallVariantViewer(self.store, **kwargs)
        self.addCleanup(plt.close, viewer.figure)
        return viewer

    def test_all_and_single_duration_keep_endpoint_and_remove_unused_axes(self):
        viewer = self.viewer()
        original_count = len(viewer.figure.axes)
        self.assertEqual(viewer.axes.shape, (3, 6))
        original_id = viewer.index
        viewer.widgets["view"].set_active(2)
        self.assertEqual(viewer.selected_variant, 1)
        self.assertEqual(viewer.axes.shape, (3, 2))
        self.assertEqual(viewer.index, original_id)
        self.assertEqual({window.end_frame for window in viewer.windows}, {205})
        viewer.widgets["view"].set_active(0)
        self.assertEqual(viewer.axes.shape, (3, 6))
        self.assertEqual(len(viewer.figure.axes), original_count)
        self.assertIn("200 Hz", viewer.sampling.get_text())

    def test_filters_slider_and_split_subject_selection_are_synchronized(self):
        viewer = self.viewer()
        viewer.widgets["class"].set_active(1)
        viewer.widgets["split"].set_active(0)
        np.testing.assert_array_equal(viewer.indices, [1, 3, 5])
        viewer.widgets["subject"].set_val("08")
        np.testing.assert_array_equal(viewer.indices, [3, 5])
        self.assertEqual(viewer.subject, "SA08")
        viewer.widgets["slider"].set_val(2)
        self.assertEqual(viewer.index, 5)
        viewer._on_key(SimpleNamespace(key="left"))
        self.assertEqual(viewer.index, 3)
        viewer.widgets["split"].set_active(1)
        self.assertEqual(viewer.split, "val")
        self.assertIsNone(viewer.subject)
        self.assertEqual(viewer.windows[0].subject, "SA15")

    def test_invalid_subject_or_failed_source_keeps_the_previous_selection(self):
        viewer = self.viewer()
        first = viewer.index
        viewer.widgets["subject"].set_val("invalid")
        self.assertEqual(viewer.index, first)
        self.assertEqual(viewer.widgets["subject"].text, "Todos")
        self.store.base.failures.add(("test", 1))
        viewer._on_key(SimpleNamespace(key="right"))
        self.assertEqual(viewer.index, first)
        self.assertIn("do not match source CSV", viewer.status.get_text())

    def test_sensor_and_reconstruction_controls_redraw_all_variants(self):
        viewer = self.viewer()
        viewer.widgets["sensor"].set_active(1)
        self.assertEqual(viewer.sensor, "gyro")
        self.assertIn("GyrX", viewer.axes[0, 0].get_ylabel())
        viewer.widgets["reconstruction"].set_active(0)
        self.assertFalse(viewer.show_reconstruction)
        self.assertFalse(any(line.get_label() == "Reconstrucción ΣΔ"
                             for axis in viewer.axes.flat for line in axis.lines))

    def test_single_duration_export_produces_a_png(self):
        viewer = self.viewer(duration=.08, interactive=False)
        self.assertEqual(viewer.selected_variant, 1)
        self.assertEqual(viewer.axes.shape, (3, 2))
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "variant.png"
            with contextlib.redirect_stdout(io.StringIO()):
                viewer.save(destination)
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_unknown_duration_does_not_leave_an_open_figure(self):
        before = plt.get_fignums()
        with self.assertRaisesRegex(ValueError, "Duration"):
            KFallVariantViewer(self.store, duration=10.)
        self.assertEqual(plt.get_fignums(), before)

    def test_save_cli_works_without_opening_a_desktop_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "comparison.png"
            with patch("view_kfall_variants.KFallVariantStore", return_value=self.store), \
                    patch("matplotlib.pyplot.show", side_effect=AssertionError("desktop opened")), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(["--split", "val", "--class", "fall", "--duration", ".08", "--save", str(destination)])
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
