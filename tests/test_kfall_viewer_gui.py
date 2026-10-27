"""Exercise interactive viewer callbacks without a desktop or training cache."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from view_kfall import KFallViewer


class FakeWindowStore:
    config = {"raw_win": 5, "out_win": 9, "sample_rate_hz": 100}
    rows = {
        "train": [("SA06", 2), ("SA06", 0), ("SA06", 1),
                  ("SA08", 0), ("SA08", 1), ("SA08", 0)],
        "val": [("SA15", 2), ("SA15", 0), ("SA15", 1)],
        "test": [("SA07", 0), ("SA07", 1), ("SA07", 2)],
    }

    def __init__(self):
        self.failures = set()

    def subjects(self, split):
        return sorted({subject for subject, _ in self.rows[split]})

    def get_indices(self, split, class_id=None, subject=None):
        if subject is not None:
            match = re.fullmatch(r"(?:SA|S)?(\d{1,2})", subject.strip().upper())
            if match is None:
                raise ValueError("Subject must be a KFall ID such as SA06, or Todos.")
            subject = f"SA{int(match.group(1)):02d}"
        return np.asarray([index for index, (record_subject, record_class) in enumerate(self.rows[split])
                           if (class_id is None or record_class == class_id)
                           and (subject is None or record_subject == subject)], dtype=np.int64)

    def get_window(self, split, index):
        if (split, index) in self.failures:
            raise ValueError("Cached events do not match source CSV.")
        subject, class_id = self.rows[split][index]
        raw = np.arange(5, dtype=np.float32)[:, None] + np.arange(6)[None, :]
        interpolated = np.linspace(0, 4, 9)[:, None] + np.arange(6)[None, :]
        task = {0: 20, 1: 4, 2: 1}[class_id]
        is_fall = class_id == 0
        return SimpleNamespace(
            split=split, index=index, raw=raw, interpolated=interpolated,
            events=np.zeros((9, 6, 2), dtype=np.uint8), reconstructed=interpolated + 0.1,
            time_raw=np.linspace(0, .04, 5), time_events=np.linspace(0, .04, 9),
            frames=np.arange(201, 206), offset=raw[0], subject=subject,
            code=f"T{task:02d}", trial="R01", path=Path(f"S{subject[2:]}T{task:02d}R01.csv"),
            start=200, stop=205, end_frame=205, lead_time_ms=40.0 if is_fall else None,
            class_id=class_id, label={0: "Fall", 1: "Dynamic", 2: "Static"}[class_id],
            fall_onset_time=.02 if is_fall else None, fall_impact_time=.08 if is_fall else None,
            theta=np.asarray([.1] * 3 + [10.] * 3),
        )


class KFallViewerGuiTests(unittest.TestCase):
    def setUp(self):
        # Widgets still update their artists and run callbacks. Skipping their
        # immediate redraws keeps these interaction tests independent of a GUI.
        drawing = patch("matplotlib.backends.backend_agg.FigureCanvasAgg.draw_idle")
        drawing.start()
        self.addCleanup(drawing.stop)
        self.store = FakeWindowStore()
        self.viewer = KFallViewer(self.store, interactive=True)
        self.addCleanup(plt.close, self.viewer.figure)

    def click(self, button):
        self.viewer.widgets[button]._observers.process("clicked", None)

    def test_slider_buttons_and_keyboard_navigate_without_callback_recursion(self):
        viewer = self.viewer
        self.assertEqual((viewer.position, viewer.index), (0, 0))
        self.assertFalse(viewer.widgets["previous"].active)
        self.assertTrue(viewer.widgets["next"].active)
        self.click("next")
        self.assertEqual((viewer.position, viewer.index), (1, 1))
        self.assertEqual(viewer.widgets["slider"].val, 2)
        self.click("previous")
        self.assertEqual(viewer.index, 0)
        viewer.widgets["slider"].set_val(4)
        self.assertEqual((viewer.position, viewer.index), (3, 3))
        viewer._on_key(SimpleNamespace(key="end"))
        self.assertEqual(viewer.index, 5)
        self.assertFalse(viewer.widgets["next"].active)
        viewer._on_key(SimpleNamespace(key="left"))
        self.assertEqual(viewer.index, 4)
        viewer._on_key(SimpleNamespace(key="home"))
        self.assertEqual(viewer.index, 0)

    def test_filtered_position_and_actual_cache_index_remain_distinct(self):
        viewer = self.viewer
        viewer.widgets["class"].set_active(1)  # Fall -> indices 1, 3, 5.
        self.assertEqual(viewer.class_id, 0)
        np.testing.assert_array_equal(viewer.indices, [1, 3, 5])
        self.assertEqual((viewer.position, viewer.index), (0, 1))
        viewer.widgets["slider"].set_val(2)
        self.assertEqual((viewer.position, viewer.index), (1, 3))
        viewer.widgets["index"].set_val("5")
        self.assertEqual((viewer.position, viewer.index), (2, 5))
        self.assertEqual(viewer.widgets["slider"].val, 3)
        viewer.widgets["index"].set_val("4")
        self.assertEqual((viewer.position, viewer.index), (2, 5))
        self.assertEqual(viewer.widgets["index"].text, "5")
        self.assertIn("filtro", viewer.status.get_text())
        viewer.widgets["index"].set_val("invalid")
        self.assertEqual(viewer.widgets["index"].text, "5")
        self.assertEqual(viewer.status.get_color(), "#b91c1c")

    def test_changing_split_resets_subject_outside_new_partition(self):
        viewer = self.viewer
        viewer.widgets["subject"].set_val("sa06")
        self.assertEqual(viewer.subject, "SA06")
        np.testing.assert_array_equal(viewer.indices, [0, 1, 2])
        viewer.widgets["split"].set_active(1)
        self.assertEqual(viewer.split, "val")
        self.assertIsNone(viewer.subject)
        self.assertEqual(viewer.window.subject, "SA15")
        self.assertEqual(viewer.widgets["subject"].text, "Todos")
        self.assertEqual(viewer.widgets["split"].value_selected, "val")

    def test_invalid_subject_and_empty_class_restore_previous_filter(self):
        viewer = self.viewer
        viewer.widgets["subject"].set_val("SA08")
        self.assertEqual(viewer.index, 3)
        viewer.widgets["subject"].set_val("unknown")
        self.assertEqual(viewer.subject, "SA08")
        self.assertEqual(viewer.widgets["subject"].text, "SA08")
        self.assertEqual(viewer.index, 3)
        viewer.widgets["class"].set_active(3)  # No Static window for SA08.
        self.assertIsNone(viewer.class_id)
        self.assertEqual(viewer.widgets["class"].value_selected, "Todas")
        self.assertEqual(viewer.window.subject, "SA08")
        self.assertEqual(viewer.index, 3)
        self.assertIn("No hay ventanas", viewer.status.get_text())

    def test_sensor_and_reconstruction_controls_preserve_other_widgets(self):
        viewer = self.viewer
        axes_count = len(viewer.figure.axes)
        viewer.widgets["slider"].set_val(3)
        viewer.widgets["sensor"].set_active(1)
        self.assertEqual(viewer.sensor, "gyro")
        self.assertEqual(viewer.axes[0, 0].get_ylabel(), "GyrX (°/s)")
        viewer.widgets["reconstruction"].set_active(0)
        self.assertFalse(viewer.show_reconstruction)
        self.assertFalse(viewer.widgets["reconstruction"].get_status()[0])
        self.assertNotIn("Reconstrucción ΣΔ", [line.get_label() for line in viewer.axes[0, 0].lines])
        self.assertEqual(viewer.index, 2)
        self.assertEqual(len(viewer.figure.axes), axes_count)
        viewer.widgets["reconstruction"].set_active(0)
        self.assertTrue(viewer.show_reconstruction)
        self.assertIn("Reconstrucción ΣΔ", [line.get_label() for line in viewer.axes[0, 0].lines])

    def test_source_validation_error_preserves_current_window(self):
        viewer = self.viewer
        self.store.failures.add(("train", 1))
        original = viewer.window
        viewer.widgets["slider"].set_val(2)
        self.assertEqual(viewer.index, 0)
        self.assertIs(viewer.window, original)
        self.assertEqual(viewer.widgets["slider"].val, 1)
        self.assertEqual(viewer.widgets["index"].text, "0")
        self.assertIn("do not match", viewer.status.get_text())
        self.assertEqual(viewer.status.get_color(), "#b91c1c")

    def test_single_matching_window_disables_navigation(self):
        viewer = self.viewer
        viewer.widgets["class"].set_active(3)  # The only Static window is index 0.
        self.assertEqual(viewer.indices.tolist(), [0])
        self.assertFalse(viewer.widgets["slider"].active)
        self.assertFalse(viewer.widgets["previous"].active)
        self.assertFalse(viewer.widgets["next"].active)
        self.click("random")
        self.assertEqual(viewer.index, 0)


if __name__ == "__main__":
    unittest.main()
