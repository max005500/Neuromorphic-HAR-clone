"""Keep old visualizations readable while displaying the new model class."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from animate_kfall import KFallReplayPlayer
from visualization.kfall_replay_plot import POSTIMPACT_CLASS_COLOR, ReplayPlot


def replay_fixture(n_classes=4):
    times = np.arange(601) / 100
    endpoints = np.array([2.04, 2.54, 3.04, 3.54, 4.04, 4.54, 5.04])
    labels = np.array([2, 0, 3, 3, 3, 2, 2]) if n_classes == 4 else np.array([2, 0, 0, 1, 1, 2, 2])
    probabilities = np.full((len(endpoints), n_classes), 0.1)
    probabilities[np.arange(len(labels)), labels] = 1 - .1 * (n_classes - 1)
    record = {"subject": "SA07", "code": "T20", "trial": "R01", "label_id": 0}
    return SimpleNamespace(
        raw=np.column_stack([np.sin(times + channel) for channel in range(6)]),
        frames=np.arange(1, len(times) + 1), times=times,
        endpoint_rows=(endpoints * 100).astype(int), endpoint_times=endpoints,
        probabilities=probabilities, predicted_classes=labels, record=record,
        fall_onset_time=2.4, fall_impact_time=3.0,
        post_impact_seconds=1.2 if n_classes == 4 else 0.0,
        post_impact_end_time=4.2 if n_classes == 4 else None,
        raw_win=205, out_win=2048, prediction_stride=50,
        checkpoint=Path("runs/example/checkpoints/best_model.pt"), best_epoch=1,
        best_val_loss=.2, config={}, cache_dir=Path("cache"), subject="SA07", label="Fall",
    )


class KFallPostimpactViewerTests(unittest.TestCase):
    def plot(self, replay):
        figure = Figure(figsize=(10, 8))
        FigureCanvasAgg(figure)
        self.addCleanup(figure.clear)
        return ReplayPlot(figure.subplots(5, 1), replay)

    def test_four_class_probabilities_and_postimpact_decision_are_rendered(self):
        replay = replay_fixture()
        plot = self.plot(replay)
        info = plot.update(3.2)
        self.assertEqual(info["predicted_class"], 3)
        self.assertEqual(info["predicted_label"], "Postimpacto")
        self.assertAlmostEqual(info["p_fall"], .1)
        self.assertTrue(any(np.allclose(color, to_rgba(POSTIMPACT_CLASS_COLOR))
                            for color in plot.prediction_collection.get_facecolors()))

    def test_player_header_includes_new_class_and_probability(self):
        with patch("matplotlib.backends.backend_agg.FigureCanvasAgg.draw_idle"):
            player = KFallReplayPlayer(replay_fixture(), interactive=False)
            self.addCleanup(plt.close, player.figure)
            player.seek(3.2)
            self.assertIn("POSTIMPACTO", player.prediction_text.get_text())
            self.assertIn("P(Postimpact)", player.prediction_text.get_text())
            self.assertEqual(player.prediction_text.get_color(), "#d97706")

    def test_three_class_replay_keeps_legacy_postimpact_semantics(self):
        plot = self.plot(replay_fixture(3))
        info = plot.update(5.5)
        self.assertEqual(info["real_phase"], "Posimpacto")
        self.assertEqual(len(plot.class_names), 3)

    def test_four_class_truth_ends_after_the_bounded_postimpact_phase(self):
        from visualization.kfall_replay_plot import POST_IMPACT_COLOR

        plot = self.plot(replay_fixture())
        self.assertEqual(plot.update(2.999)["real_phase"], "Caída (preimpacto)")
        self.assertEqual(plot.update(3.0)["real_phase"], "Postimpacto")
        self.assertEqual(plot.update(4.199)["real_phase"], "Postimpacto")
        self.assertEqual(plot.update(4.2)["real_phase"], "Fuera del intervalo etiquetado")
        plot.update(5)
        colors = plot.truth_collection.get_facecolors()
        self.assertTrue(any(np.allclose(color, to_rgba(POSTIMPACT_CLASS_COLOR)) for color in colors))
        self.assertTrue(any(np.allclose(color, to_rgba(POST_IMPACT_COLOR)) for color in colors))
        self.assertTrue(any(np.isclose(moment, 4.2) for moment, _ in plot.annotation_lines))

    def test_cached_postimpact_window_keeps_its_fall_source_and_new_filter(self):
        from loaders.kfall_event_loader import load_or_build_kfall_event_splits
        from test_kfall_viewer_data import KFallViewerDataTests
        from view_kfall import KFallViewer
        from visualization.kfall_data import KFallWindowStore

        fixture = KFallViewerDataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = replace(fixture.config, post_impact_seconds=1.2)
        load_or_build_kfall_event_splits(fixture.root, cache_dir=fixture.cache_parent,
                                        run_name="post", cfg=config, verbose=False)
        store = KFallWindowStore(fixture.cache_parent / "kfall_event_loader_post")
        index = int(store.get_indices("train", class_id=3)[0])
        window = store.get_window("train", index)
        self.assertEqual(window.class_id, 3)
        self.assertEqual(window.label, "Postimpact")
        self.assertLessEqual(window.lead_time_ms, 0)
        self.assertAlmostEqual(window.post_impact_end_time - window.fall_impact_time, 1.2)
        record = store.records[int(store._splits["train"]["file_idx"][index])]
        self.assertEqual(record["label_id"], 0)
        with patch("matplotlib.backends.backend_agg.FigureCanvasAgg.draw_idle"):
            viewer = KFallViewer(store, class_id=3, interactive=True)
            self.addCleanup(plt.close, viewer.figure)
            self.assertEqual(viewer.class_filters["Postimpact"], 3)
            self.assertEqual(viewer.widgets["class"].value_selected, "Postimpact")
            self.assertIn("tiempo desde el impacto", viewer.details.get_text())
            self.assertTrue(viewer.axes[0, 0].patches)

    def test_unrecognized_class_count_fails_before_rendering(self):
        replay = replay_fixture()
        replay.probabilities = np.zeros((len(replay.endpoint_times), 5))
        with self.assertRaisesRegex(ValueError, "three or four"):
            self.plot(replay)


if __name__ == "__main__":
    unittest.main()
