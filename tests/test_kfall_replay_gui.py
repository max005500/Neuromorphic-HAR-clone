"""Exercise the moving KFall signal player's controls without a desktop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from animate_kfall import KFallReplayPlayer


def replay_fixture() -> SimpleNamespace:
    times = np.arange(801, dtype=np.float64) / 100
    endpoint_rows = np.arange(204, 801, 10, dtype=np.int64)
    endpoint_times = times[endpoint_rows]
    predicted = np.where(endpoint_times < 3.24, 2,
                         np.where(endpoint_times < 4.14, 0, 1)).astype(np.int64)
    probability_rows = np.asarray([[.8, .1, .1], [.15, .75, .1], [.1, .2, .7]])
    signals = np.column_stack([np.sin(times + channel) * (channel + 1)
                               for channel in range(6)]).astype(np.float32)
    record = {
        "file": "data/KFall/sensor_data/SA07/S07T20R01.csv",
        "subject": "SA07", "code": "T20", "trial": "R01", "label_id": 0,
        "label": "FALL", "T": len(times), "fall_onset_frame": 301,
        "fall_impact_frame": 401, "onset_idx": 300, "impact_idx": 400,
    }
    return SimpleNamespace(
        record=record, path=Path(record["file"]), raw=signals,
        frames=np.arange(1, 802), times=times, endpoint_rows=endpoint_rows,
        endpoint_times=endpoint_times, probabilities=probability_rows[predicted],
        predicted_classes=predicted, raw_win=205, out_win=2048,
        prediction_stride=10, checkpoint=Path("runs/example/checkpoints/best_model.pt"),
        best_epoch=3, best_val_loss=.28, fall_onset_time=3., fall_impact_time=4.,
        config={"sample_rate_hz": 100., "fall_policy": "pre_impact"},
        cache_dir=Path("data/kfall_cache/kfall_event_loader_default"),
        subject="SA07", label="Fall",
    )


class KFallReplayGuiTests(unittest.TestCase):
    def setUp(self):
        drawing = patch("matplotlib.backends.backend_agg.FigureCanvasAgg.draw_idle")
        drawing.start()
        self.addCleanup(drawing.stop)
        self.replay = replay_fixture()
        self.player = KFallReplayPlayer(self.replay, fps=25, speed=1., interactive=True)
        self.addCleanup(self.close_player)

    def close_player(self):
        for name in ("animation", "_animation"):
            animation = getattr(self.player, name, None)
            if animation is not None:
                if animation.event_source is not None:
                    animation.event_source.stop()
                animation._draw_was_started = True
        plt.close(self.player.figure)

    def widget(self, *names):
        for name in names:
            if name in self.player.widgets:
                return self.player.widgets[name]
        self.fail(f"Missing player control: {names}.")

    @staticmethod
    def click(button):
        button._observers.process("clicked", None)

    def test_tick_respects_pause_and_playback_speed(self):
        player = self.player
        player.seek(2.5)
        player.playing = False
        player._on_tick()
        self.assertAlmostEqual(player.current_time, 2.5)
        self.click(self.widget("play", "play_pause"))
        self.assertTrue(player.playing)
        player._on_tick()
        self.assertAlmostEqual(player.current_time, 2.5 + 1 / 25)
        self.widget("speed").set_val(2.)
        player._on_tick()
        self.assertAlmostEqual(player.current_time, 2.5 + 3 / 25)
        self.click(self.widget("play", "play_pause"))
        self.assertFalse(player.playing)
        player._on_tick()
        self.assertAlmostEqual(player.current_time, 2.5 + 3 / 25)

    def test_seeking_clamps_to_recording_and_synchronizes_slider(self):
        player = self.player
        player.seek(-3.)
        self.assertEqual(player.current_time, 0.)
        self.assertEqual(self.widget("time", "slider").val, 0.)
        player.seek(100.)
        self.assertAlmostEqual(player.current_time, self.replay.times[-1])
        self.assertAlmostEqual(self.widget("time", "slider").val, self.replay.times[-1])
        self.widget("time", "slider").set_val(3.5)
        self.assertAlmostEqual(player.current_time, 3.5)

    def test_prediction_navigation_uses_available_endpoint_times(self):
        player = self.player
        player.seek(.5)
        player.step_prediction(1)
        self.assertAlmostEqual(player.current_time, self.replay.endpoint_times[0])
        player.seek(3.11)
        player.step_prediction(-1)
        self.assertAlmostEqual(player.current_time, 3.04)
        player.seek(3.11)
        player.step_prediction(1)
        self.assertAlmostEqual(player.current_time, 3.14)

    def test_restart_resets_to_zero_and_end_of_recording_stops(self):
        player = self.player
        player.seek(4.)
        self.click(self.widget("restart"))
        self.assertEqual(player.current_time, 0.)
        self.assertEqual(self.widget("time", "slider").val, 0.)
        player.seek(self.replay.times[-1] - .01)
        player.playing = True
        player._on_tick()
        self.assertAlmostEqual(player.current_time, self.replay.times[-1])
        self.assertFalse(player.playing)

    def test_sensor_selection_changes_channels_without_changing_time(self):
        player = self.player
        player.seek(3.5)
        axes_count = len(player.figure.axes)
        self.widget("sensor").set_active(1)
        self.assertAlmostEqual(player.current_time, 3.5)
        self.assertIn("GyrX", player.axes[0].get_ylabel())
        self.assertEqual(len(player.figure.axes), axes_count)
        self.widget("sensor").set_active(0)
        self.assertAlmostEqual(player.current_time, 3.5)
        self.assertIn("AccX", player.axes[0].get_ylabel())

    def test_saved_snapshot_is_png_and_does_not_advance_playback(self):
        player = self.player
        player.seek(3.5)
        player.playing = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.png"
            player.save_frame(path)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertAlmostEqual(player.current_time, 3.5)


if __name__ == "__main__":
    unittest.main()
