"""Synchronized model comparison, without trained weights or desktop."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from tests.test_kfall_replay_gui import replay_fixture
from animate_kfall_comparison import KFallComparisonPlayer, prepare_comparison, validate_comparison


def fixtures():
    result = []
    for size in (205, 330, 410):
        replay = replay_fixture()
        replay.raw_win = size
        replay.endpoint_rows = np.arange(409, 801, 10)
        replay.endpoint_times = replay.times[replay.endpoint_rows]
        replay.predicted_classes = np.zeros(len(replay.endpoint_rows), dtype=int)
        replay.probabilities = np.tile([.8, .1, .1], (len(replay.endpoint_rows), 1))
        result.append(replay)
    return result


class ComparisonTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_seek_controls_and_export(self):
        player = KFallComparisonPlayer(fixtures(), interactive=False)
        player.seek(4.08)
        self.assertTrue(all(p.current_info['prediction_index'] is None for p in player.plots))
        player.seek(4.09)
        self.assertTrue(all(p.current_info['prediction_index'] == 0 for p in player.plots))
        player.step_prediction(1)
        self.assertAlmostEqual(player.current_time, 4.19)
        player._on_sensor('Giroscopio')
        self.assertTrue(all(p.sensor == 'gyro' for p in player.plots))
        for plot in player.plots:
            self.assertLessEqual(max(plot.raw_lines[0].get_xdata()), player.current_time)
        with tempfile.TemporaryDirectory() as root:
            self.assertTrue(player.save_frame(Path(root) / 'frame.png').is_file())

    def test_mismatches_rejected(self):
        for attribute in ('endpoint_times', 'raw', 'frames'):
            replays = fixtures()
            setattr(replays[1], attribute, getattr(replays[1], attribute) + 1)
            with self.assertRaises(ValueError):
                validate_comparison(replays)
        replays = fixtures()
        replays[1].record['fall_onset_frame'] += 1
        with self.assertRaises(ValueError):
            validate_comparison(replays)

    def test_common_recording_and_context_passed_to_all_models(self):
        from types import SimpleNamespace
        replays = fixtures()
        with patch('animate_kfall_comparison._load_checkpoint', side_effect=[
            ({}, SimpleNamespace(raw_win=n)) for n in (330, 205, 410)]), patch(
                'animate_kfall_comparison.prepare_replay', side_effect=[replays[2], replays[0], replays[1]]) as prepare, patch(
                'pathlib.Path.read_text', return_value='{"split_subjects": {"test": ["SA07"]}}'):
            prepare_comparison(['b', 'a', 'c'])
            self.assertTrue(all(call.kwargs['endpoint_context'] == 410 for call in prepare.call_args_list))
            self.assertEqual(prepare.call_args_list[1].kwargs['recording'], replays[2].path.name)

    def test_interactive_controls(self):
        with patch('matplotlib.animation.FuncAnimation'):
            player = KFallComparisonPlayer(fixtures(), interactive=True)
        player.widgets['time'].set_val(5)
        self.assertEqual(player.current_time, 5)
        player.toggle_play()
        player.restart()
        self.assertEqual(player.current_time, 0)
