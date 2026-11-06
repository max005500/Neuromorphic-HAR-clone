"""Exercise the in-window recording catalog without a desktop or GPU."""
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import unittest
import threading
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from test_kfall_replay_gui import replay_fixture
from visualization.kfall_temporal_browser import TemporalReplayBrowser


class ImmediateExecutor:
    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


class PendingExecutor:
    def submit(self, function, *args, **kwargs):
        self.future = Future()
        return self.future


class FakeSource:
    def __init__(self):
        self.calls, self.fail_files, self.fail_splits = [], set(), set()
        self.catalog = {
            'test': [dict(file=f'S{subject[2:]}T{20 + number % 12:02d}R01.csv',
                          subject=subject, windows=30, split='test')
                     for number, subject in enumerate(['SA07'] * 12 + ['SA21'] * 12)],
            'val': [dict(file='S15T20R01.csv', subject='SA15', windows=30, split='val')],
            'train': [dict(file='S06T20R01.csv', subject='SA06', windows=30, split='train')],
        }

    def records(self, split_name):
        if split_name in self.fail_splits:
            raise ValueError('Partition cache is unavailable')
        return self.catalog[split_name]

    def replay(self, split_name, *, recording=None):
        self.calls.append((split_name, recording))
        if recording in self.fail_files:
            raise ValueError('Cannot validate this recording')
        row = next(row for row in self.catalog[split_name] if row['file'] == recording)
        replay = copy.deepcopy(replay_fixture())
        replay.path = Path(row['file'])
        replay.subject = row['subject']
        replay.record.update(file=row['file'], subject=row['subject'], code=row['file'][3:6])
        replay.config['replay_split'] = split_name
        # A different duration exercises slider reconfiguration on selection.
        if recording.endswith('T21R01.csv'):
            count = 501
            keep = replay.endpoint_rows < count
            replay.times, replay.frames, replay.raw = replay.times[:count], replay.frames[:count], replay.raw[:count]
            replay.endpoint_rows, replay.endpoint_times = replay.endpoint_rows[keep], replay.endpoint_times[keep]
            replay.predicted_classes, replay.probabilities = replay.predicted_classes[keep], replay.probabilities[keep]
        replay.probabilities = np.column_stack((replay.probabilities * .95,
                                                np.full(len(replay.probabilities), .05)))
        replay.post_impact_seconds, replay.post_impact_end_time = 1.2, 5.2
        return replay


class TemporalBrowserGuiTests(unittest.TestCase):
    def setUp(self):
        drawing = patch('matplotlib.backends.backend_agg.FigureCanvasAgg.draw_idle')
        drawing.start()
        self.addCleanup(drawing.stop)
        self.source = FakeSource()
        initial = self.source.replay('test', recording='S07T20R01.csv')
        self.source.calls.clear()
        self.player = TemporalReplayBrowser(self.source, initial, executor=ImmediateExecutor())
        self.addCleanup(self.close)

    def close(self):
        self.player.close_browser()
        if self.player.animation is not None:
            self.player.animation.event_source.stop()
            self.player.animation._draw_was_started = True
        plt.close(self.player.figure)

    def test_catalog_paginates_without_accumulating_axes_or_running_inference(self):
        original = len(self.player.figure.axes)
        self.assertEqual(len(self.player.file_buttons), 10)
        for _ in range(3):
            self.player._move_page(1)
            self.player._move_page(1)
            self.assertEqual(len(self.player.file_buttons), 4)
            self.player._move_page(-1)
            self.player._move_page(-1)
        self.assertEqual(len(self.player.figure.axes), original)
        self.assertEqual(self.source.calls, [])
        self.assertTrue(all(axis.get_position().x0 >= .29 for axis in self.player.axes))

    def test_subject_and_file_filters_change_list_only(self):
        self.player.search_box.set_val('t21')
        self.assertEqual(len(self.player.filtered), 2)
        self.player.subject_box.set_val('sa21')
        self.assertEqual([row['file'] for row in self.player.filtered], ['S21T21R01.csv'])
        self.player.search_box.set_val('absent')
        self.assertEqual(self.player.file_buttons, [])
        self.assertIn('Sin coincidencias', self.player.status_text.get_text())
        self.assertEqual(self.source.calls, [])

    def test_editing_a_filter_does_not_trigger_playback_keyboard_shortcuts(self):
        self.player.seek(3.5)
        with patch.object(self.player.search_box, 'capturekeystrokes', True):
            self.player._on_key(SimpleNamespace(key='left'))
            self.assertEqual(self.player.current_time, 3.5)
        self.player._on_key(SimpleNamespace(key='left'))
        self.assertLess(self.player.current_time, 3.5)

    def test_file_selection_waits_for_result_then_replaces_plot_and_slider(self):
        self.player.seek(4.)
        self.player.widgets['sensor'].set_active(1)
        old = self.player.replay
        row, button = self.player.file_buttons[1]
        button._observers.process('clicked', None)
        self.assertTrue(self.player._busy)
        self.assertIs(self.player.replay, old)
        self.assertFalse(self.player.playing)
        self.player._poll_load()
        self.assertFalse(self.player._busy)
        self.assertEqual(self.player.replay.path.name, row['file'])
        self.assertEqual(self.player.current_time, 0.)
        self.assertEqual(self.player.widgets['time'].valmax, 5.)
        self.assertIn('GyrX', self.player.axes[0].get_ylabel())
        self.player.seek(99.)
        self.assertEqual(self.player.current_time, 5.)
        self.assertEqual(self.source.calls, [('test', 'S07T21R01.csv')])

    def test_partition_switch_clears_filters_and_labels_the_actual_partition(self):
        self.player.subject_box.set_val('SA07')
        self.player.search_box.set_val('T21')
        self.player.split_selector.set_active(1)
        self.assertEqual(self.player.subject_box.text, '')
        self.assertEqual(self.player.search_box.text, '')
        self.player._poll_load()
        self.assertEqual(self.player.replay.subject, 'SA15')
        self.assertIn('VAL ·', self.player.recording_text.get_text())
        self.assertEqual(self.source.calls, [('val', 'S15T20R01.csv')])

    def test_failed_recording_preserves_previous_plot_and_reenables_selection(self):
        previous, plot = self.player.replay, self.player.plot
        row = self.player.file_buttons[1][0]
        self.source.fail_files.add(row['file'])
        self.player._select_file(row)
        self.player._poll_load()
        self.assertIs(self.player.replay, previous)
        self.assertIs(self.player.plot, plot)
        self.assertFalse(self.player._busy)
        self.assertTrue(self.player.search_box.active)
        self.assertIn('Cannot validate', self.player.status_text.get_text())

    def test_failed_partition_restores_the_previous_selector(self):
        self.source.fail_splits.add('val')
        self.player.split_selector.set_active(1)
        self.assertEqual(self.player.split_name, 'test')
        self.assertEqual(self.player.split_selector.value_selected, 'Test')
        self.assertIn('unavailable', self.player.status_text.get_text())
        self.assertEqual(self.source.calls, [])

    def test_pending_selection_does_not_replace_plot_or_allow_concurrent_requests(self):
        pending = PendingExecutor()
        self.player._executor = pending
        previous = self.player.replay
        self.player._select_file(self.player.file_buttons[1][0])
        self.player._select_file(self.player.file_buttons[2][0])
        self.player._poll_load()
        self.assertIs(self.player.replay, previous)
        self.assertTrue(self.player._busy)
        self.assertFalse(self.player.split_selector.active)
        self.player.close_browser()
        self.assertTrue(pending.future.cancelled())

    def test_real_executor_computes_off_the_gui_thread_and_installs_on_poll_only(self):
        original, main_thread, worker_threads = self.source.replay, threading.get_ident(), []
        def compute(*args, **kwargs):
            worker_threads.append(threading.get_ident())
            return original(*args, **kwargs)
        with ThreadPoolExecutor(max_workers=1) as pool, patch.object(self.source, 'replay', side_effect=compute):
            self.player._executor = pool
            before = self.player.replay
            self.player._select_file(self.player.file_buttons[1][0])
            self.player._future.result(timeout=5)
            self.assertIs(self.player.replay, before)
            self.assertNotEqual(worker_threads, [main_thread])
            self.player._poll_load()
            self.assertEqual(self.player.replay.path.name, 'S07T21R01.csv')
