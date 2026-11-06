"""Replay selection must explain missing recordings and expose real partitions."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from animate_kfall_temporal import list_temporal_recordings, main, prepare_temporal_replay


class TemporalRecordingSelectionTests(unittest.TestCase):
    def load(self, run, device='auto', *, split_name='test'):
        subjects = ['SA06', 'SA15', 'SA07']
        split_files = {'train': [0], 'val': [1], 'test': [2]}
        records = [dict(file=f'/dataset/S{subject[2:]}T04R01.csv', subject=subject,
                        task_id=4, trial_id=1, code='T04', trial='R01', T=3, label_id=1)
                   for subject in subjects]
        metadata = dict(config=dict(raw_win=2, stride=1, label_policy='temporal', post_impact_seconds=1.2),
                        recordings=records, split_file_indices=split_files)
        file_index = split_files[split_name][0]
        arrays = dict(file_idx=np.array([file_index, file_index]), start=np.array([0, 1]),
                      end_frame=np.array([2, 3]), subject=np.array([subjects[file_index]] * 2),
                      y_h1=np.array([1, 1]))
        cfg = SimpleNamespace(boundary_ignore_seconds=0.)
        return object(), cfg, {}, arrays, metadata, Path('/cache'), torch.device('cpu'), {}

    def test_listing_uses_selected_partition_on_cpu_without_csv_or_inference(self):
        with patch('animate_kfall_temporal.load_temporal_run', side_effect=self.load) as loader, patch(
                'animate_kfall_temporal.load_record', side_effect=AssertionError('Listing must not read CSVs')), patch(
                'animate_kfall_temporal.run_temporal_epoch', side_effect=AssertionError('Listing must not infer')):
            rows = list_temporal_recordings('/run', split_name='val')
        self.assertEqual(rows, [dict(file='S15T04R01.csv', subject='SA15', windows=2, split='val')])
        loader.assert_called_once_with('/run', 'cpu', split_name='val')

    def test_cli_listing_does_not_require_a_gui_or_gpu(self):
        output = io.StringIO()
        with patch('animate_kfall_temporal.load_temporal_run', side_effect=self.load), patch(
                'animate_kfall_temporal.prepare_temporal_replay', side_effect=AssertionError('No inference')), patch(
                'matplotlib.get_backend', side_effect=AssertionError('No desktop needed')), contextlib.redirect_stdout(output):
            main(['--run-dir', '/run', '--split', 'train', '--subject', 'SA06', '--list-recordings'])
        self.assertIn('[train] recordings=1', output.getvalue())
        self.assertIn('S06T04R01.csv', output.getvalue())
        self.assertNotIn('S07T04R01.csv', output.getvalue())

    def test_recording_from_another_partition_explains_which_split_to_select(self):
        with patch('animate_kfall_temporal.load_temporal_run', side_effect=self.load), patch(
                'animate_kfall_temporal.load_record', side_effect=AssertionError('Reject before source IO')):
            with self.assertRaisesRegex(ValueError, r'belongs to train; select --split train'):
                prepare_temporal_replay('/run', recording='SA06T04R01.csv')

    def test_unknown_subject_recording_lists_actual_subjects_and_listing_hint(self):
        with patch('animate_kfall_temporal.load_temporal_run', side_effect=self.load):
            with self.assertRaises(ValueError) as raised:
                prepare_temporal_replay('/run', recording='SA01T23R01.csv')
        message = str(raised.exception)
        self.assertIn('absent from this run', message)
        self.assertIn('Available test subjects: SA07', message)
        self.assertIn('--list-recordings --split test', message)

    def test_present_recording_with_no_saved_window_has_a_specific_error(self):
        def load(*args, **kwargs):
            values = self.load(*args, **kwargs)
            metadata = values[4]
            metadata['recordings'].append(dict(file='S07T23R01.csv', subject='SA07',
                task_id=23, trial_id=1, code='T23', trial='R01', T=1, label_id=1))
            metadata['split_file_indices']['test'].append(3)
            return values
        with patch('animate_kfall_temporal.load_temporal_run', side_effect=load):
            with self.assertRaisesRegex(ValueError, 'no complete window'):
                prepare_temporal_replay('/run', recording='SA07T23R01.csv')

    def test_invalid_split_rejected_before_loading_checkpoint(self):
        with patch('animate_kfall_temporal.load_temporal_run', side_effect=AssertionError('No checkpoint IO')):
            for function in (prepare_temporal_replay, list_temporal_recordings):
                with self.subTest(function=function.__name__), self.assertRaisesRegex(ValueError, 'train, val or test'):
                    function('/run', split_name='unknown')
