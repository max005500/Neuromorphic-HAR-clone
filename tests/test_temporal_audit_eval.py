"""Regression tests for replay work, data identity, and validation indices."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from animate_kfall_temporal import prepare_temporal_replay
from compare_kfall_temporal import validate_comparison_rows
from compare_kfall_validation import read_validation
from evaluation.kfall_temporal import data_provenance_sha256
from loaders.kfall_event_loader import KFallEventLoaderConfig
from visualization.kfall_data import source_frame_labels
from visualization.kfall_replay_data import _validate_source


class ReplayAuditTests(unittest.TestCase):
    def replay(self, corrupt=False):
        frames = np.arange(1, 6)
        raw = np.zeros((5, 6), dtype=np.float32)
        record = dict(T=5, file='S01T04R01.csv', subject='SA01', code='T04', trial='R01', label_id=1)
        loader = KFallEventLoaderConfig(raw_win=2, out_win=4, stride=1,
                                        label_policy='temporal', post_impact_seconds=1.2)
        cfg = SimpleNamespace(boundary_ignore_seconds=0., raw_win=2, out_win=4,
                              stride=1, post_impact_seconds=1.2, head_mode='single')
        arrays = dict(file_idx=np.zeros(4, dtype=int), start=np.arange(4), end_frame=frames[1:],
            subject=np.array(['SA01'] * 4), y_h1=np.array([2] * 4),
            x_ev=np.zeros((4, 4, 6, 2), dtype=np.uint8), offset=np.zeros((4, 6), dtype=np.float32))
        if corrupt:
            arrays['x_ev'][0, 0, 0, 0] = 1
        metadata = dict(config=dict(label_policy='temporal', post_impact_seconds=1.2,
                                    raw_win=2, stride=1), recordings=[record])
        prediction = dict(indices=np.arange(4), probability=np.tile([.1, .2, .6, .1], (4, 1)),
                          predicted=np.array([2] * 4))
        store = SimpleNamespace(cfg=loader, _record_path=lambda rec: Path(rec['file']),
            get_window=lambda *a: self.fail('Replay must not recompute full-record labels per window.'))
        with patch('animate_kfall_temporal.load_temporal_run', return_value=(object(), cfg,
                dict(epoch=1, best_val_loss=.1), arrays, metadata, Path('/cache'), torch.device('cpu'), {})), patch(
                'animate_kfall_temporal.KFallWindowStore', return_value=store), patch(
                'animate_kfall_temporal.load_record', return_value=(raw, frames, record)), patch(
                'animate_kfall_temporal._validate_source', wraps=_validate_source) as validate, patch(
                'animate_kfall_temporal.source_frame_labels', wraps=source_frame_labels) as labels, patch(
                'animate_kfall_temporal.run_temporal_epoch') as infer:
            def predict(*a, **kw):
                self.assertEqual(validate.call_count, 1)
                self.assertEqual(labels.call_count, 1)
                return {'predictions': prediction}
            infer.side_effect = predict
            if corrupt:
                with self.assertRaisesRegex(ValueError, 'differs'):
                    prepare_temporal_replay('/run')
                self.assertFalse(infer.called)
            else:
                result = prepare_temporal_replay('/run')
                self.assertEqual(labels.call_count, 1)
                self.assertEqual(len(result.endpoint_times), 4)
                np.testing.assert_array_equal(result.true_labels, [2] * 5)

    def test_full_record_labels_computed_once_before_inference(self):
        self.replay()

    def test_corrupt_cached_inputs_rejected_before_inference(self):
        self.replay(corrupt=True)


class DataIdentityAuditTests(unittest.TestCase):
    def metadata(self):
        return dict(version=5, dataset='kfall', source_signature={'sha256': 'source'},
            split_subjects=dict(train=['SA01'], val=['SA02'], test=['SA03']),
            split_file_indices=dict(train=[0], val=[1], test=[2]),
            channels=['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ'], units=['g'] * 3 + ['degrees/s'] * 3,
            fall_label_rule='onset <= final frame < impact',
            config=dict(raw_win=2, theta_sd=[.1] * 3 + [10.] * 3, batch_size=256),
            recordings=[dict(file=f'/original/record{i}.csv', subject=f'SA0{i + 1}', label_id=1) for i in range(3)])

    def test_data_hash_detects_encoder_and_source_changes_not_runtime(self):
        base = self.metadata()
        baseline = data_provenance_sha256(base)
        runtime = copy.deepcopy(base)
        runtime['config']['batch_size'] = 32
        self.assertEqual(data_provenance_sha256(runtime), baseline)
        relocated = copy.deepcopy(base)
        relocated['recordings'][0]['file'] = '/relocated/record0.csv'
        self.assertEqual(data_provenance_sha256(relocated), baseline)
        for mutation in ('source', 'theta', 'record'):
            changed = copy.deepcopy(base)
            if mutation == 'source':
                changed['source_signature']['sha256'] = 'changed'
            elif mutation == 'theta':
                changed['config']['theta_sd'][3] = 5.
            else:
                changed['recordings'][0]['spike_label_sensor'] = 'acc'
            self.assertNotEqual(data_provenance_sha256(changed), baseline)

    def test_same_cohort_is_insufficient_when_encoder_changes(self):
        rows = [dict(cohort_sha256='same', data_provenance_sha256='a'),
                dict(cohort_sha256='same', data_provenance_sha256='b')]
        with self.assertRaisesRegex(ValueError, 'provenance'):
            validate_comparison_rows(rows)
        rows[1]['data_provenance_sha256'] = 'a'
        validate_comparison_rows(rows)


class ValidationIndexAuditTests(unittest.TestCase):
    def read(self, indices):
        arrays = dict(y_h1=np.array([0, 1, 2]), file_idx=np.array([0, 1, 2]), end_frame=np.array([100, 100, 100]))
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / 'metrics/validation').mkdir(parents=True)
            (run / 'config.json').write_text('{}')
            index = np.asarray(indices)
            target = arrays['y_h1'][index] if index.dtype.kind in 'iu' and not np.any(index >= 3) else np.array([0, 1])
            np.save(run / 'metrics/y_true_val.npy', target)
            np.save(run / 'metrics/y_pred_val.npy', target)
            np.save(run / 'metrics/validation/window_indices.npy', index)
            with patch('compare_kfall_validation.load_evaluation_data', return_value=(arrays, {}, {}, run)):
                return read_validation(run)

    def test_duplicate_negative_noninteger_and_nonvector_indices_rejected(self):
        for indices in ([0, 0], [-1, 0], [0., 1.], [[0, 1]], [0, 3]):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, 'indices'):
                self.read(indices)

    def test_different_prediction_order_has_canonical_cohort(self):
        ordered = self.read([0, 1, 2])
        shuffled = self.read([2, 0, 1])
        np.testing.assert_array_equal(ordered['cohort'], shuffled['cohort'])
        self.assertEqual(ordered['cohort_sha256'], shuffled['cohort_sha256'])
