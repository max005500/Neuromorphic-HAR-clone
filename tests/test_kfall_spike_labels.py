"""Whole-record event cluster labels and persisted evaluation provenance."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import shutil
import numpy as np
import torch
from preprocess.kfall_labels import sustained_spike_activity, record_frame_labels, label_manifest
from loaders.kfall_event_loader import KFallEventLoaderConfig, window_targets_for_record
from train_kfall import get_args, main
from evaluation.kfall_data import load_evaluation_data
from test_kfall_training import write_fixture, fixture_args


class SpikeLabelTests(unittest.TestCase):
    def clusters(self, event_times, protected=None):
        times = np.arange(201) / 100
        events = np.zeros((201, 6, 2), dtype=np.uint8)
        events[np.rint(np.array(event_times) * 100).astype(int), 0, 0] = 1
        return sustained_spike_activity(times, events, protected=protected)

    def test_no_events_and_isolated_events_are_static(self):
        self.assertFalse(self.clusters([]).any())
        self.assertFalse(self.clusters([.1, .5, .9, 1.3]).any())

    def test_gyro_events_alone_do_not_make_a_dynamic_cluster(self):
        times = np.arange(201) / 100
        events = np.zeros((201, 6, 2), dtype=np.uint8)
        events[10:80, 3:, :] = 1
        self.assertFalse(sustained_spike_activity(times, events).any())
        self.assertTrue(sustained_spike_activity(times, events, sensor='imu').any())
        events[10:80, 2, 1] = 1
        self.assertTrue(sustained_spike_activity(times, events).any())

    def test_full_record_with_only_gyro_movement_is_static(self):
        frames = np.arange(1, 301)
        signal = np.zeros((300, 6), dtype=np.float32)
        signal[:, 3] = np.arange(300) * 10
        labels = record_frame_labels(frames, dict(T=300, label_id=1), signal=signal,
                                     label_policy='spikes')
        self.assertTrue(np.all(labels == 2))

    def test_strict_duration_boundary_and_internal_gap(self):
        self.assertFalse(self.clusters([.1, .2, .3, .4]).any())
        moving = self.clusters([.1, .2, .3, .4, .41])
        np.testing.assert_array_equal(np.flatnonzero(moving), np.arange(10, 42))
        self.assertFalse(self.clusters([.1, .2, .31, .41]).any())

    def test_fall_and_postimpact_break_clusters(self):
        protected = np.zeros(201, dtype=bool)
        protected[25:31] = True
        moving = self.clusters([.1, .2, .25, .30, .31, .4, .5], protected)
        self.assertFalse(moving.any())

    def test_continuous_signal_retains_official_labels_and_cache_targets(self):
        frames = np.arange(1, 501)
        signal = np.zeros((500, 6), dtype=np.float32)
        signal[50:150, 0] = np.arange(100) * .03
        record = dict(T=500, label_id=0, fall_onset_frame=201, fall_impact_frame=251,
                      onset_idx=200, impact_idx=250, frames=frames)
        cfg = KFallEventLoaderConfig(raw_win=100, out_win=16, stride=20, label_policy='spikes',
                                    post_impact_seconds=1.2)
        labels = record_frame_labels(frames, record, signal=signal, label_policy='spikes', post_impact_seconds=1.2)
        self.assertEqual(labels[0], 2)
        self.assertEqual(labels[100], 1)
        np.testing.assert_array_equal(labels[200:250], 0)
        np.testing.assert_array_equal(labels[250:370], 3)
        record.update(label_manifest(frames, labels))
        for window in (100, 205):
            from dataclasses import replace
            starts, targets = window_targets_for_record(np.empty((500, 0)), record,
                                                        replace(cfg, raw_win=window, endpoint_context=205))
            np.testing.assert_array_equal(targets, labels[starts + window - 1])
        cached = record_frame_labels(frames, record, label_policy='spikes')
        np.testing.assert_array_equal(cached, labels)

    def test_cli_cache_identity_and_validation(self):
        baseline = get_args(['--label-policy', 'temporal'])
        spikes = get_args(['--label-policy', 'spikes'])
        self.assertNotEqual(baseline.cache_name, spikes.cache_name)
        different = get_args(['--label-policy', 'spikes', '--spike-max-gap-seconds', '.05'])
        self.assertNotEqual(spikes.cache_name, different.cache_name)
        for kwargs in ({'spike_min_duration_seconds': -.1}, {'spike_max_gap_seconds': 0}, {'spike_label_hz': 50}):
            with self.assertRaises(ValueError):
                KFallEventLoaderConfig(**kwargs)

    def test_training_and_raw_free_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root)
            args = fixture_args(root) + ['--label-policy', 'spikes', '--post-impact-seconds', '1.2', '--cache-name', 'spikes', '--class-weight', 'none']
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            run = next((root / 'runs').iterdir())
            saved = torch.load(run / 'checkpoints/best_model.pt', map_location='cpu', weights_only=True)
            shutil.rmtree(root / 'sensor_data')
            shutil.rmtree(root / 'label_data')
            arrays, _, metadata, _ = load_evaluation_data(run, saved['config'])
            self.assertGreater(len(arrays['y_h1']), 0)
            self.assertEqual(metadata['config']['spike_label_hz'], 200.)
            self.assertEqual(metadata['version'], 5)
            self.assertTrue(all(r['spike_label_sensor'] == 'acc' for r in metadata['recordings']))
            with self.assertRaisesRegex(ValueError, 'spike_max_gap_seconds'):
                load_evaluation_data(run, dict(saved['config'], spike_max_gap_seconds=.05))
