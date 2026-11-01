"""Boundary, provenance and causal motion tests for KFall relabeling."""
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from preprocess.kfall_labels import (record_frame_labels, label_manifest, record_frames,
                                     label_at_frame, labels_from_intervals)
from loaders.kfall_event_loader import (KFallEventLoaderConfig, window_targets_for_record,
                                        load_or_build_kfall_event_splits)
import test_kfall_events as event_fixtures


class TemporalLabelsTests(unittest.TestCase):
    def setUp(self):
        self.frames = np.arange(101, 1101)
        self.record = {'T': 1000, 'label_id': 0, 'fall_onset_frame': 501,
                       'fall_impact_frame': 601, 'onset_idx': 400, 'impact_idx': 500,
                       'frames': self.frames}
        self.kwargs = dict(label_policy='temporal', post_impact_seconds=1.2)

    def test_fall_boundaries_and_rest_static(self):
        labels = record_frame_labels(self.frames, self.record, **self.kwargs)
        for frame, expected in ((101, 2), (380, 2), (381, 1), (500, 1),
                                (501, 0), (600, 0), (601, 3), (720, 3),
                                (721, 1), (920, 1), (921, 2), (1100, 2)):
            self.assertEqual(labels[frame - 101], expected, frame)

    def test_adl_task_family_no_longer_controls_middle(self):
        for task_label in (1, 2):
            record = {'T': 1000, 'label_id': task_label}
            labels = record_frame_labels(self.frames, record, **self.kwargs)
            np.testing.assert_array_equal(labels[:100], 2)
            np.testing.assert_array_equal(labels[100:-100], 1)
            np.testing.assert_array_equal(labels[-100:], 2)
        for length in (100, 199, 200):
            labels = record_frame_labels(self.frames[:length], {'T': length, 'label_id': 1}, **self.kwargs)
            self.assertTrue(np.all(labels == 2))
        labels = record_frame_labels(self.frames[:201], {'T': 201, 'label_id': 1}, **self.kwargs)
        np.testing.assert_array_equal(np.flatnonzero(labels == 1), [100])

    def test_explicit_saved_adl_margin_remains_supported(self):
        record = {'T': 1000, 'label_id': 2}
        old = record_frame_labels(self.frames, record, adl_margin_seconds=1.5, **self.kwargs)
        current = record_frame_labels(self.frames, record, **self.kwargs)
        self.assertEqual(old[100], 2)
        self.assertEqual(current[100], 1)
        np.testing.assert_array_equal(old[:150], 2)
        np.testing.assert_array_equal(old[150:-150], 1)
        np.testing.assert_array_equal(old[-150:], 2)

    def test_explicit_saved_two_second_policy_keeps_its_original_labels(self):
        old = record_frame_labels(self.frames, self.record, pre_fall_seconds=2.0, **self.kwargs)
        current = record_frame_labels(self.frames, self.record, **self.kwargs)
        self.assertEqual(old[300 - 101], 2)
        self.assertEqual(old[301 - 101], 1)
        self.assertEqual(current[301 - 101], 2)
        np.testing.assert_array_equal(old[400:], current[400:])

    def test_clipped_fall_intervals_and_missing_frames(self):
        frames = np.array([450, 500, 501, 599, 600, 601, 700, 720, 721, 920, 921])
        record = dict(self.record, T=len(frames), frames=frames)
        expected = [1, 1, 0, 0, 0, 3, 3, 3, 1, 1, 2]
        labels = record_frame_labels(frames, record, **self.kwargs)
        np.testing.assert_array_equal(labels, expected)
        manifest = label_manifest(frames, labels)
        record = {key: value for key, value in record.items() if key != 'frames'} | manifest
        np.testing.assert_array_equal(record_frames(record), frames)
        np.testing.assert_array_equal(labels_from_intervals(frames, manifest['frame_label_intervals']), labels)
        for frame, label in zip(frames, labels):
            self.assertEqual(label_at_frame(int(frame), record, **self.kwargs), label)

    def test_window_endpoint_target_and_raw_free_reconstruction(self):
        signal = np.zeros((1000, 6), dtype=np.float32)
        cfg = KFallEventLoaderConfig(raw_win=100, out_win=16, stride=50, **self.kwargs)
        starts, labels = window_targets_for_record(signal, self.record, cfg)
        full = record_frame_labels(self.frames, self.record, **self.kwargs)
        np.testing.assert_array_equal(labels, full[starts + 99])
        self.assertEqual(set(labels), {0, 1, 2, 3})
        record = {key: value for key, value in self.record.items() if key != 'frames'}
        record.update(label_manifest(self.frames, full))
        offline_starts, offline_labels = window_targets_for_record(np.empty((1000, 0)), record, cfg)
        np.testing.assert_array_equal(offline_starts, starts)
        np.testing.assert_array_equal(offline_labels, labels)

    def test_motion_uses_past_only_and_preserves_protected_classes(self):
        signal = np.zeros((1000, 6), dtype=np.float32)
        signal[250:, 0] = .2
        kwargs = self.kwargs | {'label_policy': 'motion'}
        labels = record_frame_labels(self.frames, self.record, signal=signal, **kwargs)
        self.assertEqual(labels[249], 2)
        self.assertEqual(labels[250], 1)
        self.assertEqual(labels[300], 2)
        np.testing.assert_array_equal(labels[400:500], 0)
        np.testing.assert_array_equal(labels[500:620], 3)
        modified = signal.copy()
        modified[700:] = 999
        other = record_frame_labels(self.frames, self.record, signal=modified, **kwargs)
        np.testing.assert_array_equal(labels[:700], other[:700])
        # An ADL flat segment is static even in the nominal temporal middle.
        flat = record_frame_labels(self.frames, {'T': 1000, 'label_id': 1},
                                   signal=np.zeros_like(signal), **kwargs)
        self.assertTrue(np.all(flat == 2))

    def test_invalid_label_settings_rejected(self):
        for changes in ({'label_policy': 'bad'}, {'adl_margin_seconds': -.1},
                        {'pre_fall_seconds': float('nan')}, {'post_recovery_seconds': .015},
                        {'motion_window_seconds': 0}, {'motion_acc_threshold': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                KFallEventLoaderConfig(**changes)

    def test_new_cache_version_manifest_and_legacy_reuse(self):
        fixture = event_fixtures.KFallEventTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'KFall'
            root.mkdir()
            fixture.write_fixture(root)
            cfg = KFallEventLoaderConfig(raw_win=5, out_win=16, stride=4, label_policy='temporal',
                                         post_impact_seconds=1.2)
            cache = Path(temporary) / 'cache'
            splits, meta = load_or_build_kfall_event_splits(root, cache_dir=cache, cfg=cfg, verbose=False)
            self.assertEqual(meta['version'], 4)
            self.assertTrue(all('frame_label_intervals' in record for record in meta['recordings']))
            for arrays in splits.values():
                self.assertTrue(np.isin(arrays['y_h1'], [0, 1, 2, 3]).all())
            with patch('loaders.kfall_event_loader.load_kfall_3class', side_effect=AssertionError('cache missed')):
                load_or_build_kfall_event_splits(root, cache_dir=cache, cfg=cfg, verbose=False)
            legacy = replace(cfg, label_policy='legacy', post_impact_seconds=0)
            _, metadata = load_or_build_kfall_event_splits(root, cache_dir=cache, run_name='legacy', cfg=legacy, verbose=False)
            self.assertEqual(metadata['version'], 1)
            self.assertNotIn('label_policy', metadata['config'])
