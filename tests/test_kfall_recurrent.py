"""Causality, state isolation, masked pairs and temporal checkpoint integration."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from models.temporal import TemporalSCN, detach_state
from train.utils import scn_model_kwargs
from train_kfall_temporal import get_args, main
from loaders.kfall_sequences import EpisodeSet, training_statistics
from train.temporal import temporal_loss
from test_kfall_training import write_fixture, fixture_args


class RecurrentTests(unittest.TestCase):
    def cfg(self):
        return get_args(['--conv-ch', '4', '4', '4', '--kernels', '3', '3', '3', '--out-win', '128',
                         '--p-drop', '0', '--class-weight', 'none'])

    def model(self):
        cfg = self.cfg()
        return TemporalSCN(scn_model_kwargs(cfg), hidden_size=4, boundary=True).eval()

    def test_causality_chunked_equals_full_and_explicit_reset(self):
        torch.manual_seed(3)
        model = self.model()
        events = torch.randint(0, 2, (2, 5, 128, 6, 2)).float()
        offsets = torch.randn(2, 5, 6)
        full, _ = model(events, offsets)
        first, state = model(events[:, :2], offsets[:, :2])
        second, _ = model(events[:, 2:], offsets[:, 2:], detach_state(state))
        torch.testing.assert_close(torch.cat((first['logits'], second['logits']), 1), full['logits'])
        changed = events.clone()
        changed[:, 3:] = 100
        future, _ = model(changed, offsets)
        torch.testing.assert_close(future['logits'][:, :3], full['logits'][:, :3])
        reset, _ = model(events, offsets, None)
        torch.testing.assert_close(reset['logits'], full['logits'])

    def test_padding_does_not_change_states_or_valid_logits(self):
        model = self.model()
        events = torch.randn(2, 4, 128, 6, 2)
        offsets = torch.randn(2, 4, 6)
        valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
        out, state = model(events, offsets, valid=valid)
        single, single_state = model(events[:1, :2], offsets[:1, :2])
        torch.testing.assert_close(out['logits'][:1, :2], single['logits'])
        for current, expected in zip(state, single_state):
            torch.testing.assert_close(current[:1], expected)

    def test_masks_and_pairs_across_chunk_boundary(self):
        logits = torch.randn(1, 3, 4, requires_grad=True)
        boundary = torch.randn(1, 3, requires_grad=True)
        target = torch.tensor([[1, 0, 3]])
        valid = torch.tensor([[1, 0, 1]], dtype=torch.bool)
        loss, parts = temporal_loss({'logits': logits, 'boundary': boundary}, target, valid,
            torch.tensor([2]), torch.tensor([True]), costs=torch.ones(4, 4),
            previous_probability=torch.ones(1, 4) / 4, lambda_transition=.1, lambda_boundary=.2)
        self.assertEqual(parts['pair_count'].item(), 1)
        self.assertEqual(parts['boundary_target'][0, 0].item(), 1)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[0, 1] == 0))
        self.assertTrue(torch.all(boundary.grad[0, 1:] == 0))

    def test_full_training_evaluation_and_train_only_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root)
            args = fixture_args(root) + ['--post-impact-seconds', '1.2', '--label-policy', 'temporal',
                '--raw-win', '50', '--stride', '25', '--memory', 'lif', '--recurrent-hidden', '4',
                '--sequence-length', '3', '--lambda-transition', '.1', '--lambda-boundary', '.1',
                '--class-weight', 'none']
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            run = next((root / 'runs').iterdir())
            config = json.loads((run / 'config.json').read_text())
            statistics = json.loads((run / 'training_statistics.json').read_text())
            metadata = json.loads((run / 'dataset_metadata.json').read_text())
            self.assertEqual(config['model_type'], 'temporal_scn')
            self.assertEqual(statistics['subjects'], metadata['split_subjects']['train'])
            self.assertTrue(np.isfinite(statistics['transition_costs']).all())
            metrics = json.loads((run / 'metrics/test_best/metrics.json').read_text())
            self.assertIn('event_metrics', metrics)
            self.assertEqual(metrics['observed_windows'], 33)
            self.assertTrue((run / 'metrics/test_best/episode_predictions.npz').is_file())
            from evaluation.kfall_temporal import load_temporal_run
            from unittest.mock import patch
            opened = []
            load = np.load
            def track(path, *a, **kw):
                opened.append(str(path)); return load(path, *a, **kw)
            with patch('numpy.load', side_effect=track):
                load_temporal_run(run, 'cpu', split_name='val')
            self.assertFalse(any('/test/' in path for path in opened))
            from compare_kfall_validation import compare_validation
            self.assertTrue(compare_validation(run, run)['non_degrading'])
            from animate_kfall_temporal import prepare_temporal_replay
            replay = prepare_temporal_replay(run, device='cpu')
            self.assertTrue(np.isin(replay.predicted_classes, [0, 1, 2, 3]).all())
            self.assertEqual(len(replay.predicted_classes), len(replay.endpoint_times))
            import csv
            with (run / 'metrics/validation/predictions.csv').open() as stream:
                self.assertEqual(next(csv.DictReader(stream))['split'], 'val')


    def test_hierarchical_memory_and_no_memory_baseline_smoke(self):
        from evaluation.kfall_temporal import evaluate_temporal
        for memory, head in (('none', 'single'), ('lif', 'hierarchical')):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_fixture(root)
                flags = fixture_args(root) + ['--post-impact-seconds', '1.2', '--label-policy', 'temporal',
                    '--raw-win', '50', '--stride', '25', '--memory', memory, '--head-mode', head,
                    '--recurrent-hidden', '4', '--sequence-length', '4', '--class-weight', 'none', '--validation-only']
                with contextlib.redirect_stdout(io.StringIO()):
                    main(flags)
                run = next((root / 'runs').iterdir())
                self.assertFalse((run / 'metrics/test_best').exists())
                if head == 'hierarchical':
                    self.assertTrue((run / 'metrics/validation/hierarchy/hierarchy_metrics.json').is_file())
                measured = evaluate_temporal(run, device='cpu')
                self.assertEqual(measured['memory'], memory)
                self.assertTrue(np.isfinite(measured['loss']))


class EpisodeProtocolTests(unittest.TestCase):
    def episodes(self):
        metadata = dict(config=dict(label_policy='temporal', post_impact_seconds=1.2, raw_win=2,
                                    endpoint_context=None, stride=2),
            recordings=[dict(T=10, subject='SA01', label_id=1), dict(T=6, subject='SA02', label_id=2)],
            split_file_indices=dict(train=[0], val=[1], test=[]))
        arrays = dict(file_idx=np.array([0, 0, 0, 0, 0, 1, 1, 1]),
            start=np.array([0, 2, 3, 4, 6, 0, 2, 4]),
            end_frame=np.array([2, 4, 5, 6, 8, 2, 4, 6]),
            subject=np.array(['SA01'] * 5 + ['SA02'] * 3),
            y_h1=np.array([2, 1, 0, 0, 3, 2, 2, 1]),
            x_ev=np.zeros((8, 16, 6, 2), dtype=np.uint8), offset=np.zeros((8, 6), dtype=np.float32))
        return EpisodeSet(arrays, metadata)

    def test_uniform_endpoints_no_cross_episode_pairs_and_padding(self):
        dataset = self.episodes()
        np.testing.assert_array_equal(dataset.episodes[0].indices, [0, 1, 3, 4])
        group = next(dataset.groups(2))
        chunks = list(dataset.chunks(group, 3))
        self.assertFalse(chunks[0]['previous_valid'].any())
        self.assertEqual(chunks[1]['targets'].shape, (2, 1))
        self.assertFalse(chunks[1]['present'][1, 0])
        self.assertEqual(chunks[1]['previous_target'][0].item(), 0)
        for chunk in chunks:
            for row, ep in enumerate(group):
                selected = chunk['indices'][row][chunk['present'][row].numpy()]
                self.assertTrue(np.all(dataset.arrays['file_idx'][selected] == ep.file_index))

    def test_train_only_statistics_reject_validation_records(self):
        dataset = self.episodes()
        with self.assertRaisesRegex(ValueError, 'training'):
            training_statistics(dataset)
        dataset.episodes = dataset.episodes[:1]
        stats = training_statistics(dataset)
        self.assertEqual(stats['source_split'], 'train')
        self.assertEqual(np.sum(stats['transition_counts']), 3)
        self.assertTrue(np.all(np.diag(stats['transition_costs']) == 0))
        self.assertTrue(np.isfinite(stats['transition_costs']).all())

    def test_synthetic_episode_detection_fragmentation_and_false_alarms(self):
        from evaluation.kfall_temporal import episode_metrics
        records = [dict(T=300, label_id=0, fall_onset_frame=101, fall_impact_frame=151,
                        subject='SA01', file='fall.csv', frame_label_intervals=[
                            [1, 101, 2], [101, 151, 0], [151, 201, 3], [201, 301, 1]]),
                   dict(T=100, label_id=0, fall_onset_frame=10, fall_impact_frame=20,
                        subject='SA02', file='early.csv', frame_label_intervals=[
                            [1, 10, 1], [10, 20, 0], [20, 101, 3]])]
        arrays = dict(file_idx=np.array([0] * 5), end_frame=np.array([51, 101, 151, 201, 251]))
        predicted = dict(indices=np.arange(5), predicted=np.array([0, 0, 3, 0, 1]))
        metadata = dict(recordings=records, split_file_indices=dict(test=[0, 1]))
        metrics = episode_metrics(predicted, arrays, metadata, 'test')
        self.assertEqual(metrics['event_recall'], .5)
        self.assertEqual(metrics['false_fall_alarms'], 1)
        self.assertAlmostEqual(metrics['normal_observed_hours'] * 3600, 1.5)
        self.assertEqual(metrics['mean_detection_latency_seconds'], -.5)
        self.assertEqual(metrics['fragments_per_detected_fall'], 1)
