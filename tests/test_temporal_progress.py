"""Progress counts actual TBPTT chunks and closes cleanly on failure."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from train.temporal import run_temporal_epoch


class ProgressBar:
    def __init__(self, **kwargs):
        self.total = kwargs['total']
        self.description = kwargs['desc']
        self.disable = kwargs['disable'] is True
        self.completed, self.closed, self.postfix = 0, False, {}
    def update(self, count):
        self.completed += count
    def close(self):
        self.closed = True
    def set_postfix(self, values, refresh=False):
        self.postfix = values


class Episodes:
    def groups(self, batch_size, **kwargs):
        yield [SimpleNamespace(indices=np.arange(3)), SimpleNamespace(indices=np.arange(1))]
        yield [SimpleNamespace(indices=np.arange(2))]
    def chunks(self, group, sequence_length):
        for begin in range(0, max(len(ep.indices) for ep in group), sequence_length):
            present = torch.tensor([[begin + t < len(ep.indices) for t in range(1)] for ep in group])
            size = len(group)
            yield dict(events=torch.zeros(size, 1, 1), offsets=torch.zeros(size, 1, 6),
                targets=torch.ones(size, 1, dtype=torch.long), valid=present, present=present,
                previous_target=torch.ones(size, dtype=torch.long), previous_valid=torch.zeros(size, dtype=torch.bool))


class Model(torch.nn.Module):
    def forward(self, events, offsets, state, valid):
        return {'logits': torch.zeros(len(events), 1, 4)}, None
    def predict(self, outputs):
        return outputs['logits'].argmax(-1)


class TemporalProgressTests(unittest.TestCase):
    def test_complete_count_padding_and_loss_refresh(self):
        cfg = SimpleNamespace(class_weight='none', batch_size=2, sequence_length=1, seed=0,
            amp=False, lambda_transition=.1, lambda_boundary=0., label_smoothing=0., epochs=10, no_progress=False)
        bars = []
        def create(**kwargs):
            bar = ProgressBar(**kwargs); bars.append(bar); return bar
        with patch('train.temporal.tqdm', side_effect=create), patch('train.temporal.time.monotonic', side_effect=range(0, 100, 6)):
            metrics = run_temporal_epoch(Model(), Episodes(), cfg, torch.device('cpu'),
                dict(transition_costs=np.ones((4, 4)), boundary_pos_weight=1.), epoch=1, phase='Val')
        self.assertEqual(bars[0].total, 5)
        self.assertEqual(bars[0].completed, 5)
        self.assertEqual(bars[0].postfix['ventanas'], 6)
        self.assertEqual(bars[0].description, 'Val e001/10')
        self.assertTrue(bars[0].closed)
        self.assertAlmostEqual(float(bars[0].postfix['loss']), metrics['loss'], places=3)

    def test_bar_closed_when_forward_raises(self):
        cfg = SimpleNamespace(class_weight='none', batch_size=2, sequence_length=1, seed=0, amp=False)
        bar = ProgressBar(total=5, desc='Train', disable=False)
        with patch('train.temporal.tqdm', return_value=bar), patch.object(Model, 'forward', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                run_temporal_epoch(Model(), Episodes(), cfg, torch.device('cpu'),
                    dict(transition_costs=np.ones((4, 4)), boundary_pos_weight=1.))
        self.assertTrue(bar.closed)
        self.assertEqual(bar.completed, 0)

    def test_disabled_progress_does_not_copy_gpu_statistics_per_chunk(self):
        cfg = SimpleNamespace(class_weight='none', batch_size=2, sequence_length=1, seed=0,
            amp=False, lambda_transition=0., lambda_boundary=0., label_smoothing=0., no_progress=True)
        bar = ProgressBar(total=5, desc='Val', disable=True)
        copies = []
        original = torch.Tensor.cpu
        def track(value, *args, **kwargs):
            copies.append(1)
            return original(value, *args, **kwargs)
        with patch('train.temporal.tqdm', return_value=bar), patch.object(torch.Tensor, 'cpu', track):
            run_temporal_epoch(Model(), Episodes(), cfg, torch.device('cpu'),
                dict(transition_costs=np.ones((4, 4)), boundary_pos_weight=1.))
        self.assertEqual(len(copies), 1)  # final epoch totals only
        self.assertTrue(bar.closed)
