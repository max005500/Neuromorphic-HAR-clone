"""Numerical parity and synchronization regression tests for single-head epochs."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from train.utils import make_grad_scaler
from train_ucihar import run_epoch


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, x, offset=None):
        return self.fc(x + offset[:, :4] * .1)


def old_epoch(model, batches, cfg, class_weight=None, optimizer=None):
    """The previous scalar accumulation and SGD training path, as reference."""
    loss_sum, correct, count = 0., 0, 0
    true, predicted = [], []
    model.train(optimizer is not None)
    with torch.set_grad_enabled(optimizer is not None):
        for batch in batches:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch['x'], offset=batch['offset'])
            loss = F.cross_entropy(logits, batch['y'], weight=class_weight,
                                   label_smoothing=cfg.label_smoothing)
            if optimizer is not None:
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            pred = logits.argmax(1)
            bs = len(batch['y'])
            loss_sum += loss.detach().item() * bs
            correct += (pred == batch['y']).sum().item()
            count += bs
            true.append(batch['y'].detach().cpu())
            predicted.append(pred.detach().cpu())
    return {'loss': loss_sum / max(count, 1), 'acc': correct / max(count, 1), 'n': count,
            'y_true': torch.cat(true).numpy() if true else np.array([], dtype=np.int64),
            'y_pred': torch.cat(predicted).numpy() if predicted else np.array([], dtype=np.int64)}


class EpochMetricsPerformanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(104)
        self.model = TinyClassifier()
        self.batches = [dict(x=torch.randn(size, 4), offset=torch.randn(size, 6),
                             y=torch.arange(size) % 3) for size in (3, 3, 1)]
        self.cfg = SimpleNamespace(amp=False, label_smoothing=.1, grad_clip=.8)
        self.weight = torch.tensor([.5, 2., 1.3])

    def assert_metrics_equal(self, current, reference):
        self.assertEqual(current['loss'], reference['loss'])
        self.assertEqual(current['acc'], reference['acc'])
        self.assertEqual(current['n'], reference['n'])
        self.assertIsInstance(current['loss'], float)
        self.assertIsInstance(current['acc'], float)
        for key in ('y_true', 'y_pred'):
            np.testing.assert_array_equal(current[key], reference[key])
            self.assertEqual(current[key].dtype, np.int64)

    def test_eval_preserves_weighted_batch_loss_and_uneven_last_batch(self):
        expected = old_epoch(self.model, self.batches, self.cfg, self.weight)
        actual = run_epoch(self.model, self.batches, torch.device('cpu'), self.cfg,
                           class_weight=self.weight, collect_predictions=True)
        self.assert_metrics_equal(actual, expected)
        self.assertEqual(actual['n'], 7)

    def test_training_preserves_optimizer_updates_and_epoch_metrics(self):
        reference = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.07, momentum=.9)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.07, momentum=.9)
        expected = old_epoch(reference, self.batches, self.cfg, self.weight, reference_optimizer)
        actual = run_epoch(self.model, self.batches, torch.device('cpu'), self.cfg,
                           optimizer=optimizer, scaler=make_grad_scaler(False),
                           class_weight=self.weight, collect_predictions=True)
        self.assert_metrics_equal(actual, expected)
        for current, old in zip(self.model.parameters(), reference.parameters()):
            torch.testing.assert_close(current, old, rtol=0, atol=0)
            torch.testing.assert_close(current.grad, old.grad, rtol=0, atol=0)

    def test_empty_epoch_and_disabled_prediction_collection(self):
        empty = run_epoch(self.model, [], torch.device('cpu'), self.cfg, collect_predictions=True)
        self.assert_metrics_equal(empty, old_epoch(self.model, [], self.cfg))
        actual = run_epoch(self.model, self.batches, torch.device('cpu'), self.cfg)
        self.assertNotIn('y_true', actual)
        self.assertNotIn('y_pred', actual)

    def test_eval_does_not_read_a_scalar_per_batch(self):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            run_epoch(self.model, self.batches, torch.device('cpu'), self.cfg)
        scalars = sum(event.count for event in profile.key_averages()
                      if event.key in {'aten::item', 'aten::_local_scalar_dense'})
        self.assertEqual(scalars, 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_matches_cpu_metrics(self):
        # Compare identical FP32 forward calculations on device, avoiding a
        # cross-device GEMM tolerance claim by using CUDA for both references.
        device = torch.device('cuda')
        model = self.model.to(device)
        batches = [{key: tensor.to(device) for key, tensor in batch.items()} for batch in self.batches]
        expected = old_epoch(model, batches, self.cfg, self.weight.to(device))
        actual = run_epoch(model, batches, device, self.cfg, class_weight=self.weight.to(device),
                           collect_predictions=True)
        self.assert_metrics_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
