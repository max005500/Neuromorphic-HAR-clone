"""Benchmark timing boundaries, CLI validation and opt-in trace export."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

from benchmark_kfall import benchmark_model, build_parser, resolve_device, _synchronize


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(18, 4)

    def forward(self, events, offset):
        features = torch.cat((events.mean(1).flatten(1), offset), dim=1)
        return self.linear(features)


class KFallBenchmarkTests(unittest.TestCase):
    def config(self):
        return SimpleNamespace(amp=False, lr=.001, wd=.0001, label_smoothing=0., grad_clip=1.)

    def batch(self):
        return {'x': torch.rand(3, 8, 6, 2), 'offset': torch.randn(3, 6), 'y': torch.tensor([0, 1, 3])}

    def test_actual_updates_and_summary(self):
        model = SmallModel()
        before = model.linear.weight.detach().clone()
        with patch('benchmark_kfall._synchronize') as sync:
            result = benchmark_model(model, self.batch(), self.config(), warmup=1, steps=2)
        self.assertEqual(sync.call_count, 2)
        self.assertFalse(torch.equal(before, model.linear.weight))
        self.assertGreater(result['samples_per_second'], 0)
        self.assertGreater(result['milliseconds_per_batch'], 0)
        self.assertEqual(result['timed_steps'], 2)
        self.assertFalse(result['includes_dataloader'])
        self.assertNotIn('profile', result)
        self.assertNotIn('cuda_memory_mib', result)

    def test_optional_profiler_creates_trace_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder) / 'profile' / 'trace.json'
            result = benchmark_model(SmallModel(), self.batch(), self.config(), warmup=1, steps=1,
                                     profile_path=trace)
            self.assertTrue(trace.is_file())
            self.assertEqual(result['timed_steps'], 1)
            self.assertEqual(result['profile']['steps'], 3)

    def test_cpu_never_synchronizes_cuda(self):
        with patch('torch.cuda.synchronize') as sync:
            _synchronize(torch.device('cpu'))
        sync.assert_not_called()

    def test_invalid_configuration(self):
        for kwargs in ({'warmup': 0}, {'steps': 0}):
            with self.assertRaises(ValueError):
                benchmark_model(SmallModel(), self.batch(), self.config(), **kwargs)
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(resolve_device('auto'), torch.device('cpu'))
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                resolve_device('cuda:0')
        with self.assertRaises(ValueError):
            resolve_device('mps')
        for flags in (['--steps', '0'], ['--warmup', '-1'], ['--batch-size', '0'],
                      ['--window-seconds', 'nan']):
            with self.assertRaises(SystemExit), patch('sys.stderr'):
                build_parser().parse_args(flags)
