"""Synthetic phase timing excludes the SCN completely after feature caching."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from benchmark_kfall_temporal import (benchmark_phase, benchmark_training_phases,
                                     build_parser, main, _synchronize)
from train_kfall_temporal import build_temporal_model, get_args


class TemporalBenchmarkTests(unittest.TestCase):
    def config(self):
        return get_args(['--architecture', 'compact', '--out-win', '128', '--batch-size', '2',
                         '--sequence-length', '2', '--p-drop', '0'])

    def test_phase_counts_and_finite_reports(self):
        cfg = self.config()
        with patch('benchmark_kfall_temporal._synchronize') as synchronize:
            report = benchmark_training_phases(cfg, 'cpu', pretrain_batch_size=3, warmup=1, steps=1)
        self.assertEqual(synchronize.call_count, 6)
        self.assertEqual(set(report['phases']), {'joint', 'window_pretraining', 'cached_temporal'})
        self.assertEqual(report['feature_dim'], 40)
        self.assertFalse(report['includes_dataloader'])
        self.assertFalse(report['includes_feature_extraction'])
        self.assertFalse(report['is_total_staged_runtime'])
        self.assertEqual(report['device_name'], 'CPU')
        self.assertIn('python_version', report)
        for name, phase in report['phases'].items():
            self.assertGreater(phase['windows_per_second'], 0)
            self.assertGreater(phase['milliseconds_per_step'], 0)
            self.assertTrue(torch.isfinite(torch.tensor(phase['final_synthetic_loss'])))
            self.assertEqual(phase['timed_steps'], 1)
            self.assertNotIn('cuda_memory_mib', phase)
            self.assertFalse(phase['includes_dataloader'])
            self.assertEqual(phase['windows_per_step'], 3 if name == 'window_pretraining' else 4)
            self.assertEqual(phase['timed_windows'], phase['windows_per_step'])
        self.assertEqual(report['phases']['joint']['parameters'], 22156)
        self.assertEqual(report['phases']['joint']['trainable_parameters'], 22156)
        self.assertEqual(report['phases']['cached_temporal']['trainable_parameters'], 2468)
        self.assertEqual(report['phases']['window_pretraining']['parameters'], 19852)

    def test_cached_training_never_calls_backbone_and_changes_temporal_weights(self):
        cfg = self.config()
        model = build_temporal_model(cfg, torch.device('cpu'))
        events = torch.zeros(2, 2, 128, 6, 2)
        offsets = torch.randn(2, 2, 6)
        model.eval()
        with torch.no_grad():
            features = model.backbone.extract_features(events.flatten(0, 1), offsets.flatten(0, 1)).reshape(2, 2, -1)
        batch = dict(features=features, targets=torch.tensor([[0, 1], [2, 3]]))
        previous = model.head.weight.detach().clone()
        with patch.object(model.backbone, 'extract_features', side_effect=AssertionError('Backbone used in cached phase')):
            report = benchmark_phase(model, batch, cfg, 'cached_temporal', warmup=1, steps=2)
        self.assertEqual(report['timed_windows'], 8)
        self.assertFalse(torch.equal(previous, model.head.weight))
        self.assertFalse(model.backbone.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))

    def test_cli_writes_report_without_real_data(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / 'benchmark.json'
            main(['--architecture', 'compact', '--window-seconds', '.13', '--batch-size', '2',
                  '--sequence-length', '2', '--pretrain-batch-size', '3', '--device', 'cpu',
                  '--warmup', '1', '--steps', '1', '--output', str(output)])
            report = json.loads(output.read_text())
        self.assertTrue(report['synthetic_only'])
        self.assertFalse(report['includes_disk_io'])
        self.assertEqual(report['phases']['window_pretraining']['batch_windows'], 3)

    def test_validation_and_cpu_synchronization(self):
        cfg = self.config()
        with self.assertRaisesRegex(ValueError, 'positive'):
            benchmark_training_phases(cfg, 'cpu', warmup=0)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            benchmark_phase(None, {}, cfg, 'unexpected')
        with patch('torch.cuda.synchronize') as synchronize:
            _synchronize(torch.device('cpu'))
        synchronize.assert_not_called()
        for flags in (['--steps', '0'], ['--warmup', '-1'], ['--batch-size', '0'],
                      ['--pretrain-batch-size', '0'], ['--sequence-length', '0']):
            with self.assertRaises(SystemExit), patch('sys.stderr'):
                build_parser().parse_args(flags)
