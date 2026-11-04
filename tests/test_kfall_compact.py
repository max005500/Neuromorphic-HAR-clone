"""Compact preset, old checkpoint defaults and isolated inference reports."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from compare_kfall_validation import compare_validation
from evaluation.kfall_temporal import evaluate_temporal, load_temporal_run
from train.utils import count_parameters
from train_kfall_temporal import build_temporal_model, get_args, main
from test_kfall_training import write_fixture


class CompactModelTests(unittest.TestCase):
    def test_presets_and_explicit_overrides_have_resolved_sizes(self):
        standard = get_args([])
        compact = get_args(['--architecture', 'compact'])
        self.assertEqual(standard.memory_reset, 'immediate')
        self.assertEqual(compact.memory_reset, 'delayed')
        for cfg, count, channels, kernels in (
                (standard, 114476, [32, 64, 64], [32, 32, 8]),
                (compact, 22156, [16, 32, 32], [16, 16, 8])):
            self.assertEqual(cfg.conv_ch, channels)
            self.assertEqual(cfg.kernels, kernels)
            model = build_temporal_model(cfg, torch.device('cpu'))
            self.assertEqual(count_parameters(model)[0], count)
            self.assertEqual(model.memory.hidden_size, 32)
            self.assertEqual(model.memory.reset_mode, cfg.memory_reset)
        override = get_args(['--conv-ch', '8', '16', '16', '--architecture', 'compact',
                             '--kernels', '32', '32', '8', '--memory-reset', 'immediate',
                             '--recurrent-hidden', '16', '--offset-hidden', '4'])
        self.assertEqual(override.conv_ch, [8, 16, 16])
        self.assertEqual(override.kernels, [32, 32, 8])
        self.assertEqual((override.memory_reset, override.recurrent_hidden, override.offset_hidden),
                         ('immediate', 16, 4))

    def test_older_config_keeps_immediate_reset_and_weights(self):
        cfg = get_args(['--conv-ch', '4', '4', '4', '--kernels', '3', '3', '3'])
        reference = build_temporal_model(cfg, torch.device('cpu')).eval()
        del cfg.memory_reset
        del cfg.architecture
        restored = build_temporal_model(cfg, torch.device('cpu')).eval()
        restored.load_state_dict(reference.state_dict(), strict=True)
        self.assertEqual(restored.memory.reset_mode, 'immediate')
        events, offsets = torch.zeros(2, 2, 128, 6, 2), torch.zeros(2, 2, 6)
        with torch.no_grad():
            expected, before = reference(events, offsets)
            actual, after = restored(events, offsets)
        torch.testing.assert_close(expected['logits'], actual['logits'], rtol=0, atol=0)
        for left, right in zip(before, after):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_architecture_comparison_requires_explicit_flag_preserves_training_checks(self):
        base = dict(run_dir='base', config=dict(conv_ch=[32, 64, 64], kernels=[32, 32, 8],
                    offset_hidden=8, lr=.001), metadata=dict(source_signature='a',
                    split_subjects={'train': ['SA01']}, version=5, channels=['acc'],
                    units=['g'], config={'theta_sd': [.1]}), cohort=np.array([[0, 100, 0]]),
                    cohort_sha256='same', accuracy=.9, macro_f1=.8, fall_recall=.85)
        candidate = copy.deepcopy(base)
        candidate['config'].update(conv_ch=[16, 32, 32], kernels=[16, 16, 8])
        with patch('compare_kfall_validation.read_validation', side_effect=[base, candidate]):
            with self.assertRaisesRegex(ValueError, 'conv_ch'):
                compare_validation('base', 'candidate')
        with patch('compare_kfall_validation.read_validation', side_effect=[base, candidate]):
            result = compare_validation('base', 'candidate', allow_architecture_changes=True)
        self.assertTrue(result['non_degrading'])
        self.assertEqual(set(result['architecture_changes']), {'conv_ch', 'kernels'})
        candidate['config']['lr'] = .01
        with patch('compare_kfall_validation.read_validation', side_effect=[base, candidate]):
            with self.assertRaisesRegex(ValueError, 'lr'):
                compare_validation('base', 'candidate', allow_architecture_changes=True)


class CompactPipelineTests(unittest.TestCase):
    def test_compact_training_fp32_fused_and_weights8_preserve_original_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            flags = ['--dataset-root', str(root), '--cache-dir', str(root / 'cache'),
                '--out-dir', str(root / 'runs'), '--split', '1', '1', '1',
                '--architecture', 'compact', '--epochs', '1', '--batch-size', '3', '--workers', '0',
                '--raw-win', '50', '--stride', '25', '--out-win', '128', '--sequence-length', '3',
                '--class-weight', 'none', '--p-drop', '0', '--label-policy', 'temporal',
                '--post-impact-seconds', '1.2', '--validation-only', '--no-progress', '--quiet-loader']
            with contextlib.redirect_stdout(io.StringIO()):
                main(flags)
            run = next((root / 'runs').iterdir())
            self.assertFalse((run / 'metrics/test_best').exists())
            cfg = json.loads((run / 'config.json').read_text())
            self.assertEqual(cfg['memory_reset'], 'delayed')
            loaded = load_temporal_run(run, 'cpu', split_name='val')[0]
            self.assertEqual(loaded.memory.reset_mode, 'delayed')
            self.assertEqual(count_parameters(loaded)[0], 22156)
            watched = [run / 'checkpoints/best_model.pt', run / 'metrics/validation/metrics.json',
                       run / 'metrics/y_pred_val.npy', run / 'config.json']
            original = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in watched}
            opened, load = [], np.load
            def track(path, *args, **kwargs):
                opened.append(str(path))
                return load(path, *args, **kwargs)
            with patch('numpy.load', side_effect=track):
                fused = evaluate_temporal(run, split_name='val', device='cpu', fuse_bn=True)
                quantized = evaluate_temporal(run, split_name='val', device='cpu', fuse_bn=True, weight_bits=8)
            self.assertFalse(any('/test/' in name for name in opened))
            self.assertEqual(len(fused['inference']['bn_fused_layers']), 4)
            self.assertEqual(fused['original_parameters'], 22156)
            self.assertEqual(fused['inference']['state_precision'], 'float32')
            self.assertEqual(quantized['inference']['weight_bits'], 8)
            self.assertFalse(quantized['inference']['native_int8'])
            self.assertGreater(quantized['inference']['quantized_weight_count'], 0)
            self.assertTrue(np.isfinite(quantized['loss']))
            self.assertNotEqual(fused['output_dir'], quantized['output_dir'])
            self.assertTrue((run / 'metrics/validation_fused_int8_weights/metrics.json').exists())
            for path, digest in original.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            with patch('evaluation.kfall_temporal.run_temporal_epoch') as infer:
                with self.assertRaisesRegex(ValueError, 'separate output directory'):
                    evaluate_temporal(run, split_name='val', device='cpu', fuse_bn=True,
                                      output_dir=run / 'metrics/validation')
                self.assertFalse(infer.called)
