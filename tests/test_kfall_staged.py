"""Staged training bypasses the SCN during temporal epochs and exports online weights."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluation.kfall_temporal import load_temporal_run
from train.temporal_staged import _temporal_stage
from train.utils import count_parameters
from train_kfall_temporal import build_temporal_model, get_args, main
from test_kfall_training import fixture_args, write_fixture


class StagedConfigTests(unittest.TestCase):
    def test_joint_default_and_staged_precision_and_trainable_count(self):
        self.assertEqual(get_args([]).training_strategy, 'joint')
        cfg = get_args(['--architecture', 'compact', '--training-strategy', 'staged', '--amp'])
        self.assertFalse(cfg.amp)
        self.assertTrue(cfg.pretrain_amp)
        model = build_temporal_model(cfg, torch.device('cpu')).set_backbone_trainable('frozen')
        self.assertEqual(count_parameters(model), (22156, 2468))

    def test_invalid_stage_options_are_rejected(self):
        for flags in (['--pretrain-epochs', '0'], ['--pretrain-batch-size', '0'],
                      ['--feature-batch-size', '0'], ['--finetune-epochs', '-1'],
                      ['--finetune-lr', 'nan'], ['--pretrain-lr', '0'], ['--finetune-epochs', '1']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    get_args(flags)


class StagedPipelineTests(unittest.TestCase):
    def test_features_only_stage_and_finetune_selection_preserve_online_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            flags = fixture_args(root) + ['--architecture', 'compact', '--conv-ch', '16', '32', '32',
                '--kernels', '16', '16', '8', '--training-strategy', 'staged',
                '--pretrain-epochs', '1', '--pretrain-batch-size', '16', '--feature-batch-size', '12',
                '--finetune-epochs', '1', '--finetune-lr', '.0001', '--sequence-length', '3',
                '--raw-win', '50', '--stride', '25', '--label-policy', 'temporal',
                '--post-impact-seconds', '1.2', '--class-weight', 'none', '--p-drop', '0',
                '--validation-only', '--no-progress']
            visited = []
            def observe(model, train, val, cfg, device, stats, **kwargs):
                name = kwargs['name']
                visited.append(name)
                if name == 'cached_temporal':
                    self.assertEqual(count_parameters(model)[1], 2468)
                    for dataset in (train, val):
                        self.assertTrue(hasattr(dataset, 'features'))
                        self.assertEqual(dataset.features.shape[1], 40)
                    with patch.object(model.backbone, 'extract_features',
                                      side_effect=AssertionError('Stage two must never execute the backbone')):
                        return _temporal_stage(model, train, val, cfg, device, stats, **kwargs)
                self.assertFalse(hasattr(train, 'features'))
                self.assertTrue(model.backbone.conv_block3.conv.weight.requires_grad)
                self.assertFalse(model.backbone.conv_block2.conv.weight.requires_grad)
                result = _temporal_stage(model, train, val, cfg, device, stats, **kwargs)
                result['best_val_loss'] = 1e6  # force rejecting the fine-tuned candidate
                return result
            with patch('train.temporal_staged._temporal_stage', side_effect=observe), contextlib.redirect_stdout(io.StringIO()):
                main(flags)
            self.assertEqual(visited, ['cached_temporal', 'last_block_finetune'])
            run = next((root / 'runs').iterdir())
            summary = json.loads((run / 'training_stages.json').read_text())
            cfg = json.loads((run / 'config.json').read_text())
            self.assertEqual(cfg['training_strategy'], 'staged')
            self.assertEqual(summary['feature_dim'], 40)
            self.assertEqual(summary['selected_stage'], 'cached_temporal')
            self.assertFalse(summary['test_used_for_selection'])
            self.assertEqual(set(summary['feature_extraction']['splits']), {'train', 'val'})
            self.assertFalse((run / 'features/test').exists())
            self.assertFalse((run / 'metrics/test_best').exists())
            self.assertTrue((run / 'stages/pretrain/metrics/loss.png').exists())
            self.assertTrue((run / 'stages/finetune/metrics/loss.png').exists())
            stage2 = torch.load(run / 'stages/temporal/checkpoints/best_model.pt', map_location='cpu', weights_only=True)
            selected = torch.load(run / 'checkpoints/best_model.pt', map_location='cpu', weights_only=True)
            self.assertEqual(selected['config']['epochs'], cfg['epochs'])
            self.assertEqual(selected['training_stage'], 'cached_temporal')
            self.assertEqual(selected['best_val_loss'], summary['stages'][1]['selection_val_loss'])
            for key, value in stage2['model_state_dict'].items():
                torch.testing.assert_close(selected['model_state_dict'][key], value, rtol=0, atol=0)
            model = load_temporal_run(run, 'cpu', split_name='val')[0]
            self.assertEqual(count_parameters(model)[0], 22156)
            self.assertEqual(model.memory.reset_mode, 'delayed')
            metrics = json.loads((run / 'metrics/validation/metrics.json').read_text())
            self.assertTrue(np.isfinite(metrics['loss']))

    def test_staged_without_finetune_is_replayable_without_feature_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            flags = fixture_args(root) + ['--training-strategy', 'staged', '--pretrain-epochs', '1',
                '--pretrain-batch-size', '16', '--feature-batch-size', '12', '--sequence-length', '3',
                '--raw-win', '50', '--stride', '25', '--label-policy', 'temporal',
                '--post-impact-seconds', '1.2', '--class-weight', 'none', '--p-drop', '0',
                '--validation-only', '--no-progress']
            with contextlib.redirect_stdout(io.StringIO()):
                main(flags)
            run = next((root / 'runs').iterdir())
            summary = json.loads((run / 'training_stages.json').read_text())
            self.assertEqual(len(summary['stages']), 2)
            self.assertEqual(summary['selected_stage'], 'cached_temporal')
            (run / 'features').rename(run / 'unused_features')
            from animate_kfall_temporal import prepare_temporal_replay
            replay = prepare_temporal_replay(run, device='cpu')
            self.assertTrue(len(replay.predicted_classes))
            self.assertTrue(np.isin(replay.predicted_classes, [0, 1, 2, 3]).all())
