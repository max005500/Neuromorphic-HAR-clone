"""Independent-window pretraining changes only SCN parameters and respects masks."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from loaders.kfall_sequences import EpisodeSet
from train.window_pretraining import pretrain_backbone, supervised_window_indices
from train_kfall_temporal import build_temporal_model, get_args


def episode_fixture(split):
    partition = {'train': 0, 'val': 1, 'test': 2}
    fi = partition[split]
    generator = np.random.default_rng(fi)
    arrays = dict(file_idx=np.full(8, fi), start=np.arange(8), end_frame=np.arange(16, 24),
        subject=np.array([f'SA0{fi + 1}'] * 8), y_h1=np.tile(np.arange(4), 2),
        y_h2=np.zeros(8, dtype=np.float32), src_h2=np.zeros(8, dtype=np.uint8),
        w_h2=np.zeros(8, dtype=np.float32),
        x_ev=generator.integers(0, 2, (8, 128, 6, 2), dtype=np.uint8),
        offset=generator.normal(size=(8, 6)).astype(np.float32))
    metadata = dict(config=dict(label_policy='temporal', post_impact_seconds=1.2,
                    stride=1, raw_win=16), split_file_indices={name: [index] for name, index in partition.items()},
                    recordings=[dict(subject=f'SA0{i + 1}', T=32, label_id=1) for i in range(3)])
    result = EpisodeSet(arrays, metadata)
    result.episodes[0].valid[-1] = False
    return result


class WindowPretrainingTests(unittest.TestCase):
    def config(self):
        return get_args(['--conv-ch', '4', '4', '4', '--kernels', '3', '3', '3',
                         '--p-drop', '0', '--recurrent-hidden', '4', '--no-progress'])

    def test_selected_windows_preserve_uniform_grid_and_mask(self):
        episodes = episode_fixture('train')
        np.testing.assert_array_equal(supervised_window_indices(episodes, 'train'), np.arange(7))
        with self.assertRaisesRegex(ValueError, 'another partition'):
            supervised_window_indices(episodes, 'val')
        with self.assertRaisesRegex(ValueError, 'only train and val'):
            supervised_window_indices(episode_fixture('test'), 'test')

    def test_pretraining_changes_only_backbone_and_restores_best_validation_weights(self):
        torch.manual_seed(8)
        cfg = self.config()
        model = build_temporal_model(cfg, torch.device('cpu'))
        original = copy.deepcopy(model.state_dict())
        train, val = episode_fixture('train'), episode_fixture('val')
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch('train.window_pretraining.run_epoch', wraps=__import__(
                    'train_ucihar', fromlist=['run_epoch']).run_epoch) as runner:
                report = pretrain_backbone(model, train, val, cfg, torch.device('cpu'),
                    epochs=2, batch_size=4, out_dir=Path(directory))
            self.assertFalse(report['test_consulted'])
            self.assertTrue(report['temporary_head_discarded'])
            self.assertEqual((report['train_windows'], report['val_windows']), (7, 7))
            expected_weights = 7 / (4 * np.bincount(train.arrays['y_h1'][:7], minlength=4))
            np.testing.assert_allclose(report['class_weights'], expected_weights)
            checkpoint = torch.load(report['checkpoint'], weights_only=True)
            self.assertEqual(checkpoint['config']['model_type'], 'window_scn_pretraining')
            self.assertEqual(checkpoint['config']['epochs'], 2)
            self.assertEqual(cfg.epochs, 100)
            self.assertEqual(len(runner.call_args_list), 4)
            self.assertTrue((Path(directory) / 'metrics/loss.png').is_file())
            np.testing.assert_array_equal(np.load(Path(directory) / 'train_window_indices.npy'), np.arange(7))
            for key, value in model.state_dict().items():
                if key.startswith('backbone.'):
                    torch.testing.assert_close(value, checkpoint['model_state_dict'][key], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(value, original[key]) for key, value in model.state_dict().items()
                                if key.startswith('backbone.') and value.is_floating_point()))
            self.assertFalse(model.backbone.training)
            self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))

    def test_balanced_weights_do_not_use_masked_or_validation_examples(self):
        train, val = episode_fixture('train'), episode_fixture('val')
        train.episodes[0].valid[train.episodes[0].targets == 3] = False
        cfg = self.config()
        model = build_temporal_model(cfg, torch.device('cpu'))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'empty classes'):
                pretrain_backbone(model, train, val, cfg, torch.device('cpu'),
                    epochs=1, batch_size=4, out_dir=directory)
            self.assertFalse((Path(directory) / 'checkpoints').exists())

    def test_invalid_schedule_and_cross_subject_data_rejected(self):
        cfg = self.config()
        model = build_temporal_model(cfg, torch.device('cpu'))
        train, val = episode_fixture('train'), episode_fixture('val')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'positive'):
                pretrain_backbone(model, train, val, cfg, torch.device('cpu'),
                    epochs=0, batch_size=4, out_dir=directory)
            val.metadata['recordings'][1]['subject'] = 'SA01'
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                pretrain_backbone(model, train, val, cfg, torch.device('cpu'),
                    epochs=1, batch_size=4, out_dir=directory)
