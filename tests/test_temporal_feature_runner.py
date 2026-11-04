"""Cached features avoid the SCN and retain causal recurrent state semantics."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.temporal import TemporalSCN, detach_state
from train.temporal import run_temporal_epoch
from train.utils import make_grad_scaler


def small_model(**options):
    kwargs = dict(n_channels=6, conv_channels=(3, 4, 4), conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 1), pool_kernel_sizes=(2, 2, 2), pool_strides=(2, 2, 2),
        p_drop=.3, offset_hidden=3)
    return TemporalSCN(kwargs, hidden_size=4, boundary=True, **options)


class FeatureEpisodes:
    """Two recordings with unequal lengths; deliberately no event/offset keys."""
    def __init__(self, feature_size):
        self.features = torch.randn(2, 5, feature_size)
        self.targets = torch.tensor([[0, 1, 2, 3, 1], [2, 0, 3, -1, -1]])
        self.lengths = [5, 3]

    def groups(self, batch_size, **kwargs):
        yield [SimpleNamespace(indices=np.arange(length)) for length in self.lengths]

    def chunks(self, group, length):
        for start in range(0, 5, length):
            stop = min(5, start + length)
            present = torch.arange(start, stop)[None] < torch.tensor(self.lengths)[:, None]
            previous_target = (self.targets[:, start - 1] if start else torch.full((2,), -1))
            previous_valid = (torch.tensor(self.lengths) > start - 1) & (start > 0)
            indices = np.broadcast_to(np.arange(start, stop), (2, stop - start)).copy()
            indices[1] += 5
            indices[~present.numpy()] = -1
            yield dict(features=self.features[:, start:stop].clone(),
                targets=self.targets[:, start:stop], present=present, valid=present.clone(),
                previous_target=previous_target, previous_valid=previous_valid, indices=indices)


class TemporalFeatureTests(unittest.TestCase):
    def test_feature_path_matches_online_outputs_states_and_padding(self):
        for memory in ('none', 'lif'):
            for reset in ('immediate', 'delayed'):
                for hierarchical in (False, True):
                    with self.subTest(memory=memory, reset=reset, hierarchical=hierarchical):
                        torch.manual_seed(27)
                        model = small_model(memory=memory, memory_reset=reset,
                                            hierarchical=hierarchical).eval()
                        events = torch.randint(0, 2, (2, 4, 128, 6, 2)).float()
                        offsets = torch.randn(2, 4, 6)
                        present = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
                        with torch.no_grad():
                            features = model.backbone.extract_features(events.flatten(0, 1),
                                offsets.flatten(0, 1)).reshape(2, 4, -1)
                            expected, previous = model(events, offsets, valid=present)
                            # This branch must not re-enter the SCN for any window.
                            with patch.object(model.backbone, 'extract_features',
                                              side_effect=AssertionError('backbone called')):
                                actual, state = model.forward_features(features, valid=present)
                        for key in expected:
                            torch.testing.assert_close(actual[key], expected[key])
                        if previous is not None:
                            for before, after in zip(previous, state):
                                torch.testing.assert_close(after, before)

    def test_feature_recurrence_is_causal_and_chunk_continuous(self):
        torch.manual_seed(19)
        model = small_model(memory_reset='delayed').eval()
        features = torch.randn(2, 6, model.backbone.fused_dim)
        present = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        whole, whole_state = model.forward_features(features, valid=present)
        first, state = model.forward_features(features[:, :2], valid=present[:, :2])
        second, state = model.forward_features(features[:, 2:], detach_state(state), present[:, 2:])
        torch.testing.assert_close(torch.cat((first['logits'], second['logits']), 1), whole['logits'])
        for before, after in zip(whole_state, state):
            torch.testing.assert_close(after, before)
        future = features.clone()
        future[:, 4:] = -100
        altered, _ = model.forward_features(future, valid=present)
        torch.testing.assert_close(altered['logits'][:, :4], whole['logits'][:, :4], rtol=0, atol=0)

    def test_frozen_mode_survives_train_and_does_not_update_bn(self):
        model = small_model().set_backbone_trainable('frozen').train()
        self.assertTrue(model.training)
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.memory.training)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        snapshot = copy.deepcopy(model.backbone.state_dict())
        model(torch.randn(2, 3, 128, 6, 2), torch.randn(2, 3, 6))[0]['logits'].sum().backward()
        for name, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertIsNotNone(model.head.weight.grad)

    def test_last_mode_keeps_prefix_frozen_and_only_last_block_learns(self):
        model = small_model().set_backbone_trainable('last').train()
        for name in ('conv_block1', 'conv_block2', 'offset_proj_spk'):
            block = getattr(model.backbone, name)
            self.assertFalse(block.training)
            self.assertTrue(all(not p.requires_grad for p in block.parameters()))
        self.assertTrue(model.backbone.conv_block3.training)
        self.assertTrue(all(p.requires_grad for p in model.backbone.conv_block3.parameters()))
        snapshot = copy.deepcopy(model.backbone.state_dict())
        model(torch.randn(2, 3, 128, 6, 2), torch.randn(2, 3, 6))[0]['logits'].sum().backward()
        for name, value in model.backbone.state_dict().items():
            if not name.startswith('conv_block3.'):
                torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
        self.assertIsNotNone(model.backbone.conv_block3.conv.weight.grad)
        self.assertGreater(model.backbone.conv_block3.bn.num_batches_tracked.item(),
                           snapshot['conv_block3.bn.num_batches_tracked'].item())
        model.eval()
        self.assertFalse(model.backbone.conv_block3.training)
        model.train()
        self.assertTrue(model.backbone.conv_block3.training)
        model.set_backbone_trainable('all')
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(model.backbone.conv_block1.training)

    def test_epoch_trains_feature_sequences_without_event_access_or_backbone_calls(self):
        torch.manual_seed(11)
        model = small_model().set_backbone_trainable('frozen')
        episodes = FeatureEpisodes(model.backbone.fused_dim)
        cfg = SimpleNamespace(class_weight='none', batch_size=2, sequence_length=2, seed=0,
            amp=False, lambda_transition=0., lambda_boundary=0., label_smoothing=0.,
            grad_clip=1., no_progress=True)
        stats = dict(transition_costs=np.zeros((4, 4)), boundary_pos_weight=1.)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
        before = copy.deepcopy(model.backbone.state_dict())
        head = model.head.bias.detach().clone()
        with patch.object(model.backbone, 'extract_features', side_effect=AssertionError('backbone called')):
            result = run_temporal_epoch(model, episodes, cfg, torch.device('cpu'), stats,
                optimizer=optimizer, scaler=make_grad_scaler(False), collect=True)
        self.assertEqual(result['n'], 8)
        np.testing.assert_array_equal(result['predictions']['indices'], np.arange(8))
        self.assertTrue(np.isfinite(result['loss']))
        self.assertFalse(torch.equal(head, model.head.bias.detach()))
        for name, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertFalse(model.backbone.training)

    def test_invalid_feature_shape_and_backbone_mode_rejected(self):
        model = small_model()
        with self.assertRaisesRegex(ValueError, 'fused features'):
            model.forward_features(torch.zeros(2, 3, model.backbone.fused_dim + 1))
        with self.assertRaisesRegex(ValueError, 'at least one'):
            model.forward_features(torch.zeros(2, 0, model.backbone.fused_dim))
        with self.assertRaisesRegex(ValueError, 'Backbone training mode'):
            model.set_backbone_trainable('invalid')
