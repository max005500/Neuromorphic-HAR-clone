"""Masked chunks must advance memory without Adam momentum/weight-decay updates."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from loaders.kfall_sequences import EpisodeSet
from train.temporal import run_temporal_epoch
from train.utils import make_grad_scaler


class MemoryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 4)
        self.seen_states = []
    def forward(self, events, offsets, state=None, valid=None):
        self.seen_states.append(None if state is None else state[0].clone())
        current = torch.ones(len(events), 1) if state is None else state[0] + 1
        return {'logits': self.head(events)}, (current, current)
    def predict(self, outputs):
        return outputs['logits'].argmax(-1)


class MaskedEpisodes:
    def groups(self, batch_size, **kwargs):
        yield [SimpleNamespace(indices=np.arange(2))]
    def chunks(self, group, length):
        for _ in range(2):
            yield dict(events=torch.ones(1, 1, 1), offsets=torch.zeros(1, 1, 6),
                targets=torch.ones(1, 1, dtype=torch.long), valid=torch.zeros(1, 1, dtype=torch.bool),
                present=torch.ones(1, 1, dtype=torch.bool), previous_target=torch.ones(1, dtype=torch.long),
                previous_valid=torch.zeros(1, dtype=torch.bool))


class MaskedUpdateTests(unittest.TestCase):
    def test_adam_weights_and_step_counters_unchanged_but_state_advances(self):
        model = MemoryModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.1)
        optimizer.zero_grad()
        model.head(torch.ones(2, 1)).square().sum().backward()
        optimizer.step()  # populate nonzero momentum
        weights = copy.deepcopy(model.state_dict())
        states = copy.deepcopy(optimizer.state_dict())
        cfg = SimpleNamespace(class_weight='none', batch_size=1, sequence_length=1, seed=0,
            amp=False, lambda_transition=0., lambda_boundary=0., label_smoothing=0., grad_clip=1., no_progress=True)
        result = run_temporal_epoch(model, MaskedEpisodes(), cfg, torch.device('cpu'),
            dict(transition_costs=np.zeros((4, 4)), boundary_pos_weight=1.),
            optimizer=optimizer, scaler=make_grad_scaler(False))
        self.assertEqual(result['n'], 0)
        for key, value in weights.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        for param, values in states['state'].items():
            for key, value in values.items():
                torch.testing.assert_close(value, optimizer.state_dict()['state'][param][key], rtol=0, atol=0)
        self.assertIsNone(model.seen_states[0])
        torch.testing.assert_close(model.seen_states[1], torch.ones(1, 1))

    def test_all_masked_dataset_rejected_before_training(self):
        arrays = dict(file_idx=np.array([0, 0]), start=np.array([0, 1]), end_frame=np.array([2, 3]),
                      subject=np.array(['SA01', 'SA01']), y_h1=np.array([1, 2]))
        meta = dict(config=dict(label_policy='temporal', post_impact_seconds=1.2, raw_win=2, stride=1),
                    recordings=[dict(subject='SA01', frame_label_intervals=[[1, 3, 1], [3, 5, 2]])])
        with self.assertRaisesRegex(ValueError, 'No supervised windows'):
            EpisodeSet(arrays, meta, boundary_ignore_seconds=1.)
