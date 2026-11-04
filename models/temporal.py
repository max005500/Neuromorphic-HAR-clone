"""Causal recurrent LIF memory across SCN windows; state is explicit per episode."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from models.models import DualHeadSCN
from models.utils import TriangularSurrogate
from models.hierarchical import hierarchy_predict


class RecurrentLIF(nn.Module):
    def __init__(self, input_size, hidden_size=32, tau=.9, threshold=.5, soft_reset=True):
        super().__init__()
        if hidden_size < 1 or not 0 <= tau <= 1 or threshold <= 0:
            raise ValueError('Invalid recurrent LIF size, leak or threshold.')
        self.hidden_size, self.tau, self.threshold = hidden_size, float(tau), float(threshold)
        self.soft_reset = soft_reset
        self.input = nn.Linear(input_size, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.orthogonal_(self.recurrent.weight, gain=.5)

    def forward(self, features, state=None):
        if state is None:
            membrane = features.new_zeros(features.shape[0], self.hidden_size)
            previous_spikes = torch.zeros_like(membrane)
        else:
            membrane, previous_spikes = state
        membrane = self.tau * membrane + self.input(features) + self.recurrent(previous_spikes)
        spikes = TriangularSurrogate.apply(membrane - self.threshold, 1.)
        membrane = membrane - spikes * self.threshold if self.soft_reset else membrane * (1 - spikes)
        return spikes, (membrane, spikes)


class TemporalSCN(nn.Module):
    def __init__(self, backbone_kwargs, *, memory='lif', hidden_size=32, memory_tau=.9,
                 memory_threshold=.5, hierarchical=False, boundary=False):
        super().__init__()
        if memory not in {'none', 'lif'}:
            raise ValueError('Memory must be none or lif.')
        self.backbone = DualHeadSCN(**{**backbone_kwargs, 'backbone': True})
        self.memory_kind, self.hierarchical = memory, hierarchical
        self.memory = (RecurrentLIF(self.backbone.fused_dim, hidden_size, memory_tau,
                                   memory_threshold, backbone_kwargs.get('soft_reset', True))
                       if memory == 'lif' else None)
        size = hidden_size if self.memory is not None else self.backbone.fused_dim
        if hierarchical:
            self.group_head = nn.Linear(size, 2)
            self.normal_head = nn.Linear(size, 2)
            self.event_head = nn.Linear(size, 2)
        else:
            self.head = nn.Linear(size, 4)
        self.boundary_head = nn.Linear(size, 1) if boundary else None

    def forward(self, events, offsets, state=None, valid=None):
        if events.ndim != 5 or offsets.shape != (*events.shape[:2], 6):
            raise ValueError('Expected events [B,K,T,6,2] and offsets [B,K,6].')
        batch, steps = events.shape[:2]
        if steps < 1:
            raise ValueError('A sequence must contain at least one window.')
        # Keep lengths on CPU; gathering valid rows avoids padding affecting BN.
        valid = torch.ones((batch, steps), dtype=torch.bool) if valid is None else valid.detach().cpu().bool()
        if valid.shape != (batch, steps):
            raise ValueError('Sequence validity mask must have shape [B,K].')
        history, activity = [], []
        for time in range(steps):
            fully_present = bool(valid[:, time].all())  # CPU-only length mask
            if fully_present:
                # Most chunks have no padding: avoid a tiny H2D index transfer,
                # gather and scatter for every temporal position.
                features = self.backbone.extract_features(events[:, time].float(), offsets[:, time])
            else:
                indices = torch.nonzero(valid[:, time], as_tuple=False).flatten().to(events.device)
                features = events.new_zeros((batch, self.backbone.fused_dim), dtype=torch.float32)
                if len(indices):
                    current = self.backbone.extract_features(events[indices, time].float(), offsets[indices, time])
                    features = features.to(current.dtype).index_copy(0, indices, current)
            if self.memory is not None:
                spikes, candidate = self.memory(features, state)
                if fully_present:
                    state = candidate
                    activity.append(state[1].sum(1))
                else:
                    previous = (tuple(torch.zeros_like(value) for value in candidate)
                                if state is None else state)
                    mask = valid[:, time].to(events.device).unsqueeze(1)
                    state = tuple(torch.where(mask, new, old) for new, old in zip(candidate, previous))
                    activity.append((state[1] * mask).sum(1))
                representation = state[1]
            else:
                representation = features
            history.append(representation)
        representation = torch.stack(history, dim=1)
        outputs = {}
        if self.hierarchical:
            for key in ('group', 'normal', 'event'):
                outputs[key] = getattr(self, key + '_head')(representation)
            group, normal, event = (F.log_softmax(outputs[key].float(), dim=-1) for key in ('group', 'normal', 'event'))
            outputs['logits'] = torch.stack((group[..., 1] + event[..., 0], group[..., 0] + normal[..., 0],
                                            group[..., 0] + normal[..., 1], group[..., 1] + event[..., 1]), -1)
        else:
            outputs['logits'] = self.head(representation)
        if self.boundary_head is not None:
            outputs['boundary'] = self.boundary_head(representation).squeeze(-1)
        if activity:
            outputs['memory_spikes'] = torch.stack(activity, dim=1)
        return outputs, state

    def predict(self, outputs):
        if not self.hierarchical:
            return outputs['logits'].argmax(-1)
        shape = outputs['logits'].shape[:2]
        return hierarchy_predict({key: outputs[key].reshape(-1, 2) for key in ('group', 'normal', 'event')}).reshape(shape)


def detach_state(state):
    return None if state is None else tuple(value.detach() for value in state)
