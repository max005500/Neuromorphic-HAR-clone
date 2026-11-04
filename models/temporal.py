"""Causal recurrent LIF memory across SCN windows; state is explicit per episode."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from models.models import DualHeadSCN
from models.utils import TriangularSurrogate
from models.hierarchical import hierarchy_predict


class RecurrentLIF(nn.Module):
    """Explicit membrane/spike state with immediate or previous-spike reset.

    ``immediate`` preserves the original post-spike reset: the stored membrane
    has already been reset. ``delayed`` stores the pre-reset membrane and uses
    the previous spike at the next decision. With soft reset this is exactly
    ``u_t = tau*u_prev + W*x_t + R*s_prev - threshold*s_prev``. Delayed hard
    reset instead clears the previous membrane before its leak/integration.
    """
    def __init__(self, input_size, hidden_size=32, tau=.9, threshold=.5, soft_reset=True,
                 reset_mode='immediate'):
        super().__init__()
        if hidden_size < 1 or not 0 <= tau <= 1 or threshold <= 0:
            raise ValueError('Invalid recurrent LIF size, leak or threshold.')
        if reset_mode not in {'immediate', 'delayed'}:
            raise ValueError('Recurrent reset mode must be immediate or delayed.')
        self.hidden_size, self.tau, self.threshold = hidden_size, float(tau), float(threshold)
        self.soft_reset, self.reset_mode = soft_reset, reset_mode
        self.input = nn.Linear(input_size, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.orthogonal_(self.recurrent.weight, gain=.5)

    def forward(self, features, state=None):
        if state is None:
            membrane = features.new_zeros(features.shape[0], self.hidden_size)
            previous_spikes = torch.zeros_like(membrane)
        else:
            membrane, previous_spikes = state
        if self.reset_mode == 'delayed':
            if self.soft_reset:
                membrane = (self.tau * membrane + self.input(features)
                            + self.recurrent(previous_spikes) - self.threshold * previous_spikes)
            else:
                membrane = (self.tau * membrane * (1 - previous_spikes)
                            + self.input(features) + self.recurrent(previous_spikes))
        else:
            membrane = self.tau * membrane + self.input(features) + self.recurrent(previous_spikes)
        spikes = TriangularSurrogate.apply(membrane - self.threshold, 1.)
        if self.reset_mode == 'immediate':
            membrane = membrane - spikes * self.threshold if self.soft_reset else membrane * (1 - spikes)
        return spikes, (membrane, spikes)


class TemporalSCN(nn.Module):
    def __init__(self, backbone_kwargs, *, memory='lif', hidden_size=32, memory_tau=.9,
                 memory_threshold=.5, memory_reset='immediate', hierarchical=False, boundary=False):
        super().__init__()
        if memory not in {'none', 'lif'}:
            raise ValueError('Memory must be none or lif.')
        if memory_reset not in {'immediate', 'delayed'}:
            raise ValueError('Recurrent reset mode must be immediate or delayed.')
        self.backbone = DualHeadSCN(**{**backbone_kwargs, 'backbone': True})
        self.memory_kind, self.hierarchical = memory, hierarchical
        self.memory = (RecurrentLIF(self.backbone.fused_dim, hidden_size, memory_tau,
                                   memory_threshold, backbone_kwargs.get('soft_reset', True), memory_reset)
                       if memory == 'lif' else None)
        size = hidden_size if self.memory is not None else self.backbone.fused_dim
        if hierarchical:
            self.group_head = nn.Linear(size, 2)
            self.normal_head = nn.Linear(size, 2)
            self.event_head = nn.Linear(size, 2)
        else:
            self.head = nn.Linear(size, 4)
        self.boundary_head = nn.Linear(size, 1) if boundary else None
        self._backbone_train_mode = 'all'

    def set_backbone_trainable(self, mode='all'):
        """Select full, frozen or last-block fine-tuning without changing weights.

        Frozen blocks stay in eval mode after ``model.train()`` so BatchNorm
        statistics and dropout cannot invalidate a saved feature cache.
        """
        if mode not in {'all', 'frozen', 'last'}:
            raise ValueError('Backbone training mode must be all, frozen or last.')
        self._backbone_train_mode = mode
        self.backbone.requires_grad_(mode == 'all')
        if mode == 'last':
            self.backbone.conv_block3.requires_grad_(True)
        self._apply_backbone_train_mode()
        return self

    def _apply_backbone_train_mode(self):
        mode = getattr(self, '_backbone_train_mode', 'all')
        self.backbone.train(self.training if mode == 'all' else False)
        if mode == 'last' and self.training:
            self.backbone.conv_block3.train(True)

    def train(self, mode=True):
        super().train(mode)
        self._apply_backbone_train_mode()
        return self

    @staticmethod
    def _validity_mask(valid, batch, steps):
        if steps < 1:
            raise ValueError('A sequence must contain at least one window.')
        # Keep lengths on CPU; gathering valid rows avoids padding affecting BN.
        valid = torch.ones((batch, steps), dtype=torch.bool) if valid is None else valid.detach().cpu().bool()
        if valid.shape != (batch, steps):
            raise ValueError('Sequence validity mask must have shape [B,K].')
        return valid

    def _advance_memory(self, features, state, present, fully_present):
        if self.memory is None:
            return features, state, None
        _, candidate = self.memory(features, state)
        if fully_present:
            state = candidate
            activity = state[1].sum(1)
        else:
            previous = (tuple(torch.zeros_like(value) for value in candidate)
                        if state is None else state)
            mask = present.to(features.device).unsqueeze(1)
            state = tuple(torch.where(mask, new, old) for new, old in zip(candidate, previous))
            activity = (state[1] * mask).sum(1)
        return state[1], state, activity

    def forward(self, events, offsets, state=None, valid=None):
        if events.ndim != 5 or offsets.shape != (*events.shape[:2], 6):
            raise ValueError('Expected events [B,K,T,6,2] and offsets [B,K,6].')
        batch, steps = events.shape[:2]
        valid = self._validity_mask(valid, batch, steps)
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
            representation, state, spikes = self._advance_memory(features, state, valid[:, time], fully_present)
            if spikes is not None:
                activity.append(spikes)
            history.append(representation)
        return self._readout(history, activity), state

    def forward_features(self, features, state=None, valid=None):
        """Run temporal memory/readouts on cached fused window features only.

        No convolution, offset projection or backbone BatchNorm is executed.
        The caller keeps the same recording order and carries/detaches state
        at the same boundaries as in the end-to-end event path.
        """
        if features.ndim != 3 or features.shape[-1] != self.backbone.fused_dim:
            raise ValueError(f'Expected fused features [B,K,{self.backbone.fused_dim}].')
        batch, steps = features.shape[:2]
        valid = self._validity_mask(valid, batch, steps)
        history, activity = [], []
        for time in range(steps):
            fully_present = bool(valid[:, time].all())
            current = features[:, time]
            if not fully_present:
                current = current.masked_fill(~valid[:, time].to(features.device).unsqueeze(1), 0)
            representation, state, spikes = self._advance_memory(current, state, valid[:, time], fully_present)
            if spikes is not None:
                activity.append(spikes)
            history.append(representation)
        return self._readout(history, activity), state

    def _readout(self, history, activity):
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
        return outputs

    def predict(self, outputs):
        if not self.hierarchical:
            return outputs['logits'].argmax(-1)
        shape = outputs['logits'].shape[:2]
        return hierarchy_predict({key: outputs[key].reshape(-1, 2) for key in ('group', 'normal', 'event')}).reshape(shape)


def detach_state(state):
    return None if state is None else tuple(value.detach() for value in state)
