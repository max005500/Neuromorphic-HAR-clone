"""Compact window encoder with explicit, causal recurrent spiking state."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

from models.models import DualHeadSCN
from models.utils import TriangularSurrogate


class LIFState(NamedTuple):
    membrane: torch.Tensor
    spikes: torch.Tensor

    def detach(self) -> LIFState:
        return LIFState(self.membrane.detach(), self.spikes.detach())


class ALIFState(NamedTuple):
    membrane: torch.Tensor
    spikes: torch.Tensor
    adaptation: torch.Tensor

    def detach(self) -> ALIFState:
        return ALIFState(*(value.detach() for value in self))


class AdaptiveRecurrentLIF(nn.Module):
    """LSNN-inspired recurrence with learned bounded decays and threshold adaptation."""

    def __init__(self, input_size: int, hidden_size: int = 32, beta: float = 0.9,
                 threshold: float = 0.5, adaptation_decay: float = 0.95,
                 adaptation_strength: float = 0.5):
        super().__init__()
        if not 0 < beta < 1 or not 0 < adaptation_decay < 1:
            raise ValueError("Adaptive decays must be strictly between zero and one.")
        if not math.isfinite(adaptation_strength) or adaptation_strength <= 0:
            raise ValueError("Adaptation strength must be finite and positive.")
        base = RecurrentLIF(input_size, hidden_size, beta, threshold)
        self.input, self.recurrent = base.input, base.recurrent
        self.hidden_size, self.threshold = hidden_size, float(threshold)
        self.membrane_logit = nn.Parameter(torch.full((hidden_size,), math.log(beta / (1 - beta))))
        self.adaptation_logit = nn.Parameter(torch.full(
            (hidden_size,), math.log(adaptation_decay / (1 - adaptation_decay))))
        self.strength_raw = nn.Parameter(torch.full(
            (hidden_size,), math.log(math.expm1(adaptation_strength))))

    def forward(self, features, state: ALIFState | None = None, mask=None):
        if features.ndim != 3 or features.shape[1] < 1:
            raise ValueError("Adaptive features must have shape (B, decisions, features).")
        if mask is not None and (mask.shape != features.shape[:2] or mask.dtype != torch.bool):
            raise ValueError("Adaptive mask must be boolean (B, decisions).")
        if state is None:
            zeros = features.new_zeros(features.shape[0], self.hidden_size)
            state = ALIFState(zeros, torch.zeros_like(zeros), torch.zeros_like(zeros))
        if not isinstance(state, ALIFState):
            raise ValueError("Adaptive recurrence requires ALIFState.")
        if any(value.shape != (features.shape[0], self.hidden_size) or value.device != features.device
               for value in state):
            raise ValueError("Adaptive state must match batch, hidden size and device.")
        beta, rho = self.membrane_logit.sigmoid(), self.adaptation_logit.sigmoid()
        strength = torch.nn.functional.softplus(self.strength_raw)
        outputs = []
        for t, current in enumerate(features.unbind(1)):
            previous_threshold = self.threshold + strength * state.adaptation
            adaptation = rho * state.adaptation + state.spikes
            threshold = self.threshold + strength * adaptation
            membrane = (beta * state.membrane + self.input(current)
                        + self.recurrent(state.spikes) - previous_threshold * state.spikes)
            spikes = TriangularSurrogate.apply(membrane - threshold, 1.0)
            updated = ALIFState(membrane, spikes, adaptation)
            if mask is None:
                state = updated
                outputs.append(spikes)
            else:
                valid = mask[:, t:t + 1]
                state = ALIFState(*(torch.where(valid, new, old) for new, old in zip(updated, state)))
                outputs.append(torch.where(valid, spikes, torch.zeros_like(spikes)))
        return torch.stack(outputs, dim=1), state


class RecurrentLIF(nn.Module):
    """u[t] = beta*u[t-1] + W*x[t] + R*s[t-1] - threshold*s[t-1]."""

    def __init__(self, input_size: int, hidden_size: int = 32,
                 beta: float = 0.9, threshold: float = 0.5):
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("Recurrent input/hidden sizes must be positive.")
        if not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("Recurrent beta must be finite in [0, 1).")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("Recurrent threshold must be finite and positive.")
        self.hidden_size = hidden_size
        self.beta, self.threshold = float(beta), float(threshold)
        self.input = nn.Linear(input_size, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.orthogonal_(self.recurrent.weight, gain=0.5)

    def forward(self, features: torch.Tensor, state: LIFState | None = None,
                mask: torch.Tensor | None = None):
        if features.ndim != 3 or features.shape[1] < 1:
            raise ValueError("Recurrent features must have shape (B, decisions, features).")
        batch = features.shape[0]
        if mask is not None and (mask.shape != features.shape[:2] or mask.dtype != torch.bool):
            raise ValueError("Recurrent mask must be boolean with shape (B, decisions).")
        if state is None:
            zeros = features.new_zeros(batch, self.hidden_size)
            state = LIFState(zeros, torch.zeros_like(zeros))
        for value in state:
            if value.shape != (batch, self.hidden_size) or value.device != features.device:
                raise ValueError("Recurrent state must match batch, hidden size and device.")
        outputs = []
        for t, current in enumerate(features.unbind(dim=1)):
            membrane = (self.beta * state.membrane + self.input(current)
                        + self.recurrent(state.spikes) - self.threshold * state.spikes)
            spikes = TriangularSurrogate.apply(membrane - self.threshold, 1.0)
            if mask is None:
                state = LIFState(membrane, spikes)
                outputs.append(spikes)
            else:
                valid = mask[:, t:t + 1]
                state = LIFState(torch.where(valid, membrane, state.membrane),
                                 torch.where(valid, spikes, state.spikes))
                outputs.append(torch.where(valid, spikes, torch.zeros_like(spikes)))
        return torch.stack(outputs, dim=1), state


class CompactRecurrentSCN(nn.Module):
    """Compact SCN -> offset fusion -> recurrent LIF -> lightweight readout.

    Only the recurrent state persists between decisions. The convolutional
    window encoder and Sigma-Delta encoding are not incremental/streaming.
    Pass state explicitly; None resets an independent recording.
    """

    def __init__(self, n_classes: int = 4, hidden_size: int = 32,
                 recurrent_beta: float = 0.9, adaptive: bool = False,
                 adaptation_decay: float = 0.95, adaptation_strength: float = 0.5,
                 **encoder_kwargs):
        super().__init__()
        encoder_kwargs = dict(encoder_kwargs)
        encoder_kwargs.setdefault("conv_channels", (16, 32, 32))
        encoder_kwargs.setdefault("conv_kernel_sizes", (16, 16, 8))
        encoder_kwargs["backbone"] = True
        encoder_kwargs["spiking_heads"] = False
        self.encoder = DualHeadSCN(**encoder_kwargs)
        if adaptive:
            self.temporal = AdaptiveRecurrentLIF(
                self.encoder.fused_dim, hidden_size, recurrent_beta,
                encoder_kwargs.get("thresh", 0.5), adaptation_decay, adaptation_strength)
        else:
            self.temporal = RecurrentLIF(self.encoder.fused_dim, hidden_size,
                                         recurrent_beta, encoder_kwargs.get("thresh", 0.5))
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, events: torch.Tensor, offset: torch.Tensor | None = None,
                state: LIFState | ALIFState | None = None, mask: torch.Tensor | None = None):
        if events.ndim != 5 or events.shape[-2:] != (6, 2) or events.shape[1] < 1:
            raise ValueError("Recurrent SCN expects (B, decisions, event_steps, 6, 2).")
        batch, steps = events.shape[:2]
        if offset is not None and offset.shape != (batch, steps, 6):
            raise ValueError("Recurrent offsets must have shape (B, decisions, 6).")
        flattened_offset = None if offset is None else offset.flatten(0, 1)
        if mask is None:
            _, features = self.encoder(events.flatten(0, 1), offset=flattened_offset)
            features = features.reshape(batch, steps, -1)
        else:
            if mask.shape != (batch, steps) or mask.dtype != torch.bool or not mask.any():
                raise ValueError("SCN mask must be boolean (B, decisions) with valid windows.")
            valid = mask.flatten()
            selected_offsets = None if flattened_offset is None else flattened_offset[valid]
            _, encoded = self.encoder(events.flatten(0, 1)[valid], offset=selected_offsets)
            features = encoded.new_zeros(batch * steps, encoded.shape[-1])
            features[valid] = encoded
            features = features.reshape(batch, steps, -1)
        spikes, state = self.temporal(features, state, mask)
        return self.head(spikes), state, spikes

    def fuse_batch_norm(self):
        """Return an independent eval-only encoder with Conv/Linear BN folded."""
        import copy
        from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

        if self.training:
            raise ValueError("Call eval() before folding BatchNorm.")
        model = copy.deepcopy(self)
        for block in (model.encoder.conv_block1, model.encoder.conv_block2, model.encoder.conv_block3):
            if isinstance(block.bn, nn.BatchNorm1d):
                block.conv = fuse_conv_bn_eval(block.conv, block.bn)
                block.bn = nn.Identity()
        if model.encoder.use_offset:
            block = model.encoder.offset_proj_spk
            if isinstance(block.bn, nn.BatchNorm1d):
                block.fc = fuse_linear_bn_eval(block.fc, block.bn)
                block.bn = nn.Identity()
        return model
