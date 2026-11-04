"""Experimental KFall hierarchy: Normal/Event -> Dynamic/Static or Fall/Postimpact."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from models.models import DualHeadSCN
from models.utils import SpikingLinearBlock


class HierarchicalSCN(DualHeadSCN):
    """Reuse the SCN backbone/offset branch with three supervised binary heads."""
    def _build_heads(self, lif_kwargs):
        if self.backbone:
            raise ValueError('HierarchicalSCN requires its classification heads.')
        self.n_classes = self.n_classes_head1 = 4
        for name in ('group', 'normal', 'event'):
            if self.spiking_heads:
                layer = SpikingLinearBlock(self.fused_dim, 2, bias=True, use_bn=self.use_bn, **lif_kwargs)
            else:
                layer = nn.Linear(self.fused_dim, 2)
            setattr(self, name + '_head', layer)

    def forward(self, x, offset=None):
        spikes, representation = self._fused_representation(x, offset)
        outputs = {}
        for name in ('group', 'normal', 'event'):
            head = getattr(self, name + '_head')
            outputs[name] = self.head_rate_scale * head(spikes).mean(2) if self.spiking_heads else head(representation)
        # Preserve original KFall class order: Fall, Dynamic, Static, Postimpact.
        group, normal, event = (F.log_softmax(outputs[key].float(), dim=1) for key in ('group', 'normal', 'event'))
        outputs['head1'] = torch.stack((group[:, 1] + event[:, 0], group[:, 0] + normal[:, 0],
                                       group[:, 0] + normal[:, 1], group[:, 1] + event[:, 1]), dim=1)
        return outputs, representation


def hierarchy_targets(target):
    """Normal=0/Event=1; Dynamic=0/Static=1; Fall=0/Postimpact=1."""
    event = (target == 0) | (target == 3)
    return event.long(), (target == 2).long(), (target == 3).long()


def hierarchy_predict(outputs):
    """Hard route through the root and then the selected child classifier."""
    group = outputs['group'].argmax(1)
    normal = outputs['normal'].argmax(1) + 1
    event = outputs['event'].argmax(1) * 3
    return torch.where(group == 0, normal, event)


def hierarchy_loss(outputs, target, *, class_weight=None, label_smoothing=0.):
    group, normal, event = hierarchy_targets(target)
    root_loss = F.cross_entropy(outputs['group'], group, reduction='none', label_smoothing=label_smoothing)
    normal_loss = F.cross_entropy(outputs['normal'], normal, reduction='none', label_smoothing=label_smoothing) * (group == 0)
    event_loss = F.cross_entropy(outputs['event'], event, reduction='none', label_smoothing=label_smoothing) * (group == 1)
    weights = torch.ones_like(root_loss) if class_weight is None else class_weight[target]
    denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    terms = {name: (loss * weights).sum() / denominator for name, loss in
             (('group', root_loss), ('normal', normal_loss), ('event', event_loss))}
    return sum(terms.values()), terms


def head_probabilities(outputs):
    return {key: torch.softmax(outputs[key].detach().float(), dim=1) for key in ('group', 'normal', 'event')}
