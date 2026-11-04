"""The no-padding fast path preserves temporal outputs and backward semantics."""
import unittest
import torch
from torch.nn import functional as F
from models.temporal import TemporalSCN


class LegacyTemporalSCN(TemporalSCN):
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
            indices = torch.nonzero(valid[:, time], as_tuple=False).flatten().to(events.device)
            features = events.new_zeros((batch, self.backbone.fused_dim), dtype=torch.float32)
            if len(indices):
                current = self.backbone.extract_features(events[indices, time].float(), offsets[indices, time])
                features = features.to(current.dtype).index_copy(0, indices, current)
            if self.memory is not None:
                spikes, candidate = self.memory(features, state)
                if state is None:
                    previous = tuple(torch.zeros_like(value) for value in candidate)
                else:
                    previous = state
                mask = valid[:, time].to(events.device).unsqueeze(1)
                state = tuple(torch.where(mask, new, old) for new, old in zip(candidate, previous))
                representation = state[1]
                activity.append((representation * mask).sum(1))
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



class TemporalFastPathTests(unittest.TestCase):
    def test_original_path_outputs_states_and_gradients_match(self):
        kwargs = dict(n_channels=6, conv_channels=(3, 4, 4), conv_kernel_sizes=(3, 3, 3),
            conv_strides=(2, 2, 1), pool_kernel_sizes=(2, 2, 2), pool_strides=(2, 2, 2),
            p_drop=0, offset_hidden=3)
        for training in (False, True):
            for memory in ('none', 'lif'):
                for hierarchical in (False, True):
                    for padded in (False, True):
                        with self.subTest(training=training, memory=memory, hierarchical=hierarchical, padded=padded):
                            torch.manual_seed(15)
                            current = TemporalSCN(kwargs, memory=memory, hidden_size=4,
                                                  hierarchical=hierarchical, boundary=True).train(training)
                            previous = LegacyTemporalSCN(kwargs, memory=memory, hidden_size=4,
                                                         hierarchical=hierarchical, boundary=True).train(training)
                            previous.load_state_dict(current.state_dict(), strict=True)
                            value = torch.randint(0, 2, (2, 4, 64, 6, 2)).float()
                            offset = torch.randn(2, 4, 6)
                            a, b = value.clone().requires_grad_(), value.clone().requires_grad_()
                            oa, ob = offset.clone().requires_grad_(), offset.clone().requires_grad_()
                            valid = torch.ones(2, 4, dtype=torch.bool)
                            if padded:
                                valid[0, 2:] = False
                            actual, actual_state = current(a, oa, valid=valid)
                            expected, expected_state = previous(b, ob, valid=valid)
                            for key in actual:
                                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                            if actual_state is not None:
                                for x, y in zip(actual_state, expected_state):
                                    torch.testing.assert_close(x, y, rtol=0, atol=0)
                            actual['logits'].sum().backward()
                            expected['logits'].sum().backward()
                            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                            torch.testing.assert_close(oa.grad, ob.grad, rtol=0, atol=0)
                            self.assertEqual(current.state_dict().keys(), previous.state_dict().keys())
                            for (name, param), (oldname, oldparam) in zip(current.named_parameters(), previous.named_parameters()):
                                self.assertEqual(name, oldname)
                                if param.grad is None:
                                    self.assertIsNone(oldparam.grad)
                                else:
                                    torch.testing.assert_close(param.grad, oldparam.grad, rtol=0, atol=0)
