"""Reference trajectories and persistent-state semantics for recurrent LIF reset."""
import copy
import unittest

import torch

from models.temporal import RecurrentLIF, TemporalSCN
from models.utils import TriangularSurrogate


class RecurrentResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @staticmethod
    def cell(mode='immediate', soft=True):
        cell = RecurrentLIF(1, 1, tau=.5, threshold=1., soft_reset=soft, reset_mode=mode).double()
        with torch.no_grad():
            cell.input.weight.fill_(1.)
            cell.input.bias.zero_()
            cell.recurrent.weight.fill_(.4)
        return cell

    @staticmethod
    def trajectory(cell, values):
        state, membranes, spikes = None, [], []
        for value in values:
            spike, state = cell(torch.tensor([[value]], dtype=torch.float64), state)
            membranes.append(state[0].squeeze())
            spikes.append(spike.squeeze())
        return torch.stack(membranes), torch.stack(spikes)

    def test_delayed_soft_reset_matches_requested_equation(self):
        membranes, spikes = self.trajectory(self.cell('delayed'), [1.2, .8, 0., 1.2, 0.])
        torch.testing.assert_close(membranes, torch.tensor([1.2, .8, .4, 1.4, .1], dtype=torch.float64))
        torch.testing.assert_close(spikes, torch.tensor([1., 0., 0., 1., 0.], dtype=torch.float64))
        # tau < 1 makes delayed subtraction differ from post-reset membrane leak.
        _, immediate = self.trajectory(self.cell(), [1.2, .8, 0., 1.2, 0.])
        self.assertEqual(float(immediate[1].detach()), 1.)
        self.assertEqual(float(spikes[1].detach()), 0.)

    def test_delayed_hard_reset_clears_previous_membrane_before_integration(self):
        membranes, spikes = self.trajectory(self.cell('delayed', soft=False), [1.2, .8, 0., 1.2, 0.])
        torch.testing.assert_close(membranes, torch.tensor([1.2, 1.2, .4, 1.4, .4], dtype=torch.float64))
        torch.testing.assert_close(spikes, torch.tensor([1., 1., 0., 1., 0.], dtype=torch.float64))

    def test_default_keeps_original_outputs_states_and_gradients(self):
        for soft in (True, False):
            with self.subTest(soft=soft):
                torch.manual_seed(32)
                current = RecurrentLIF(3, 4, soft_reset=soft).double()
                previous = copy.deepcopy(current)
                a = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
                b = a.detach().clone().requires_grad_()
                state = None
                actual = []
                for time in range(5):
                    spike, state = current(a[:, time], state)
                    actual.append(spike)
                membrane = torch.zeros(2, 4, dtype=torch.float64)
                old_spikes = torch.zeros_like(membrane)
                expected = []
                for time in range(5):
                    membrane = previous.tau * membrane + previous.input(b[:, time]) + previous.recurrent(old_spikes)
                    old_spikes = TriangularSurrogate.apply(membrane - previous.threshold, 1.)
                    membrane = (membrane - old_spikes * previous.threshold if soft
                                else membrane * (1 - old_spikes))
                    expected.append(old_spikes)
                torch.testing.assert_close(torch.stack(actual), torch.stack(expected), rtol=0, atol=0)
                torch.testing.assert_close(state[0], membrane, rtol=0, atol=0)
                torch.stack(actual).sum().backward()
                torch.stack(expected).sum().backward()
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                for current_param, previous_param in zip(current.parameters(), previous.parameters()):
                    torch.testing.assert_close(current_param.grad, previous_param.grad, rtol=0, atol=0)

    def test_temporal_delayed_memory_is_chunk_invariant_and_differentiable(self):
        kwargs = dict(n_channels=6, conv_channels=(3, 4, 4), conv_kernel_sizes=(3, 3, 3),
            conv_strides=(2, 2, 1), pool_kernel_sizes=(2, 2, 2), pool_strides=(2, 2, 2),
            offset_hidden=3, p_drop=0.)
        for soft in (True, False):
            with self.subTest(soft=soft):
                torch.manual_seed(44)
                model = TemporalSCN({**kwargs, 'soft_reset': soft}, hidden_size=4,
                                    memory_reset='delayed').eval()
                other = copy.deepcopy(model)
                a = torch.randint(0, 2, (2, 5, 64, 6, 2)).float().requires_grad_()
                b = a.detach().clone().requires_grad_()
                offsets = torch.randn(2, 5, 6)
                full, state = model(a, offsets)
                first, carry = other(b[:, :2], offsets[:, :2])
                second, end = other(b[:, 2:], offsets[:, 2:], carry)
                combined = torch.cat((first['logits'], second['logits']), dim=1)
                torch.testing.assert_close(full['logits'], combined, rtol=0, atol=0)
                for left, right in zip(state, end):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                full['logits'].square().sum().backward()
                combined.square().sum().backward()
                self.assertTrue(torch.isfinite(a.grad).all())
                torch.testing.assert_close(a.grad, b.grad)
                for left, right in zip(model.parameters(), other.parameters()):
                    if left.grad is not None:
                        self.assertTrue(torch.isfinite(left.grad).all())
                        torch.testing.assert_close(left.grad, right.grad)

    def test_reset_mode_adds_no_checkpoint_tensors(self):
        immediate = self.cell()
        delayed = self.cell('delayed')
        self.assertEqual(immediate.state_dict().keys(), delayed.state_dict().keys())
        delayed.load_state_dict(immediate.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in immediate.parameters()),
                         sum(p.numel() for p in delayed.parameters()))

    def test_unknown_reset_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reset mode'):
            RecurrentLIF(1, reset_mode='unknown')
        with self.assertRaisesRegex(ValueError, 'reset mode'):
            TemporalSCN({}, memory='none', memory_reset='unknown')
