"""Keep surrogate and LIF numerics while removing device scalar synchronization."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from models.models import SingleHeadSCN
from models.utils import LIFSpike, TriangularSurrogate, _surrogate_gamma


class LegacySurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, gamma):
        gamma_tensor = input_tensor.new_tensor(float(gamma))
        ctx.save_for_backward(input_tensor, gamma_tensor)
        return (input_tensor > 0).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, gamma_tensor = ctx.saved_tensors
        gamma = float(gamma_tensor.item())
        grad = (1.0 / (gamma * gamma)) * (gamma - input_tensor.abs()).clamp(min=0.0)
        return grad_output * grad, None


class LegacyLIF(LIFSpike):
    def _spike(self, value):
        return LegacySurrogate.apply(value, self.gamma)


class SurrogateOptimizationTests(unittest.TestCase):
    def compare_surrogate(self, device):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for gamma in (1.0, .7, 1.123456789, 2.0):
                with self.subTest(device=device, dtype=dtype, gamma=gamma):
                    value = torch.linspace(-1.5, 1.5, 257, device=device, dtype=dtype)
                    # Include both signs, zero and values at/near the surrogate cutoff.
                    value = torch.cat((value, torch.tensor([-gamma, 0, gamma], dtype=dtype, device=device)))
                    new = value.detach().clone().requires_grad_()
                    old = value.detach().clone().requires_grad_()
                    weights = torch.linspace(-.8, 1.3, len(value), dtype=dtype, device=device)
                    new_out = TriangularSurrogate.apply(new, gamma)
                    old_out = LegacySurrogate.apply(old, gamma)
                    new_out.backward(weights)
                    old_out.backward(weights)
                    torch.testing.assert_close(new_out, old_out, rtol=0, atol=0)
                    torch.testing.assert_close(new.grad, old.grad, rtol=0, atol=0)

    def test_surrogate_exact_cpu_parity_for_supported_precisions(self):
        self.compare_surrogate('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_surrogate_exact_cuda_parity_for_supported_precisions(self):
        self.compare_surrogate('cuda')

    def test_lif_recurrence_and_input_gradients_match_for_both_reset_modes(self):
        torch.manual_seed(7)
        for soft_reset in (True, False):
            for gamma in (1.0, .7):
                new = LIFSpike(gamma=gamma, soft_reset=soft_reset)
                old = LegacyLIF(gamma=gamma, soft_reset=soft_reset)
                inputs = torch.randn(3, 5, 37)
                new_inputs = inputs.clone().requires_grad_()
                old_inputs = inputs.clone().requires_grad_()
                weights = torch.randn_like(inputs)
                new_output = new(new_inputs)
                old_output = old(old_inputs)
                new_output.backward(weights)
                old_output.backward(weights)
                torch.testing.assert_close(new_output, old_output, rtol=0, atol=0)
                torch.testing.assert_close(new_inputs.grad, old_inputs.grad, rtol=0, atol=0)

    def test_scn_checkpoint_logits_and_parameter_gradients_unchanged(self):
        torch.manual_seed(3)
        model = SingleHeadSCN(n_channels=6, n_classes=4, conv_channels=(4, 6, 6),
                              conv_kernel_sizes=(3, 3, 3), conv_strides=(2, 2, 1),
                              pool_kernel_sizes=(2, 2, 2), pool_strides=(2, 2, 2),
                              p_drop=0, gamma=.7)
        reference = copy.deepcopy(model)
        events = torch.randint(0, 2, (3, 128, 6, 2)).float()
        offsets = torch.randn(3, 6)
        model.eval()
        reference.eval()
        actual = model(events, offset=offsets)[0]
        actual.sum().backward()
        with patch('models.utils.TriangularSurrogate', LegacySurrogate):
            expected = reference(events, offset=offsets)[0]
            expected.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(model.state_dict().keys(), reference.state_dict().keys())
        model.load_state_dict(reference.state_dict(), strict=True)
        for (name, current), (old_name, old) in zip(model.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, old_name)
            if old.grad is None:
                self.assertIsNone(current.grad)
            else:
                torch.testing.assert_close(current.grad, old.grad, rtol=0, atol=0)

    def test_warmed_up_surrogate_forward_backward_has_no_scalar_reads(self):
        for gamma in (1.0, .7):
            _surrogate_gamma(gamma, torch.float32)
            value = torch.randn(5, 7, requires_grad=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                output = TriangularSurrogate.apply(value, gamma)
                output.sum().backward()
            keys = {entry.key for entry in profiler.key_averages()}
            self.assertNotIn('aten::item', keys)
            self.assertNotIn('aten::_local_scalar_dense', keys)


if __name__ == '__main__':
    unittest.main()
