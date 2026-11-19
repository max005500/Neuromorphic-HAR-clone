"""Surrogate-gradient parity and regression coverage for scalar synchronization."""

import unittest

import torch

from models.utils import LIFSpike, RecurrentLIF, TriangularSurrogate


class SpikeGradientTests(unittest.TestCase):
    def test_recurrent_feedback_is_causal_binary_and_resets_between_windows(self):
        layer = RecurrentLIF(1, 1, tau=0, thresh=0.5)
        with torch.no_grad():
            layer.input_proj.weight.fill_(1)
            layer.input_proj.bias.zero_()
            layer.recurrent.weight.fill_(1)
        x = torch.tensor([[[1., 0., 0.]]])
        torch.testing.assert_close(layer(x), torch.ones_like(x))
        torch.testing.assert_close(layer(torch.zeros_like(x)), torch.zeros_like(x))
        changed_future = x.clone()
        changed_future[:, :, 2] = -4
        torch.testing.assert_close(layer(x)[:, :, :2], layer(changed_future)[:, :, :2])
        with torch.no_grad():
            layer.recurrent.weight.zero_()
        torch.testing.assert_close(layer(x), x)

    def test_zero_feedback_matches_feedforward_lif_with_both_reset_modes(self):
        x = torch.tensor([[[0.8, 0.4, -0.3, 0.6]]])
        for soft_reset in (False, True):
            with self.subTest(soft_reset=soft_reset):
                layer = RecurrentLIF(1, 1, tau=0.75, soft_reset=soft_reset)
                with torch.no_grad():
                    layer.input_proj.weight.fill_(1)
                    layer.input_proj.bias.zero_()
                    layer.recurrent.weight.zero_()
                torch.testing.assert_close(layer(x), LIFSpike(tau=0.75, soft_reset=soft_reset)(x))

    def test_recurrent_surrogate_backpropagates_through_previous_spikes(self):
        layer = RecurrentLIF(1, 1, tau=0, thresh=0.5)
        with torch.no_grad():
            layer.input_proj.weight.fill_(1)
            layer.input_proj.bias.zero_()
            layer.recurrent.weight.fill_(0.6)
        x = torch.tensor([[[0.6, 0.2, 0.2]]], requires_grad=True)
        spikes = layer(x)
        self.assertTrue(torch.all((spikes == 0) | (spikes == 1)))
        spikes[:, :, -1].sum().backward()
        self.assertGreater(float(x.grad[0, 0, 0]), 0)
        for parameter in layer.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_spikes_and_surrogate_gradients_preserve_the_triangular_rule(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for gamma in (0.5, 1.0, 1.7):
                with self.subTest(device=device, gamma=gamma):
                    x = torch.tensor(
                        [-2.0, -0.4, 0.0, 0.3, 2.0], device=device, requires_grad=True,
                    )
                    upstream = torch.tensor([1.0, -2.0, 0.5, 3.0, 1.0], device=device)
                    spikes = TriangularSurrogate.apply(x, gamma)
                    spikes.backward(upstream)
                    torch.testing.assert_close(spikes, (x.detach() > 0).float())
                    expected = upstream * (gamma - x.detach().abs()).clamp(min=0) / gamma**2
                    torch.testing.assert_close(x.grad, expected)

    def test_backward_does_not_read_a_tensor_scalar(self):
        x = torch.randn(8, requires_grad=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            TriangularSurrogate.apply(x, 1.0).sum().backward()
        operations = {event.key for event in profile.key_averages()}
        self.assertNotIn("aten::item", operations)
        self.assertNotIn("aten::_local_scalar_dense", operations)


if __name__ == "__main__":
    unittest.main()
