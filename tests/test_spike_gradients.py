"""Surrogate-gradient parity and regression coverage for scalar synchronization."""

import unittest

import torch

from models.utils import TriangularSurrogate


class SpikeGradientTests(unittest.TestCase):
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
