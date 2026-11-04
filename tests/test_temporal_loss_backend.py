"""Transition expectation parity without einsum/batched outer-product backward."""
import unittest
import torch
from train.temporal import temporal_loss


class TemporalLossBackendTests(unittest.TestCase):
    def test_broadcast_expectation_and_gradients_match_einsum(self):
        torch.manual_seed(17)
        values = torch.randn(2, 5, 4)
        costs = torch.rand(4, 4)
        target = torch.randint(0, 4, (2, 5))
        previous = torch.softmax(torch.randn(2, 4), -1)
        valid = torch.ones(2, 5, dtype=torch.bool)
        optimized = values.clone().requires_grad_()
        reference = values.clone().requires_grad_()
        loss, parts = temporal_loss({'logits': optimized}, target, valid, target[:, 0],
            torch.ones(2, dtype=torch.bool), costs=costs, previous_probability=previous,
            lambda_transition=.2)
        p = reference.softmax(-1)
        before = torch.cat((previous[:, None], p[:, :-1]), 1)
        expected_transition = torch.einsum('bti,ij,btj->bt', before, costs, p).mean()
        expected = torch.nn.functional.cross_entropy(reference.flatten(0, 1), target.flatten()) + .2 * expected_transition
        loss.backward(); expected.backward()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(parts['transition_sum'] / parts['pair_count'], expected_transition)
        torch.testing.assert_close(optimized.grad, reference.grad)

    def test_no_bmm_and_inactive_auxiliary_terms_have_no_graph(self):
        for weight in (0., .2):
            logits = torch.randn(2, 3, 4, requires_grad=True)
            boundary = torch.randn(2, 3, requires_grad=True)
            target = torch.tensor([[1, 0, 3], [2, 1, 1]])
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                loss, parts = temporal_loss({'logits': logits, 'boundary': boundary}, target,
                    torch.ones_like(target, dtype=torch.bool), target[:, 0], torch.ones(2, dtype=torch.bool),
                    costs=torch.ones(4, 4), lambda_transition=weight)
                loss.backward()
            keys = {row.key for row in profile.key_averages()}
            self.assertFalse(any('bmm' in key.lower() for key in keys))
            self.assertIsNone(boundary.grad)
            self.assertFalse(parts['boundary_sum'].requires_grad)
            self.assertEqual(parts['transition_sum'].requires_grad, weight > 0)
