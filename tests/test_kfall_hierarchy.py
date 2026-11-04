"""Hierarchy target mapping, masked gradients, routing and checkpoint integration."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from models.hierarchical import HierarchicalSCN, hierarchy_loss, hierarchy_predict, hierarchy_targets
from train_kfall import get_args, build_model, main
from evaluate_kfall import evaluate_run
from visualization.kfall_replay_data import prepare_replay
from test_kfall_training import fixture_args, write_fixture


class HierarchyTests(unittest.TestCase):
    def test_mapping_and_hard_route_can_differ_from_joint_map(self):
        target = torch.tensor([0, 1, 2, 3])
        group, normal, event = hierarchy_targets(target)
        torch.testing.assert_close(group, torch.tensor([1, 0, 0, 1]))
        # Root chooses Normal, although Fall is the highest individual joint probability.
        outputs = {'group': torch.tensor([[.6, .4]]).log(),
                   'normal': torch.tensor([[.5, .5]]).log(),
                   'event': torch.tensor([[.99, .01]]).log()}
        self.assertEqual(hierarchy_predict(outputs).item(), 1)

    def test_absent_branch_has_zero_loss_gradient_for_both_groups(self):
        for target, unused in ((torch.tensor([1, 2]), 'event'), (torch.tensor([0, 3]), 'normal')):
            outputs = {key: torch.randn(2, 2, requires_grad=True) for key in ('group', 'normal', 'event')}
            loss, terms = hierarchy_loss(outputs, target)
            loss.backward()
            self.assertEqual(terms[unused].item(), 0)
            self.assertTrue(torch.all(outputs[unused].grad == 0))
            self.assertGreater(outputs['group'].grad.abs().sum().item(), 0)

    def test_loss_is_joint_negative_log_likelihood_when_unweighted(self):
        outputs = {key: torch.randn(4, 2, requires_grad=True) for key in ('group', 'normal', 'event')}
        target = torch.tensor([0, 1, 2, 3])
        group = torch.log_softmax(outputs['group'], 1)
        normal = torch.log_softmax(outputs['normal'], 1)
        event = torch.log_softmax(outputs['event'], 1)
        joint = torch.stack((group[:, 1] + event[:, 0], group[:, 0] + normal[:, 0],
                             group[:, 0] + normal[:, 1], group[:, 1] + event[:, 1]), 1)
        loss, _ = hierarchy_loss(outputs, target)
        torch.testing.assert_close(loss, torch.nn.functional.nll_loss(joint, target))

    def test_dense_and_spiking_heads_are_supervised_and_joint_probs_normalized(self):
        for spiking in (False, True):
            cfg = get_args(['--head-mode', 'hierarchical', '--out-win', '128', '--conv-ch', '4', '4', '4',
                            '--kernels', '3', '3', '3'] + (['--spiking-heads'] if spiking else []))
            model = build_model(cfg, torch.device('cpu'))
            self.assertIsInstance(model, HierarchicalSCN)
            outputs, _ = model(torch.randint(0, 2, (4, 128, 6, 2)).float(), offset=torch.randn(4, 6))
            for key in ('group', 'normal', 'event'):
                self.assertEqual(outputs[key].shape, (4, 2))
            torch.testing.assert_close(outputs['head1'].exp().sum(1), torch.ones(4))
            loss, _ = hierarchy_loss(outputs, torch.tensor([0, 1, 2, 3]))
            loss.backward()
            for key in ('group', 'normal', 'event'):
                self.assertTrue(any(p.grad is not None for p in getattr(model, key + '_head').parameters()))

    def test_train_evaluate_replay_and_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root)
            flags = fixture_args(root) + ['--head-mode', 'hierarchical', '--post-impact-seconds', '1.2',
                                         '--raw-win', '50', '--stride', '25', '--label-policy', 'temporal']
            with contextlib.redirect_stdout(io.StringIO()):
                main(flags)
            run = next((root / 'runs').iterdir())
            for name in ('hierarchy_accuracy.png', 'hierarchy_loss.png'):
                self.assertTrue((run / 'metrics' / name).is_file())
            report = json.loads((run / 'metrics/hierarchy_test/hierarchy_metrics.json').read_text())
            self.assertGreater(report['group']['n'], 0)
            result = evaluate_run(run, device='cpu')
            true = np.load(run / 'metrics/y_true.npy')
            pred = np.load(run / 'metrics/y_pred.npy')
            self.assertEqual(result['accuracy'], float(np.mean(true == pred)))
            np.testing.assert_array_equal(np.load(run / 'metrics/test_best/hierarchy/routed_predictions.npy'), pred)
            replay = prepare_replay(run, device='cpu', prediction_stride=50)
            self.assertEqual(replay.probabilities.shape[1], 4)
            self.assertEqual(len(replay.predicted_classes), len(replay.endpoint_times))
            self.assertTrue(np.isfinite(replay.probabilities).all())
            self.assertIsNotNone(replay.hierarchy_probabilities)
            # Force a case where routing selects Dynamic but global joint MAP
            # selects Fall, then verify both offline evaluation and replay.
            checkpoint = torch.load(run / 'checkpoints/best_model.pt', map_location='cpu', weights_only=True)
            for key, probabilities in (('group', [.6, .4]), ('normal', [.5, .5]), ('event', [.99, .01])):
                checkpoint['model_state_dict'][key + '_head.weight'].zero_()
                checkpoint['model_state_dict'][key + '_head.bias'] = torch.tensor(probabilities).log()
            explicit = root / 'routing.pt'
            torch.save(checkpoint, explicit)
            evaluated = evaluate_run(run, checkpoint=explicit, output_dir=root / 'routed_report', device='cpu')
            self.assertAlmostEqual(evaluated['accuracy'], float(np.mean(true == 1)))
            np.testing.assert_array_equal(np.load(root / 'routed_report/y_pred.npy'), np.ones(len(true)))
            probabilities = np.load(root / 'routed_report/probabilities.npy')
            self.assertTrue(np.all(probabilities.argmax(1) == 0))
            routed_replay = prepare_replay(run, checkpoint=explicit, device='cpu', prediction_stride=50)
            self.assertTrue(np.all(routed_replay.predicted_classes == 1))
            self.assertTrue(np.all(routed_replay.probabilities.argmax(1) == 0))
