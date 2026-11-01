"""Carry label rules through commands, caches, checkpoints, and raw-free evaluation."""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train_kfall import get_args, label_cache_tag, main
from compare_kfall_windows import build_parser, build_experiment_specs, training_command
from tune_kfall_thresholds import build_parser as threshold_parser, build_candidates, search_settings, candidate_name
from evaluation.kfall_data import load_evaluation_data
from test_kfall_training import fixture_args, write_fixture


class TemporalCLI(unittest.TestCase):
    def test_new_runs_default_temporal_and_old_label_scheme_is_explicit(self):
        temporal = get_args([])
        legacy = get_args(['--label-policy', 'legacy'])
        self.assertEqual(temporal.label_policy, 'temporal')
        self.assertEqual(temporal.pre_fall_seconds, 1.2)
        self.assertEqual(temporal.adl_margin_seconds, 1.0)
        self.assertEqual(temporal.post_recovery_seconds, 2.0)
        self.assertNotEqual(temporal.cache_name, legacy.cache_name)
        for flags in (['--adl-margin-seconds', '1.5'], ['--pre-fall-seconds', '2'],
                      ['--post-recovery-seconds', '1'], ['--label-policy', 'motion']):
            self.assertNotEqual(get_args(flags).cache_name, temporal.cache_name)

    def test_comparison_passes_exact_label_rules_to_child_training(self):
        args = build_parser().parse_args(['--label-policy', 'motion', '--pre-fall-seconds', '1',
                                          '--motion-gyr-threshold', '15'])
        spec = build_experiment_specs([1.00], label_tag=label_cache_tag(args))[0]
        config = get_args(training_command(args, spec)[2:])
        self.assertEqual(config.label_policy, 'motion')
        self.assertEqual(config.pre_fall_seconds, 1)
        self.assertEqual(config.motion_gyr_threshold, 15)
        self.assertIn('_labelsmotion_', spec.cache_name)

    def test_threshold_search_identity_separates_target_rules(self):
        candidate = build_candidates()[0]
        parser = threshold_parser()
        temporal = candidate_name(candidate, search_settings(parser.parse_args([])))
        for flags in (['--label-policy', 'legacy'], ['--post-recovery-seconds', '3']):
            self.assertNotEqual(temporal, candidate_name(candidate, search_settings(parser.parse_args(flags))))

    def test_new_temporal_run_can_be_evaluated_without_original_signals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root)
            args = fixture_args(root) + ['--label-policy', 'temporal', '--post-impact-seconds', '1.2',
                                        '--raw-win', '50', '--stride', '25', '--cache-name', 'temporal']
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            run = next((root / 'runs').iterdir())
            saved = torch.load(run / 'checkpoints/best_model.pt', map_location='cpu', weights_only=True)
            shutil.rmtree(root / 'sensor_data')
            shutil.rmtree(root / 'label_data')
            arrays, _, snapshot, _ = load_evaluation_data(run, saved['config'])
            self.assertEqual(set(arrays['y_h1']), {0, 1, 2, 3})
            self.assertEqual(snapshot['config']['label_policy'], 'temporal')
            self.assertIn('frame_label_intervals', snapshot['recordings'][0])
            changed = {**saved['config'], 'label_policy': 'legacy'}
            with self.assertRaisesRegex(ValueError, 'Checkpoint label_policy'):
                load_evaluation_data(run, changed)

    def test_optimizer_receives_same_temporal_policy(self):
        from optimize_theta import build_parser as optimizer_parser
        from theta_optimization.utils import RunConfig
        parser = optimizer_parser()
        cfg = RunConfig.from_args(parser.parse_args(['--dataset', 'kfall', '--ups', '10', '--dead-zone', '.5']))
        self.assertEqual(cfg.label_policy, 'temporal')
        self.assertIn('_labelstemporal_', str(cfg.result_dir))
        old = RunConfig.from_args(parser.parse_args(['--dataset', 'kfall', '--ups', '10', '--dead-zone', '.5',
                                                    '--label-policy', 'legacy']))
        self.assertNotEqual(cfg.result_dir, old.result_dir)
