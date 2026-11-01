"""Use each run's saved temporal targets in cached and continuous viewers."""
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba

from preprocess.kfall_labels import record_frame_labels
from visualization.kfall_replay_plot import ReplayPlot, CLASS_COLORS
from visualization.kfall_data import KFallWindowStore
from test_kfall_postimpact_viewers import replay_fixture


class KFallTemporalReplayPlotTests(unittest.TestCase):
    def plot(self, replay):
        figure = Figure(figsize=(10, 8))
        FigureCanvasAgg(figure)
        self.addCleanup(figure.clear)
        return ReplayPlot(figure.subplots(5, 1), replay)

    def replay(self):
        replay = replay_fixture()
        replay.record.update(fall_onset_frame=241, fall_impact_frame=301)
        replay.true_labels = record_frame_labels(replay.frames, replay.record, label_policy='temporal',
                                                post_impact_seconds=1.2)
        return replay

    def test_real_labels_follow_temporal_policy_at_each_boundary(self):
        plot = self.plot(self.replay())
        for moment, expected in ((1.19, 'Estático'), (1.20, 'Dinámico'), (2.4, 'Caída'),
                                 (3., 'Postimpacto'), (4.2, 'Dinámico')):
            self.assertEqual(plot.update(moment)['real_phase'], expected)
        self.assertLessEqual(max(right for _, right, _ in plot._truth_intervals(4.2)), 4.2)

    def test_each_class_appears_and_seek_hides_future_labels(self):
        plot = self.plot(self.replay())
        plot.update(5.9)
        colors = plot.truth_collection.get_facecolors()
        self.assertTrue(all(any(np.allclose(color, to_rgba(expected)) for color in colors)
                            for expected in CLASS_COLORS.values()))
        plot.update(.2)
        np.testing.assert_allclose(plot.truth_collection.get_facecolors(), [to_rgba(CLASS_COLORS[2])])
        self.assertTrue(all(path.vertices[:, 0].max() <= .2 for path in plot.truth_collection.get_paths()))

    def test_adl_uses_margins_even_for_legacy_static_task(self):
        replay = replay_fixture()
        replay.record['label_id'] = 2
        replay.fall_onset_time = replay.fall_impact_time = replay.post_impact_end_time = None
        replay.true_labels = record_frame_labels(replay.frames, replay.record, label_policy='temporal')
        plot = self.plot(replay)
        for moment, expected in ((.99, 'Estático'), (1.0, 'Dinámico'), (5.0, 'Dinámico'), (5.01, 'Estático')):
            self.assertEqual(plot.update(moment)['real_phase'], expected)

    def test_misaligned_labels_are_rejected(self):
        replay = self.replay()
        replay.true_labels = replay.true_labels[:-1]
        with self.assertRaisesRegex(ValueError, 'align'):
            self.plot(replay)


class KFallTemporalCachedViewerTests(unittest.TestCase):
    def setUp(self):
        from test_kfall_viewer_data import KFallViewerDataTests
        self.fixture = KFallViewerDataTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_window_uses_full_record_label_policy_and_draws_class_regions(self):
        from loaders.kfall_event_loader import load_or_build_kfall_event_splits
        from visualization.kfall_plotting import draw_window
        config = replace(self.fixture.config, label_policy='temporal', post_impact_seconds=1.2)
        load_or_build_kfall_event_splits(self.fixture.root, cache_dir=self.fixture.cache_parent,
                                       run_name='temporal', cfg=config, verbose=False)
        store = KFallWindowStore(self.fixture.cache_parent / 'kfall_event_loader_temporal')
        window = store.get_window('train', 0)
        self.assertEqual(window.frame_labels[-1], window.class_id)
        self.assertEqual(len(window.frame_labels), len(window.raw))
        figure = Figure(figsize=(10, 8))
        FigureCanvasAgg(figure)
        self.addCleanup(figure.clear)
        axes = figure.subplots(3, 2)
        draw_window(axes, window)
        self.assertTrue(axes[0, 0].patches)
        self.assertFalse(window.frame_labels.flags.writeable)

    def test_corrupt_saved_temporal_manifest_is_rejected(self):
        from visualization.kfall_data import source_frame_labels
        frames = np.arange(1, 601)
        signal = np.zeros((len(frames), 6))
        record = {'label_id': 2, 'frame_label_intervals': [[1, 601, 1]]}
        with self.assertRaisesRegex(ValueError, 'manifest'):
            source_frame_labels(signal, frames, record, {'label_policy': 'temporal'})

    def test_replay_uses_saved_policy_and_sample_targets(self):
        import contextlib
        import io
        import tempfile
        import torch
        from pathlib import Path
        from unittest.mock import patch
        from preprocess.utils import save_json
        from test_kfall_training import fixture_args, write_fixture
        from train_kfall import build_data, build_model, get_args
        from visualization.kfall_replay_data import prepare_replay

        class ConstantModel(torch.nn.Module):
            def load_state_dict(self, *args, **kwargs):
                pass
            def forward(self, events, offset=None):
                return torch.zeros((len(events), 4), device=events.device)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_fixture(root)
            cfg = get_args(fixture_args(root) + ['--label-policy', 'temporal', '--post-impact-seconds', '1.2'])
            with contextlib.redirect_stdout(io.StringIO()):
                data = build_data(cfg, torch.device('cpu'))
            run = root / 'run'
            (run / 'checkpoints').mkdir(parents=True)
            save_json(run / 'dataset_metadata.json', data['metadata'])
            model = build_model(cfg, torch.device('cpu'))
            torch.save({'config': vars(cfg), 'model_state_dict': model.state_dict(), 'epoch': 1,
                        'best_val_loss': .5}, run / 'checkpoints' / 'best_model.pt')
            with patch('visualization.kfall_replay_data.build_model', return_value=ConstantModel()):
                replay = prepare_replay(run, device='cpu')
            self.assertIsNotNone(replay.true_labels)
            expected = record_frame_labels(replay.frames, replay.record, signal=replay.raw,
                                           label_policy='temporal', post_impact_seconds=1.2)
            np.testing.assert_array_equal(replay.true_labels, expected)
            self.assertFalse(replay.true_labels.flags.writeable)
            self.assertEqual(replay.config['label_policy'], 'temporal')


if __name__ == '__main__':
    unittest.main()
