"""Interactive gyro tuning keeps cached labels and acceleration input fixed."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from preprocess.event_encoding import sigma_delta
from train_kfall import read_theta_file
from tune_kfall_gyro import GyroThresholdTuner, preview_gyro, main
from visualization.kfall_data import KFallWindowStore
import test_kfall_viewer_data as viewer_fixtures


class GyroTunerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = viewer_fixtures.KFallViewerDataTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(plt.close, 'all')
        self.store = KFallWindowStore(self.fixture.cache)
        self.output = self.fixture.directory / 'theta.json'

    def tuner(self, interactive=False, **kwargs):
        return GyroThresholdTuner(self.store, interactive=interactive, output=self.output, **kwargs)

    def test_preview_matches_encoder_and_keeps_acceleration_immutable(self):
        window = self.store.get_window('train', 0)
        before = window.events.copy()
        candidate, metrics = preview_gyro(window, [1, 2, 3], dead_zone=self.store.cfg.dead_zone)
        expected, _, rec = sigma_delta(window.interpolated[:, 3:], theta=[1, 2, 3],
                                      init='x0', dead_zone=.5, return_reconstruction=True)
        np.testing.assert_array_equal(candidate.events[:, 3:], expected)
        np.testing.assert_array_equal(candidate.reconstructed[:, 3:], rec)
        np.testing.assert_array_equal(candidate.events[:, :3], before[:, :3])
        np.testing.assert_array_equal(candidate.reconstructed[:, :3], window.reconstructed[:, :3])
        np.testing.assert_array_equal(candidate.offset, window.offset)
        np.testing.assert_array_equal(window.events, before)
        self.assertEqual(candidate.class_id, window.class_id)
        self.assertTrue(np.isfinite(metrics['rmse']).all())
        baseline, _ = preview_gyro(window, window.theta[3:], dead_zone=.5)
        np.testing.assert_array_equal(baseline.events, window.events)
        np.testing.assert_array_equal(baseline.reconstructed, window.reconstructed)

    def test_widget_updates_link_reset_and_validation(self):
        viewer = self.tuner(interactive=True)
        viewer.widgets['theta_text_0'].set_val('2.5')
        self.assertAlmostEqual(viewer.gyro_theta[0], 2.5)
        viewer.widgets['linked'].set_active(0)
        viewer.widgets['theta_slider_1'].set_val(np.log10(4))
        np.testing.assert_allclose(viewer.gyro_theta, [4, 4, 4])
        viewer.widgets['theta_text_1'].set_val('nan')
        np.testing.assert_allclose(viewer.gyro_theta, [4, 4, 4])
        viewer.reset_theta()
        np.testing.assert_array_equal(viewer.gyro_theta, self.store.cfg.theta_sd[3:])

    def test_filters_keep_tuning_and_saving_preserves_all_cache_files(self):
        stamps = {p: p.stat().st_mtime_ns for p in self.fixture.cache.rglob('*') if p.is_file()}
        viewer = self.tuner(gyro_theta=[1, 2, 3])
        viewer.set_filters('val', None, None)
        viewer.set_position(1)
        viewer.sensor = 'acc'
        viewer.redraw()
        with contextlib.redirect_stdout(io.StringIO()):
            path = viewer.save_theta()
            viewer.save(self.fixture.directory / 'preview.png')
        theta = read_theta_file(path)
        np.testing.assert_array_equal(theta[:3], self.store.cfg.theta_sd[:3])
        np.testing.assert_array_equal(theta[3:], [1, 2, 3])
        self.assertTrue(json.loads(path.read_text())['calibration']['accelerometer_fixed'])
        self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in stamps})

    def test_cli_headless_and_reopen_json(self):
        png = self.fixture.directory / 'preview.png'
        flags = ['--cache-dir', str(self.fixture.cache), '--gyro-theta', '1', '2', '3',
                 '--output', str(self.output), '--save', str(png)]
        with contextlib.redirect_stdout(io.StringIO()):
            main(flags)
            main(['--cache-dir', str(self.fixture.cache), '--theta-file', str(self.output),
                  '--output', str(self.output), '--save', str(png)])
        self.assertEqual(png.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        np.testing.assert_array_equal(read_theta_file(self.output)[3:], [1, 2, 3])

    def test_invalid_theta_rejected(self):
        window = self.store.get_window('train', 0)
        for values in ([0, 1, 1], [np.inf, 1, 1], [1, 2]):
            with self.assertRaises(ValueError):
                preview_gyro(window, values)
