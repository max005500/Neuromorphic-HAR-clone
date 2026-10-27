"""Verify replay inference remains causal and uses only held-out recordings."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from preprocess.utils import save_json
from test_kfall_training import fixture_args, write_fixture
from train_kfall import build_data, build_model, get_args
from visualization.kfall_replay_data import prepare_replay, prediction_index_at


class ProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.observations = []

    def load_state_dict(self, state, strict=True):
        self.loaded_strictly = strict

    def forward(self, events, offset=None):
        self.observations.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                                  events.detach().cpu().numpy().copy(), offset.detach().cpu().numpy().copy()))
        score = events[:, :, 0, 0].sum(dim=1) / 20 + offset[:, 0]
        return torch.stack((score, torch.zeros_like(score), torch.ones_like(score)), dim=1)


class KFallReplayDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        write_fixture(cls.root)
        cls.cfg = get_args(fixture_args(cls.root))
        with contextlib.redirect_stdout(io.StringIO()):
            data = build_data(cls.cfg, torch.device("cpu"))
        cls.metadata = data["metadata"]
        cls.cache = Path(cls.cfg.cache_dir) / "kfall_event_loader_default"
        cls.run_dir = cls.root / "saved_run"
        (cls.run_dir / "checkpoints").mkdir(parents=True)
        save_json(cls.run_dir / "dataset_metadata.json", cls.metadata)
        model = build_model(cls.cfg, torch.device("cpu"))
        cls.checkpoint = cls.run_dir / "checkpoints" / "best_model.pt"
        torch.save({"config": vars(cls.cfg), "model_state_dict": model.state_dict(),
                    "epoch": 1, "best_val_loss": 0.5}, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def replay(self, **kwargs):
        probes = []

        def construct(*args, **kwargs):
            model = ProbeModel()
            probes.append(model)
            return model

        with patch("visualization.kfall_replay_data.build_model", side_effect=construct):
            replay = prepare_replay(self.run_dir, device="cpu", **kwargs)
        self.assertEqual(len(probes), 1)
        return replay, probes[0]

    def test_fixed_grid_covers_full_record_without_annotation_anchor(self):
        replay, _ = self.replay(prediction_stride=10)
        self.assertEqual(replay.raw.shape, (301, 6))
        np.testing.assert_array_equal(replay.endpoint_rows, np.arange(204, 301, 10))
        self.assertEqual(replay.probabilities.shape, (10, 3))
        np.testing.assert_allclose(replay.probabilities.sum(axis=1), 1, atol=1e-6)
        self.assertTrue(np.any(replay.endpoint_times >= replay.fall_impact_time))
        offline_starts = np.load(self.cache / "test" / "start.npy")
        file_indices = np.load(self.cache / "test" / "file_idx.npy")
        file_index = next(index for index, record in enumerate(self.metadata["recordings"])
                          if Path(record["file"]).name == replay.path.name)
        # The offline pre-impact anchor is not added to the continuous grid.
        anchor = int(offline_starts[file_indices == file_index].max())
        self.assertEqual(anchor, 44)
        self.assertNotIn(anchor + replay.raw_win - 1, replay.endpoint_rows)
        self.assertFalse(replay.raw.flags.writeable)
        self.assertFalse(replay.probabilities.flags.writeable)

    def test_warmup_and_predictions_are_visible_only_after_their_endpoint(self):
        replay, _ = self.replay()
        self.assertIsNone(prediction_index_at(replay, 0.0))
        self.assertIsNone(prediction_index_at(replay, replay.endpoint_times[0] - 1e-8))
        self.assertEqual(prediction_index_at(replay, replay.endpoint_times[0]), 0)
        self.assertEqual(prediction_index_at(replay, replay.endpoint_times[1] - 1e-8), 0)
        self.assertEqual(prediction_index_at(replay, replay.endpoint_times[1]), 1)
        self.assertEqual(prediction_index_at(replay, replay.times[-1]), len(replay.endpoint_rows) - 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            prediction_index_at(replay, np.nan)

    def test_inference_receives_exact_trailing_windows_without_gradients(self):
        replay, probe = self.replay(prediction_stride=20, batch_size=2)
        self.assertTrue(probe.loaded_strictly)
        self.assertTrue(all(observation[:3] == (False, False, True) for observation in probe.observations))
        observed_offsets = np.concatenate([observation[4] for observation in probe.observations])
        starts = replay.endpoint_rows - replay.raw_win + 1
        np.testing.assert_array_equal(observed_offsets, replay.raw[starts])
        self.assertTrue(all(observation[3].shape[1:] == (128, 6, 2) for observation in probe.observations))

    def test_future_source_changes_cannot_change_earlier_predictions(self):
        before, _ = self.replay()
        original = before.path.read_bytes()
        try:
            with before.path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            # The original cached pre-impact windows end before row 260.
            for row in rows[261:]:
                for channel in range(2, 8):
                    row[channel] = str(float(row[channel]) + 20)
            with before.path.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            after, _ = self.replay()
        finally:
            before.path.write_bytes(original)
        past = before.endpoint_rows < 260
        np.testing.assert_array_equal(before.probabilities[past], after.probabilities[past])
        self.assertFalse(np.array_equal(before.probabilities[~past], after.probabilities[~past]))

    def test_selectors_accept_only_original_test_recordings(self):
        replay, _ = self.replay()
        direct, _ = self.replay(recording=replay.path.stem)
        self.assertEqual(replay.path, direct.path)
        documented, _ = self.replay(recording="SA" + replay.path.name[1:])
        self.assertEqual(documented.path, replay.path)
        training_record = self.metadata["recordings"][self.metadata["split_file_indices"]["train"][0]]
        with self.assertRaisesRegex(ValueError, "test partition"):
            self.replay(recording=Path(training_record["file"]).name)
        with self.assertRaisesRegex(ValueError, "test partition"):
            self.replay(subject=training_record["subject"])

    def test_actual_test_index_selects_whole_source_and_normal_activity(self):
        classes = np.load(self.cache / "test" / "y_h1.npy")
        index = int(np.flatnonzero(classes == 1)[0])
        replay, _ = self.replay(window=index)
        self.assertEqual(replay.label, "Dynamic")
        self.assertIsNone(replay.fall_onset_time)
        self.assertIsNone(replay.fall_impact_time)
        self.assertEqual(len(replay.raw), 301)
        other, _ = self.replay()
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.replay(window=index, recording=other.path.name)

    def test_cache_and_checkpoint_remain_unchanged_without_training(self):
        paths = [self.checkpoint, self.cache / "metadata.json", *self.cache.rglob("*.npy")]

        def fingerprints():
            return {path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths}

        before = fingerprints()
        forbidden = AssertionError("Replay must not train or preprocess the dataset.")
        with contextlib.ExitStack() as stack:
            for name in ("train_kfall.main", "train_kfall.build_data",
                         "preprocess.preprocess_kfall.load_kfall_3class",
                         "loaders.kfall_event_loader.load_or_build_kfall_event_splits"):
                stack.enter_context(patch(name, side_effect=forbidden))
            self.replay()
        self.assertEqual(before, fingerprints())

    def test_invalid_schedule_and_changed_cached_source_are_rejected(self):
        for stride in (0, -1, 1.5):
            with self.subTest(stride=stride), self.assertRaisesRegex(ValueError, "prediction_stride"):
                self.replay(prediction_stride=stride)
        replay, _ = self.replay()
        original = replay.path.read_bytes()
        try:
            with replay.path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            rows[1][2] = str(float(rows[1][2]) + 10)
            with replay.path.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            with self.assertRaisesRegex(ValueError, "cached test windows"):
                self.replay()
        finally:
            replay.path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
