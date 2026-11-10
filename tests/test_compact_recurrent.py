"""Recurrent dynamics, sequence isolation and compact KFall training."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from evaluate_kfall import evaluate_run
from models.recurrent import ALIFState, AdaptiveRecurrentLIF, CompactRecurrentSCN, RecurrentLIF
from train.kfall_recurrent import RecordingChunkSampler, make_sequence_loader, recording_sequences, run_epoch
from train_kfall import KFallTrainingDataset, build_model, get_args, main
from test_kfall_training import fixture_args, write_fixture


class CompactRecurrentTests(unittest.TestCase):
    def test_adaptive_dynamics_learnable_decays_and_masked_state(self):
        cell = AdaptiveRecurrentLIF(1, 1, beta=0.5, threshold=0.5,
                                   adaptation_decay=0.5, adaptation_strength=0.5)
        with torch.no_grad():
            cell.input.weight.fill_(1)
            cell.input.bias.zero_()
            cell.recurrent.weight.fill_(0.25)
        features = torch.tensor([[[0.8], [0.7], [0.9]]], requires_grad=True)
        spikes, state = cell(features)
        # u: .8 -> .85 -> 1.325; a: 0 -> 1 -> .5; thresholds: .5, 1, .75.
        torch.testing.assert_close(spikes, torch.tensor([[[1.], [0.], [1.]]]))
        torch.testing.assert_close(state.membrane, torch.tensor([[1.325]]))
        torch.testing.assert_close(state.adaptation, torch.tensor([[0.5]]))
        spikes.sum().backward()
        for parameter in (cell.membrane_logit, cell.adaptation_logit, cell.strength_raw,
                          cell.recurrent.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        frozen = state.detach()
        self.assertIsInstance(frozen, ALIFState)
        self.assertTrue(all(value.grad_fn is None for value in frozen))
        output, unchanged = cell(features.detach(), frozen, torch.zeros((1, 3), dtype=torch.bool))
        self.assertEqual(float(output.detach().sum()), 0)
        for actual, expected in zip(unchanged, frozen):
            torch.testing.assert_close(actual, expected)

    def test_adaptive_chunk_equivalence_causality_and_checkpoint_replay(self):
        torch.manual_seed(19)
        cfg = get_args(["--architecture", "compact_adaptive"])
        model = build_model(cfg, torch.device("cpu")).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 22252)
        events = torch.randint(0, 2, (2, 4, 128, 6, 2)).float()
        offsets = torch.randn(2, 4, 6)
        with torch.inference_mode():
            full, final, _ = model(events, offsets)
            first, state, _ = model(events[:, :2], offsets[:, :2])
            last, end, _ = model(events[:, 2:], offsets[:, 2:], state)
            torch.testing.assert_close(torch.cat([first, last], 1), full)
            for value, expected in zip(end, final):
                torch.testing.assert_close(value, expected)
            changed = events.clone()
            changed[:, 2:] = 0
            other, _, _ = model(changed, offsets)
            torch.testing.assert_close(other[:, :2], full[:, :2])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            args = fixture_args(root) + [
                "--architecture", "compact_adaptive", "--post-impact-seconds", "1.2",
                "--raw-win", "120", "--stride", "30", "--sequence-length", "2",
                "--recurrent-batch-size", "2", "--no-progress",
            ]
            with patch("train_kfall.get_device", return_value=torch.device("cpu")), contextlib.redirect_stdout(io.StringIO()):
                main(args)
                run = next((root / "runs").iterdir())
                metrics = evaluate_run(run, device="cpu", output_dir=root / "replay")
            np.testing.assert_allclose(
                np.load(run / "metrics" / "test_best" / "probabilities.npy"),
                np.load(root / "replay" / "probabilities.npy"), atol=1e-6,
            )
            self.assertEqual(metrics["architecture"], "compact_adaptive")
            self.assertEqual(metrics["sequence_resets"], 3)
            saved = torch.load(run / "checkpoints" / "best_model.pt", weights_only=True)
            self.assertIn("temporal.adaptation_logit", saved["model_state_dict"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_parallel_streams_amp_and_worker_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            args = fixture_args(root) + [
                "--architecture", "compact_adaptive", "--post-impact-seconds", "1.2",
                "--raw-win", "120", "--stride", "30", "--sequence-length", "2",
                "--recurrent-batch-size", "2", "--cache-format", "pt",
                "--no-progress", "--amp", "--workers", "2",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
                run = next((root / "runs").iterdir())
                evaluate_run(run, device="cuda:0", workers=2, output_dir=root / "replay")
            np.testing.assert_allclose(
                np.load(run / "metrics" / "test_best" / "probabilities.npy"),
                np.load(root / "replay" / "probabilities.npy"), atol=1e-6,
            )

    def test_parallel_streams_match_serial_replay_without_padding_or_state_leakage(self):
        torch.manual_seed(7)
        n = 14
        files = np.repeat([0, 1, 2, 3], [5, 3, 2, 4])
        starts = np.concatenate([np.arange(length) * 10 for length in (5, 3, 2, 4)])
        split = {
            "x_ev": np.random.default_rng(7).integers(0, 2, (n, 256, 6, 2), dtype=np.uint8),
            "offset": np.ones((n, 6), dtype=np.float32), "y_h1": np.arange(n) % 4,
            "y_h2": np.zeros(n, dtype=np.float32), "src_h2": np.zeros(n, dtype=np.uint8),
            "w_h2": np.zeros(n, dtype=np.float32), "file_idx": files,
            "start": starts, "end_frame": starts + 120,
            "subject": np.asarray([f"SA{file:02d}" for file in files]),
        }
        cfg = get_args(["--architecture", "compact_recurrent", "--sequence-length", "2",
                        "--stride", "10", "--workers", "0", "--no-progress"])
        dev = torch.device("cpu")
        model = build_model(cfg, dev)
        dataset = KFallTrainingDataset(split)
        serial = run_epoch(model, make_sequence_loader(dataset, split, cfg, dev), dev, cfg,
                           collect_predictions=True)
        cfg.recurrent_batch_size = 3
        parallel = run_epoch(model, make_sequence_loader(dataset, split, cfg, dev), dev, cfg,
                             collect_predictions=True)
        np.testing.assert_array_equal(parallel["window_indices"], np.arange(n))
        np.testing.assert_allclose(parallel["probabilities"], serial["probabilities"], atol=1e-6)
        self.assertEqual(parallel["n"], n)
        self.assertEqual(parallel["sequence_resets"], 4)
        self.assertAlmostEqual(parallel["recurrent_spike_rate"], serial["recurrent_spike_rate"])

    def test_equation_spikes_gradients_and_explicit_reset(self):
        cell = RecurrentLIF(1, 1, beta=0.5, threshold=0.5)
        with torch.no_grad():
            cell.input.weight.fill_(1)
            cell.input.bias.zero_()
            cell.recurrent.weight.fill_(0.25)
        x = torch.tensor([[[0.8], [0.1], [0.6]]], requires_grad=True)
        spikes, state = cell(x)
        torch.testing.assert_close(spikes, torch.tensor([[[1.0], [0.0], [1.0]]]))
        # 0.8 -> 0.5*0.8 + 0.1 + 0.25 - 0.5 = 0.25 -> 0.725.
        torch.testing.assert_close(state.membrane, torch.tensor([[0.725]]))
        spikes.sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertGreater(float(cell.recurrent.weight.grad.abs().sum()), 0)
        self.assertGreater(float(cell.input.weight.grad.abs().sum()), 0)
        detached = state.detach()
        self.assertIsNone(detached.membrane.grad_fn)
        _, fresh = cell(x.detach())
        torch.testing.assert_close(fresh.membrane, state.membrane)

    def test_parameter_budget_chunked_inference_and_bn_folding(self):
        torch.manual_seed(7)
        for enabled in (True, False):
            with self.subTest(use_offset=enabled):
                model = CompactRecurrentSCN(use_offset=enabled, p_drop=0).eval()
                count = sum(p.numel() for p in model.parameters())
                self.assertTrue(20000 <= count <= 30000, count)
                self.assertEqual(hasattr(model.encoder, "offset_proj_spk"), enabled)
                events = torch.randint(0, 2, (1, 4, 256, 6, 2)).float()
                offset = torch.randn(1, 4, 6)
                with torch.inference_mode():
                    full, final, spikes = model(events, offset)
                    first, state, _ = model(events[:, :2], offset[:, :2])
                    second, chunk_final, _ = model(events[:, 2:], offset[:, 2:], state)
                    torch.testing.assert_close(torch.cat([first, second], dim=1), full)
                    torch.testing.assert_close(chunk_final.membrane, final.membrane)
                    future_changed = events.clone()
                    future_changed[:, 2:] = 0
                    other, _, _ = model(future_changed, offset)
                    torch.testing.assert_close(other[:, :2], full[:, :2])
                    folded = model.fuse_batch_norm()
                    actual, _, _ = folded(events, offset)
                    torch.testing.assert_close(actual, full)
                    self.assertTrue(torch.isin(spikes, torch.tensor([0., 1.])).all())
                with self.assertRaisesRegex(ValueError, "eval"):
                    model.train().fuse_batch_norm()

    def test_recordings_and_gaps_are_never_joined_or_shuffled_internally(self):
        split = {
            "file_idx": np.array([1, 0, 0, 0, 1]),
            "start": np.array([10, 0, 10, 50, 0]),
            "end_frame": np.array([130, 120, 130, 170, 120]),
            "subject": np.array(["SA07", "SA06", "SA06", "SA06", "SA07"]),
        }
        groups = recording_sequences(split, 10)
        self.assertEqual([group.tolist() for group in groups], [[1, 2], [3], [4, 0]])
        self.assertEqual(list(RecordingChunkSampler(groups, 1, False)), [[1], [2], [3], [4], [0]])
        for _ in range(3):
            batches = list(RecordingChunkSampler(groups, 1, True))
            flattened = [item for batch in batches for item in batch]
            self.assertLess(flattened.index(1), flattened.index(2))
            self.assertLess(flattened.index(4), flattened.index(0))
        split["start"][2] = 0
        with self.assertRaisesRegex(ValueError, "unique"):
            recording_sequences(split, 10)

    def test_train_checkpoint_evaluate_and_temporal_state_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            # Add more chronological decisions to exercise state across chunks.
            args = fixture_args(root) + [
                "--architecture", "compact_recurrent", "--post-impact-seconds", "1.2",
                "--raw-win", "120", "--stride", "30", "--sequence-length", "2",
                "--cache-format", "pt",
                "--recurrent-batch-size", "2", "--no-progress",
            ]
            with patch("train_kfall.get_device", return_value=torch.device("cpu")), contextlib.redirect_stdout(io.StringIO()):
                main(args)
            run = next((root / "runs").iterdir())
            config = json.loads((run / "config.json").read_text())
            self.assertEqual(config["architecture"], "compact_recurrent")
            checkpoint = torch.load(run / "checkpoints" / "best_model.pt", weights_only=True)
            cfg = get_args(args)
            build_model(cfg, torch.device("cpu")).load_state_dict(checkpoint["model_state_dict"], strict=True)
            original = np.load(run / "metrics" / "test_best" / "probabilities.npy")
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_run(run, device="cpu", output_dir=root / "evaluated")
            actual = np.load(root / "evaluated" / "probabilities.npy")
            np.testing.assert_allclose(actual, original, atol=1e-6)
            self.assertEqual(result["architecture"], "compact_recurrent")
            self.assertTrue(0 <= result["recurrent_spike_rate"] <= 1)
            self.assertEqual(result["sequence_resets"], 3)
            with self.assertRaisesRegex(ValueError, "preceding state"):
                evaluate_run(run, device="cpu", window=0)


if __name__ == "__main__":
    unittest.main()
