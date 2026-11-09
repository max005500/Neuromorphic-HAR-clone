"""Tensor cache parity, reuse, and source invalidation."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from loaders.kfall_tensor_cache import load_or_build_tensor_dataset
from test_kfall_training import fixture_args, write_fixture
from train_kfall import KFallTrainingDataset, build_data, get_args, main
from evaluate_kfall import evaluate_run


class KFallTensorCacheTests(unittest.TestCase):
    def test_tensor_views_preserve_all_labels_and_reuse_saved_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            cfg = get_args(fixture_args(root) + ["--cache-format", "pt"])
            with contextlib.redirect_stdout(io.StringIO()):
                data = build_data(cfg, torch.device("cpu"))
            cache = root / "cache" / "kfall_event_loader_default"
            for name, split in data["splits"].items():
                dataset = data["loaders"][name].dataset
                original = KFallTrainingDataset(split)
                for index in range(len(dataset)):
                    self.assertEqual(dataset[index]["x"].dtype, torch.uint8)
                    for key, expected in original[index].items():
                        torch.testing.assert_close(dataset[index][key].to(expected.dtype), expected)
                payload = torch.load(cache / f"{name}.pt", weights_only=True, mmap=True)
                self.assertEqual(payload["subjects"], split["subject"].tolist())
                self.assertEqual(payload["metadata"]["split_subjects"], data["metadata"]["split_subjects"])
                for key, array in split.items():
                    if key != "subject":
                        np.testing.assert_array_equal(payload["tensors"][key].numpy(), array)
                with patch("loaders.kfall_tensor_cache.torch.save", side_effect=AssertionError("cache rebuilt")):
                    with contextlib.redirect_stdout(io.StringIO()):
                        reloaded = load_or_build_tensor_dataset(cache, name, split)
                self.assertEqual(reloaded.tensors["x_ev"].shape, dataset.tensors["x_ev"].shape)
                item = reloaded[0]["x"]
                self.assertEqual(item.data_ptr(), reloaded.tensors["x_ev"][0].data_ptr())

    def test_changed_source_arrays_rebuild_pt_instead_of_using_stale_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            cfg = get_args(fixture_args(root) + ["--cache-format", "pt"])
            with contextlib.redirect_stdout(io.StringIO()):
                data = build_data(cfg, torch.device("cpu"))
            cache = root / "cache" / "kfall_event_loader_default"
            split = dict(data["splits"]["train"])
            split["y_h1"] = (split["y_h1"] + 1) % 3
            np.save(cache / "train" / "y_h1.npy", split["y_h1"])
            with patch("loaders.kfall_tensor_cache.torch.save", wraps=torch.save) as save:
                with contextlib.redirect_stdout(io.StringIO()):
                    dataset = load_or_build_tensor_dataset(cache, "train", split)
            self.assertEqual(save.call_count, 1)
            np.testing.assert_array_equal(dataset.tensors["y_h1"].numpy(), split["y_h1"])

    def test_pt_training_predictions_match_npy_and_saved_checkpoint_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            with patch("train_kfall.get_device", return_value=torch.device("cpu")), contextlib.redirect_stdout(io.StringIO()):
                main(fixture_args(root) + ["--cache-format", "npy"])
                main(fixture_args(root) + ["--cache-format", "pt"])
            runs = sorted((root / "runs").iterdir())
            original, tensor_run = runs
            for name in ("y_true.npy", "y_pred.npy"):
                np.testing.assert_array_equal(np.load(original / "metrics" / name), np.load(tensor_run / "metrics" / name))
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = evaluate_run(tensor_run, device="cpu")
            saved = json.loads((tensor_run / "metrics" / "test_metrics.json").read_text())
            self.assertAlmostEqual(metrics["loss"], saved["loss"], places=6)


if __name__ == "__main__":
    unittest.main()
