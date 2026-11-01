"""Compact binary host batches retain exact model inputs and cache immutability."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_kfall import KFallTrainingDataset
from train_ucihar import unpack_batch


class CompactKFallBatchTests(unittest.TestCase):
    def test_read_only_events_are_compact_until_device_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            rng = np.random.default_rng(42)
            original = rng.integers(0, 2, size=(5, 32, 6, 2), dtype=np.uint8)
            path = Path(temporary) / 'events.npy'
            np.save(path, original)
            mapped = np.load(path, mmap_mode='r')
            offsets = rng.normal(size=(5, 6)).astype(np.float32)
            targets = np.array([0, 1, 2, 3, 1], dtype=np.int64)
            split = dict(x_ev=mapped, offset=offsets, y_h1=targets,
                         y_h2=np.full(5, .5, dtype=np.float32),
                         src_h2=np.zeros(5, dtype=np.uint8), w_h2=np.zeros(5, dtype=np.float32))
            dataset = KFallTrainingDataset(split, n_classes=4)
            position = 0
            for batch in DataLoader(dataset, batch_size=3, shuffle=False):
                self.assertEqual(batch['x'].dtype, torch.uint8)
                x, offset, y = unpack_batch(batch, torch.device('cpu'))
                size = len(y)
                self.assertEqual(x.dtype, torch.float32)
                self.assertEqual(batch['x'].numel() * batch['x'].element_size() * 4,
                                 x.numel() * x.element_size())
                np.testing.assert_array_equal(x.numpy(), original[position:position + size])
                np.testing.assert_array_equal(offset.numpy(), offsets[position:position + size])
                np.testing.assert_array_equal(y.numpy(), targets[position:position + size])
                position += size
            self.assertEqual(position, 5)
            # Mutating an item must never mutate the original read-only cache.
            dataset[0]['x'].zero_()
            np.testing.assert_array_equal(mapped, original)
