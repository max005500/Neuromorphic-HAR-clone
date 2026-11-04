"""Frozen feature provenance, independent extraction and chronological replay."""
import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from loaders.kfall_features import extract_feature_cache
from loaders.kfall_sequences import EpisodeSet


class ToyBackbone(torch.nn.Module):
    fused_dim = 7
    tau = .75

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(18, self.fused_dim)
        self.bn = torch.nn.BatchNorm1d(self.fused_dim)
        self.drop = torch.nn.Dropout(.5)
        self.grad_modes = []

    def extract_features(self, events, offsets):
        self.grad_modes.append(torch.is_grad_enabled())
        value = torch.cat((events.mean(1).flatten(1), offsets), 1)
        return self.drop(self.bn(self.fc(value)))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ToyBackbone()
        self.head = torch.nn.Linear(7, 4)


class NoEventReads:
    def __getitem__(self, key):
        raise AssertionError('Stage two must never read source events or offsets.')


def episodes():
    metadata = dict(version=5, dataset='kfall', source_signature={'sha256': 'source'},
        config=dict(label_policy='temporal', post_impact_seconds=1.2, raw_win=2, stride=2,
                    theta_sd=[.1] * 6),
        recordings=[dict(T=10, subject='SA01', label_id=1),
                    dict(T=6, subject='SA02', label_id=2),
                    dict(T=10, subject='SA03', label_id=1)],
        split_file_indices=dict(train=[0, 1], val=[2], test=[]),
        split_subjects=dict(train=['SA01', 'SA02'], val=['SA03'], test=[]))
    random = np.random.default_rng(0)
    arrays = dict(file_idx=np.array([0, 0, 0, 0, 0, 1, 1, 1]),
        start=np.array([0, 2, 3, 4, 6, 0, 2, 4]),
        end_frame=np.array([2, 4, 5, 6, 8, 2, 4, 6]),
        subject=np.array(['SA01'] * 5 + ['SA02'] * 3),
        y_h1=np.array([2, 1, 0, 0, 3, 2, 2, 1]),
        x_ev=random.integers(0, 2, size=(8, 16, 6, 2), dtype=np.uint8),
        offset=random.normal(size=(8, 6)).astype(np.float32))
    return EpisodeSet(arrays, metadata)


class FrozenFeatureTests(unittest.TestCase):
    def extract(self, model, source, path, **kwargs):
        return extract_feature_cache(model, source, path, device='cpu', batch_size=3,
                                     split_name='train', no_progress=True, **kwargs)

    def test_independent_window_features_are_compact_readonly_and_restore_modes(self):
        source, model = episodes(), ToyModel().train()
        model.backbone.bn.eval()  # Preserve an intentionally mixed mode tree.
        model.backbone.fc.weight.requires_grad_(False)
        before_modes = [module.training for module in model.modules()]
        before_grad = [parameter.requires_grad for parameter in model.parameters()]
        snapshot = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as temporary:
            cached = self.extract(model, source, Path(temporary) / 'features')
            selected = np.array([0, 1, 3, 4, 5, 6, 7])
            np.testing.assert_array_equal(cached.indices, selected)
            self.assertIsInstance(cached.features, np.memmap)
            self.assertEqual(cached.features.shape, (7, 7))
            self.assertFalse(cached.features.flags.writeable)
            self.assertEqual(cached.manifest['extraction_precision'], 'float32')
            self.assertEqual(cached.manifest['subjects'], ['SA01', 'SA02'])
            self.assertEqual(model.backbone.grad_modes, [False, False, False])
            self.assertEqual([module.training for module in model.modules()], before_modes)
            self.assertEqual([parameter.requires_grad for parameter in model.parameters()], before_grad)
            model.eval()
            with torch.no_grad():
                expected = model.backbone.extract_features(
                    torch.from_numpy(source.arrays['x_ev'][selected]).float(),
                    torch.from_numpy(source.arrays['offset'][selected]))
            np.testing.assert_allclose(cached.features, expected.numpy(), rtol=1e-5, atol=1e-6)
            # The comparison above deliberately changed modes after extraction.
            for module, training in zip(model.modules(), before_modes):
                module.training = training
        self.assertEqual([module.training for module in model.modules()], before_modes)
        self.assertEqual([parameter.requires_grad for parameter in model.parameters()], before_grad)
        for key, value in snapshot.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_stage_two_chunks_match_masks_padding_and_cross_chunk_targets_without_events(self):
        source = episodes()
        # Include a masked but physically present window in state progression.
        ep = source.episodes[0]
        source.episodes[0] = replace(ep, valid=np.array([True, False, True, True]))
        with tempfile.TemporaryDirectory() as temporary:
            cached = self.extract(ToyModel(), source, Path(temporary) / 'features')
            source_groups = list(source.groups(2, shuffle=True, seed=10))
            cached_groups = list(cached.groups(2, shuffle=True, seed=10))
            self.assertEqual([[ep.file_index for ep in group] for group in source_groups],
                             [[ep.file_index for ep in group] for group in cached_groups])
            expected = list(source.chunks(source_groups[0], 3))
            source.arrays['x_ev'] = source.arrays['offset'] = NoEventReads()
            actual = list(cached.chunks(cached_groups[0], 3))
            for before, after in zip(expected, actual):
                self.assertNotIn('events', after)
                self.assertNotIn('offsets', after)
                for key in ('targets', 'present', 'valid', 'indices', 'previous_target', 'previous_valid'):
                    np.testing.assert_array_equal(before[key], after[key])
                mask = after['present'].numpy()
                ids = after['indices'][mask]
                positions = np.searchsorted(cached.indices, ids)
                np.testing.assert_array_equal(after['features'].numpy()[mask], cached.features[positions])
                self.assertTrue(np.all(after['features'].numpy()[~mask] == 0))

    def test_valid_cache_reuses_without_backbone_and_stale_provenance_is_rejected(self):
        source, model = episodes(), ToyModel()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'features'
            self.extract(model, source, path)
            with patch.object(model.backbone, 'extract_features', side_effect=AssertionError('Must reuse')):
                reused = self.extract(model, source, path)
            self.assertEqual(reused.window_count, source.window_count)
            variants = []
            weights = copy.deepcopy(model)
            with torch.no_grad():
                weights.backbone.fc.weight.add_(.1)
            variants.append((weights, source))
            buffers = copy.deepcopy(model)
            buffers.backbone.bn.running_mean.add_(1)
            variants.append((buffers, source))
            dynamics = copy.deepcopy(model)
            dynamics.backbone.tau = .9
            variants.append((dynamics, source))
            data = copy.deepcopy(source)
            data.metadata['source_signature']['sha256'] = 'changed'
            variants.append((model, data))
            masks = copy.deepcopy(source)
            masks.episodes[0] = replace(masks.episodes[0], valid=np.array([True, False, True, True]))
            variants.append((model, masks))
            for changed_model, changed_source in variants:
                with self.subTest(model=changed_model, source=changed_source), self.assertRaisesRegex(ValueError, 'stale'):
                    self.extract(changed_model, changed_source, path)
            refreshed = self.extract(weights, source, path, refresh=True)
            self.assertNotEqual(refreshed.manifest['backbone_sha256'], reused.manifest['backbone_sha256'])

    def test_event_memmaps_unchanged_and_output_inside_event_cache_rejected(self):
        source, model = episodes(), ToyModel()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = root / 'event_cache/train'
            events.mkdir(parents=True)
            files = []
            for key in ('x_ev', 'offset'):
                path = events / (key + '.npy')
                np.save(path, source.arrays[key])
                source.arrays[key] = np.load(path, mmap_mode='r')
                files.append(path)
            before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
            self.extract(model, source, root / 'features')
            self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in files])
            with self.assertRaisesRegex(ValueError, 'outside'):
                self.extract(model, source, root / 'event_cache/features')

    def test_test_split_and_mixed_recordings_are_rejected_before_backbone(self):
        source, model = episodes(), ToyModel()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'features'
            with self.assertRaisesRegex(ValueError, 'train/val'):
                extract_feature_cache(model, source, path, device='cpu', split_name='test')
            mixed = copy.deepcopy(source)
            mixed.episodes[0] = replace(mixed.episodes[0], file_index=2)
            with self.assertRaisesRegex(ValueError, 'partition'):
                self.extract(model, mixed, path)
            self.assertEqual(model.backbone.grad_modes, [])

    def test_failure_leaves_no_partial_cache_and_corrupt_contents_are_rejected(self):
        source, model = episodes(), ToyModel().train()
        before_modes = [module.training for module in model.modules()]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'features'
            with patch.object(model.backbone, 'extract_features', side_effect=RuntimeError('interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    self.extract(model, source, path)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])
            self.assertEqual([module.training for module in model.modules()], before_modes)
            self.extract(model, source, path)
            values = np.load(path / 'features.npy', mmap_mode='r+')
            values[0, 0] += .5
            values.flush()
            del values
            with self.assertRaisesRegex(ValueError, 'checksums'):
                self.extract(model, source, path)


if __name__ == '__main__':
    unittest.main()
