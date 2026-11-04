"""Frozen SCN features for training recurrent memory without replaying IMU windows."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
import torch
from tqdm.auto import tqdm


_KIND = 'kfall_frozen_backbone_features'
_VERSION = 1
_FILES = {'metadata.json', 'features.npy', 'indices.npy'}


def _json(value):
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_json(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _backbone_digest(backbone):
    # LIF dynamics, convolution strides and BN eps are not in state_dict.
    attributes = ('n_channels', 'merge_polarities', 'event_scale', 'len_sw', 'time_steps',
        'offset_dim', 'offset_hidden', 'offset_scale', 'out_dim', 'fused_dim', 'tau',
        'thresh', 'gamma', 'smooth', 'soft_reset', 'region', 'stride', 'padding', 'dilation',
        'groups', 'eps', 'track_running_stats', 'ceil_mode', 'kernel_size', 'p',
        'padding_mode', 'in_features', 'out_features')
    structure = {name: {'class': type(module).__module__ + '.' + type(module).__qualname__,
                       **{key: getattr(module, key) for key in attributes if hasattr(module, key)}}
                 for name, module in backbone.named_modules()}
    digest = hashlib.sha256(_digest(structure).encode())
    for name, tensor in sorted(backbone.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(_digest((name, str(tensor.dtype), list(tensor.shape))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _identity(model, episodes, split_name):
    if split_name not in {'train', 'val'}:
        raise ValueError('Frozen feature preparation is restricted to train/val; evaluate test with the full model.')
    allowed = episodes.metadata.get('split_file_indices', {}).get(split_name)
    allowed_set = set(allowed or ())
    if allowed is None or any(ep.file_index not in allowed_set for ep in episodes.episodes):
        raise ValueError(f'Feature episodes must belong only to the {split_name} recording partition.')
    if not episodes.episodes:
        raise ValueError('Cannot extract an empty feature split.')
    selected = np.sort(np.concatenate([ep.indices for ep in episodes.episodes]).astype(np.int64))
    if (selected.ndim != 1 or not len(selected) or len(np.unique(selected)) != len(selected)
            or np.any(selected < 0) or np.any(selected >= len(episodes.arrays['y_h1']))):
        raise ValueError('Feature endpoints must have unique, valid original cache indices.')
    selection = [dict(file_index=ep.file_index, indices=ep.indices, targets=ep.targets,
                      valid=ep.valid, frames=ep.frames) for ep in episodes.episodes]
    # Runtime loader settings do not affect the stored signal representation.
    config = {key: value for key, value in episodes.metadata['config'].items()
              if key not in {'batch_size', 'num_workers', 'pin_memory', 'verbose_every'}}
    data = {key: episodes.metadata.get(key) for key in ('version', 'dataset', 'source_signature',
            'split_strategy', 'split_subjects', 'split_file_indices', 'channels', 'units',
            'fall_label_rule', 'recordings')}
    data['config'] = config
    identity = dict(kind=_KIND, version=_VERSION, split=split_name,
                    backbone_sha256=_backbone_digest(model.backbone), data_sha256=_digest(data),
                    selection_sha256=_digest(selection), feature_dim=int(model.backbone.fused_dim),
                    window_count=len(selected), extraction_precision='float32',
                    subjects=sorted({episodes.metadata['recordings'][ep.file_index]['subject']
                                     for ep in episodes.episodes}))
    identity['identity_sha256'] = _digest(identity)
    return selected, identity


def _check_destination(destination, arrays):
    if destination.is_symlink():
        raise ValueError('Use a dedicated feature directory, not a symbolic link.')
    for value in arrays.values():
        filename = getattr(value, 'filename', None)
        if filename is None:
            continue
        parent = Path(filename).resolve().parent
        source = parent.parent if parent.name in {'train', 'val', 'test'} else parent
        if destination.resolve().is_relative_to(source):
            raise ValueError('Store frozen features outside the read-only event cache.')


class FeatureEpisodeSet:
    """Original episode identities, backed by a compact N-selected by F matrix."""
    def __init__(self, episodes, features, indices, cache_dir, manifest):
        self.source = episodes
        self.arrays, self.metadata, self.episodes = episodes.arrays, episodes.metadata, episodes.episodes
        self.stride, self.raw_win = episodes.stride, episodes.raw_win
        self.features, self.indices = features, indices
        self.cache_dir, self.manifest = Path(cache_dir), manifest

    @property
    def window_count(self):
        return sum(len(ep.indices) for ep in self.episodes)

    def groups(self, batch_size, *, shuffle=False, seed=0):
        order = np.arange(len(self.episodes))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for begin in range(0, len(order), batch_size):
            yield [self.episodes[int(index)] for index in order[begin:begin + batch_size]]

    def chunks(self, episodes, sequence_length):
        if sequence_length < 1:
            raise ValueError('Sequence length must be positive.')
        maximum = max(len(ep.indices) for ep in episodes)
        for start in range(0, maximum, sequence_length):
            length = min(sequence_length, maximum - start)
            batch = len(episodes)
            features = np.zeros((batch, length, self.features.shape[1]), dtype=np.float32)
            targets = np.full((batch, length), -1, dtype=np.int64)
            present = np.zeros((batch, length), dtype=bool)
            valid = np.zeros_like(present)
            indices = np.full((batch, length), -1, dtype=np.int64)
            previous_target = np.full(batch, -1, dtype=np.int64)
            previous_valid = np.zeros(batch, dtype=bool)
            for row, ep in enumerate(episodes):
                selected = ep.indices[start:start + length]
                count = len(selected)
                if count:
                    positions = np.searchsorted(self.indices, selected)
                    if np.any(positions >= len(self.indices)) or not np.array_equal(self.indices[positions], selected):
                        raise ValueError('Episode endpoint is missing from the frozen feature cache.')
                    features[row, :count] = self.features[positions]
                    targets[row, :count] = ep.targets[start:start + count]
                    present[row, :count] = True
                    valid[row, :count] = ep.valid[start:start + count]
                    indices[row, :count] = selected
                if start and start <= len(ep.indices):
                    previous_target[row] = ep.targets[start - 1]
                    previous_valid[row] = ep.valid[start - 1]
            yield dict(features=torch.from_numpy(features), targets=torch.from_numpy(targets),
                       present=torch.from_numpy(present), valid=torch.from_numpy(valid), indices=indices,
                       previous_target=torch.from_numpy(previous_target), previous_valid=torch.from_numpy(previous_valid))


def _load(episodes, destination, selected, identity):
    try:
        manifest = json.loads((destination / 'metadata.json').read_text())
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError('Frozen feature cache is stale: backbone, data or endpoint provenance changed.')
        features = np.load(destination / 'features.npy', mmap_mode='r', allow_pickle=False)
        indices = np.load(destination / 'indices.npy', mmap_mode='r', allow_pickle=False)
        if (features.dtype != np.float32 or features.shape != (len(selected), identity['feature_dim'])
                or indices.dtype != np.int64 or not np.array_equal(indices, selected)):
            raise ValueError('Frozen feature cache shape, dtype or endpoint indices are invalid.')
        for begin in range(0, len(features), 4096):
            if not np.isfinite(features[begin:begin + 4096]).all():
                raise ValueError('Frozen feature cache contains nonfinite values.')
        if any(_file_digest(destination / name) != manifest.get(name + '_sha256')
               for name in ('features.npy', 'indices.npy')):
            raise ValueError('Frozen feature cache contents differ from their saved checksums.')
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError('Frozen feature cache is incomplete; rebuild with refresh=True.') from exc
    return FeatureEpisodeSet(episodes, features, indices, destination, manifest)


def extract_feature_cache(model, episodes, output_dir, *, batch_size=128, device,
                          split_name='train', no_progress=False, refresh=False):
    """Extract or reuse frozen train/val features, without running recurrent state.

    Frozen features use eval-mode FP32 and are tied to both the backbone and
    original endpoint identities. A stale or partial cache raises unless the
    caller explicitly requests ``refresh=True``. Input arrays stay read-only.
    """
    if batch_size < 1:
        raise ValueError('Feature extraction batch size must be positive.')
    destination = Path(output_dir).expanduser()
    _check_destination(destination, episodes.arrays)
    destination = destination.resolve()
    selected, identity = _identity(model, episodes, split_name)
    if destination.exists() and not refresh:
        return _load(episodes, destination, selected, identity)
    if destination.exists() and (not destination.is_dir() or
            any(path.name not in _FILES for path in destination.iterdir())):
        raise ValueError('Feature output must be a dedicated cache directory.')
    dev = torch.device(device)
    if dev.type == 'cuda' and dev.index is None:
        dev = torch.device('cuda', torch.cuda.current_device())
    for parameter in model.backbone.parameters():
        if parameter.dtype != torch.float32 or parameter.device != dev:
            raise ValueError('Extract features with an FP32 backbone on the requested device.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.', dir=destination.parent))
    modes = [(module, module.training) for module in model.modules()]
    progress = None
    try:
        features = np.lib.format.open_memmap(temporary / 'features.npy', mode='w+', dtype=np.float32,
                                            shape=(len(selected), identity['feature_dim']))
        np.save(temporary / 'indices.npy', selected, allow_pickle=False)
        model.eval()
        progress = tqdm(total=math.ceil(len(selected) / batch_size), desc=f'Features {split_name}',
                        unit='batch', mininterval=.5, dynamic_ncols=True,
                        disable=True if no_progress else None)
        with torch.inference_mode(), torch.autocast(device_type=dev.type, enabled=False):
            for begin in range(0, len(selected), batch_size):
                ids = selected[begin:begin + batch_size]
                events = torch.from_numpy(np.asarray(episodes.arrays['x_ev'][ids])).to(dev).float()
                offsets = torch.from_numpy(np.asarray(episodes.arrays['offset'][ids])).to(dev).float()
                value = model.backbone.extract_features(events, offsets)
                if value.shape != (len(ids), identity['feature_dim']) or value.dtype != torch.float32:
                    raise ValueError('Backbone returned unexpected feature dimensions or precision.')
                value = value.cpu().numpy()
                if not np.isfinite(value).all():
                    raise ValueError('Backbone produced nonfinite frozen features.')
                features[begin:begin + len(ids)] = value
                progress.update(1)
        features.flush()
        del features
        manifest = {**identity, **{name + '_sha256': _file_digest(temporary / name)
                                  for name in ('features.npy', 'indices.npy')}}
        (temporary / 'metadata.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # A concurrent extraction may have completed while this one ran.
        # Reuse a compatible committed cache; never replace it implicitly.
        if destination.exists() and not refresh:
            return _load(episodes, destination, selected, identity)
        previous = None
        if destination.exists():
            previous = destination.with_name('.' + destination.name + '.old.' + uuid.uuid4().hex)
            os.replace(destination, previous)
        try:
            os.replace(temporary, destination)
        except BaseException:
            if previous is not None:
                os.replace(previous, destination)
            raise
        if previous is not None:
            shutil.rmtree(previous)
    finally:
        if progress is not None:
            progress.close()
        for module, training in modes:
            module.training = training
        if temporary.exists():
            shutil.rmtree(temporary)
    return _load(episodes, destination, selected, identity)
