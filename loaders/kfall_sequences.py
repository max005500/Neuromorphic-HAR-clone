"""Chronological KFall episode batches with explicit padding and TBPTT chunks."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class Episode:
    file_index: int
    indices: np.ndarray
    targets: np.ndarray
    valid: np.ndarray
    frames: np.ndarray


class EpisodeSet:
    def __init__(self, arrays, metadata, *, boundary_ignore_seconds=0.):
        if not np.isfinite(boundary_ignore_seconds) or boundary_ignore_seconds < 0:
            raise ValueError('Boundary tolerance must be nonnegative and finite.')
        self.arrays, self.metadata = arrays, metadata
        cfg = metadata['config']
        if cfg.get('label_policy', 'legacy') == 'legacy' or cfg.get('post_impact_seconds', 0) <= 0:
            raise ValueError('Temporal experiments require complete four-class labeling (temporal/motion/spikes).')
        self.stride, self.raw_win = int(cfg['stride']), int(cfg['raw_win'])
        context = cfg.get('endpoint_context') or self.raw_win
        self.episodes = []
        for fi in np.unique(arrays['file_idx']):
            indices = np.flatnonzero(arrays['file_idx'] == fi)
            ends = arrays['start'][indices] + self.raw_win - 1
            # Remove the annotation-dependent extra endpoints of the old loader.
            indices = indices[(ends >= context - 1) & ((ends - (context - 1)) % self.stride == 0)]
            indices = indices[np.argsort(arrays['start'][indices], kind='stable')]
            if not len(indices):
                continue
            if np.any(np.diff(arrays['start'][indices]) != self.stride):
                raise ValueError('Temporal episodes require a complete uniform window grid, without skipped phases.')
            record = metadata['recordings'][int(fi)]
            if not np.all(arrays['subject'][indices] == record['subject']):
                raise ValueError('An episode contains windows from different subjects.')
            targets = np.asarray(arrays['y_h1'][indices], dtype=np.int64)
            frames = np.asarray(arrays['end_frame'][indices], dtype=np.int64)
            valid = np.isin(targets, [0, 1, 2, 3])
            if boundary_ignore_seconds > 0:
                boundaries = np.asarray([row[0] for row in record.get('frame_label_intervals', [])[1:]])
                if len(boundaries):
                    distances = np.abs(frames[:, None] - boundaries[None, :]).min(1)
                    valid &= distances >= boundary_ignore_seconds * 100
            self.episodes.append(Episode(int(fi), indices, targets, valid, frames))
        if not self.episodes:
            raise ValueError('No complete episodes remain after selecting uniform endpoints.')

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
        maximum = max(len(ep.indices) for ep in episodes)
        shape = self.arrays['x_ev'].shape[1:]
        for start in range(0, maximum, sequence_length):
            length = min(sequence_length, maximum - start)
            batch = len(episodes)
            events = np.zeros((batch, length, *shape), dtype=np.uint8)
            offsets = np.zeros((batch, length, 6), dtype=np.float32)
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
                    events[row, :count] = self.arrays['x_ev'][selected]
                    offsets[row, :count] = self.arrays['offset'][selected]
                    targets[row, :count] = ep.targets[start:start + count]
                    present[row, :count] = True
                    valid[row, :count] = ep.valid[start:start + count]
                    indices[row, :count] = selected
                if start and start <= len(ep.indices):
                    previous_target[row] = ep.targets[start - 1]
                    previous_valid[row] = ep.valid[start - 1]
            yield dict(events=torch.from_numpy(events), offsets=torch.from_numpy(offsets),
                       targets=torch.from_numpy(targets), present=torch.from_numpy(present),
                       valid=torch.from_numpy(valid), indices=indices,
                       previous_target=torch.from_numpy(previous_target),
                       previous_valid=torch.from_numpy(previous_valid))


def training_statistics(episodes, *, laplace=1.):
    """Only the training EpisodeSet is supplied: no validation/test estimation."""
    if not np.isfinite(laplace) or laplace <= 0:
        raise ValueError('Laplace smoothing must be positive.')
    allowed = set(episodes.metadata['split_file_indices']['train'])
    if any(ep.file_index not in allowed for ep in episodes.episodes):
        raise ValueError('Transition statistics may only use training recordings.')
    counts = np.zeros(4, dtype=np.int64)
    transitions = np.zeros((4, 4), dtype=np.int64)
    boundary_positive = boundary_total = 0
    for ep in episodes.episodes:
        counts += np.bincount(ep.targets[ep.valid], minlength=4)
        mask = ep.valid[1:] & ep.valid[:-1]
        left, right = ep.targets[:-1][mask], ep.targets[1:][mask]
        np.add.at(transitions, (left, right), 1)
        boundary_positive += int(np.sum(left != right))
        boundary_total += len(left)
    probability = (transitions + laplace) / (transitions.sum(1, keepdims=True) + 4 * laplace)
    costs = -np.log(probability)
    np.fill_diagonal(costs, 0.)
    costs /= max(float(costs.max()), 1e-12)
    return dict(class_counts=counts.tolist(), transition_counts=transitions.tolist(),
                transition_costs=costs.tolist(), laplace=laplace,
                boundary_positive=boundary_positive, boundary_pairs=boundary_total,
                boundary_pos_weight=(boundary_total - boundary_positive + 1) / (boundary_positive + 1),
                source_split='train',
                subjects=sorted({episodes.metadata['recordings'][ep.file_index]['subject'] for ep in episodes.episodes}))
