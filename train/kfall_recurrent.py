"""Chronological, recording-isolated truncated BPTT for KFall windows."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from train.utils import make_autocast


def recording_sequences(split, stride):
    groups = []
    for file_index in np.unique(split["file_idx"]):
        indices = np.flatnonzero(split["file_idx"] == file_index)
        indices = indices[np.argsort(split["start"][indices], kind="stable")]
        starts = np.asarray(split["start"])[indices]
        ends = np.asarray(split["end_frame"])[indices]
        if np.any(np.diff(starts) <= 0) or np.any(np.diff(ends) <= 0):
            raise ValueError("Temporal windows must have unique, increasing starts/endpoints.")
        if len(np.unique(np.asarray(split["subject"])[indices])) != 1:
            raise ValueError("One recording cannot contain windows from multiple subjects.")
        cuts = np.flatnonzero((np.diff(starts) > stride) | (np.diff(ends) > stride)) + 1
        groups.extend(np.split(indices, cuts))
    if not groups:
        raise ValueError("No recording sequences available.")
    return groups


class SequenceWindows(Dataset):
    def __init__(self, dataset, split, stride):
        self.dataset = dataset
        self.groups = recording_sequences(split, stride)
        self.reset = {int(group[0]) for group in self.groups}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, slot, position = index
        else:
            slot, position = 0, 0
        return {**self.dataset[index], "reset": index in self.reset, "window_index": index,
                "slot": slot, "position": position}


class RecordingChunkSampler(Sampler):
    def __init__(self, groups, sequence_length, shuffle):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive.")
        self.groups, self.sequence_length, self.shuffle = groups, sequence_length, shuffle

    def __iter__(self):
        order = torch.randperm(len(self.groups)).tolist() if self.shuffle else range(len(self.groups))
        for i in order:
            group = self.groups[i]
            for start in range(0, len(group), self.sequence_length):
                yield group[start:start + self.sequence_length].tolist()

    def __len__(self):
        return sum((len(group) + self.sequence_length - 1) // self.sequence_length for group in self.groups)


class ParallelRecordingSampler(Sampler):
    """Fixed stream slots per recording cohort; never shuffle chunks."""

    def __init__(self, groups, sequence_length, streams, shuffle=False):
        if sequence_length < 1 or streams < 1:
            raise ValueError("sequence_length and recurrent batch size must be positive.")
        self.groups, self.sequence_length = groups, sequence_length
        self.streams, self.shuffle = min(streams, len(groups)), shuffle

    def __iter__(self):
        order = torch.randperm(len(self.groups)).tolist() if self.shuffle else range(len(self.groups))
        ordered = [self.groups[i] for i in order]
        for start in range(0, len(ordered), self.streams):
            cohort = ordered[start:start + self.streams]
            for offset in range(0, max(map(len, cohort)), self.sequence_length):
                yield [(int(index), slot, position)
                       for slot, group in enumerate(cohort)
                       for position, index in enumerate(group[offset:offset + self.sequence_length])]

    def __len__(self):
        # Training shuffles cohorts, so its exact number of batches can vary.
        if self.shuffle:
            raise TypeError("Shuffled parallel loader has variable batch count.")
        return sum((max(map(len, self.groups[i:i + self.streams])) + self.sequence_length - 1)
                   // self.sequence_length for i in range(0, len(self.groups), self.streams))


def make_sequence_loader(dataset, split, cfg, dev, train=False):
    windows = SequenceWindows(dataset, split, cfg.stride)
    return DataLoader(
        windows, batch_sampler=ParallelRecordingSampler(
            windows.groups, cfg.sequence_length, getattr(cfg, "recurrent_batch_size", 1), train),
        num_workers=cfg.workers, persistent_workers=cfg.workers > 0,
        pin_memory=not cfg.no_pin_memory and dev.type == "cuda",
    )


def run_epoch(model, loader, dev, cfg, optimizer=None, scaler=None,
              class_weight=None, collect_predictions=False):
    training = optimizer is not None
    model.train(training)
    state = None
    total, correct, loss_sum, spike_sum, neurons, resets = 0, 0, 0.0, 0.0, 0, 0
    truth, predictions, probabilities, indices = [], [], [], []
    with (torch.enable_grad() if training else torch.inference_mode()), tqdm(
        total=len(loader.dataset), desc="Train" if training else "Eval", unit="window",
        dynamic_ncols=True, disable=getattr(cfg, "no_progress", False),
    ) as progress:
        for batch in loader:
            slots = batch["slot"].to(dev)
            positions = batch["position"].to(dev)
            streams = loader.batch_sampler.streams
            steps = int(batch["position"].max()) + 1
            reset = torch.zeros(streams, dtype=torch.bool, device=dev)
            if bool((batch["reset"] & (batch["position"] != 0)).any()):
                raise ValueError("A temporal chunk cannot cross a recording/gap boundary.")
            reset[slots[batch["reset"]]] = True
            resets += int(batch["reset"].sum())
            if state is not None:
                state = type(state)(*(torch.where(reset[:, None], torch.zeros_like(value), value) for value in state))
            if training:
                optimizer.zero_grad(set_to_none=True)
            values = batch["x"].to(dev, non_blocking=True).float()
            x = values.new_zeros((streams, steps, *values.shape[1:]))
            x[slots, positions] = values
            offsets = torch.zeros(streams, steps, 6, device=dev)
            offsets[slots, positions] = batch["offset"].to(dev, non_blocking=True).float()
            mask = torch.zeros(streams, steps, device=dev, dtype=torch.bool)
            mask[slots, positions] = True
            y = batch["y"].to(dev, non_blocking=True).long()
            with make_autocast(bool(cfg.amp and dev.type == "cuda")):
                logits, state, spikes = model(x, offset=offsets, state=state, mask=mask)
                logits = logits[slots, positions]
                spikes = spikes[slots, positions]
                loss = F.cross_entropy(logits, y, weight=class_weight,
                                       label_smoothing=cfg.label_smoothing)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite recurrent loss; inspect inputs and learning rate.")
            if training:
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            state = state.detach()
            predicted = logits.argmax(dim=1)
            total += len(y)
            correct += int((predicted == y).sum().item())
            loss_sum += float(loss.detach().item()) * len(y)
            spike_sum += float(spikes.detach().sum().item())
            neurons += spikes.numel()
            if collect_predictions:
                truth.append(y.cpu().numpy())
                predictions.append(predicted.cpu().numpy())
                probabilities.append(logits.detach().float().softmax(dim=1).cpu().numpy())
                indices.append(batch["window_index"].numpy())
            progress.update(len(y))
            progress.set_postfix(loss=f"{loss_sum / total:.4f}", acc=f"{correct / total:.1%}", refresh=False)
    if total == 0:
        raise ValueError("No temporal decisions evaluated.")
    result = {"loss": loss_sum / total, "acc": correct / total, "n": total,
              "recurrent_spike_rate": spike_sum / neurons, "sequence_resets": resets}
    if collect_predictions:
        window_indices = np.concatenate(indices)
        order = np.argsort(window_indices)
        result.update(y_true=np.concatenate(truth)[order], y_pred=np.concatenate(predictions)[order],
                      probabilities=np.concatenate(probabilities)[order], window_indices=window_indices[order])
    return result
