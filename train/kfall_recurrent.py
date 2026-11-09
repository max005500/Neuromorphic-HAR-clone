"""Chronological, recording-isolated truncated BPTT for KFall windows."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

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
        return {**self.dataset[index], "reset": index in self.reset, "window_index": index}


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


def make_sequence_loader(dataset, split, cfg, dev, train=False):
    windows = SequenceWindows(dataset, split, cfg.stride)
    return DataLoader(
        windows, batch_sampler=RecordingChunkSampler(windows.groups, cfg.sequence_length, train),
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
    with torch.enable_grad() if training else torch.inference_mode():
        for batch in loader:
            if bool(batch["reset"][0]):
                state = None
                resets += 1
            if bool(batch["reset"][1:].any()):
                raise ValueError("A temporal chunk cannot cross a recording/gap boundary.")
            if training:
                optimizer.zero_grad(set_to_none=True)
            x = batch["x"].to(dev, non_blocking=True).float().unsqueeze(0)
            offsets = batch["offset"].to(dev, non_blocking=True).float().unsqueeze(0)
            y = batch["y"].to(dev, non_blocking=True).long()
            with make_autocast(bool(cfg.amp and dev.type == "cuda")):
                logits, state, spikes = model(x, offset=offsets, state=state)
                logits = logits.squeeze(0)
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
    if total == 0:
        raise ValueError("No temporal decisions evaluated.")
    result = {"loss": loss_sum / total, "acc": correct / total, "n": total,
              "recurrent_spike_rate": spike_sum / neurons, "sequence_resets": resets}
    if collect_predictions:
        result.update(y_true=np.concatenate(truth), y_pred=np.concatenate(predictions),
                      probabilities=np.concatenate(probabilities), window_indices=np.concatenate(indices))
    return result
