#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared preprocessing helpers for splits, labels, and JSON I/O."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

Array = np.ndarray


def json_ready(obj: Any) -> Any:
    """Convert common scientific Python objects into JSON-serializable values."""
    if isinstance(obj, argparse.Namespace):
        return {str(k): json_ready(v) for k, v in vars(obj).items()}

    if is_dataclass(obj) and not isinstance(obj, type):
        return json_ready(asdict(obj))

    if isinstance(obj, Mapping):
        return {str(k): json_ready(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]

    if isinstance(obj, set):
        return [json_ready(v) for v in sorted(obj, key=str)]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, Path):
        return str(obj)

    if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "tolist"):
        return obj.detach().cpu().tolist()

    return obj

def load_json(path: str | os.PathLike[str]) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str | os.PathLike[str], obj: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(obj), f, ensure_ascii=False, indent=indent)
    return path


def save_json_atomic(path: str | os.PathLike[str], obj: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(json_ready(obj), f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)
    return path


def class_counts(y: Any, label_names: Mapping[int, str] | Sequence[str] | None = None) -> dict[int, int] | dict[str, int]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    if label_names is None:
        labels = sorted(np.unique(y).astype(np.int64).tolist())
        return {int(k): int(np.sum(y == k)) for k in labels}

    if isinstance(label_names, Mapping):
        out: dict[str, int] = {}
        for label_id, name in sorted(label_names.items(), key=lambda item: int(item[0])):
            label_id = int(label_id)
            out[str(name)] = int(np.sum(y == label_id))
        return out

    out = {}
    for label_id, name in enumerate(label_names):
        out[str(name)] = int(np.sum(y == label_id))
    return out


def named_class_counts(y: Any, labels: Mapping[int, str] | Sequence[str]) -> dict[str, int]:
    counts = class_counts(y, label_names=labels)
    return {str(k): int(v) for k, v in counts.items()}


def _normalize_split(split: Sequence[float]) -> tuple[float, float, float]:
    if len(split) != 3:
        raise ValueError("split must contain exactly three values: train, val, test.")

    values = tuple(float(v) for v in split)
    if any(not np.isfinite(v) or v < 0.0 for v in values):
        raise ValueError("split values must be finite and non-negative.")

    total = sum(values)
    if total <= 0.0:
        raise ValueError("At least one split value must be positive.")

    return values[0] / total, values[1] / total, values[2] / total

def split_counts(n: int, split: Sequence[float]) -> tuple[int, int, int]:
    fractions = np.asarray(_normalize_split(split), dtype=np.float64)
    raw = fractions * int(n)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(n) - int(counts.sum())

    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        for idx in order[:remainder]:
            counts[int(idx)] += 1

    return int(counts[0]), int(counts[1]), int(counts[2])

def stratified_split_indices(
    y: Any,
    split: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[Array, Array, Array]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(int(seed))

    train_parts: list[Array] = []
    val_parts: list[Array] = []
    test_parts: list[Array] = []

    for label in sorted(np.unique(y).astype(np.int64).tolist()):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_train, n_val, n_test = split_counts(idx.size, split)
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train:n_train + n_val])
        test_parts.append(idx[n_train + n_val:n_train + n_val + n_test])

    train_idx = np.concatenate(train_parts) if train_parts else np.empty((0,), dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.empty((0,), dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.empty((0,), dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)

def stratified_file_split(
    y_file: Any,
    split: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[Array, Array, Array]:
    y_file = np.asarray(y_file, dtype=np.int64).reshape(-1)
    return stratified_split_indices(y_file, split=split, seed=seed)


def subject_split_indices(
    subjects: Any,
    split: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[Array, Array, Array]:
    """Assign whole subjects to train/val/test before selecting their samples.

    Fractions apply to the number of subjects, not recordings or windows. Each
    positive fraction receives at least one subject; impossible requests fail
    explicitly. Sorting subject IDs before shuffling makes assignments stable
    when recordings are reordered or a subject has more recordings.
    """
    fractions = np.asarray(_normalize_split(split), dtype=np.float64)
    subjects = np.asarray(subjects)
    if subjects.ndim != 1:
        raise ValueError("subjects must be a one-dimensional array of subject IDs.")
    if any(
        value is None
        or (isinstance(value, (str, bytes)) and not value.strip())
        or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
        for value in subjects.tolist()
    ):
        raise ValueError("Every sample must have a valid subject ID.")

    unique = np.unique(subjects)
    positive = np.flatnonzero(fractions > 0)
    if unique.size < positive.size:
        raise ValueError(
            f"Subject split needs at least {positive.size} subjects for the "
            f"requested non-empty partitions; found {unique.size}."
        )
    counts = np.asarray(split_counts(unique.size, fractions), dtype=np.int64)
    for target in positive[counts[positive] == 0]:
        donors = np.flatnonzero(counts > 1)
        donor = int(donors[np.argmax(counts[donors] - fractions[donors] * unique.size)])
        counts[donor] -= 1
        counts[target] += 1

    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique)
    boundaries = np.cumsum(counts)
    groups = np.split(unique, boundaries[:-1])
    return tuple(
        np.flatnonzero(np.isin(subjects, group)).astype(np.int64)
        for group in groups
    )


def train_val_split_indices(
    y: Any,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[Array, Array]:
    val_ratio = float(val_ratio)
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    train_idx, val_idx, _ = stratified_split_indices(
        y,
        split=(1.0 - val_ratio, val_ratio, 0.0),
        seed=seed,
    )
    return train_idx, val_idx
