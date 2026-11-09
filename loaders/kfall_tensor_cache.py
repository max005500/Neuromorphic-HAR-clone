"""Reusable, memory-mapped PyTorch caches derived from KFall event arrays."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from loaders.kfall_event_loader import ARRAY_KEYS, CLASS_NAMES


class KFallTensorDataset(Dataset):
    """Return tensor views; events remain uint8 until batch transfer/conversion."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors
        n = len(tensors["y_h1"])
        if tensors["x_ev"].ndim != 4 or tensors["x_ev"].shape[0] != n or tensors["x_ev"].shape[2:] != (6, 2):
            raise ValueError("Tensor cache events must have shape (N, Te, 6, 2).")
        for key, value in tensors.items():
            expected = (n, 6) if key == "offset" else (n,)
            if key != "x_ev" and tuple(value.shape) != expected:
                raise ValueError(f"Tensor cache {key} has inconsistent shape.")
        if tensors["x_ev"].dtype != torch.uint8:
            raise ValueError("Tensor cache events must be uint8.")
        if tensors["y_h1"].dtype != torch.int64 or not torch.isin(
            tensors["y_h1"], torch.tensor(list(CLASS_NAMES)),
        ).all():
            raise ValueError("Tensor cache has invalid KFall activity labels.")

    def __len__(self) -> int:
        return len(self.tensors["y_h1"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tensors = self.tensors
        return {
            "x": tensors["x_ev"][index],
            "offset": tensors["offset"][index],
            "y": tensors["y_h1"][index],
            "y_h1": tensors["y_h1"][index],
            "y_h2": tensors["y_h2"][index],
            "src_h2": tensors["src_h2"][index],
            "w_h2": tensors["w_h2"][index],
        }


def load_or_build_tensor_dataset(
    cache_root: Path, name: str, split: dict[str, np.ndarray],
) -> KFallTensorDataset:
    """Invalidate PT data whenever the source metadata or array files change."""
    metadata = (cache_root / "metadata.json").read_bytes()
    inventory = []
    for key in ARRAY_KEYS:
        stat = (cache_root / name / f"{key}.npy").stat()
        inventory.append((key, stat.st_size, stat.st_mtime_ns))
    signature = hashlib.sha256(metadata + json.dumps(inventory).encode()).hexdigest()
    path = cache_root / f"{name}.pt"
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError(f"Invalid tensor cache {path}; remove that file to rebuild it.")
        if payload.get("source_signature") == signature:
            dataset = KFallTensorDataset(payload["tensors"])
            print(f"[cache:pt:load] {path} | N={len(dataset)}")
            return dataset
        print(f"[cache:pt:rebuild] Source metadata/arrays changed: {path}")
        del payload

    tensors = {
        key: torch.from_numpy(np.array(split[key], copy=True))
        for key in ARRAY_KEYS if key != "subject"
    }
    dataset = KFallTensorDataset(tensors)
    payload: dict[str, Any] = {
        "version": 1, "source_signature": signature, "split": name,
        "metadata": json.loads(metadata),
        "subjects": np.asarray(split["subject"]).tolist(), "tensors": tensors,
    }
    with tempfile.NamedTemporaryFile(prefix=f"{name}_", suffix=".pt", dir=cache_root, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    del dataset, tensors, payload
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    dataset = KFallTensorDataset(saved["tensors"])
    print(f"[cache:pt:save] {path} | N={len(dataset)}")
    return dataset
