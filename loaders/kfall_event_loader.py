"""KFall temporal labels, subject splits, and Sigma-Delta event caches.

The prediction time is the final sample of each window. In ``pre_impact``
mode it must be at or after onset and strictly before impact. Interpolation
and offsets use only samples inside that window.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from loaders import utils as u
from preprocess.event_encoding import sigma_delta
from preprocess.interpolate_signal import interpolate_to_times
from preprocess.preprocess_kfall import load_kfall_3class
from preprocess.utils import subject_split_indices

SPLIT_NAMES = ("train", "val", "test")
ARRAY_KEYS = (
    "x_ev", "offset", "y_h1", "y_h2", "src_h2", "w_h2", "file_idx",
    "start", "subject", "end_frame", "lead_time_ms",
)
DEFAULT_THETA_KFALL = (0.1, 0.1, 0.1, 10.0, 10.0, 10.0)


@dataclass(frozen=True)
class KFallEventLoaderConfig:
    seed: int = 0
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    raw_win: int = 205
    out_win: int = 2048
    stride: int = 102
    n_ch: int = 6
    sample_rate_hz: float = 100.0
    fall_policy: str = "pre_impact"
    theta_sd: tuple[float, ...] = DEFAULT_THETA_KFALL
    dead_zone: float = 0.5
    sd_init: str = "x0"
    verbose_every: int = 2000
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self) -> None:
        if self.raw_win < 2 or self.out_win < 2 or self.stride < 1:
            raise ValueError("raw_win/out_win must be >= 2 and stride must be >= 1.")
        if self.n_ch != 6:
            raise ValueError("KFall uses exactly six accelerometer/gyroscope channels.")
        theta = np.asarray(self.theta_sd, dtype=np.float32)
        if theta.shape != (6,) or not np.all(np.isfinite(theta)) or np.any(theta <= 0):
            raise ValueError("theta_sd must contain six finite positive values in g and degrees/s.")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz != 100.0:
            raise ValueError("KFall frame annotations use a 100 Hz sample rate.")
        if not np.isfinite(self.dead_zone) or self.dead_zone < 0:
            raise ValueError("dead_zone must be finite and non-negative.")
        if self.sd_init not in {"x0", "zero"}:
            raise ValueError("sd_init must be 'x0' or 'zero'.")
        if self.fall_policy not in {"pre_impact", "annotated_interval"}:
            raise ValueError("fall_policy must be 'pre_impact' or 'annotated_interval'.")


def window_starts_for_record(
    record: np.ndarray, meta: dict[str, Any], cfg: KFallEventLoaderConfig,
) -> np.ndarray:
    """Select windows by their last observed frame; never label an entire fall trial."""
    length = int(len(record))
    if length < cfg.raw_win:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, length - cfg.raw_win + 1, cfg.stride, dtype=np.int64)
    if int(meta["label_id"]) != 0:
        return starts

    onset = int(meta["onset_idx"])
    impact = int(meta["impact_idx"])
    # Add the last admissible endpoint so short fall phases are not missed
    # between two stride positions. The endpoint still has no future samples.
    last_end = impact - (1 if cfg.fall_policy == "pre_impact" else 0)
    anchor = last_end - cfg.raw_win + 1
    if anchor >= 0:
        starts = np.unique(np.append(starts, anchor))
    ends = starts + cfg.raw_win - 1
    return starts[(ends >= onset) & (ends <= last_end)]


def signal_to_events(
    signal: np.ndarray, frames: np.ndarray, cfg: KFallEventLoaderConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate within the observed window, preserving physical channel units."""
    signal = np.asarray(signal, dtype=np.float32)
    frames = np.asarray(frames, dtype=np.int64)
    times = (frames - frames[0]).astype(np.float64) / cfg.sample_rate_hz
    destination = np.linspace(0.0, times[-1], cfg.out_win, dtype=np.float32)
    interpolated = interpolate_to_times(signal, times, destination, method="linear")
    events, _, _ = sigma_delta(
        interpolated, theta=np.asarray(cfg.theta_sd, dtype=np.float32),
        init=cfg.sd_init, dead_zone=cfg.dead_zone, return_reconstruction=False,
    )
    return events, signal[0].copy()


def build_event_split(
    recordings: np.ndarray, labels: np.ndarray, metadata: list[dict[str, Any]],
    file_indices: Iterable[int], cfg: KFallEventLoaderConfig,
    *, name: str = "split", output_dir: Path | None = None, verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Encode a split directly to disk when an output directory is supplied."""
    selected = [(int(fi), window_starts_for_record(recordings[int(fi)], metadata[int(fi)], cfg))
                for fi in file_indices]
    count = sum(len(starts) for _, starts in selected)
    if count == 0:
        raise RuntimeError(f"No KFall windows generated for '{name}'; check subjects, labels and raw_win.")
    shapes = {
        "x_ev": ((count, cfg.out_win, 6, 2), np.uint8),
        "offset": ((count, 6), np.float32), "y_h1": ((count,), np.int64),
        "y_h2": ((count,), np.float32), "src_h2": ((count,), np.uint8),
        "w_h2": ((count,), np.float32), "file_idx": ((count,), np.int32),
        "start": ((count,), np.int32), "subject": ((count,), "U4"),
        "end_frame": ((count,), np.int64), "lead_time_ms": ((count,), np.float32),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: (np.empty(shape, dtype=dtype) if output_dir is None else
              np.lib.format.open_memmap(output_dir / f"{key}.npy", mode="w+", dtype=dtype, shape=shape))
        for key, (shape, dtype) in shapes.items()
    }
    cursor = 0
    for fi, starts in selected:
        signal = np.asarray(recordings[fi], dtype=np.float32)
        info = metadata[fi]
        frames = np.asarray(info["frames"], dtype=np.int64)
        for start in starts:
            start = int(start)
            stop = start + cfg.raw_win
            events, offset = signal_to_events(signal[start:stop], frames[start:stop], cfg)
            end_frame = int(frames[stop - 1])
            arrays["x_ev"][cursor] = events
            arrays["offset"][cursor] = offset
            arrays["y_h1"][cursor] = int(labels[fi])
            # KFall supplies fall timing, not temporal posture-transition labels.
            arrays["y_h2"][cursor] = 0.5
            arrays["src_h2"][cursor] = 0
            arrays["w_h2"][cursor] = 0.0
            arrays["file_idx"][cursor] = fi
            arrays["start"][cursor] = start
            arrays["subject"][cursor] = info["subject"]
            arrays["end_frame"][cursor] = end_frame
            arrays["lead_time_ms"][cursor] = (
                (int(info["fall_impact_frame"]) - end_frame) * 1000.0 / cfg.sample_rate_hz
                if int(labels[fi]) == 0 else np.nan
            )
            cursor += 1
            if verbose and cfg.verbose_every and cursor % cfg.verbose_every == 0:
                print(f"[kfall:{name}] Encoded {cursor}/{count} windows.")
    for array in arrays.values():
        if isinstance(array, np.memmap):
            array.flush()
    if verbose:
        print(f"[kfall:{name}] {count} windows, subjects={sorted(np.unique(arrays['subject']).tolist())}")
    return arrays


def _source_signature(root: Path, label_dir: Path) -> dict[str, Any]:
    files = sorted({path for base, suffix in ((root, ".csv"), (label_dir, ".xlsx"))
                    for path in base.rglob("*") if path.is_file() and path.suffix.lower() == suffix})
    inventory = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                 for path in files if not path.name.startswith("~$")]
    return {"sha256": hashlib.sha256(json.dumps(inventory).encode()).hexdigest(), "files": len(inventory)}


def load_event_splits(
    root: str | os.PathLike[str], mmap_mode: str | None = None, verbose: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    return u.load_event_splits(root, split_names=SPLIT_NAMES, array_keys=ARRAY_KEYS,
                               mmap_mode=mmap_mode, mmap_keys=("x_ev",), verbose=verbose)


def load_or_build_kfall_event_splits(
    dataset_root: str | os.PathLike[str] = "data/KFall",
    label_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] = "data/kfall_cache", run_name: str = "default",
    cfg: KFallEventLoaderConfig | None = None, refresh: bool = False,
    mmap_mode: str | None = None, verbose: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Split subjects before windowing, then build or reuse a source-aware cache."""
    cfg = KFallEventLoaderConfig() if cfg is None else cfg
    root = Path(dataset_root)
    labels_root = Path(label_dir) if label_dir is not None else root / "label_data"
    if not root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(f"KFall sensor/label directories missing: {root}, {labels_root}")
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a simple cache name.")
    expected = {
        "version": 1, "dataset": "kfall", "split_strategy": "subject",
        "dataset_root": str(root.resolve()), "label_dir": str(labels_root.resolve()),
        "config": asdict(cfg), "source_signature": _source_signature(root, labels_root),
    }
    output = Path(cache_dir) / f"kfall_event_loader_{run_name}"
    if not refresh and u.event_cache_is_valid(
        output, output / "metadata.json", expected, split_names=SPLIT_NAMES, array_keys=ARRAY_KEYS,
        metadata_keys=("version", "dataset_root", "label_dir", "config", "source_signature"),
    ):
        return load_event_splits(output, mmap_mode, verbose), u.load_json(output / "metadata.json")

    recordings, labels, records, skipped = load_kfall_3class(root, label_dir=labels_root)
    indices = subject_split_indices([record["subject"] for record in records], cfg.split, cfg.seed)
    if any(not len(index) for index in indices):
        raise ValueError("KFall training requires nonempty train, validation and test subject groups.")
    subjects = {name: sorted({records[int(i)]["subject"] for i in idx})
                for name, idx in zip(SPLIT_NAMES, indices)}
    metadata = {
        **expected, "split_subjects": subjects,
        "split_file_indices": {name: idx.tolist() for name, idx in zip(SPLIT_NAMES, indices)},
        "recordings": [{key: value for key, value in record.items() if key not in {"frames", "timestamps"}}
                       for record in records],
        "raw_data": {"recordings": len(records), "skipped": skipped},
        "channels": ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"],
        "units": ["g", "g", "g", "degrees/s", "degrees/s", "degrees/s"],
        "head2_labels_available": False,
        "fall_label_rule": "onset <= final frame < impact" if cfg.fall_policy == "pre_impact"
                           else "onset <= final frame <= impact",
        "split_summary": {},
    }
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kfall_build_", dir=cache_dir) as temporary:
        temporary_root = Path(temporary) / "cache"
        for name, idx in zip(SPLIT_NAMES, indices):
            arrays = build_event_split(recordings, labels, records, idx, cfg, name=name,
                                       output_dir=temporary_root / name, verbose=verbose)
            metadata["split_summary"][name] = {
                "windows": len(arrays["y_h1"]), "subjects": subjects[name],
                "class_counts": u.class_counts(arrays["y_h1"]),
                "recordings_without_windows": sum(not len(window_starts_for_record(recordings[int(i)], records[int(i)], cfg)) for i in idx),
            }
            del arrays
        u.save_json(temporary_root / "metadata.json", metadata)
        if output.exists():
            u.safe_rmtree(output)
        temporary_root.rename(output)
    return load_event_splits(output, mmap_mode, verbose), metadata
