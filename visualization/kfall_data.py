"""Read cached KFall windows and their original signals without rebuilding data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from loaders.kfall_event_loader import KFallEventLoaderConfig, signal_to_events, window_targets_for_record
from preprocess.kfall_labels import class_names
from preprocess.interpolate_signal import interpolate_to_times
from preprocess.kfall_resampling import resample_kfall_signal
from preprocess.preprocess_kfall import load_record


SPLIT_NAMES = ("train", "val", "test")
ARRAY_KEYS = ("x_ev", "offset", "y_h1", "file_idx", "start", "subject", "end_frame", "lead_time_ms")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WindowData:
    split: str
    index: int
    raw: np.ndarray
    interpolated: np.ndarray
    events: np.ndarray
    reconstructed: np.ndarray
    time_raw: np.ndarray
    time_events: np.ndarray
    frames: np.ndarray
    offset: np.ndarray
    subject: str
    code: str
    trial: str
    path: Path
    start: int
    stop: int
    end_frame: int
    lead_time_ms: float | None
    class_id: int
    label: str
    fall_onset_time: float | None
    fall_impact_time: float | None
    theta: np.ndarray
    post_impact_end_time: float | None = None
    resample_hz: float | None = None
    analog: np.ndarray | None = None
    time_analog: np.ndarray | None = None


@lru_cache(maxsize=8)
def _read_record(path: str, mtime_ns: int, size: int):
    # Stat values invalidate this small cache when a source recording changes.
    signal, frames, metadata = load_record(path)
    signal.setflags(write=False)
    frames.setflags(write=False)
    return signal, frames, metadata


class KFallWindowStore:
    """Memory-map existing split arrays and validate each displayed source window.

    This viewer never runs preprocessing or modifies the cache. Cached events
    are the plotted input; re-encoding only verifies that the CSV still matches.
    """

    def __init__(
        self, cache_dir: str | Path = "data/kfall_cache/kfall_event_loader_default",
        dataset_root: str | Path | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"KFall event cache not found: {metadata_path}. "
                "Prepare it first with python train_kfall.py --preprocess-only."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("dataset") != "kfall":
            raise ValueError(f"Expected a KFall cache at {self.cache_dir}.")
        self.config = dict(self.metadata.get("config", {}))
        allowed = {field.name for field in fields(KFallEventLoaderConfig)}
        self.cfg = KFallEventLoaderConfig(**{key: value for key, value in self.config.items() if key in allowed})
        self.n_classes = 4 if getattr(self.cfg, "post_impact_seconds", 0.0) > 0 else 3
        self.class_names = class_names(self.n_classes)
        self.records = self.metadata.get("recordings", [])
        if not isinstance(self.records, list) or not self.records:
            raise ValueError("KFall cache metadata has no source-recording manifest.")
        self.dataset_root = Path(dataset_root).expanduser().resolve() if dataset_root is not None else None
        if self.dataset_root is not None and not self.dataset_root.is_dir():
            raise FileNotFoundError(f"KFall dataset directory not found: {self.dataset_root}")
        self._splits: dict[str, dict[str, np.ndarray]] = {}

    def _load_split(self, split: str) -> dict[str, np.ndarray]:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {split!r}; choose train, val, or test.")
        if split not in self._splits:
            arrays = {}
            for key in ARRAY_KEYS:
                path = self.cache_dir / split / f"{key}.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"Incomplete KFall event cache: {path}")
                arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
            count = len(arrays["y_h1"])
            shapes = {"x_ev": (count, self.cfg.out_win, 6, 2), "offset": (count, 6)}
            shapes.update({key: (count,) for key in ARRAY_KEYS if key not in shapes})
            for key, shape in shapes.items():
                if arrays[key].shape != shape:
                    raise ValueError(f"Invalid cached {key} shape in {split}: expected {shape}, got {arrays[key].shape}.")
            self._splits[split] = arrays
        return self._splits[split]

    def split_size(self, split: str) -> int:
        return len(self._load_split(split)["y_h1"])

    def subjects(self, split: str) -> list[str]:
        return np.unique(self._load_split(split)["subject"]).tolist()

    def get_indices(
        self, split: str, class_id: int | None = None, subject: str | None = None,
    ) -> np.ndarray:
        arrays = self._load_split(split)
        mask = np.ones(len(arrays["y_h1"]), dtype=bool)
        if class_id is not None:
            if class_id not in self.class_names:
                raise ValueError(f"class_id must belong to this cache's classes: {self.class_names}.")
            mask &= arrays["y_h1"] == class_id
        if subject is not None and str(subject).strip().lower() not in {"", "all", "todos"}:
            match = re.fullmatch(r"(?:SA|S)?(\d{1,2})", str(subject).strip().upper())
            if match is None:
                raise ValueError("Subject must be a KFall ID such as SA06, or Todos.")
            canonical = f"SA{int(match.group(1)):02d}"
            mask &= arrays["subject"] == canonical
        return np.flatnonzero(mask).astype(np.int64)

    def _record_path(self, record: dict[str, Any]) -> Path:
        recorded = Path(record["file"])
        candidates = []
        roots = []
        if self.dataset_root is not None:
            roots.append(self.dataset_root)
        if self.metadata.get("dataset_root"):
            roots.append(Path(self.metadata["dataset_root"]).expanduser())
        roots.append(PROJECT_ROOT / "data" / "KFall")
        for root in roots:
            # Accept either a KFall root or its sensor_data directory.
            candidates.extend((root / "sensor_data" / record["subject"] / recorded.name,
                               root / record["subject"] / recorded.name))
        candidates.extend((recorded, PROJECT_ROOT / recorded))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"Source CSV not found for {record['subject']}/{recorded.name}. "
            "Use --dataset-root to point to the local KFall directory."
        )

    def get_window(self, split: str, index: int) -> WindowData:
        arrays = self._load_split(split)
        if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < len(arrays["y_h1"]):
            raise IndexError(f"Window index must be in [0, {len(arrays['y_h1']) - 1}] for {split}.")
        index = int(index)
        file_index = int(arrays["file_idx"][index])
        if not 0 <= file_index < len(self.records):
            raise ValueError(f"Cached window {split}[{index}] has an invalid file_idx.")
        record = self.records[file_index]
        path = self._record_path(record)
        stat = path.stat()
        signal, all_frames, source = _read_record(str(path), stat.st_mtime_ns, stat.st_size)
        class_id = int(arrays["y_h1"][index])
        subject = str(arrays["subject"][index])
        for key in ("subject", "code", "trial", "label_id"):
            if source[key] != record[key]:
                raise ValueError(f"Source CSV {path.name} does not match cached recording metadata ({key}).")
        if class_id not in self.class_names or subject != record["subject"]:
            raise ValueError(f"Cached window {split}[{index}] has inconsistent subject or activity labels.")
        start = int(arrays["start"][index])
        stop = start + self.cfg.raw_win
        if start < 0 or stop > len(signal):
            raise ValueError(f"Cached window {split}[{index}] is outside source CSV {path.name}.")
        raw, frames = signal[start:stop], all_frames[start:stop]
        starts, targets = window_targets_for_record(signal, {**record, "frames": all_frames}, self.cfg)
        positions = np.flatnonzero(starts == start)
        if len(positions) != 1 or int(targets[int(positions[0])]) != class_id:
            raise ValueError(f"Cached window {split}[{index}] has an inconsistent temporal activity label.")
        events = np.asarray(arrays["x_ev"][index])
        offset = np.asarray(arrays["offset"][index])
        if not np.isin(events, (0, 1)).all():
            raise ValueError(f"Cached window {split}[{index}] has non-binary events.")
        if np.any((events[:, :, 0] != 0) & (events[:, :, 1] != 0)):
            raise ValueError(f"Cached window {split}[{index}] has simultaneous ON/OFF events.")
        if int(arrays["end_frame"][index]) != int(frames[-1]):
            raise ValueError(f"Source CSV frame counters do not match cached window {split}[{index}].")
        expected_events, expected_offset = signal_to_events(raw, frames, self.cfg)
        if not np.array_equal(offset, expected_offset) or not np.array_equal(events, expected_events):
            raise ValueError(
                f"Cached events/offsets do not match source CSV {path.name} for {split}[{index}]. "
                "The CSV or cache may have changed; explicitly rebuild preprocessing before viewing."
            )

        time_raw = (frames - frames[0]).astype(np.float64) / self.cfg.sample_rate_hz
        time_events_grid = np.linspace(0.0, time_raw[-1], self.cfg.out_win, dtype=np.float32)
        analog, analog_times = resample_kfall_signal(
            raw, frames, sample_rate_hz=self.cfg.sample_rate_hz, resample_hz=self.cfg.resample_hz,
        )
        interpolated = interpolate_to_times(analog, analog_times, time_events_grid, method="linear")
        theta = np.asarray(self.cfg.theta_sd, dtype=np.float32)
        state = offset.copy() if self.cfg.sd_init == "x0" else np.zeros(6, dtype=np.float32)
        signed = events[:, :, 0].astype(np.float32) - events[:, :, 1].astype(np.float32)
        reconstructed = np.empty_like(interpolated)
        # Preserve the encoder's float32 recurrence, including any event at t=0.
        for step in range(len(events)):
            state = state + signed[step] * theta
            reconstructed[step] = state

        onset_time = impact_time = lead_time = post_end_time = None
        if int(record["label_id"]) == 0:
            onset, impact = int(record["fall_onset_frame"]), int(record["fall_impact_frame"])
            onset_time = (onset - int(frames[0])) / self.cfg.sample_rate_hz
            impact_time = (impact - int(frames[0])) / self.cfg.sample_rate_hz
            lead_time = float(arrays["lead_time_ms"][index])
            expected_lead = (impact - int(frames[-1])) * 1000.0 / self.cfg.sample_rate_hz
            if not np.isfinite(lead_time) or not np.isclose(lead_time, expected_lead):
                raise ValueError(f"Cached lead time does not match fall annotations for {split}[{index}].")
            if getattr(self.cfg, "post_impact_seconds", 0.0) > 0:
                end_frame = int(record.get("post_impact_end_frame", impact + round(
                    self.cfg.post_impact_seconds * self.cfg.sample_rate_hz)))
                post_end_time = (end_frame - int(frames[0])) / self.cfg.sample_rate_hz

        return WindowData(
            split=split, index=index, raw=raw, interpolated=interpolated, events=events,
            reconstructed=reconstructed, time_raw=time_raw, time_events=time_events_grid.astype(np.float64),
            frames=frames, offset=offset, subject=subject, code=record["code"], trial=record["trial"],
            path=path, start=start, stop=stop, end_frame=int(frames[-1]), lead_time_ms=lead_time,
            class_id=class_id, label=self.class_names[class_id], fall_onset_time=onset_time,
            fall_impact_time=impact_time, theta=theta, post_impact_end_time=post_end_time,
            resample_hz=self.cfg.resample_hz, analog=analog, time_analog=analog_times,
        )
