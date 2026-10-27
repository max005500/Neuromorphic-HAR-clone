"""Read the native KFall sensor CSVs and per-subject fall annotations.

Signals retain KFall's physical units (g and degrees/s). Temporal labels refer
to ``FrameCounter`` values, not row numbers. Task grouping follows the existing
SisFall three-class protocol: the static family includes posture changes.

References:
    https://sites.google.com/view/kfalldataset
    https://doi.org/10.3389/fnagi.2021.692865 (Table 2)
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


CHANNELS = ("AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ")
SAMPLE_RATE_HZ = 100
STABLE_TASK_IDS = frozenset({1, 11, 12, 17})
STATIC_TASK_IDS = frozenset({1, 2, 3, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19})
DYNAMIC_TASK_IDS = frozenset({4, 6, 7, 8, 9, 10, 35, 36})
FALL_TASK_IDS = frozenset(range(20, 35))
STATIC_CODES = frozenset(f"T{task:02d}" for task in STATIC_TASK_IDS)
DYNAMIC_CODES = frozenset(f"T{task:02d}" for task in DYNAMIC_TASK_IDS)
LABELS = {0: "FALL", 1: "NORMAL_DYNAMIC", 2: "NORMAL_STATIC"}

# Native downloads use S06T20R01.csv; the documentation uses SA06T20R01.csv.
FILE_RE = re.compile(r"^S(?:A)?(\d{2})T(\d{2})R(\d{2})\.csv$", re.IGNORECASE)
LABEL_FILE_RE = re.compile(r"^S(?:A)?(\d{2})_label\.xlsx$", re.IGNORECASE)


def _column_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _positive_int(value: Any, context: str) -> int:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context}: expected a positive integer, got {value!r}.") from exc
    if not np.isfinite(number) or number <= 0 or number != int(number):
        raise ValueError(f"{context}: expected a positive integer, got {value!r}.")
    return int(number)


def _task_id(value: Any, context: str) -> int:
    text = str(value).strip()
    # E.g. F01 (20), as used by the official spreadsheets.
    match = re.search(r"\(\s*(\d+)\s*\)", text)
    if match is None:
        match = re.fullmatch(r"[Tt]?(\d+)", text)
    if match is None:
        raise ValueError(f"{context}: cannot read Task ID from {value!r}.")
    task = _positive_int(match.group(1), context)
    if task not in FALL_TASK_IDS:
        raise ValueError(f"{context}: Task ID {task} is not a KFall fall task (20..34).")
    return task


def load_fall_labels(
    label_dir: str | os.PathLike[str],
) -> dict[tuple[str, int, int], tuple[int, int]]:
    """Return ``(subject, task, trial) -> (onset_frame, impact_frame)``.

    Blank task cells are forward-filled within each sheet, including merged
    cells. Duplicate annotations and malformed temporal bounds are errors.
    """
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise ImportError("KFall XLSX labels require openpyxl; install requirements.txt.") from exc

    root = Path(label_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"KFall label directory not found: {root}")
    paths = sorted(path for path in root.rglob("*") if LABEL_FILE_RE.fullmatch(path.name))
    if not paths:
        raise FileNotFoundError(f"No SAxx_label.xlsx annotation files found in {root}.")

    annotations: dict[tuple[str, int, int], tuple[int, int]] = {}
    required = ("taskcodetaskid", "trialid", "fallonsetframe", "fallimpactframe")
    for path in paths:
        match = LABEL_FILE_RE.fullmatch(path.name)
        assert match is not None
        subject = f"SA{match.group(1)}"
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in workbook:
                rows = sheet.iter_rows(values_only=True)
                header = next(rows, None)
                if header is None or not any(value is not None for value in header):
                    continue
                columns = {_column_key(value): index for index, value in enumerate(header) if value is not None}
                missing = set(required) - columns.keys()
                if missing:
                    raise ValueError(f"{path} [{sheet.title}]: missing label columns {sorted(missing)}.")
                current_task: int | None = None
                for row_number, row in enumerate(rows, start=2):
                    if not any(value is not None for value in row):
                        continue
                    context = f"{path} [{sheet.title}], row {row_number}"
                    values = {key: row[columns[key]] if columns[key] < len(row) else None for key in required}
                    task_value = values["taskcodetaskid"]
                    if task_value is not None and str(task_value).strip():
                        current_task = _task_id(task_value, context)
                    if current_task is None:
                        raise ValueError(f"{context}: trial has no Task ID to inherit.")
                    trial = _positive_int(values["trialid"], f"{context}, Trial ID")
                    onset = _positive_int(values["fallonsetframe"], f"{context}, Fall_onset_frame")
                    impact = _positive_int(values["fallimpactframe"], f"{context}, Fall_impact_frame")
                    if onset >= impact:
                        raise ValueError(f"{context}: fall onset {onset} must precede impact {impact}.")
                    key = (subject, current_task, trial)
                    if key in annotations:
                        raise ValueError(f"{context}: duplicate fall annotation for {key}.")
                    annotations[key] = (onset, impact)
        finally:
            workbook.close()
    if not annotations:
        raise ValueError(f"No fall annotations found in {root}.")
    return annotations


def load_record(path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read one CSV as six float32 channels, frame counters and metadata.

    Metadata also retains the timestamps and frame grid so downstream code can
    interpolate in physical time when frames are missing. No Euler channel or
    sensor-unit conversion is applied.
    """
    path = Path(path)
    match = FILE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not a KFall recording filename: {path.name}")
    subject = f"SA{match.group(1)}"
    task, trial = int(match.group(2)), int(match.group(3))
    if task not in STATIC_TASK_IDS | DYNAMIC_TASK_IDS | FALL_TASK_IDS:
        raise ValueError(f"{path}: unknown KFall Task ID {task}.")
    if trial < 1:
        raise ValueError(f"{path}: Trial ID must be positive.")
    folder_match = re.fullmatch(r"S(?:A)?(\d{2})", path.parent.name, re.IGNORECASE)
    if folder_match and folder_match.group(1) != match.group(1):
        raise ValueError(f"{path}: subject in filename does not match subject directory.")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    keys = [_column_key(value) for value in header]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate CSV column names.")
    requested = ("TimeStamp(s)", "FrameCounter", *CHANNELS)
    missing = [name for name in requested if _column_key(name) not in keys]
    if missing:
        raise ValueError(f"{path}: missing sensor columns {missing}.")
    indices = tuple(keys.index(_column_key(name)) for name in requested)
    try:
        values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=indices, dtype=np.float64, ndmin=2,
                            encoding="utf-8-sig")
    except ValueError as exc:
        raise ValueError(f"{path}: invalid numeric sensor data: {exc}") from exc
    if values.shape[0] == 0:
        raise ValueError(f"{path}: recording is empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: sensor values, timestamps and frame counters must be finite.")
    frame_values = values[:, 1]
    if np.any(frame_values < 1) or np.any(frame_values != np.floor(frame_values)):
        raise ValueError(f"{path}: FrameCounter must contain positive integers.")
    if np.any(frame_values > np.iinfo(np.int64).max):
        raise ValueError(f"{path}: FrameCounter exceeds int64 range.")
    frames = frame_values.astype(np.int64)
    timestamps = values[:, 0]
    if np.any(np.diff(frames) <= 0):
        raise ValueError(f"{path}: FrameCounter must be strictly increasing.")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{path}: timestamps must be strictly increasing.")

    label_id = 0 if task in FALL_TASK_IDS else 1 if task in DYNAMIC_TASK_IDS else 2
    signals = values[:, 2:].astype(np.float32)
    if not np.all(np.isfinite(signals)):
        raise ValueError(f"{path}: sensor values exceed float32 range.")
    metadata = {
        "file": str(path), "subject": subject, "code": f"T{task:02d}", "trial": f"R{trial:02d}",
        "task_id": task, "trial_id": trial, "T": int(signals.shape[0]),
        "label_id": label_id, "label": LABELS[label_id], "channels": list(CHANNELS),
        "sample_rate_hz": SAMPLE_RATE_HZ, "acc_unit": "g", "gyr_unit": "deg/s",
        "frames": frames, "timestamps": timestamps,
    }
    return signals, frames, metadata


def load_kfall_3class(
    root: str | os.PathLike[str],
    label_dir: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], int]:
    """Load KFall recordings and validated temporal annotations.

    ``root`` may name the KFall directory or its ``sensor_data`` directory.
    Matching sensor CSVs with missing or inconsistent annotations fail loudly;
    unrelated CSVs are counted in ``skipped``. Each recording remains intact so
    subject separation can happen before windowing.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"KFall dataset directory not found: {root}")
    sensor_root = root / "sensor_data" if (root / "sensor_data").is_dir() else root
    if label_dir is None:
        label_dir = (root / "label_data") if (root / "label_data").is_dir() else root.parent / "label_data"
    labels = load_fall_labels(label_dir)

    recordings: list[np.ndarray] = []
    targets: list[int] = []
    metadata: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    skipped = 0
    for path in sorted(sensor_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        if FILE_RE.fullmatch(path.name) is None:
            skipped += 1
            continue
        signal, frames, record = load_record(path)
        key = (record["subject"], record["task_id"], record["trial_id"])
        if key in seen:
            raise ValueError(f"{path}: duplicate sensor recording for {key}.")
        seen.add(key)
        if record["label_id"] == 0:
            if key not in labels:
                raise ValueError(f"{path}: missing fall annotation for {key}.")
            onset, impact = labels[key]
            onset_idx = int(np.searchsorted(frames, onset))
            impact_idx = int(np.searchsorted(frames, impact))
            if onset_idx >= frames.size or frames[onset_idx] != onset:
                raise ValueError(f"{path}: Fall_onset_frame {onset} is absent from FrameCounter.")
            if impact_idx >= frames.size or frames[impact_idx] != impact:
                raise ValueError(f"{path}: Fall_impact_frame {impact} is absent from FrameCounter.")
            record.update(fall_onset_frame=onset, fall_impact_frame=impact,
                          onset_idx=onset_idx, impact_idx=impact_idx)
        recordings.append(signal)
        targets.append(record["label_id"])
        metadata.append(record)
    if not recordings:
        raise ValueError(f"No KFall sensor recordings found in {sensor_root}.")
    # Explicit object allocation also works when all recordings have equal lengths.
    x = np.empty(len(recordings), dtype=object)
    x[:] = recordings
    return x, np.asarray(targets, dtype=np.int64), metadata, skipped
