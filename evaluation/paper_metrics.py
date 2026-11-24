"""Reproducible metrics and raw-signal perturbations for paper evaluations.

The four SisFall classes use the fixed order Fall, Dynamic, Stable, Transition.
In dual-head evaluations, stable/transition labels are admitted only when
``src_h2`` is ``SRC_MANUAL``; autotags are never treated as ground truth.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

CLASS_NAMES = ("Fall", "Dynamic", "Stable", "Transition")
SRC_MANUAL = 1
H1_FALL = 0
H1_DYNAMIC = 1
H1_STATIC = 2


def _as_1d(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return array


def _class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = np.arange(len(CLASS_NAMES))
    if not y_true.size:
        matrix = np.zeros((len(labels), len(labels)), dtype=int).tolist()
        return {
            "count": 0,
            "accuracy": None,
            "macro_f1": None,
            "confusion": matrix,
            "confusion_matrix": matrix,
            "classes": {
                name: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                for name in CLASS_NAMES
            },
        }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "count": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "confusion": matrix,
        "confusion_matrix": matrix,
        "classes": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
    }


def classification_metrics(truth: Sequence[int], pred: Sequence[int], n_classes: int) -> dict[str, Any]:
    """Return fixed-label accuracy, macro-F1 and confusion matrix."""
    y_true = _as_1d("truth", truth)
    y_pred = _as_1d("pred", pred)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("truth and pred must have the same non-zero length")
    if not isinstance(n_classes, (int, np.integer)) or n_classes < 1:
        raise ValueError("n_classes must be a positive integer")
    labels = np.arange(int(n_classes))
    if not np.all(np.isin(y_true, labels)) or not np.all(np.isin(y_pred, labels)):
        raise ValueError("truth and pred labels must be in range(n_classes)")
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion": matrix,
        "confusion_matrix": matrix,
    }


def evaluate_dual_head_sisfall(
    y_h1_true: Sequence[int],
    y_h2_true: Sequence[int],
    src_h2: Sequence[int],
    y_h1_pred: Sequence[int],
    y_h2_pred: Sequence[int],
    subjects: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate dual-head SisFall predictions as four classes.

    Head 1 labels are Fall=0, Dynamic=1, Static=2; Head 2 labels are
    Stable=0 and Transition=1. Static rows whose source is not manual are
    excluded entirely (including from the denominator), never inferred from
    autotags. If supplied, ``subjects`` enables the SE01--SE15 subgroup.
    """
    arrays = {
        "y_h1_true": _as_1d("y_h1_true", y_h1_true),
        "y_h2_true": _as_1d("y_h2_true", y_h2_true),
        "src_h2": _as_1d("src_h2", src_h2),
        "y_h1_pred": _as_1d("y_h1_pred", y_h1_pred),
        "y_h2_pred": _as_1d("y_h2_pred", y_h2_pred),
    }
    lengths = {len(a) for a in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all label and prediction arrays must have equal length")
    if subjects is not None and len(subjects) != len(arrays["y_h1_true"]):
        raise ValueError("subjects must have one entry per prediction row")

    raw_true_h1 = arrays["y_h1_true"]
    raw_true_h2 = arrays["y_h2_true"]
    raw_source = arrays["src_h2"]
    raw_pred_h1 = arrays["y_h1_pred"]
    raw_pred_h2 = arrays["y_h2_pred"]
    if not np.all(np.isin(raw_true_h1, (H1_FALL, H1_DYNAMIC, H1_STATIC))):
        raise ValueError("true Head1 labels must be 0 (Fall), 1 (Dynamic), or 2 (Static)")
    if not np.all(np.isin(raw_source, (0, SRC_MANUAL))):
        raise ValueError("src_h2 values must be 0 (autotag) or 1 (manual)")
    true_static_manual = (raw_true_h1 == H1_STATIC) & (raw_source == SRC_MANUAL)
    if np.any(true_static_manual & ~np.isin(raw_true_h2, (0, 1))):
        raise ValueError("manual Head2 labels must be 0 (Stable) or 1 (Transition)")
    if not np.all(np.isin(raw_pred_h1, (H1_FALL, H1_DYNAMIC, H1_STATIC))):
        raise ValueError("predicted Head1 labels must be 0..2")
    if np.any((raw_pred_h1 == H1_STATIC) & ~np.isin(raw_pred_h2, (0, 1))):
        raise ValueError("predicted Head2 labels must be 0 (Stable) or 1 (Transition) when Head1 predicts Static")

    true_h1 = raw_true_h1.astype(int)
    true_h2 = np.zeros(len(raw_true_h2), dtype=int)
    true_h2[true_static_manual] = raw_true_h2[true_static_manual].astype(int)
    source = raw_source.astype(int)
    pred_h1 = raw_pred_h1.astype(int)
    pred_h2 = raw_pred_h2.astype(int)
    valid = (true_h1 != H1_STATIC) | (source == SRC_MANUAL)
    invalid_manual_label = valid & (true_h1 == H1_STATIC) & ~np.isin(true_h2, (0, 1))
    if np.any(invalid_manual_label):
        raise ValueError("manual Head2 labels must be 0 (Stable) or 1 (Transition)")

    truth = true_h1[valid].copy()
    prediction = pred_h1[valid].copy()
    manual_static = true_h1[valid] == H1_STATIC
    truth[manual_static] = 2 + true_h2[valid][manual_static]
    predicted_static = pred_h1[valid] == H1_STATIC
    prediction[predicted_static] = 2 + pred_h2[valid][predicted_static]
    result: dict[str, Any] = _class_metrics(truth, prediction)
    result["excluded_autotag_static"] = int(np.sum((true_h1 == H1_STATIC) & (source != SRC_MANUAL)))

    if subjects is not None:
        subject_ids = np.char.upper(np.asarray(subjects, dtype=str))
        major_mask = np.isin(subject_ids, [f"SE{i:02d}" for i in range(1, 16)]) & valid
        major_truth = true_h1[major_mask].copy()
        major_prediction = pred_h1[major_mask].copy()
        major_static = true_h1[major_mask] == H1_STATIC
        major_truth[major_static] = 2 + true_h2[major_mask][major_static]
        major_predicted_static = pred_h1[major_mask] == H1_STATIC
        major_prediction[major_predicted_static] = 2 + pred_h2[major_mask][major_predicted_static]
        result["major_subjects_SE01_SE15"] = _class_metrics(major_truth, major_prediction)
    else:
        result["major_subjects_SE01_SE15"] = None
    return result


def hierarchical_metrics(
    y_h1: Sequence[int],
    y_h2: Sequence[int],
    src_h2: Sequence[int],
    pred_h1: Sequence[int],
    pred_h2: Sequence[int],
    subjects: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Short alias for :func:`evaluate_dual_head_sisfall` (ground truth first)."""
    return evaluate_dual_head_sisfall(y_h1, y_h2, src_h2, pred_h1, pred_h2, subjects)


def evaluate_continuous_alarms(
    windows: Sequence[Mapping[str, Any]],
    fall_events: Sequence[Mapping[str, Any]] | None,
    recording_info: Mapping[str, Mapping[str, Any]] | None,
    *,
    refractory_seconds: float = 0.0,
) -> dict[str, Any]:
    """Score alarm windows against explicit official fall intervals.

    Windows require ``recording_id`` and ``timestamp_seconds``. Fall events
    require ``recording_id``, ``start_seconds`` and ``end_seconds``. Recording
    metadata requires ``duration_seconds`` and boolean ``is_adl``. An explicit empty
    ``fall_events=[]`` means that no official events were annotated; omitted
    events/timestamps/durations are errors, never interpreted as no falls.

    For each recording, alarm windows are time-sorted and refractory suppression
    keeps the first alarm, suppressing later ones less than the refractory
    interval away. Each retained alarm matches at most one event containing its
    timestamp, choosing the eligible event with the earliest end. Events are
    likewise matched at most once. False alarms are unmatched retained alarms
    on recordings explicitly marked ADL. The denominator is the sum of actual
    durations only for ADL recordings represented by timestamped windows;
    metadata-only recordings are reported as lacking window coverage.
    """
    if fall_events is None:
        raise ValueError("fall_events must be supplied explicitly (use [] for no events)")
    if recording_info is None or not recording_info:
        raise ValueError("recording_info with real recording durations is required")
    if not windows:
        raise ValueError("at least one timestamped prediction window is required")
    if not np.isfinite(refractory_seconds) or refractory_seconds < 0:
        raise ValueError("refractory_seconds must be finite and non-negative")

    durations: dict[str, float] = {}
    adl: dict[str, bool] = {}
    for rec, info in recording_info.items():
        if "duration_seconds" not in info or "is_adl" not in info:
            raise ValueError(f"recording_info[{rec!r}] requires duration_seconds and is_adl")
        if not isinstance(info["is_adl"], (bool, np.bool_)):
            raise ValueError(f"recording_info[{rec!r}].is_adl must be a boolean")
        duration = float(info["duration_seconds"])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"recording {rec!r} must have a positive real duration")
        durations[str(rec)] = duration
        adl[str(rec)] = bool(info["is_adl"])

    grouped_windows: dict[str, list[float]] = {}
    recordings_with_windows: set[str] = set()
    for row in windows:
        if "recording_id" not in row or "timestamp_seconds" not in row:
            raise ValueError("every window requires recording_id and timestamp_seconds")
        rec = str(row["recording_id"])
        timestamp = float(row["timestamp_seconds"])
        if rec not in durations:
            raise ValueError(f"window recording {rec!r} has no recording_info")
        if not np.isfinite(timestamp) or timestamp < 0 or timestamp > durations[rec]:
            raise ValueError(f"window timestamp for {rec!r} is outside its recording")
        recordings_with_windows.add(rec)
        alarm_value = row.get("alarm", row.get("prediction"))
        if alarm_value is None:
            raise ValueError("every window requires an alarm or prediction value")
        if isinstance(alarm_value, str):
            alarm_value = alarm_value.strip().lower() in {"1", "true", "yes", "fall"}
        if bool(alarm_value):
            grouped_windows.setdefault(rec, []).append(timestamp)

    grouped_events: dict[str, list[tuple[float, float, int]]] = {}
    for event_index, event in enumerate(fall_events):
        for key in ("recording_id", "start_seconds", "end_seconds"):
            if key not in event:
                raise ValueError(f"every fall event requires {key}")
        rec = str(event["recording_id"])
        start, end = float(event["start_seconds"]), float(event["end_seconds"])
        if rec not in durations:
            raise ValueError(f"event recording {rec!r} has no recording_info")
        if "is_adl" in event and not isinstance(event["is_adl"], (bool, np.bool_)):
            raise ValueError(f"fall event {event_index}.is_adl must be a boolean")
        if event.get("is_adl", False) or adl[rec]:
            raise ValueError(f"official fall event {event_index} cannot be marked as ADL")
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or start >= end or end > durations[rec]:
            raise ValueError(f"fall event interval for {rec!r} is invalid or outside its recording")
        grouped_events.setdefault(rec, []).append((start, end, event_index))

    retained: dict[str, list[float]] = {}
    suppressed = 0
    for rec, timestamps in grouped_windows.items():
        selected: list[float] = []
        for timestamp in sorted(timestamps):
            if not selected or timestamp - selected[-1] >= refractory_seconds:
                selected.append(timestamp)
            else:
                suppressed += 1
        retained[rec] = selected

    matched_events: set[int] = set()
    matches: list[dict[str, Any]] = []
    for rec, timestamps in retained.items():
        events = grouped_events.get(rec, [])
        for timestamp in timestamps:
            eligible = [
                event for event in events
                if event[2] not in matched_events and event[0] <= timestamp <= event[1]
            ]
            if eligible:
                chosen = min(eligible, key=lambda event: (event[1], event[0], event[2]))
                matched_events.add(chosen[2])
                matches.append({
                    "recording_id": rec,
                    "timestamp_seconds": timestamp,
                    "event_index": chosen[2],
                })

    adl_duration = sum(
        duration for rec, duration in durations.items()
        if adl[rec] and rec in recordings_with_windows
    )
    adl_hours = adl_duration / 3600.0
    false_alarm_count = sum(
        1 for rec, timestamps in retained.items()
        if adl.get(rec, False)
        for timestamp in timestamps
        if not any(match["recording_id"] == rec and match["timestamp_seconds"] == timestamp for match in matches)
    )
    total_events = len(fall_events)
    return {
        "fall_events": total_events,
        "detected_events": len(matched_events),
        "event_sensitivity": float(len(matched_events) / total_events) if total_events else None,
        "false_alarms_on_adl": int(false_alarm_count),
        "adl_duration_seconds": float(adl_duration),
        "adl_recordings_without_window_coverage": sorted(
            rec for rec in durations if adl[rec] and rec not in recordings_with_windows
        ),
        "false_alarms_per_hour": float(false_alarm_count / adl_hours) if adl_hours else None,
        "retained_alarms": int(sum(map(len, retained.values()))),
        "refractory_suppressed_alarms": int(suppressed),
        "matches": matches,
    }


def perturb_raw_signal(
    signal: np.ndarray,
    *,
    units: str,
    sample_rate_hz: float,
    noise_std: float = 0.0,
    noise_channels: Sequence[int] | None = None,
    drift_per_second: float = 0.0,
    zero_channels: Sequence[int] = (),
    seed: int | None = 0,
) -> np.ndarray:
    """Perturb a raw ``(samples, channels)`` signal before event encoding.

    Noise standard deviation is in ``units`` (for example sensor counts);
    ``drift_per_second`` is in those units per second. ``noise_channels=None``
    applies Gaussian noise to all channels. The input is not modified.
    """
    raw = np.asarray(signal)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("signal must be a non-empty (samples, channels) array")
    if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
        raise ValueError("signal must contain finite numeric values")
    if not isinstance(units, str) or not units.strip():
        raise ValueError("units must explicitly name the raw signal units")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if not np.isfinite(noise_std) or noise_std < 0 or not np.isfinite(drift_per_second):
        raise ValueError("noise_std must be non-negative and drift_per_second finite")

    channels = raw.shape[1]
    noise_idx = np.arange(channels) if noise_channels is None else np.asarray(noise_channels, dtype=int)
    zero_idx = np.asarray(zero_channels, dtype=int)
    if np.any(noise_idx < 0) or np.any(noise_idx >= channels):
        raise ValueError("noise channel index is out of range")
    if np.any(zero_idx < 0) or np.any(zero_idx >= channels):
        raise ValueError("zero channel index is out of range")

    result = raw.astype(np.result_type(raw.dtype, np.float64), copy=True)
    if drift_per_second:
        elapsed = np.arange(raw.shape[0], dtype=float) / float(sample_rate_hz)
        result += elapsed[:, None] * float(drift_per_second)
    if noise_std:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, float(noise_std), size=(raw.shape[0], len(noise_idx)))
        result[:, noise_idx] += noise
    if zero_idx.size:
        result[:, zero_idx] = 0.0
    return result


__all__ = [
    "CLASS_NAMES",
    "SRC_MANUAL",
    "classification_metrics",
    "hierarchical_metrics",
    "evaluate_dual_head_sisfall",
    "evaluate_continuous_alarms",
    "perturb_raw_signal",
]


def _classification_csv_rows(report: Mapping[str, Any], subset: str) -> list[dict[str, Any]]:
    metrics = report if subset == "all" else report.get("major_subjects_SE01_SE15")
    if metrics is None:
        return []
    return [
        {
            "subset": subset,
            "class": name,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1"],
            "support": values["support"],
        }
        for name, values in metrics["classes"].items()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SisFall dual-head predictions from CSV. Required columns: "
            "y_h1_true (0 Fall, 1 Dynamic, 2 Static), y_h2_true (0 Stable, "
            "1 Transition), src_h2 (1 manual; other values are autotag), "
            "y_h1_pred (0..2), y_h2_pred (0..1). Optional subject enables "
            "the SE01..SE15 metrics. Static rows with non-manual src_h2 are "
            "excluded. Writes JSON summary and per-class CSV."
        )
    )
    parser.add_argument("--input", required=True, help="Input prediction CSV with the documented columns.")
    parser.add_argument("--json-out", required=True, help="Path for the full JSON metrics report.")
    parser.add_argument("--csv-out", required=True, help="Path for per-class precision/recall/F1 CSV.")
    args = parser.parse_args(argv)

    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = ("y_h1_true", "y_h2_true", "src_h2", "y_h1_pred", "y_h2_pred")
    if not rows:
        parser.error("input CSV must contain at least one prediction row")
    missing = [column for column in required if column not in rows[0]]
    if missing:
        parser.error(f"input CSV is missing columns: {', '.join(missing)}")
    report = evaluate_dual_head_sisfall(
        *([int(row[column]) for row in rows] for column in required),
        subjects=[row.get("subject", "") for row in rows] if "subject" in rows[0] else None,
    )
    Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_rows = _classification_csv_rows(report, "all")
    output_rows.extend(_classification_csv_rows(report, "SE01_SE15"))
    with Path(args.csv_out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subset", "class", "precision", "recall", "f1", "support"))
        writer.writeheader()
        writer.writerows(output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
