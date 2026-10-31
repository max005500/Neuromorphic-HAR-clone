"""KFall endpoint labels and duration conversion shared by all workflows."""

from __future__ import annotations

from typing import Any

import numpy as np

FALL = 0
DYNAMIC = 1
STATIC = 2
POSTIMPACT = 3


def class_names(n_classes: int = 3) -> dict[int, str]:
    if n_classes not in (3, 4):
        raise ValueError("KFall supports three legacy classes or four classes including Postimpact.")
    names = {FALL: "Fall", DYNAMIC: "Dynamic", STATIC: "Static"}
    if n_classes == 4:
        names[POSTIMPACT] = "Postimpact"
    return names


def window_samples(seconds: float, sample_rate_hz: float = 100.0) -> int:
    samples = float(seconds) * float(sample_rate_hz)
    if not np.isfinite(samples) or samples < 2 or not np.isclose(samples, round(samples), rtol=0, atol=1e-8):
        raise ValueError("Window duration must represent at least two whole samples at the sensor rate.")
    return int(round(samples))


def event_steps_for_samples(raw_win: int) -> int:
    """Preserve the original 205-to-2048 temporal scale when context grows."""
    if not isinstance(raw_win, (int, np.integer)) or raw_win < 2:
        raise ValueError("raw_win must be an integer >= 2.")
    return int(round(int(raw_win) * 2048 / 205))


def label_at_frame(
    frame: int, record: dict[str, Any], *, post_impact_seconds: float = 0.0,
    sample_rate_hz: float = 100.0, fall_policy: str = "pre_impact", **policy: Any,
) -> int | None:
    """Label a window's last observed frame; unknown fall-trial phases stay unknown.

    Fall occupies [onset, impact). With four classes Postimpact occupies
    [impact, impact + duration * sample_rate). A whole input window may contain
    several phases: its endpoint determines the target.
    """
    if policy.get("label_policy", "legacy") != "legacy":
        if "frame_label_intervals" in record:
            value = int(labels_from_intervals(np.array([frame]), record["frame_label_intervals"])[0])
        else:
            frames = record_frames(record)
            index = int(np.searchsorted(frames, frame))
            if index >= len(frames) or frames[index] != frame:
                return None
            value = int(record_frame_labels(frames, record, post_impact_seconds=post_impact_seconds,
                sample_rate_hz=sample_rate_hz, fall_policy=fall_policy, **policy)[index])
        return None if value < 0 else value
    recording_label = int(record["label_id"])
    if recording_label in (DYNAMIC, STATIC):
        return recording_label
    if recording_label != FALL:
        raise ValueError("KFall recording labels must be Fall, Dynamic, or Static.")
    onset, impact = int(record["fall_onset_frame"]), int(record["fall_impact_frame"])
    if onset <= frame < impact:
        return FALL
    if post_impact_seconds > 0:
        if impact <= frame < impact + post_impact_seconds * sample_rate_hz:
            return POSTIMPACT
    elif fall_policy == "annotated_interval" and frame == impact:
        return FALL
    return None


LABEL_POLICY_DEFAULTS = {
    "label_policy": "legacy", "adl_margin_seconds": 1.5,
    "pre_fall_seconds": 1.2, "post_recovery_seconds": 2.0,
    "motion_window_seconds": 0.25, "motion_acc_threshold": 0.08,
    "motion_gyr_threshold": 10.0,
}


def label_policy_kwargs(config: Any) -> dict[str, Any]:
    """Normalize missing fields from old caches/checkpoints to the legacy rules."""
    return {key: config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)
            for key, default in LABEL_POLICY_DEFAULTS.items()}


def record_frames(record: dict[str, Any]) -> np.ndarray:
    """Reconstruct the original frame grid from its lossless contiguous runs."""
    if "frames" in record:
        return np.asarray(record["frames"], dtype=np.int64)
    if "frame_runs" in record:
        result = np.empty(int(record["T"]), dtype=np.int64)
        position = 0
        for row, first, count in record["frame_runs"]:
            if row != position or count < 1 or position + count > len(result):
                raise ValueError("Invalid saved FrameCounter runs.")
            result[position:position + count] = np.arange(first, first + count, dtype=np.int64)
            position += count
        if position != len(result) or np.any(np.diff(result) <= 0):
            raise ValueError("Saved FrameCounter runs do not cover the recording.")
        return result
    return np.arange(int(record.get("first_frame", 1)),
                     int(record.get("first_frame", 1)) + int(record["T"]), dtype=np.int64)


def labels_from_intervals(frames: np.ndarray, intervals: list) -> np.ndarray:
    """Read persisted targets without requiring source signals at evaluation."""
    frames = np.asarray(frames, dtype=np.int64)
    bounds = np.asarray(intervals, dtype=np.int64)
    if bounds.ndim != 2 or bounds.shape[1] != 3 or not len(bounds):
        raise ValueError("Invalid frame_label_intervals metadata.")
    if (np.any(bounds[:, 1] <= bounds[:, 0]) or np.any(bounds[1:, 0] < bounds[:-1, 1])
            or not np.isin(bounds[:, 2], [FALL, DYNAMIC, STATIC, POSTIMPACT]).all()):
        raise ValueError("Overlapping or invalid saved label intervals.")
    indices = np.searchsorted(bounds[:, 0], frames, side="right") - 1
    valid = (indices >= 0) & (frames < bounds[np.maximum(indices, 0), 1])
    result = np.full(frames.shape, -1, dtype=np.int64)
    result[valid] = bounds[indices[valid], 2]
    return result


def record_frame_labels(
    frames: np.ndarray, record: dict[str, Any], *, post_impact_seconds: float = 0.0,
    sample_rate_hz: float = 100.0, fall_policy: str = "pre_impact", label_policy: str = "legacy",
    adl_margin_seconds: float = 1.5, pre_fall_seconds: float = 1.2,
    post_recovery_seconds: float = 2.0, signal: np.ndarray | None = None,
    motion_window_seconds: float = 0.25, motion_acc_threshold: float = 0.08,
    motion_gyr_threshold: float = 10.0,
) -> np.ndarray:
    """Assign native samples, preserving official fall/postimpact intervals.

    ``temporal`` ignores the old activity family: ADLs are dynamic only in the
    middle; fall trials add pre-fall and recovery dynamic intervals. All other
    samples are static. ``motion`` explicitly replaces dynamic/static decisions
    using trailing per-axis ranges; its fixed physical thresholds are heuristic,
    not calibrated ground truth. Neither policy changes Fall or Postimpact.
    """
    frames = np.asarray(frames, dtype=np.int64)
    if frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError("Labels require a nonempty increasing FrameCounter grid.")
    if label_policy not in {"legacy", "temporal", "motion"}:
        raise ValueError("label_policy must be legacy, temporal or motion.")
    if label_policy == "legacy":
        return np.array([value if (value := label_at_frame(int(frame), record,
            post_impact_seconds=post_impact_seconds, sample_rate_hz=sample_rate_hz,
            fall_policy=fall_policy)) is not None else -1 for frame in frames], dtype=np.int64)
    if signal is None and "frame_label_intervals" in record:
        return labels_from_intervals(frames, record["frame_label_intervals"])
    result = np.full(len(frames), STATIC, dtype=np.int64)
    if int(record["label_id"]) != FALL:
        # The observed recording covers [first_frame, last_frame + 1).
        left = frames[0] + round(adl_margin_seconds * sample_rate_hz)
        right = frames[-1] + 1 - round(adl_margin_seconds * sample_rate_hz)
        result[(frames >= left) & (frames < right)] = DYNAMIC
    else:
        onset, impact = int(record["fall_onset_frame"]), int(record["fall_impact_frame"])
        post_end = impact + round(post_impact_seconds * sample_rate_hz)
        result[(frames >= onset - round(pre_fall_seconds * sample_rate_hz)) & (frames < onset)] = DYNAMIC
        result[(frames >= post_end) &
               (frames < post_end + round(post_recovery_seconds * sample_rate_hz))] = DYNAMIC
        result[(frames >= onset) & (frames < impact)] = FALL
        if post_impact_seconds > 0:
            result[(frames >= impact) & (frames < post_end)] = POSTIMPACT
        elif fall_policy == "annotated_interval":
            result[frames == impact] = FALL
    if label_policy == "motion":
        if signal is None:
            raise ValueError("Motion labels require the original six-channel signal or saved label intervals.")
        signal = np.asarray(signal)
        if signal.shape != (len(frames), 6) or not np.isfinite(signal).all():
            raise ValueError("Motion labeling requires finite aligned six-channel signals.")
        # Work in physical time so missing native frames do not stretch history.
        starts = np.searchsorted(frames, frames - motion_window_seconds * sample_rate_hz, side="left")
        moving = np.zeros(len(frames), dtype=bool)
        for end, start in enumerate(starts):
            ranges = np.ptp(signal[start:end + 1], axis=0)
            moving[end] = bool(np.any(ranges[:3] >= motion_acc_threshold)
                               or np.any(ranges[3:] >= motion_gyr_threshold))
        unprotected = (result != FALL) & (result != POSTIMPACT)
        result[unprotected] = np.where(moving[unprotected], DYNAMIC, STATIC)
    return result


def label_manifest(frames: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Compact, lossless frame grid and labels for raw-free provenance checks."""
    frames = np.asarray(frames, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if frames.shape != labels.shape or not len(frames):
        raise ValueError("Frame and label manifests must align.")
    starts = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1]
    stops = np.r_[starts[1:], len(frames)]
    runs = [[int(a), int(frames[a]), int(b - a)] for a, b in zip(starts, stops)]
    starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
    stops = np.r_[starts[1:], len(frames)]
    intervals = [[int(frames[a]), int(frames[b - 1]) + 1, int(labels[a])]
                 for a, b in zip(starts, stops)]
    return {"first_frame": int(frames[0]), "last_frame": int(frames[-1]),
            "frame_runs": runs, "frame_label_intervals": intervals}
