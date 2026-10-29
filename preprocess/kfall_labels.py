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
    sample_rate_hz: float = 100.0, fall_policy: str = "pre_impact",
) -> int | None:
    """Label a window's last observed frame; unknown fall-trial phases stay unknown.

    Fall occupies [onset, impact). With four classes Postimpact occupies
    [impact, impact + duration * sample_rate). A whole input window may contain
    several phases: its endpoint determines the target.
    """
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
