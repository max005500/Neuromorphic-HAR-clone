"""Read manual normal-activity intervals in native KFall frame coordinates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ACTIVITY_NAMES = (
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING",
    "STANDING", "LAYING", "ASCENDING_TRANSITION", "DESCENDING_TRANSITION",
    "CROUCHING_BENT",
)


def load_manual_labels(path: str | Path, *, extended: bool = False) -> tuple[dict[str, Any], dict[str, str]]:
    path = Path(path)
    content = path.read_bytes()
    data = json.loads(content)
    if not isinstance(data, dict) or data.get("format") != "kfall_manual_normal_labels":
        raise ValueError(f"{path}: expected kfall_manual_normal_labels format.")
    if data.get("version") != 1 or data.get("dataset") != "kfall" or data.get("sample_rate_hz") != 100:
        raise ValueError(f"{path}: unsupported manual-label version, dataset or sample rate.")
    if extended and (data.get("activity_label_scheme") != "uci_har_extended"
                     or data.get("activity_labels") != {str(i): name for i, name in enumerate(ACTIVITY_NAMES)}):
        raise ValueError(f"{path}: expected the uci_har_extended activity label definitions.")
    records = data.get("recordings")
    if not isinstance(records, dict):
        raise ValueError(f"{path}: missing recording annotations.")
    for key, record in records.items():
        if not isinstance(record, dict) or record.get("reviewed") is not True:
            raise ValueError(f"{path}: {key} is not a reviewed recording.")
        intervals = record.get("intervals")
        if not isinstance(intervals, list):
            raise ValueError(f"{path}: {key} has no interval list.")
        previous_end = 0
        for interval in intervals:
            if (not isinstance(interval, list) or len(interval) != 3
                    or any(type(value) is not int for value in interval)):
                raise ValueError(f"{path}: {key} has a malformed interval.")
            start, stop, label = interval
            if start < 1 or stop <= start or start < previous_end or label not in (1, 2):
                raise ValueError(f"{path}: {key} has invalid or overlapping normal intervals.")
            previous_end = stop
        if extended:
            activities = record.get("activity_intervals", [])
            if not isinstance(activities, list):
                raise ValueError(f"{path}: {key} has invalid activity intervals.")
            previous_end = 0
            for interval in activities:
                if (not isinstance(interval, list) or len(interval) != 3
                        or any(type(value) is not int for value in interval)):
                    raise ValueError(f"{path}: {key} has a malformed activity interval.")
                start, stop, label = interval
                if start < 1 or stop <= start or start < previous_end or not 0 <= label < len(ACTIVITY_NAMES):
                    raise ValueError(f"{path}: {key} has invalid or overlapping activity intervals.")
                previous_end = stop
    return records, {"sha256": hashlib.sha256(content).hexdigest()}
