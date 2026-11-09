"""Read manual normal-activity intervals in native KFall frame coordinates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_manual_labels(path: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = Path(path)
    content = path.read_bytes()
    data = json.loads(content)
    if not isinstance(data, dict) or data.get("format") != "kfall_manual_normal_labels":
        raise ValueError(f"{path}: expected kfall_manual_normal_labels format.")
    if data.get("version") != 1 or data.get("dataset") != "kfall" or data.get("sample_rate_hz") != 100:
        raise ValueError(f"{path}: unsupported manual-label version, dataset or sample rate.")
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
    return records, {"sha256": hashlib.sha256(content).hexdigest()}
