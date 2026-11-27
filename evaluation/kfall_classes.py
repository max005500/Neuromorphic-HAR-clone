"""Resolve the saved KFall activity scheme without changing old checkpoints."""

from __future__ import annotations

from typing import Any


LEGACY_NAMES = ("Fall", "Dynamic", "Static", "PostImpact")


def class_count(config: dict[str, Any]) -> int:
    scheme = config.get("label_scheme", "legacy")
    if scheme not in ("legacy", "extended"):
        raise ValueError(f"Unknown KFall label_scheme: {scheme!r}.")
    return (10 if scheme == "extended" else 3) + int(config.get("post_impact_seconds", 0.0) > 0)


def resolve_class_names(metadata: dict[str, Any], config: dict[str, Any]) -> tuple[str, ...]:
    count = class_count(config)
    if "n_classes" in config and config["n_classes"] != count:
        raise ValueError("Checkpoint class count disagrees with its label scheme and post-impact configuration.")
    names = metadata.get("class_names")
    if names is None:
        names = config.get("class_names")
    if names is None and config.get("label_scheme", "legacy") == "legacy":
        names = LEGACY_NAMES[:count]
    if isinstance(names, dict):
        if set(map(str, names)) != {str(index) for index in range(count)}:
            raise ValueError("KFall class_names must contain exactly one name for each class ID.")
        names = [names[str(i)] if str(i) in names else names[i] for i in range(count)]
    if not isinstance(names, (list, tuple)) or len(names) != count:
        raise ValueError(f"KFall class_names must contain {count} ordered names.")
    if (any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != count or set(names) & {"accuracy", "macro avg", "weighted avg", "micro avg"}):
        raise ValueError("KFall class_names must be unique nonempty names, distinct from report aggregate keys.")
    return tuple(names)
