"""Read the held-out KFall cache recorded by a training run, without rebuilding it."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from loaders.kfall_event_loader import KFallEventLoaderConfig, window_targets_for_record
from preprocess.kfall_labels import class_names, label_at_frame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "val", "test")
ARRAY_KEYS = (
    "x_ev", "offset", "y_h1", "y_h2", "src_h2", "w_h2", "file_idx",
    "start", "subject", "end_frame", "lead_time_ms",
)
RUNTIME_CONFIG = {"batch_size", "num_workers", "pin_memory", "verbose_every"}
CHECKPOINT_DATA_KEYS = {
    "seed": "seed", "split": "split", "raw_win": "raw_win", "out_win": "out_win",
    "stride": "stride", "n_ch": "n_ch", "fall_policy": "fall_policy",
    "theta": "theta_sd", "dead_zone": "dead_zone", "sd_init": "sd_init",
    "post_impact_seconds": "post_impact_seconds", "endpoint_context": "endpoint_context",
    "resample_hz": "resample_hz",
}
OPTIONAL_DATA_DEFAULTS = {"post_impact_seconds": 0.0, "endpoint_context": None, "resample_hz": None}


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation metadata not found: {path}. Use the original training run and its saved event cache.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read evaluation metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation metadata must be a JSON object: {path}")
    return value


def _normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalized(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _data_config(metadata: dict[str, Any]) -> dict[str, Any]:
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("KFall evaluation metadata has no preprocessing config.")
    normalized = {key: value for key, value in config.items() if key not in RUNTIME_CONFIG}
    for key, default in OPTIONAL_DATA_DEFAULTS.items():
        normalized.setdefault(key, default)
    return normalized


def _record_identity(record: dict[str, Any]) -> dict[str, Any]:
    # File prefixes can change when the source dataset or cache is relocated.
    identity = dict(record)
    identity["file"] = str(identity["file"]).replace("\\", "/").rsplit("/", 1)[-1]
    return identity


def _resolve_cache(checkpoint_config: dict[str, Any], override: str | Path | None) -> Path:
    if override is not None:
        candidate = Path(override).expanduser()
    else:
        if "cache_dir" not in checkpoint_config or "cache_name" not in checkpoint_config:
            raise ValueError("Checkpoint config has no cache_dir/cache_name; specify --cache-dir explicitly.")
        name = str(checkpoint_config["cache_name"])
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Checkpoint cache_name must be a simple cache name.")
        candidate = Path(checkpoint_config["cache_dir"]).expanduser() / f"kfall_event_loader_{name}"
    if not candidate.is_absolute() and not candidate.exists() and (PROJECT_ROOT / candidate).exists():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _validate_partition_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("dataset") != "kfall" or metadata.get("split_strategy") != "subject":
        raise ValueError("Evaluation requires a KFall cache with split_strategy='subject'.")
    subjects = metadata.get("split_subjects")
    files = metadata.get("split_file_indices")
    records = metadata.get("recordings")
    if not isinstance(subjects, dict) or not isinstance(files, dict) or not isinstance(records, list) or not records:
        raise ValueError("KFall metadata is missing subject, file, or recording manifests.")
    for record in records:
        required = {"file", "subject", "code", "trial", "T", "label_id"}
        if not isinstance(record, dict) or required - record.keys():
            raise ValueError("KFall recording manifest has incomplete recording metadata.")
        if type(record["T"]) is not int or record["T"] < 1 or record["label_id"] not in (0, 1, 2):
            raise ValueError("KFall recording manifest has an invalid length or activity label.")
        if record["label_id"] == 0:
            temporal = {"onset_idx", "impact_idx", "fall_onset_frame", "fall_impact_frame"}
            if temporal - record.keys():
                raise ValueError("KFall fall recording is missing temporal annotation metadata.")
    subject_sets, file_sets = {}, {}
    for split in SPLIT_NAMES:
        subject_list, file_list = subjects.get(split), files.get(split)
        if not isinstance(subject_list, list) or not subject_list or len(set(subject_list)) != len(subject_list):
            raise ValueError(f"Invalid or empty {split} subject manifest.")
        if not isinstance(file_list, list) or not file_list or any(type(index) is not int for index in file_list):
            raise ValueError(f"Invalid or empty {split} recording manifest.")
        if len(set(file_list)) != len(file_list) or any(not 0 <= index < len(records) for index in file_list):
            raise ValueError(f"Invalid {split} recording indices.")
        subject_sets[split], file_sets[split] = set(subject_list), set(file_list)
        if any(records[index].get("subject") not in subject_sets[split] for index in file_list):
            raise ValueError(f"{split} recording subjects disagree with the subject manifest.")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if subject_sets[left] & subject_sets[right] or file_sets[left] & file_sets[right]:
            raise ValueError(f"Subject or recording leakage between {left} and {right}.")
    if set.union(*file_sets.values()) != set(range(len(records))):
        raise ValueError("Recording manifests do not cover all saved recordings.")


def load_evaluation_data(
    run_dir: str | Path, checkpoint_config: dict[str, Any], cache_dir: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any], Path]:
    """Return the original held-out test arrays after checking training provenance.

    Only the test arrays are mapped, all read-only. Metadata from the run is
    required so an overwritten cache cannot silently change the evaluation.
    No source CSV, workbook, raw preprocessing, or train/validation arrays are
    accessed. Source signatures are compared to the saved training snapshot.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    run_metadata = _read_metadata(run_dir / "dataset_metadata.json")
    resolved_cache = _resolve_cache(checkpoint_config, cache_dir)
    cache_metadata = _read_metadata(resolved_cache / "metadata.json")
    _validate_partition_metadata(run_metadata)
    _validate_partition_metadata(cache_metadata)
    config = _data_config(run_metadata)
    if _normalized(config) != _normalized(_data_config(cache_metadata)):
        raise ValueError("Event cache preprocessing config differs from the training run.")
    for saved_key, data_key in CHECKPOINT_DATA_KEYS.items():
        if (saved_key not in checkpoint_config and saved_key not in OPTIONAL_DATA_DEFAULTS) or data_key not in config:
            raise ValueError(f"Missing checkpoint/preprocessing setting: {saved_key}.")
        saved_value = checkpoint_config.get(saved_key, OPTIONAL_DATA_DEFAULTS.get(saved_key))
        if _normalized(saved_value) != _normalized(config[data_key]):
            raise ValueError(f"Checkpoint {saved_key} differs from the training dataset config.")
    if checkpoint_config.get("dataset", "kfall") != "kfall":
        raise ValueError("The checkpoint belongs to a different dataset.")
    for key in ("version", "source_signature", "channels", "units", "fall_label_rule", "head2_labels_available", "raw_data", "split_summary", "split_file_indices"):
        if key not in run_metadata or key not in cache_metadata or run_metadata[key] != cache_metadata[key]:
            raise ValueError(f"Event cache {key} differs from the training snapshot.")
    for split in SPLIT_NAMES:
        if set(run_metadata["split_subjects"][split]) != set(cache_metadata["split_subjects"][split]):
            raise ValueError(f"Event cache {split} subjects differ from the training snapshot.")
    original_records = [_record_identity(record) for record in run_metadata["recordings"]]
    cached_records = [_record_identity(record) for record in cache_metadata["recordings"]]
    if original_records != cached_records:
        raise ValueError("Event cache recording manifest differs from the training snapshot.")

    allowed = {field.name for field in fields(KFallEventLoaderConfig)}
    cfg = KFallEventLoaderConfig(**{key: value for key, value in run_metadata["config"].items() if key in allowed})
    n_classes = 4 if cfg.post_impact_seconds > 0 else 3
    names = list(class_names(n_classes).values())
    for label, source in (("training snapshot", run_metadata), ("event cache", cache_metadata)):
        if source.get("n_classes", n_classes) != n_classes:
            raise ValueError(f"The {label} class count differs from its post-impact label configuration.")
        if source.get("class_names", names) != names:
            raise ValueError(f"The {label} class names differ from the checkpoint's KFall label scheme.")
    if checkpoint_config.get("n_classes", n_classes) != n_classes:
        raise ValueError("Checkpoint class count differs from the training dataset class scheme.")
    arrays = {}
    for key in ARRAY_KEYS:
        path = resolved_cache / "test" / f"{key}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Original test cache is incomplete: {path}. Evaluation will not rebuild it.")
        arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    count = len(arrays["y_h1"])
    if count == 0:
        raise ValueError("The held-out test split has no windows.")
    shapes = {key: (count,) for key in ARRAY_KEYS}
    shapes.update(x_ev=(count, cfg.out_win, 6, 2), offset=(count, 6))
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"Invalid test {key} shape: expected {shape}, got {arrays[key].shape}.")
    for key in ("y_h1", "file_idx", "start", "end_frame", "src_h2"):
        if not np.issubdtype(arrays[key].dtype, np.integer):
            raise ValueError(f"Test {key} must use an integer dtype.")
    if np.any(arrays["end_frame"] < 1):
        raise ValueError("Test end_frame must contain positive FrameCounter values.")
    if arrays["x_ev"].dtype != np.uint8:
        raise ValueError("Test x_ev must contain uint8 binary events.")
    if not np.isin(arrays["y_h1"], range(n_classes)).all():
        raise ValueError(f"Test activity labels must use the checkpoint's {n_classes}-class KFall scheme.")
    for key in ("offset", "y_h2", "w_h2"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Test {key} contains non-finite values.")
    # Bound temporary memory when validating a large event cache.
    for begin in range(0, count, 256):
        events = arrays["x_ev"][begin:begin + 256]
        if not np.isin(events, (0, 1)).all() or np.any(events[..., 0] & events[..., 1]):
            raise ValueError("Test x_ev contains invalid or simultaneous ON/OFF events.")
    expected_files, expected_starts, expected_targets = [], [], []
    records = run_metadata["recordings"]
    for index in run_metadata["split_file_indices"]["test"]:
        record = records[index]
        starts, targets = window_targets_for_record(np.empty((int(record["T"]), 0)), record, cfg)
        expected_files.extend([index] * len(starts))
        expected_starts.extend(starts.tolist())
        expected_targets.extend(targets.tolist())
    if not np.array_equal(arrays["file_idx"], expected_files) or not np.array_equal(arrays["start"], expected_starts):
        raise ValueError("Test window file/start manifest differs from the original training protocol.")
    if not np.array_equal(arrays["y_h1"], expected_targets):
        raise ValueError("Test window activity labels differ from the original temporal labeling protocol.")
    if count != run_metadata["split_summary"]["test"].get("windows"):
        raise ValueError("Test window count differs from the training snapshot.")
    if set(np.unique(arrays["subject"]).tolist()) != set(run_metadata["split_subjects"]["test"]):
        raise ValueError("Test window subjects differ from the training snapshot.")
    for index in np.unique(arrays["file_idx"]):
        record = records[int(index)]
        mask = arrays["file_idx"] == index
        if not np.all(arrays["subject"][mask] == record["subject"]):
            raise ValueError("Test window subject/activity labels disagree with their source recording.")
        if int(record["label_id"]) == 0:
            endpoints = arrays["end_frame"][mask]
            expected_labels = [label_at_frame(int(frame), record,
                post_impact_seconds=cfg.post_impact_seconds, sample_rate_hz=cfg.sample_rate_hz,
                fall_policy=cfg.fall_policy) for frame in endpoints]
            valid = np.array([label is not None and label == target
                              for label, target in zip(expected_labels, arrays["y_h1"][mask])])
            impact = int(record["fall_impact_frame"])
            lead = (impact - endpoints) * 1000.0 / cfg.sample_rate_hz
            if not valid.all() or not np.allclose(arrays["lead_time_ms"][mask], lead):
                raise ValueError("Test fall windows violate their annotated temporal policy.")
        else:
            if not np.all(arrays["y_h1"][mask] == record["label_id"]):
                raise ValueError("Test window subject/activity labels disagree with their source recording.")
            if not np.isnan(arrays["lead_time_ms"][mask]).all():
                raise ValueError("Non-fall test windows must not have an annotated impact lead time.")
    counts = {str(label): int(np.sum(arrays["y_h1"] == label)) for label in np.unique(arrays["y_h1"])}
    if counts != run_metadata["split_summary"]["test"].get("class_counts"):
        raise ValueError("Test class counts differ from the training snapshot.")
    return arrays, cache_metadata, run_metadata, resolved_cache
