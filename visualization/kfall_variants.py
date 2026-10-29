"""Join existing KFall variants by recording and prediction time, read-only."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from visualization.kfall_data import KFallWindowStore, SPLIT_NAMES, WindowData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON = Path("results/kfall_window_comparison_200hz_calibrated/comparison.json")
PREPARE_COMMAND = (
    "python compare_kfall_windows.py --mode preprocess --durations 2.05 3.30 4.10 "
    "--resample-hz 200 --theta-file configs/kfall_theta_200hz.json "
    "--post-impact-seconds 1.20 --endpoint-policy matched --workers 0 "
    "--results-dir results/kfall_window_comparison_200hz_calibrated"
)
_VARYING_CONFIG = {
    "raw_win", "out_win", "endpoint_context", "batch_size", "num_workers",
    "pin_memory", "verbose_every",
}


def _basename(path: str | Path) -> str:
    return str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return str(record["subject"]), str(record["code"]), str(record["trial"])


def _subjects(metadata: dict[str, Any]) -> dict[str, list[str]]:
    groups = metadata.get("split_subjects")
    if not isinstance(groups, dict) or metadata.get("split_strategy") != "subject":
        raise ValueError("KFall variant comparison requires subject-separated cache metadata.")
    normalized = {}
    for split in SPLIT_NAMES:
        values = groups.get(split)
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"Invalid {split} subject manifest in KFall variant.")
        normalized[split] = sorted(values)
    if any(set(normalized[a]) & set(normalized[b])
           for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("KFall variants contain subjects in more than one partition.")
    return normalized


def _check_manifest_row(store: KFallWindowStore, row: dict[str, Any]) -> None:
    for name in ("raw_win", "out_win", "stride", "resample_hz", "post_impact_seconds", "endpoint_context"):
        if name in row and row[name] != getattr(store.cfg, name):
            raise ValueError(f"Comparison manifest {name} differs from cache {store.cache_dir.name}.")
    if "duration_seconds" in row and not np.isclose(
        float(row["duration_seconds"]), store.cfg.raw_win / store.cfg.sample_rate_hz, rtol=0, atol=1e-8,
    ):
        raise ValueError("Comparison manifest duration differs from the cached window length.")
    for name in ("source_signature", "split_subjects"):
        if name in row and row[name] != store.metadata.get(name):
            raise ValueError(f"Comparison manifest {name} differs from cache {store.cache_dir.name}.")


class KFallVariantStore:
    """Compare only common endpoint-labelled windows from compatible caches.

    Split arrays are memory-mapped only on first access to that partition. Source
    CSVs are read and encoded solely when ``get_windows`` verifies a selection.
    Common indices identify sorted recording/endpoint/label keys, rather than an
    assumed common numeric row number in the individual caches.
    """

    def __init__(
        self, comparison: str | Path | None = None, *,
        cache_dirs: Sequence[str | Path] | None = None,
        dataset_root: str | Path | None = None,
    ) -> None:
        if comparison is not None and cache_dirs is not None:
            raise ValueError("Use a comparison manifest or explicit cache directories, not both.")
        self.comparison_path: Path | None = None
        if cache_dirs is None:
            manifest = Path(comparison or DEFAULT_COMPARISON).expanduser()
            if not manifest.is_file() and not manifest.is_absolute() and (PROJECT_ROOT / manifest).is_file():
                manifest = PROJECT_ROOT / manifest
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"KFall comparison manifest not found: {manifest}. Prepare the variants first: "
                    f"{PREPARE_COMMAND}. Alternatively, specify --cache-dirs with existing caches."
                )
            self.comparison_path = manifest.resolve()
            rows = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or any(not isinstance(row, dict) or "cache_dir" not in row for row in rows):
                raise ValueError("comparison.json must contain a list of variants with cache_dir.")
        else:
            rows = [{"cache_dir": str(path)} for path in cache_dirs]
        if len(rows) < 2:
            raise ValueError("Select at least two KFall window variants for comparison.")
        selected = []
        for row in rows:
            source = Path(row["cache_dir"]).expanduser()
            candidates = [source, PROJECT_ROOT / "data" / "kfall_cache" / _basename(source)]
            if self.comparison_path is not None:
                candidates += [self.comparison_path.parent / source,
                               self.comparison_path.parent / _basename(source)]
                if not source.is_absolute():
                    candidates.append(PROJECT_ROOT / source)
                    candidates.extend(parent / source for parent in self.comparison_path.parents)
            found, failures = None, []
            for candidate in dict.fromkeys(path.resolve() for path in candidates):
                if not (candidate / "metadata.json").is_file():
                    continue
                try:
                    store = KFallWindowStore(candidate, dataset_root=dataset_root)
                    _check_manifest_row(store, row)
                except (ValueError, KeyError, OSError) as exc:
                    failures.append(str(exc))
                    continue
                found = store
                break
            if found is None:
                detail = f" Cache validation: {'; '.join(failures)}" if failures else ""
                raise FileNotFoundError(
                    f"KFall variant cache not found or incompatible: {source}.{detail} "
                    f"Prepare it with: {PREPARE_COMMAND}"
                )
            selected.append((found, row))
        selected.sort(key=lambda pair: pair[0].cfg.raw_win)
        self.stores = [pair[0] for pair in selected]
        self._manifest_rows = [pair[1] for pair in selected]
        self.durations = [store.cfg.raw_win / store.cfg.sample_rate_hz for store in self.stores]
        if len(set(self.durations)) != len(self.durations):
            raise ValueError("Select distinct KFall window lengths; duplicate durations cannot be compared.")
        self.cache_dirs = [store.cache_dir for store in self.stores]
        self.class_names = dict(self.stores[0].class_names)
        self.n_classes = self.stores[0].n_classes
        self._validate_compatibility()
        self._keys: dict[str, list[tuple[str, str, str, int, int]]] = {}
        self._joined: dict[str, np.ndarray] = {}
        self.coverage: dict[str, dict[str, Any]] = {}

    def _validate_compatibility(self) -> None:
        reference = None
        for store in self.stores:
            cfg = {key: value for key, value in asdict(store.cfg).items() if key not in _VARYING_CONFIG}
            subjects = _subjects(store.metadata)
            if store.metadata.get("n_classes", store.n_classes) != store.n_classes or (
                store.metadata.get("class_names", list(store.class_names.values())) != list(store.class_names.values())
            ):
                raise ValueError("KFall cache class metadata disagrees with its temporal label policy.")
            records = {}
            for record in store.records:
                key = _record_key(record)
                if key in records:
                    raise ValueError("KFall recording manifest has duplicate subject/task/trial identities.")
                records[key] = {**record, "file": _basename(record["file"])}
            signature = {
                "config": cfg, "subjects": subjects, "recordings": records,
                "source_signature": store.metadata.get("source_signature"),
                "channels": store.metadata.get("channels"), "units": store.metadata.get("units"),
                "classes": store.class_names,
            }
            if any(signature[name] is None for name in ("source_signature", "channels", "units")):
                raise ValueError("KFall variant metadata is missing source, channel, or physical-unit provenance.")
            if reference is None:
                reference = signature
            else:
                for name, value in signature.items():
                    if value != reference[name]:
                        raise ValueError(f"KFall variants have incompatible {name}; compare matching preprocessing and subjects.")

    def _ensure_split(self, split: str) -> None:
        if split in self._joined:
            return
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown partition {split!r}; choose train, val, or test.")
        lookups = []
        endpoint_labels: dict[tuple[str, str, str, int], int] = {}
        counts = []
        for store, manifest_row in zip(self.stores, self._manifest_rows):
            arrays = store._load_split(split)
            for field in ("file_idx", "end_frame", "y_h1"):
                if not np.issubdtype(arrays[field].dtype, np.integer):
                    raise ValueError(f"Cached {split} {field} must contain integers.")
            expected = manifest_row.get("endpoint_signatures", {}).get(split)
            if expected is not None:
                manifest = np.column_stack([arrays[key] for key in ("file_idx", "end_frame", "y_h1")])
                actual = hashlib.sha256(manifest.astype("<i8", copy=False).tobytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"Cached {split} endpoints differ from the comparison manifest.")
            lookup = {}
            allowed_subjects = set(store.metadata["split_subjects"][split])
            for index, (file_index, frame, label) in enumerate(zip(
                arrays["file_idx"], arrays["end_frame"], arrays["y_h1"],
            )):
                if not 0 <= int(file_index) < len(store.records):
                    raise ValueError("Cached variant has an invalid recording index.")
                record = store.records[int(file_index)]
                subject = str(arrays["subject"][index])
                if subject != record["subject"] or subject not in allowed_subjects:
                    raise ValueError("Cached variant subjects disagree with their partition or recording manifest.")
                if int(label) not in self.class_names or int(frame) < 1:
                    raise ValueError("Cached variant has an invalid target class or endpoint frame.")
                endpoint = (*_record_key(record), int(frame))
                if endpoint in endpoint_labels and endpoint_labels[endpoint] != int(label):
                    raise ValueError("KFall variants assign different labels to the same recording endpoint.")
                endpoint_labels[endpoint] = int(label)
                key = (*endpoint, int(label))
                if key in lookup:
                    raise ValueError("Cached variant contains a duplicate recording/endpoint/label window.")
                lookup[key] = index
            counts.append(len(lookup))
            lookups.append(lookup)
        common = sorted(set(lookups[0]).intersection(*(set(lookup) for lookup in lookups[1:])))
        self._keys[split] = common
        self._joined[split] = np.asarray([[lookup[key] for lookup in lookups] for key in common], dtype=np.int64).reshape(-1, len(self.stores))
        self.coverage[split] = {
            "common_windows": len(common), "variant_windows": counts,
            "discarded_windows": [count - len(common) for count in counts],
        }

    def split_size(self, split: str) -> int:
        self._ensure_split(split)
        return len(self._keys[split])

    def subjects(self, split: str) -> list[str]:
        self._ensure_split(split)
        return sorted({key[0] for key in self._keys[split]})

    def get_indices(self, split: str, class_id: int | None = None, subject: str | None = None) -> np.ndarray:
        self._ensure_split(split)
        if class_id is not None and class_id not in self.class_names:
            raise ValueError(f"class_id must belong to this comparison's classes: {self.class_names}.")
        selected_subject = None
        if subject is not None and str(subject).strip().lower() not in {"", "all", "todos", "todas"}:
            match = re.fullmatch(r"(?:SA|S)?(\d{1,2})", str(subject).strip().upper())
            if match is None:
                raise ValueError("Subject must be a KFall ID such as SA06, or Todos.")
            selected_subject = f"SA{int(match.group(1)):02d}"
        return np.asarray([
            index for index, key in enumerate(self._keys[split])
            if (class_id is None or key[-1] == class_id)
            and (selected_subject is None or key[0] == selected_subject)
        ], dtype=np.int64)

    def get_windows(self, split: str, index: int) -> list[WindowData]:
        self._ensure_split(split)
        if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < len(self._keys[split]):
            raise IndexError(f"Common window index must be in [0, {len(self._keys[split]) - 1}] for {split}.")
        windows = [store.get_window(split, int(row))
                   for store, row in zip(self.stores, self._joined[split][int(index)])]
        if any((window.subject, window.code, window.trial, window.end_frame, window.class_id)
               != self._keys[split][int(index)] for window in windows):
            raise ValueError("Selected variant windows no longer share the same recording, endpoint and label.")
        return windows
