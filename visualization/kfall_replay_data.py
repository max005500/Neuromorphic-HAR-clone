"""Causal sliding-window inference on one original held-out KFall recording."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_kfall import _device, _load_checkpoint, _subject
from evaluation.kfall_data import load_evaluation_data
from loaders.kfall_event_loader import KFallEventLoaderConfig, signal_to_events
from preprocess.preprocess_kfall import FILE_RE, load_record
from preprocess.kfall_labels import class_names
from train.utils import extract_single_logits, make_autocast, set_seed
from train_kfall import build_model
from visualization.kfall_data import KFallWindowStore


@dataclass(frozen=True)
class ReplayData:
    record: dict[str, Any]
    path: Path
    raw: np.ndarray
    frames: np.ndarray
    times: np.ndarray
    endpoint_rows: np.ndarray
    endpoint_times: np.ndarray
    probabilities: np.ndarray
    predicted_classes: np.ndarray
    raw_win: int
    out_win: int
    prediction_stride: int
    checkpoint: Path
    best_epoch: int
    best_val_loss: float
    fall_onset_time: float | None
    fall_impact_time: float | None
    config: dict[str, Any]
    cache_dir: Path
    subject: str
    label: str
    post_impact_seconds: float = 0.0
    post_impact_end_time: float | None = None


def prediction_index_at(replay: ReplayData, time: float) -> int | None:
    """Return the last prediction already available at the displayed time."""
    if not np.isfinite(time):
        raise ValueError("Replay time must be finite.")
    index = int(np.searchsorted(replay.endpoint_times, time, side="right")) - 1
    return index if index >= 0 else None


def _matches_recording(record: dict[str, Any], selection: str | Path) -> bool:
    name = str(selection).replace("\\", "/").rsplit("/", 1)[-1]
    recorded = Path(record["file"])
    if name.lower() in {recorded.name.lower(), recorded.stem.lower()}:
        return True
    match = FILE_RE.fullmatch(name if name.lower().endswith(".csv") else name + ".csv")
    return bool(match and (f"SA{match.group(1)}", int(match.group(2)), int(match.group(3)))
                == (record["subject"], record["task_id"], record["trial_id"]))


def _select_recording(
    metadata: dict[str, Any], split: dict[str, np.ndarray], *,
    recording: str | Path | None, window: int | None, subject: str | None, raw_win: int,
) -> int:
    test_files = metadata["split_file_indices"]["test"]
    records = metadata["recordings"]
    if subject is not None and subject not in metadata["split_subjects"]["test"]:
        raise ValueError(f"Subject {subject} is not in this run's held-out test partition.")
    candidates = [index for index in test_files if subject is None or records[index]["subject"] == subject]
    if recording is not None:
        candidates = [index for index in candidates if _matches_recording(records[index], recording)]
        if not candidates:
            raise ValueError(f"Recording {recording!s} is not in the selected original test partition.")
    if window is not None:
        if not isinstance(window, (int, np.integer)) or not 0 <= int(window) < len(split["file_idx"]):
            raise ValueError(f"window must be an actual test-cache index in [0, {len(split['file_idx']) - 1}].")
        file_index = int(split["file_idx"][int(window)])
        if file_index not in candidates:
            raise ValueError("The selected test window conflicts with the recording or subject filter.")
        candidates = [file_index]
    if not candidates:
        raise ValueError("No recording matches the selected test subject.")
    eligible = [index for index in candidates if records[index]["T"] >= raw_win]
    if not eligible:
        raise ValueError(f"The selected test recording has fewer than {raw_win} samples.")
    # Choose a fall for the default demonstration. Explicit recording/window
    # selections retain their requested activity, including normal activities.
    return next((index for index in eligible if records[index]["label_id"] == 0), eligible[0])


def _validate_source(
    signal: np.ndarray, frames: np.ndarray, source: dict[str, Any], record: dict[str, Any],
    file_index: int, split: dict[str, np.ndarray], cfg: KFallEventLoaderConfig,
) -> None:
    for key in ("subject", "code", "trial", "label_id", "T"):
        if source[key] != record[key]:
            raise ValueError(f"Original CSV does not match the saved recording manifest ({key}).")
    if record["label_id"] == 0:
        for index_key, frame_key in (("onset_idx", "fall_onset_frame"), ("impact_idx", "fall_impact_frame")):
            index = int(record[index_key])
            if not 0 <= index < len(frames) or int(frames[index]) != int(record[frame_key]):
                raise ValueError("Original CSV frame counters do not match the saved fall annotations.")
    # The replay also covers samples not selected by the offline window protocol.
    # Verify all existing cached windows of this recording before new inference.
    for index in np.flatnonzero(split["file_idx"] == file_index):
        start = int(split["start"][index])
        stop = start + cfg.raw_win
        events, offset = signal_to_events(signal[start:stop], frames[start:stop], cfg)
        if (not np.array_equal(events, split["x_ev"][index])
                or not np.array_equal(offset, split["offset"][index])
                or int(frames[stop - 1]) != int(split["end_frame"][index])):
            raise ValueError("Original CSV differs from this recording's cached test windows.")


def prepare_replay(
    run_dir: str | Path, *, checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None, dataset_root: str | Path | None = None,
    recording: str | Path | None = None, window: int | None = None,
    subject: str | None = None, device: str | torch.device = "auto",
    prediction_stride: int = 10, batch_size: int | None = None,
) -> ReplayData:
    """Load one test trace and predict trailing windows on a fixed sample grid.

    Prediction endpoints depend only on recording length, raw window length,
    and stride. Onset/impact annotations are returned solely for visualization;
    they neither select windows nor enter the network. The final incomplete
    stride is not anchored to an annotation or the recording's final sample.
    """
    if not isinstance(prediction_stride, (int, np.integer)) or prediction_stride < 1:
        raise ValueError("prediction_stride must be a positive integer sample count.")
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint_path = (run_dir / "checkpoints" / "best_model.pt" if checkpoint is None
                       else Path(checkpoint).expanduser().resolve())
    saved, cfg = _load_checkpoint(checkpoint_path)
    size = cfg.batch_size if batch_size is None else batch_size
    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("batch_size must be a positive integer.")
    dev = _device(device)
    subject = _subject(subject)
    split, metadata, run_metadata, resolved_cache = load_evaluation_data(run_dir, saved["config"], cache_dir)
    allowed = {field.name for field in fields(KFallEventLoaderConfig)}
    encoder = KFallEventLoaderConfig(**{key: value for key, value in run_metadata["config"].items() if key in allowed})
    file_index = _select_recording(run_metadata, split, recording=recording, window=window,
                                   subject=subject, raw_win=encoder.raw_win)
    record = dict(run_metadata["recordings"][file_index])
    store = KFallWindowStore(resolved_cache, dataset_root=dataset_root)
    path = store._record_path(record)
    signal, frames, source = load_record(path)
    _validate_source(signal, frames, source, record, file_index, split, encoder)

    # This schedule is intentionally independent of the saved fall-window starts.
    starts = np.arange(0, len(signal) - encoder.raw_win + 1, int(prediction_stride), dtype=np.int64)
    endpoints = starts + encoder.raw_win - 1
    set_seed(cfg.seed)
    model = build_model(cfg, dev)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    probability_batches = []
    with torch.inference_mode():
        for begin in range(0, len(starts), int(size)):
            events, offsets = [], []
            for start in starts[begin:begin + int(size)]:
                stop = int(start) + encoder.raw_win
                encoded, offset = signal_to_events(signal[int(start):stop], frames[int(start):stop], encoder)
                events.append(encoded)
                offsets.append(offset)
            batch_events = torch.as_tensor(np.stack(events), dtype=torch.float32, device=dev)
            batch_offsets = torch.as_tensor(np.stack(offsets), dtype=torch.float32, device=dev)
            with make_autocast(bool(cfg.amp and dev.type == "cuda")):
                logits = extract_single_logits(model(batch_events, offset=batch_offsets))
                if logits.shape != (len(events), cfg.n_classes) or not torch.isfinite(logits).all():
                    raise ValueError("The saved checkpoint produced invalid replay activity logits.")
            probability_batches.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probabilities = np.concatenate(probability_batches)
    predicted = probabilities.argmax(axis=1).astype(np.int64)
    times = (frames - frames[0]).astype(np.float64) / encoder.sample_rate_hz
    endpoint_times = times[endpoints]
    onset_time = impact_time = post_end_time = None
    if record["label_id"] == 0:
        onset_time = (int(record["fall_onset_frame"]) - int(frames[0])) / encoder.sample_rate_hz
        impact_time = (int(record["fall_impact_frame"]) - int(frames[0])) / encoder.sample_rate_hz
        if encoder.post_impact_seconds > 0:
            post_end_frame = int(record.get("post_impact_end_frame", int(record["fall_impact_frame"]) +
                                            round(encoder.post_impact_seconds * encoder.sample_rate_hz)))
            post_end_time = (post_end_frame - int(frames[0])) / encoder.sample_rate_hz
    for array in (signal, frames, times, endpoints, endpoint_times, probabilities, predicted):
        array.setflags(write=False)
    return ReplayData(
        record=record, path=path, raw=signal, frames=frames, times=times,
        endpoint_rows=endpoints, endpoint_times=endpoint_times,
        probabilities=probabilities, predicted_classes=predicted,
        raw_win=encoder.raw_win, out_win=encoder.out_win,
        prediction_stride=int(prediction_stride), checkpoint=checkpoint_path,
        best_epoch=saved["epoch"], best_val_loss=float(saved["best_val_loss"]),
        fall_onset_time=onset_time, fall_impact_time=impact_time,
        config=dict(saved["config"]), cache_dir=resolved_cache,
        subject=record["subject"], label=class_names(cfg.n_classes)[int(record["label_id"])],
        post_impact_seconds=float(getattr(encoder, "post_impact_seconds", 0.0)),
        post_impact_end_time=post_end_time,
    )
