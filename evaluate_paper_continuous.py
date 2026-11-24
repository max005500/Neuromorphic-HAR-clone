#!/usr/bin/env python3
"""Sliding inference on held-out SisFall/KFall recordings without annotation-selected windows."""

import argparse
import csv
import json
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluation.paper_metrics import evaluate_continuous_alarms
from preprocess.frontends import encode_window
from preprocess.preprocess_sisfall import FILE_RE, load_sisfall_3class
from train.utils import extract_dual_outputs, extract_single_logits, get_device, save_json


def evaluate_continuous(run, truth_path=None, *, stride=50, batch_size=32, gpu=0,
                        refractory_seconds=2.0, fall_threshold=None, dataset="sisfall"):
    if stride < 1 or batch_size < 1:
        raise ValueError("stride and batch_size must be positive.")
    if fall_threshold is not None and not 0 <= fall_threshold <= 1:
        raise ValueError("fall_threshold must lie in [0,1] and be fixed using validation.")
    cfg = SimpleNamespace(**json.loads((run / "config.json").read_text()))
    cfg.input_mode = getattr(cfg, "input_mode", "sigma_delta")
    metadata = json.loads((run / "dataset_metadata.json").read_text())
    held_out = set(metadata["split_subjects"]["test"])
    encoder_cfg = None
    if dataset == "kfall":
        from preprocess.preprocess_kfall import load_kfall_3class
        from loaders.kfall_event_loader import KFallEventLoaderConfig, _source_signature, signal_to_events
        fs = 100
        if _source_signature(Path(cfg.dataset_root), Path(metadata["label_dir"])) != metadata["source_signature"]:
            raise ValueError("KFall recordings or official annotations changed since training.")
        signals, _, records, _ = load_kfall_3class(cfg.dataset_root, label_dir=metadata["label_dir"])
        selected = [i for i, rec in enumerate(records) if rec["subject"] in held_out and rec["code"] not in cfg.exclude_tasks]
        encoder_cfg = KFallEventLoaderConfig(
            raw_win=cfg.raw_win, out_win=cfg.out_win, input_mode=cfg.input_mode, theta_sd=tuple(cfg.theta),
            dead_zone=cfg.dead_zone, sd_init=cfg.sd_init)
    elif dataset == "sisfall":
        fs = 200
        if truth_path is None:
            raise ValueError("SisFall requires explicit annotated event intervals via --truth.")
        signals, _, records, _ = load_sisfall_3class(
            cfg.dataset_root, acc=cfg.acc, dtype=cfg.dtype, cache_dir=cfg.cache_dir)
        expected_ids = {path.stem for path in Path(cfg.dataset_root).rglob("*") if FILE_RE.fullmatch(path.name)}
        if expected_ids != {Path(record["file"]).stem for record in records}:
            raise ValueError("Some SisFall recordings could not be loaded; resolve them before continuous evaluation.")
        selected = [i for i, record in enumerate(records) if record["subject"] in held_out]
    else:
        raise ValueError("Continuous evaluation supports sisfall or kfall.")
    if truth_path is None:
        recording_info, events = {}, []
        for index in selected:
            record = records[index]
            rec_id = Path(record["file"]).stem
            frames = record["frames"]
            recording_info[rec_id] = dict(duration_seconds=(int(frames[-1]) - int(frames[0]) + 1) / fs,
                                         is_adl=record["label_id"] != 0)
            if record["label_id"] == 0:
                events.append(dict(recording_id=rec_id,
                                   start_seconds=(record["fall_onset_frame"] - int(frames[0])) / fs,
                                   end_seconds=(record["fall_impact_frame"] - int(frames[0])) / fs))
    else:
        truth = json.loads(truth_path.read_text())
        recording_info, events = truth["recordings"], truth["fall_events"]
    ids = {Path(records[i]["file"]).stem for i in selected}
    if not ids or set(recording_info) != ids:
        raise ValueError("Event truth must explicitly cover exactly every held-out recording.")
    device = get_device(gpu)
    model = importlib.import_module(f"train_{dataset}").build_model(cfg, device).eval()
    checkpoint = torch.load(run / "checkpoints/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    rows = []
    with torch.no_grad():
        for index in selected:
            signal = np.asarray(signals[index], dtype=np.float32)
            record = records[index]
            rec_id = Path(record["file"]).stem
            info = recording_info[rec_id]
            times = ((record["frames"] - record["frames"][0]) / fs if dataset == "kfall" else
                     np.arange(len(signal)) / fs)
            if (not np.isclose(info["duration_seconds"], times[-1] + 1 / fs, rtol=0, atol=1 / fs)
                    or info["is_adl"] != (record["label_id"] != 0)):
                raise ValueError(f"Incorrect physical duration or ADL status for {rec_id}.")
            if len(signal) < cfg.raw_win:
                raise ValueError(f"{rec_id} is shorter than one full inference window.")
            starts = list(range(0, len(signal) - cfg.raw_win + 1, stride))
            if starts[-1] != len(signal) - cfg.raw_win:
                starts.append(len(signal) - cfg.raw_win)
            for first in range(0, len(starts), batch_size):
                group = starts[first:first + batch_size]
                windows = [signal[start:start + cfg.raw_win] for start in group]
                if encoder_cfg is not None:
                    encoded = np.stack([signal_to_events(
                        window, record["frames"][start:start + cfg.raw_win], encoder_cfg)[0]
                        for window, start in zip(windows, group)])
                else:
                    encoded = np.stack([encode_window(
                        window, mode=cfg.input_mode, out_win=cfg.out_win, theta=cfg.theta,
                        dead_zone=cfg.dead_zone, sd_init=cfg.sd_init) for window in windows])
                x = torch.as_tensor(encoded, dtype=torch.float32, device=device)
                offset = torch.as_tensor(np.stack([window[0] for window in windows]), device=device)
                output = model(x, offset=offset)
                logits = extract_dual_outputs(output)["head1"] if cfg.head_mode == "dual" else extract_single_logits(output)
                probability = logits.softmax(dim=1)[:, 0].cpu().numpy()
                predicted = logits.argmax(dim=1).cpu().numpy()
                for start, prob, pred in zip(group, probability, predicted):
                    rows.append(dict(recording_id=rec_id, subject=record["subject"], start_sample=start,
                                     timestamp_seconds=float(times[start + cfg.raw_win - 1]),
                                     alarm=bool(pred == 0 if fall_threshold is None else prob >= fall_threshold),
                                     fall_probability=float(prob)))
    metrics = evaluate_continuous_alarms(rows, events, recording_info, refractory_seconds=refractory_seconds)
    older_ids = {row["recording_id"] for row in rows if row["subject"].startswith("SE")}
    metrics["older_SE01_SE15"] = evaluate_continuous_alarms(
        [row for row in rows if row["recording_id"] in older_ids],
        [event for event in events if event["recording_id"] in older_ids],
        {key: value for key, value in recording_info.items() if key in older_ids},
        refractory_seconds=refractory_seconds) if older_ids else None
    metrics.update(dataset=dataset, stride_samples=stride, sample_rate_hz=fs, raw_win=cfg.raw_win,
                   initial_warmup_seconds=(cfg.raw_win - 1) / fs,
                   fall_threshold=fall_threshold, checkpoint_epoch=checkpoint["epoch"],
                   truth_source=str(truth_path.resolve()) if truth_path is not None else "official_KFall_XLSX_onset_to_impact",
                   test_subjects=sorted(held_out),
                   duration_scope="full physical recording including initial window warmup")
    destination = run / "metrics" / "paper_continuous"
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / "result.json", metrics)
    save_json(destination / "truth_used.json", dict(recordings=recording_info, fall_events=events))
    with (destination / "windows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("sisfall", "kfall"), default="sisfall")
    parser.add_argument("--truth", type=Path,
                        help="JSON with recordings{id:{duration_seconds,is_adl}} and fall_events[{recording_id,start_seconds,end_seconds}].")
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--refractory-seconds", type=float, default=2.0)
    parser.add_argument("--fall-threshold", type=float, help="Omit to use argmax; never optimize this on test.")
    args = parser.parse_args(argv)
    result = evaluate_continuous(args.run_dir, args.truth, stride=args.stride, batch_size=args.batch_size,
                                 gpu=args.gpu, refractory_seconds=args.refractory_seconds,
                                 fall_threshold=args.fall_threshold, dataset=args.dataset)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
