#!/usr/bin/env python3
"""Evaluate one frozen SisFall/UCI/KFall checkpoint with auditable labels."""

import argparse
import csv
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import re
import time

import numpy as np
import torch

from energy.measure_consumption import SCNEnergyProfiler
from evaluation.paper_metrics import classification_metrics
from evaluation.kfall_classes import resolve_class_names
from train.utils import count_parameters, format_seconds, get_device, save_classification_outputs, save_json


def _progress_batches(loader, *, label, every=5.0, enabled=True):
    if not enabled:
        yield from loader
        return
    total = len(loader)
    samples_total = len(loader.dataset)
    started = last_report = time.perf_counter()
    processed = 0
    print(f"{label} batches=0/{total} samples=0/{samples_total} ETA=pending", flush=True)
    for index, batch in enumerate(loader, 1):
        yield batch
        processed += len(batch["x"])
        now = time.perf_counter()
        if index == 1 or index == total or now - last_report >= every:
            elapsed = now - started
            eta = elapsed * (total - index) / index
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(
                f"{label} batches={index}/{total} samples={processed}/{samples_total} "
                f"progress={100 * index / total:.1f}% rate={rate:.1f}samples/s "
                f"elapsed={format_seconds(elapsed)} ETA={format_seconds(eta)}", flush=True)
            last_report = now


def save_confusion_figure(path, truth, prediction, names):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from sklearn.metrics import ConfusionMatrixDisplay
    figure = Figure(figsize=(12, 10) if len(names) > 4 else (8, 7))
    FigureCanvasAgg(figure)
    ConfusionMatrixDisplay.from_predictions(
        truth, prediction, labels=list(range(len(names))), display_labels=names,
        ax=figure.subplots(), cmap="Blues", xticks_rotation=45, colorbar=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def evaluate(run, *, dataset, split="val", gpu=0, perturbation=None, tag=None,
             progress=True, progress_every=5.0):
    if not np.isfinite(progress_every) or progress_every <= 0:
        raise ValueError("progress_every must be positive and finite.")
    if perturbation is not None and (not tag or re.fullmatch(r"[a-zA-Z0-9_-]+", tag) is None):
        raise ValueError("Perturbation requires a unique alphanumeric/underscore/hyphen tag.")
    started = time.perf_counter()
    label = f"[paper:{split}]"

    def log(message):
        if progress:
            print(f"{label} {message} elapsed={format_seconds(time.perf_counter() - started)}", flush=True)

    log(f"stage=initializing run={run} dataset={dataset}")
    module = importlib.import_module(f"train_{dataset}")
    cfg = SimpleNamespace(**json.loads((run / "config.json").read_text()))
    cfg.input_mode = getattr(cfg, "input_mode", "sigma_delta")
    device = get_device(gpu)
    log(f"stage=loading_data device={device} batch_size={cfg.batch_size} "
        f"raw_win={cfg.raw_win} out_win={cfg.out_win} input_mode={cfg.input_mode}")
    data_cfg = SimpleNamespace(**vars(cfg))
    if perturbation is not None:
        data_cfg.input_mode = "raw"
        data_cfg.out_win = cfg.raw_win
        data_cfg.cache_name += "_paper_native"
    data = module.build_data(data_cfg, device)
    log(f"stage=data_ready samples={len(data['loaders'][split].dataset)} batches={len(data['loaders'][split])}")
    if perturbation is not None:
        from evaluation.paper_robustness import PerturbedWindowDataset
        original_loader = data["loaders"][split]
        frames = None
        if dataset == "kfall":
            from loaders.kfall_event_loader import window_frame_arrays
            frames = window_frame_arrays(data["splits"][split], data["metadata"], cfg.raw_win)
        data["loaders"][split] = torch.utils.data.DataLoader(
            PerturbedWindowDataset(original_loader.dataset, cfg, perturbation, frames=frames),
            batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers,
            pin_memory=device.type == "cuda")
    log("stage=loading_checkpoint")
    checkpoint = torch.load(run / "checkpoints/best_model.pt", map_location=device, weights_only=False)
    model = module.build_model(cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    weights = checkpoint.get("class_weight")
    if weights is not None:
        weights = torch.as_tensor(weights, device=device)
    profiler = SCNEnergyProfiler(model)
    profiler.register()
    handle = model.register_forward_hook(lambda _m, inputs, _out: profiler.add_batch(len(inputs[0])))
    try:
        kwargs = dict(class_weight=weights, collect_predictions=True)
        if dataset == "sisfall":
            kwargs["pos_weight_h2"] = data["pos_weight_h2"]
        log("stage=inference_and_energy")
        batches = _progress_batches(data["loaders"][split], label=label, every=progress_every, enabled=progress)
        metrics = module.run_epoch(model, batches, device, cfg, **kwargs)
    finally:
        handle.remove()
        profiler.remove()
    energy, layers = profiler.summarize()
    log("stage=saving_reports")
    destination = run / "metrics" / f"paper_{split}{'_' + tag if perturbation is not None else ''}"
    destination.mkdir(parents=True, exist_ok=True)
    if dataset == "sisfall" and cfg.head_mode == "dual":
        if not cfg.manual_eval_only:
            raise ValueError("Paper dual-head evaluation requires --manual-eval-only during training.")
        main_metrics = metrics["cascade_manual"]
        accuracy, macro_f1 = main_metrics["accuracy"], main_metrics["macro_f1"]
        metric_scope = "four_class_manual_static_plus_all_fall_dynamic"
        truth, prediction = metrics["y_true_cascade"], metrics["y_pred_cascade"]
        part = data["splits"][split]
        selected = ((part["y_h1"] != 2) | ((part["src_h2"] == 1) & (part["w_h2"] > 0)
                    & np.isin(part["y_h2"], [0, 1])))
        indices = np.flatnonzero(selected)
    elif dataset == "kfall":
        if cfg.head_mode == "dual":
            accuracy, macro_f1 = metrics["acc_h1"], metrics["macro_f1_h1"]
            truth, prediction = metrics["y_true_h1"], metrics["y_pred_h1"]
        else:
            accuracy, macro_f1 = metrics["acc"], metrics["macro_f1"]
            truth, prediction = metrics["y_true"], metrics["y_pred"]
        normal_source = "manual_normal" if cfg.annotation_file else "task_normal"
        metric_scope = f"kfall_{cfg.n_classes}_class_endpoint_{cfg.fall_policy}_{normal_source}"
        indices = np.arange(len(truth))
    else:
        if dataset == "sisfall" and (not cfg.manual_eval_only or cfg.single_ambiguous != "drop"):
            raise ValueError("Paper single-head evaluation requires manual-only, ambiguous-drop gold.")
        accuracy, macro_f1 = metrics["acc"], metrics["macro_f1"]
        metric_scope = "six_class_official" if dataset == "ucihar" else "four_class_manual_static_plus_all_fall_dynamic"
        truth, prediction = metrics["y_true"], metrics["y_pred"]
        indices = data["datasets"][split].indices if dataset == "sisfall" else np.arange(len(truth))
    if dataset == "kfall":
        saved_metadata = json.loads((run / "dataset_metadata.json").read_text())
        names = list(resolve_class_names(saved_metadata, vars(cfg)))
    else:
        class_names = module.SINGLE_HEAD_NAMES if dataset == "sisfall" else module.ACTIVITY_LABELS
        names = list(class_names.values())
    save_classification_outputs(destination, truth, prediction, range(len(names)), names)
    save_confusion_figure(destination / "confusion_matrix.png", truth, prediction, names)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    spiking_layers = [row for row in layers if row["lif_ops_total"] > 0]
    if spiking_layers:
        figure = Figure(figsize=(7, 4))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.bar([row["layer"] for row in spiking_layers], [row["lif_output_rate"] for row in spiking_layers])
        axis.set(ylabel="Output spike fraction", ylim=(0, 1))
        figure.tight_layout()
        figure.savefig(destination / "firing_rates.png", dpi=150)
    if dataset == "sisfall":
        subject_by_file = {record["file_idx"]: record["subject"] for record in data["metadata"]["recordings"]}
        subjects = np.asarray([subject_by_file[int(idx)] for idx in data["splits"][split]["file_idx"][indices]])
    elif dataset == "kfall":
        subjects = np.asarray(data["splits"][split]["subject"])
    else:
        subjects = np.asarray(data["splits"][split]["domain"]).astype(str)
    if len(subjects) != len(truth):
        raise RuntimeError("Prediction/subject alignment failed.")
    subgroup_metrics = {str(subject): classification_metrics(truth[subjects == subject], prediction[subjects == subject], len(names))
                        for subject in np.unique(subjects)}
    if dataset == "kfall":
        rows = []
        for subject in np.unique(subjects):
            log(f"stage=subject_report subject={subject}")
            mask = subjects == subject
            folder = destination / "by_subject" / str(subject)
            save_classification_outputs(folder, truth[mask], prediction[mask], range(len(names)), names)
            save_confusion_figure(folder / "confusion_matrix.png", truth[mask], prediction[mask], names)
            rows.append(dict(subject=str(subject), n=int(mask.sum()),
                             accuracy=subgroup_metrics[str(subject)]["accuracy"],
                             macro_f1=subgroup_metrics[str(subject)]["macro_f1"]))
        with (destination / "by_subject/summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["subject", "n", "accuracy", "macro_f1"])
            writer.writeheader()
            writer.writerows(rows)
    older = np.isin(subjects, [f"SE{i:02d}" for i in range(1, 16)])
    if dataset == "sisfall":
        subgroup_metrics["older_SE01_SE15"] = (
            classification_metrics(truth[older], prediction[older], 4) if older.any() else None)
    save_json(destination / "by_subject.json", subgroup_metrics)
    np.save(destination / "subjects.npy", subjects, allow_pickle=False)
    if dataset == "sisfall":
        module.save_test_outputs(cfg, destination, dict(metrics))
    for key in list(metrics):
        if isinstance(metrics[key], np.ndarray):
            np.save(destination / f"{key}.npy", metrics.pop(key), allow_pickle=False)
    if metrics.get("n_h2") == 0:
        metrics["acc_h2"] = None
    result = dict(dataset=dataset, seed=cfg.seed, split=split, accuracy=accuracy, macro_f1=macro_f1,
                  parameters=count_parameters(model)[0], metric_scope=metric_scope,
                  checkpoint_epoch=checkpoint["epoch"], metrics=metrics, energy=energy,
                  energy_unit="normalized_to_one_dense_MAC_not_joules",
                  energy_model="v2_dense_input_and_data_movement_accounted",
                  perturbation=perturbation,
                  window_raw_samples=cfg.raw_win,
                  split_subjects=data["metadata"]["split_subjects"])
    if dataset == "kfall":
        result["label_provenance"] = {
            key: data["metadata"][key] for key in
            ("manual_annotation_signature", "manual_annotation_missing_recordings", "fall_label_rule", "raw_data")}
        if getattr(cfg, "label_scheme", "legacy") == "extended":
            result["label_scheme"] = "extended"
            result["class_names"] = names
    save_json(destination / "result.json", result)
    save_json(destination / "energy_layers.json", layers)
    log(f"stage=done accuracy={accuracy} macro_f1={macro_f1} output={destination}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("sisfall", "ucihar", "kfall"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--perturbation", type=Path,
                        help="JSON kwargs for perturb_raw_signal, including units and native sample_rate_hz.")
    parser.add_argument("--tag", help="Separate output suffix, required with --perturbation.")
    parser.add_argument("--progress-every", type=float, default=5.0,
                        help="Seconds between batch progress updates (first/last always shown).")
    parser.add_argument("--quiet-progress", action="store_true", help="Suppress stage and batch progress.")
    args = parser.parse_args()
    perturbation = json.loads(args.perturbation.read_text()) if args.perturbation else None
    result = evaluate(args.run_dir, dataset=args.dataset, split=args.split, gpu=args.gpu,
                      perturbation=perturbation, tag=args.tag,
                      progress=not args.quiet_progress, progress_every=args.progress_every)
    if args.quiet_progress:
        print(f"[paper:{args.split}] macro_f1={result['macro_f1']} | accuracy={result['accuracy']}", flush=True)


if __name__ == "__main__":
    main()
