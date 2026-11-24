#!/usr/bin/env python3
"""Reproduce the subject-independent B1--B5 matrix; prepare commands unless --execute."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "B1": dict(input_mode="raw", activation="relu", use_offset=False, native=True),
    "B2": dict(input_mode="raw", activation="lif", use_offset=False, native=True),
    "B3": dict(input_mode="sigma_delta", activation="lif", use_offset=False, native=False),
    "B4": dict(input_mode="sigma_delta", activation="lif", use_offset=True, native=False),
    "B5": dict(input_mode="sigma_delta", activation="lif", use_offset=True, native=True),
}
PROFILE_KEYS = {"conv_ch", "kernels", "strides", "pool_kernels", "pool_strides", "pool_paddings",
                "tau", "thresh", "p_drop", "offset_hidden", "hard_reset"}


def cli_options(settings):
    args = []
    for key, value in settings.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(option)
        else:
            args.extend([option, *map(str, value if isinstance(value, list) else [value])])
    return args


def build_manifest(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.calibration_per_class < 1:
        raise ValueError("epochs/batch-size/calibration-per-class must be positive; workers nonnegative.")
    if not args.datasets or len(set(args.datasets)) != len(args.datasets) or not args.seeds:
        raise ValueError("Require nonempty distinct datasets and nonempty seeds.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be nonnegative.")
    if not args.pilot and (len(set(args.seeds)) < 5 or len(set(args.seeds)) != len(args.seeds)):
        raise ValueError("Publication experiments require at least five distinct seeds.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds cannot repeat.")
    profiles = {"reference": {}} if args.profiles is None else json.loads(args.profiles.read_text())
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("profiles must be a nonempty object mapping names to model settings.")
    for name, profile in profiles.items():
        if (re.fullmatch(r"[a-zA-Z0-9_-]+", name) is None
                or not isinstance(profile, dict) or set(profile) - PROFILE_KEYS):
            raise ValueError(f"Unknown architecture settings in profile {name}.")
    output = args.output.resolve()
    commands, trials = [], []
    if not args.kfall_windows or len(set(args.kfall_windows)) != len(args.kfall_windows):
        raise ValueError("KFall windows must be distinct and nonempty.")
    conditions = [(dataset, window) for dataset in args.datasets
                  for window in (args.kfall_windows if dataset == "kfall" else [None])]
    for dataset, window in conditions:
        native, event = ((window, window * 10) if dataset == "kfall" else
                         (410, 2048) if dataset == "sisfall" else (128, 640))
        window_token = f"_w{native}" if dataset == "kfall" else ""
        for seed in args.seeds:
            shared = dict(seed=seed, raw_win=native, epochs=args.epochs, batch_size=args.batch_size,
                          workers=args.workers, gpu=args.gpu, publication_protocol=not args.pilot)
            if dataset == "sisfall":
                shared.update(dataset_root=str(args.sisfall_root.resolve()), label_dir=str(args.label_dir.resolve()),
                              split=[0.64, 0.16, 0.20], manual_eval_only=True,
                              autotag_weight=0.6, result_dir=str(output / "posture" / str(seed)))
            elif dataset == "ucihar":
                shared.update(data_root=str(args.ucihar_root.resolve().parent), dataset_dir=args.ucihar_root.name,
                              case="official", val_ratio=0.2, sample_rate_hz=50)
            else:
                shared.update(dataset_root=str(args.kfall_root.resolve()), stride=native // 2,
                              split=args.kfall_split, post_impact_seconds=1.2, fall_policy="pre_impact",
                              head_mode="single")
                if args.kfall_label_dir is not None:
                    shared["label_dir"] = str(args.kfall_label_dir.resolve())
                if args.kfall_annotation_file is not None:
                    shared["annotation_file"] = str(args.kfall_annotation_file.resolve())
                if args.kfall_exclude_tasks is not None:
                    shared["exclude_tasks"] = args.kfall_exclude_tasks
            shared_args = cli_options(shared)
            if args.amp:
                shared_args.append("--amp")
            calibration = output / "calibration" / f"{dataset}{window_token}_s{seed}.json"
            commands.append([sys.executable, str(ROOT / "calibrate_paper_theta.py"), "--dataset", dataset,
                             "--output", str(calibration), "--event-out-win", str(event),
                             "--per-class", str(args.calibration_per_class), "--",
                             *shared_args, "--cache-dir", str(output / "cache"),
                             "--cache-name", f"{dataset}{window_token}_s{seed}_calibration"])
            for profile_name, settings in profiles.items():
                for variant, options in VARIANTS.items():
                    modes = ("dual", "single") if dataset == "sisfall" and args.compare_single_head and variant == "B4" else (
                        ("dual",) if dataset == "sisfall" else ("single",))
                    weights = (0.4, 0.6, 0.8, 1.0) if dataset == "sisfall" and args.omega_sweep and variant == "B4" else (0.6,)
                    for mode in modes:
                        for weight in weights:
                            trial_id = f"{dataset}{window_token}_{profile_name}_{variant}_{mode}_w{weight}_s{seed}"
                            trial_dir = output / "trials" / trial_id
                            cache_name = f"{dataset}_s{seed}_{options['input_mode']}_{native if options['native'] else event}"
                            training = [sys.executable, str(ROOT / f"train_{dataset}.py"),
                                        *shared_args, *cli_options(settings), "--cache-dir", str(output / "cache"),
                                        "--cache-name", cache_name, "--out-win", str(native if options["native"] else event),
                                        "--theta-file", str(calibration), "--input-mode", options["input_mode"],
                                        "--activation", options["activation"], "--skip-test", "--out-dir", str(trial_dir),
                                        "--run", trial_id]
                            if dataset == "sisfall":
                                training.extend(["--head-mode", mode, "--autotag-weight", str(weight)])
                            if not options["use_offset"]:
                                training.append("--no-offset")
                            commands.append(training)
                            evaluation = [sys.executable, str(ROOT / "evaluate_paper_run.py"),
                                          "--dataset", dataset, "--gpu", str(args.gpu)]
                            trials.append(dict(id=trial_id, dataset=dataset, raw_win=native, profile=profile_name, variant=variant,
                                               head_mode=mode, omega=weight, seed=seed, trial_dir=str(trial_dir),
                                               command=training, evaluation_command=evaluation))
    return dict(publication_eligible=not args.pilot, seeds=args.seeds, profiles=profiles, trials=trials, commands=commands,
                test_evaluation=args.evaluate_test,
                split_protocol="sisfall_subject_64_16_20__uci_official_subject_val__kfall_subject",
                kfall_split=args.kfall_split,
                input_units={"sisfall": "native ADC counts, 200 Hz", "ucihar": "native g and rad/s, 50 Hz",
                             "kfall": "native g and degrees/s, 100 Hz"},
                note="reference profile is not an inferred SCNsel/SCNacc/SCNminE configuration")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("sisfall", "ucihar", "kfall"),
                        default=["sisfall", "ucihar", "kfall"])
    parser.add_argument("--sisfall-root", type=Path, default=Path("data/SisFall_dataset"))
    parser.add_argument("--ucihar-root", type=Path, default=Path("data/UCI HAR Dataset"))
    parser.add_argument("--label-dir", type=Path, default=Path("data/labels_transitions"))
    parser.add_argument("--kfall-root", type=Path, default=Path("data/KFall"))
    parser.add_argument("--kfall-label-dir", type=Path, help="Override the official KFall XLSX directory.")
    parser.add_argument("--kfall-annotation-file", type=Path, help="Manual labels; otherwise use the trainer's JSON/.bak discovery.")
    parser.add_argument("--kfall-windows", type=int, nargs="+", choices=(120, 206, 320), default=[120, 206, 320],
                        help="Native sample lengths at 100 Hz.")
    parser.add_argument("--kfall-split", type=float, nargs=3, default=[0.6, 0.3, 0.1])
    parser.add_argument("--kfall-exclude-tasks", nargs="*", help="Override trainer exclusions; an empty list retains all tasks.")
    parser.add_argument("--output", type=Path, required=True, help="New experiment directory.")
    parser.add_argument("--profiles", type=Path, help="JSON with named, explicitly supplied SCN configurations.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Permit fewer seeds; not publication results.")
    parser.add_argument("--calibration-per-class", type=int, default=32)
    parser.add_argument("--compare-single-head", action="store_true")
    parser.add_argument("--omega-sweep", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually calibrate/train; otherwise save manifest only.")
    parser.add_argument("--resume", action="store_true",
                        help="Execute an identical saved manifest, skipping successfully journaled commands.")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate frozen runs after all training is complete.")
    args = parser.parse_args(argv)
    try:
        if args.resume and not args.execute:
            raise ValueError("--resume requires --execute.")
        manifest = build_manifest(args)
        if args.resume:
            saved = json.loads((args.output / "manifest.json").read_text())
            if saved != manifest:
                raise ValueError("Saved manifest differs; resume with exactly the original experiment arguments.")
        else:
            args.output.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] {manifest_path} | {len(manifest['trials'])} trainings", flush=True)
    if not args.execute:
        return manifest
    journal_path = args.output / "completed_commands.json"
    completed = set(json.loads(journal_path.read_text())) if journal_path.exists() else set()
    if not completed.issubset(set(range(len(manifest["commands"])))):
        raise ValueError("Invalid completed-command journal.")
    for index, command in enumerate(manifest["commands"]):
        if index in completed:
            continue
        if "--out-dir" in command:
            trial_dir = Path(command[command.index("--out-dir") + 1])
            if trial_dir.exists() and any(trial_dir.iterdir()):
                raise RuntimeError(f"Unfinished training output exists at {trial_dir}; archive it before resuming.")
        log_path = args.output / f"command_{index:03d}.log"
        try:
            with log_path.open("w") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            tail = log_path.read_text()[-4000:]
            raise RuntimeError(f"Experiment command {index} failed; see {log_path}.\n{tail}") from exc
        completed.add(index)
        temporary = journal_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(sorted(completed)) + "\n")
        temporary.replace(journal_path)
    rows = []
    for trial in manifest["trials"]:
        runs = list(Path(trial["trial_dir"]).glob("*/checkpoints/best_model.pt"))
        if len(runs) != 1:
            raise RuntimeError(f"Expected exactly one saved checkpoint for {trial['id']}.")
        run = runs[0].parents[1]
        for split in (("val", "test") if args.evaluate_test else ("val",)):
            subprocess.run([*trial["evaluation_command"], "--run-dir", str(run), "--split", split], cwd=ROOT, check=True)
            result_path = run / "metrics" / f"paper_{split}" / "result.json"
            result = json.loads(result_path.read_text())
            protocol = dict(manifest=manifest["split_protocol"], head=trial["head_mode"], omega=trial["omega"],
                            raw_win=trial["raw_win"], metric_scope=result["metric_scope"],
                            epochs=args.epochs, batch_size=args.batch_size, amp=args.amp,
                            kfall_split=args.kfall_split if trial["dataset"] == "kfall" else None,
                            label_provenance=result.get("label_provenance"),
                            energy_model=result["energy_model"])
            protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
            profile_id = hashlib.sha256(
                json.dumps(manifest["profiles"][trial["profile"]], sort_keys=True).encode()).hexdigest()[:12]
            rows.append(dict(experiment="frontend", dataset=trial["dataset"], variant=trial["variant"],
                             model_id=f"{trial['profile']}_{profile_id}",
                             window_raw_samples=trial["raw_win"],
                             seed=trial["seed"], protocol=protocol_id, split=split, **{
                                 key: result[key] for key in ("macro_f1", "accuracy", "parameters")},
                             energy_pe_norm=result["energy"]["energy_pe_total_norm_per_sample"],
                             source=str(result_path)))
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


if __name__ == "__main__":
    main()
