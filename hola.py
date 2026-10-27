#!/usr/bin/env python3
"""Generate KFall event tensors for Neuromorphic-HAR, without editing original KFall.

Input, as shown in the KFall distribution:
    data/KFall/sensor_data/SA06/S06T20R01.csv
    data/KFall/label_data/SA06_label.xlsx

Output:
    data/kfall_events/{train,val,test}.npz
    data/kfall_events/{records.csv,windows.csv,manifest.json,skipped.csv}

Requires NumPy, pandas, openpyxl, and the original repo's
`preprocess.event_encoding.sigma_delta` (no model modifications).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from preprocess.event_encoding import sigma_delta
except ImportError as exc:
    raise SystemExit(
        "Ejecuta este script desde la raiz de Neuromorphic-HAR; "
        "se necesita preprocess/event_encoding.py."
    ) from exc

# KFall official task order: D01-D19 -> 1-19, F01-F15 -> 20-34,
# D20-D21 -> 35-36. Assignment of ADLs to Dynamic/Static is
# OUR coarse task taxonomy for the paper's Head 1, not a KFall label.
FALL_TASKS = frozenset(range(20, 35))
DYNAMIC_TASKS = frozenset({4, 6, 7, 8, 9, 10, 35, 36})
STATIC_TASKS = frozenset({1, 2, 3, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19})
assert FALL_TASKS | DYNAMIC_TASKS | STATIC_TASKS == set(range(1, 37))
assert not (FALL_TASKS & (DYNAMIC_TASKS | STATIC_TASKS))
CLASS_NAMES = {0: "Fall", 1: "Dynamic", 2: "Static"}
CHANNELS = ("AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ")
NAME_PATTERN = re.compile(r"^(?:SA|S)(\d{2})T(\d{2})R(\d{2})$", re.I)
SUBJECT_LABEL_PATTERN = re.compile(r"SA(\d{2})_label$", re.I)


def class_of(task: int) -> int:
    if task in FALL_TASKS:
        return 0
    if task in DYNAMIC_TASKS:
        return 1
    if task in STATIC_TASKS:
        return 2
    raise ValueError(f"Task ID {task} fuera de 1..36")


def list_recordings(sensor_root: Path) -> list[tuple[Path, str, int, int]]:
    if not sensor_root.is_dir():
        raise FileNotFoundError(f"No existe sensor_data: {sensor_root}")
    records = []
    for path in sorted(sensor_root.rglob("*.csv")):
        m = NAME_PATTERN.fullmatch(path.stem)
        if not m:
            continue
        s, task, trial = map(int, m.groups())
        if task not in range(1, 37):
            raise ValueError(f"Task ID incorrecto en {path}")
        subject = f"SA{s:02d}"
        # Respect the name of the actual CSV file (often S06), and
        # the SA06 subject ID used by XLSX labels.
        records.append((path, subject, task, trial))
    if not records:
        raise FileNotFoundError(f"Sin CSV de KFall validos en {sensor_root}")
    return records


def load_labels(label_root: Path) -> dict[tuple[str, int, int], tuple[int, int]]:
    if not label_root.is_dir():
        raise FileNotFoundError(f"No existe label_data: {label_root}")
    labels = {}
    books = sorted(label_root.rglob("*_label.xlsx"))
    if not books:
        raise FileNotFoundError(f"No se encontraron *_label.xlsx en {label_root}")
    for path in books:
        subj_match = SUBJECT_LABEL_PATTERN.fullmatch(path.stem)
        if not subj_match:
            continue
        subject = f"SA{int(subj_match.group(1)):02d}"
        df = pd.read_excel(path, engine="openpyxl")
        if df.shape[1] < 5:
            raise ValueError(f"Excel sin las 5 columnas requeridas: {path}")
        # In actual KFall spreadsheets, the task code cell is merged across
        # successive trials. Forward-fill only this field, not the timestamps.
        df = df.iloc[:, :5].copy()
        df.iloc[:, 0] = df.iloc[:, 0].ffill()
        for _, row in df.iterrows():
            if pd.isna(row.iloc[2]) or pd.isna(row.iloc[3]) or pd.isna(row.iloc[4]):
                continue
            task_str = str(row.iloc[0]).strip()
            m = re.search(r"\((\d{1,2})\)", task_str)
            if m:
                task = int(m.group(1))
            else:
                m = re.search(r"F\s*0*(\d{1,2})", task_str, re.I)
                if m is None:
                    raise ValueError(f"Codigo de caida no reconocido: {task_str!r} en {path}")
                task = int(m.group(1)) + 19
            if task not in FALL_TASKS:
                raise ValueError(f"Etiqueta de caida fuera de 20..34: {path}, {task_str}")
            trial = int(row.iloc[2])
            onset, impact = int(row.iloc[3]), int(row.iloc[4])
            if onset >= impact:
                raise ValueError(f"Inicio >= impacto en {path}: {(task, trial, onset, impact)}")
            key = (subject, task, trial)
            if key in labels:
                raise ValueError(f"Etiqueta duplicada para {key} en {path}")
            labels[key] = (onset, impact)
    if not labels:
        raise ValueError(f"No se pudieron interpretar etiquetas de caida en {label_root}")
    return labels


def load_signal(path: Path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    if not all(k in df.columns for k in CHANNELS):
        raise ValueError(f"Faltan columnas {CHANNELS} en {path}; columnas={df.columns.tolist()}")
    if "FrameCounter" not in df.columns:
        raise ValueError(f"Falta FrameCounter en {path}")
    x = df.loc[:, list(CHANNELS)].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32, copy=True
    )
    frames_raw = pd.to_numeric(df["FrameCounter"], errors="coerce").to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(frames_raw).all():
        raise ValueError(f"NaN/Inf en {path}")
    frames = frames_raw.astype(np.int64)
    if not np.array_equal(frames.astype(np.float64), frames_raw.astype(np.float64)):
        raise ValueError(f"FrameCounter contiene valores no enteros en {path}")
    if len(frames) < 2 or np.any(np.diff(frames) <= 0):
        raise ValueError(f"FrameCounter no monotono en {path}")

    # Official KFall units: g and deg/s -> SI units.
    x[:, :3] *= np.float32(9.80665)
    x[:, 3:] *= np.float32(np.pi / 180.0)

    # Use actual timestamps if reliable. FrameCounter provides fallback.
    if "TimeStamp(s)" in df.columns:
        t = pd.to_numeric(df["TimeStamp(s)"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        t = np.empty(0, dtype=np.float64)
    if t.size != len(x) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        t = (frames - frames[0]).astype(np.float64) / 100.0
    # Prefer FrameCounter for onset/impact matching; never row indices.
    return x, frames, t


def split_subjects(records, seed: int):
    subs = sorted({r[1] for r in records})
    if len(subs) < 3:
        raise ValueError("Se necesitan >=3 sujetos para train/val/test por participante")
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n = len(subs)
    n_val = max(1, round(n * 0.1))
    n_test = max(1, round(n * 0.1))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("No quedaron sujetos de entrenamiento")
    split = {
        "train": sorted(subs[:n_train]),
        "val": sorted(subs[n_train:n_train + n_val]),
        "test": sorted(subs[n_train + n_val:]),
    }
    inverse = {subject: name for name, subjects in split.items() for subject in subjects}
    return split, inverse


def estimate_theta(records, inverse, theta_k: float, max_per_class: int, seed: int):
    """Heuristic (not optimization): class-balanced median per-record std on TRAIN only."""
    if theta_k <= 0 or max_per_class <= 0:
        raise ValueError("theta-k y theta-records-per-class deben ser positivos")
    rng = np.random.default_rng(seed)
    acc = []
    for label in range(3):
        pool = [r for r in records if inverse[r[1]] == "train" and class_of(r[2]) == label]
        if not pool:
            raise ValueError(f"No hay grabaciones train de clase {CLASS_NAMES[label]}")
        ids = rng.choice(len(pool), size=min(len(pool), max_per_class), replace=False)
        stds = []
        for ix in ids:
            try:
                x, _, _ = load_signal(pool[int(ix)][0])
                stds.append(np.std(x.astype(np.float64), axis=0))
            except (OSError, ValueError) as exc:
                print(f"[WARN theta] {pool[int(ix)][0]}: {exc}")
        if not stds:
            raise ValueError(f"Sin registros validos de {CLASS_NAMES[label]} para theta")
        acc.append(np.median(np.stack(stds), axis=0))
    sd = np.median(np.stack(acc), axis=0)
    theta = np.maximum(theta_k * sd, 1e-5).astype(np.float32)
    return theta


def interpolate_window(raw_x, raw_t, out_win: int):
    t_src = raw_t - raw_t[0]
    if np.any(np.diff(t_src) <= 0):
        raise ValueError("Timestamps no monotonos dentro de ventana")
    t_dst = np.linspace(0, t_src[-1], num=out_win, dtype=np.float64)
    out = np.empty((out_win, 6), dtype=np.float32)
    for c in range(6):
        out[:, c] = np.interp(t_dst, t_src, raw_x[:, c]).astype(np.float32)
    return out


def window_starts(n: int, w: int, hop: int):
    if n < w:
        return []
    starts = list(range(0, n - w + 1, hop))
    if starts[-1] != n - w:
        starts.append(n - w)
    return starts


def locate_frame(frames, frame_no: int) -> int:
    idx = int(np.searchsorted(frames, frame_no))
    if idx >= len(frames) or int(frames[idx]) != int(frame_no):
        raise ValueError(f"Frame etiquetado {frame_no} no esta en FrameCounter")
    return idx


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(args):
    dataset_root = Path(args.dataset_root)
    sensor_root = Path(args.sensor_root) if args.sensor_root else dataset_root / "sensor_data"
    label_root = Path(args.label_root) if args.label_root else dataset_root / "label_data"
    output = Path(args.output).resolve()
    # Prevent overwriting anything inside the original dataset tree.
    if output == dataset_root.resolve() or dataset_root.resolve() in output.parents:
        raise ValueError("--output debe estar FUERA de data/KFall para proteger los originales")
    if args.window_seconds <= 0 or args.hop_seconds <= 0 or args.out_win < 2:
        raise ValueError("window-seconds y hop-seconds >0; out-win >=2")
    if args.max_frame_gap < 1:
        raise ValueError("max-frame-gap >=1")
    raw_win = (args.raw_win if args.raw_win is not None
               else int(round(args.window_seconds * 100.0)))
    hop = (args.stride if args.stride is not None
           else int(round(args.hop_seconds * 100.0)))
    if raw_win < 2 or hop < 1:
        raise ValueError("Ventana u hop demasiado pequenos")
    records = list_recordings(sensor_root)
    labels = load_labels(label_root)
    subjects, group = split_subjects(records, args.seed)

    if args.theta is not None:
        theta = np.asarray(args.theta, dtype=np.float32)
        theta_source = "manual"
    else:
        theta = estimate_theta(records, group, args.theta_k, args.theta_records_per_class, args.seed)
        theta_source = "heuristic_training_only"
    if np.any(~np.isfinite(theta)) or np.any(theta <= 0):
        raise ValueError("Todos los theta deben ser finitos y >0")

    print(f"Grabaciones encontradas: {len(records)} | sujetos: {sum(map(len, subjects.values()))}")
    print(f"Window={raw_win} samples @100Hz, hop={hop}, events={args.out_win}")
    print(f"Theta ({theta_source}) en SI: {theta.tolist()}")
    print(f"Split por sujeto: { {k: len(v) for k, v in subjects.items()} }")

    keys = ("x_ev", "offset", "y_h1", "y_h2", "src_h2", "w_h2", "file_idx", "start",
            "start_frame", "end_frame", "onset_frame", "impact_frame", "task_id", "trial_id")
    dtypes = {"x_ev": np.uint8, "offset": np.float32, "y_h1": np.int64,
              "y_h2": np.float32, "src_h2": np.uint8, "w_h2": np.float32,
              "file_idx": np.int32, "start": np.int32,
              "start_frame": np.int32, "end_frame": np.int32,
              "onset_frame": np.int32, "impact_frame": np.int32,
              "task_id": np.int16, "trial_id": np.int16}
    data = {s: {k: [] for k in keys} for s in subjects}
    record_rows, window_rows, skipped = [], [], []
    records_processed = 0

    for file_idx, (path, subject, task, trial) in enumerate(records):
        target = group[subject]
        label = class_of(task)
        annotation = labels.get((subject, task, trial)) if label == 0 else None
        try:
            if label == 0 and annotation is None:
                raise ValueError("Falta etiqueta onset/impact en el Excel")
            x, frames, times = load_signal(path)
            if len(x) < raw_win:
                raise ValueError(f"Grabacion con {len(x)} muestras < ventana de {raw_win}")
            if annotation is not None:
                onset, impact = annotation
                onset_ix = locate_frame(frames, onset)
                impact_ix = locate_frame(frames, impact)
                if onset_ix >= impact_ix:
                    raise ValueError("Onset posterior al impacto")
                if args.fall_policy == "impact":
                    starts = [max(0, min(impact_ix - raw_win // 2, len(x) - raw_win))]
                elif args.fall_policy == "preimpact":
                    if impact_ix < raw_win:
                        raise ValueError(
                            f"Insuficiente preimpacto: impacto en indice {impact_ix}, "
                            f"ventana={raw_win}; prueba --window-seconds mas corta"
                        )
                    starts = [impact_ix - raw_win]
                    if not (int(frames[starts[0]]) <= onset <= int(frames[impact_ix - 1])):
                        raise ValueError("Inicio de caida no aparece en la ventana preimpacto")
                else:
                    raise ValueError("Politica de caidas desconocida")
            else:
                onset, impact = -1, -1
                starts = window_starts(len(x), raw_win, hop)

            n_windows_record = 0
            for start in starts:
                stop = start + raw_win
                frame_delta = np.diff(frames[start:stop])
                if np.any(frame_delta > args.max_frame_gap):
                    skipped.append({"file": str(path), "reason":
                                    f"Ventana [{start}:{stop}] con FrameCounter gap>{args.max_frame_gap}"})
                    continue
                raw = x[start:stop]
                window_t = times[start:stop]
                interp = interpolate_window(raw, window_t, args.out_win)
                events, _, _ = sigma_delta(
                    interp, theta=theta, init="x0", dead_zone=args.dead_zone,
                    return_reconstruction=False
                )
                if events.shape != (args.out_win, 6, 2):
                    raise RuntimeError(f"Sigma-Delta produjo shape inesperado: {events.shape}")
                row = data[target]
                row["x_ev"].append(events)
                row["offset"].append(raw[0].copy())
                row["y_h1"].append(label)
                # KFall has no per-window stable/transition ground truth for Head 2.
                row["y_h2"].append(0.5)
                row["src_h2"].append(0)
                row["w_h2"].append(0.0)
                row["file_idx"].append(file_idx)
                row["start"].append(start)
                row["start_frame"].append(int(frames[start]))
                row["end_frame"].append(int(frames[stop - 1]))
                row["onset_frame"].append(onset)
                row["impact_frame"].append(impact)
                row["task_id"].append(task)
                row["trial_id"].append(trial)
                window_rows.append({"split": target, "file_idx": file_idx,
                                    "subject": subject, "task": task, "trial": trial,
                                    "class_id": label, "start_idx": start,
                                    "start_frame": int(frames[start]),
                                    "end_frame": int(frames[stop - 1]),
                                    "onset_frame": onset, "impact_frame": impact,
                                    "n_events": int(events.sum())})
                n_windows_record += 1
            record_rows.append({"file_idx": file_idx, "file": str(path), "subject": subject,
                                "task": task, "trial": trial, "class_id": label,
                                "split": target, "n_samples": len(x),
                                "n_windows": n_windows_record})
            records_processed += 1
        except (ValueError, OSError) as exc:
            detail = {"file": str(path), "reason": str(exc)}
            skipped.append(detail)
            if args.strict:
                raise RuntimeError(f"Error procesando {path}: {exc}") from exc
            print(f"[WARN] {path.name}: {exc}")

    if not window_rows:
        raise RuntimeError("No se generaron ventanas; revisa duracion, theta y anotaciones")
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        d = data[split]
        arrays = {}
        for k in keys:
            if d[k]:
                arrays[k] = np.asarray(d[k], dtype=dtypes[k])
            else:
                tail = (args.out_win, 6, 2) if k == "x_ev" else (6,) if k == "offset" else ()
                arrays[k] = np.empty((0,) + tail, dtype=dtypes[k])
        np.savez_compressed(output / f"{split}.npz", **arrays)
        counts = Counter(int(v) for v in arrays["y_h1"])
        summary[split] = {"n_windows": len(arrays["y_h1"]),
                          "classes": {CLASS_NAMES[k]: counts.get(k, 0) for k in range(3)},
                          "subjects": subjects[split]}
        print(f"{split}: {summary[split]['n_windows']} ventanas, "
              f"{summary[split]['classes']}")

    write_csv(output / "records.csv", record_rows,
              ["file_idx", "file", "subject", "task", "trial", "class_id", "split", "n_samples", "n_windows"])
    write_csv(output / "windows.csv", window_rows,
              ["split", "file_idx", "subject", "task", "trial", "class_id", "start_idx",
               "start_frame", "end_frame", "onset_frame", "impact_frame", "n_events"])
    write_csv(output / "skipped.csv", skipped, ["file", "reason"])
    manifest = {"source": str(dataset_root), "sensor_root": str(sensor_root),
                "label_root": str(label_root), "input_recordings": len(records),
                "processed_recordings": records_processed,
                "skipped_entries": len(skipped), "theta": theta.tolist(),
                "theta_source": theta_source, "theta_k": args.theta_k,
                "units": ["m/s^2"] * 3 + ["rad/s"] * 3,
                "channels": list(CHANNELS), "out_win": args.out_win,
                "raw_window_samples": raw_win, "raw_fs_hz": 100,
                "hop_samples": hop, "dead_zone": args.dead_zone,
                "fall_policy": args.fall_policy, "seed": args.seed,
                "class_map": {str(k): v for k, v in CLASS_NAMES.items()},
                "dynamic_tasks": sorted(DYNAMIC_TASKS),
                "static_tasks": sorted(STATIC_TASKS),
                "fall_tasks": sorted(FALL_TASKS), "splits": summary,
                "head2": "unlabeled (y_h2=0.5, w_h2=0)"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nArchivos guardados en: {output}")
    print("KFall original no se modifico. Theta automatico es heuristico, no optimizado.")
    print("Estas NPZ aun no se combinan automaticamente con los caches de SisFall.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/KFall")
    parser.add_argument("--sensor-root", default=None)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--output", default="data/kfall_events")
    parser.add_argument("--window-seconds", type=float, default=2.05)
    parser.add_argument("--hop-seconds", type=float, default=1.02)
    parser.add_argument("--raw-win", type=int, default=None,
                        help="Alias: numero de muestras originales, sobreescribe --window-seconds")
    parser.add_argument("--stride", type=int, default=None,
                        help="Alias: salto en muestras originales, sobreescribe --hop-seconds")
    parser.add_argument("--out-win", type=int, default=2048)
    parser.add_argument("--fall-policy", choices=["impact", "preimpact"], default="impact")
    parser.add_argument("--theta", type=float, nargs=6, default=None,
                        metavar=("AX", "AY", "AZ", "GX", "GY", "GZ"),
                        help="Six manual Sigma-Delta thresholds in SI units. Default: class-balanced TRAIN heuristic.")
    parser.add_argument("--theta-k", type=float, default=0.1)
    parser.add_argument("--theta-records-per-class", type=int, default=128)
    parser.add_argument("--dead-zone", type=float, default=0.5)
    parser.add_argument("--max-frame-gap", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="Fail instead of warning for invalid/missing records")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
