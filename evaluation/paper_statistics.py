"""Reproducible, fail-closed paper statistics (no training or model evaluation).

CSV contract
------------
Required: experiment,dataset,protocol,split,variant,seed,macro_f1.
``protocol`` must identify the full evaluation protocol (split assignment,
preprocessing, windowing, energy normalization, etc.). Splits are train/val/test;
seeds are nonnegative integers; classification scores are fractions in [0,1].
Optional numeric metrics: accuracy,params,parameters,energy_network_j,
energy_pe_norm (or explicit metrics). ``parameters`` is accepted under its own
name; energy_pe_norm is normalized PE, NEVER joules or a system-energy input.
An optional model_id distinguishes candidate configurations; it defaults to
variant. Every other column is a grouping dimension, except run_id/source_uri
(provenance), as is ``source``. Thus different protocols, front ends and configurations never
silently pool. Missing metric cells stay missing; NaN/Inf are invalid.

Optional system-energy inputs: input_mode (raw/analog/events), frontend
(adc/sigma_delta/adc+sigma_delta/none), duration_s, duration_source,
energy_network_source, frontend_source, adc_power_w/adc_source and/or
sigma_delta_power_w/sigma_delta_source. frontend_source documents the assumed
topology, including an explicit bypass for "none". All powers are total W,
duration is seconds for the *same inference unit* as network J. Raw requires
ADC; events require Sigma-Delta. Analog B1 and raw B1 are distinct groups:
there is no variant-name heuristic or implicit free converter. No hardware
values or citations are supplied. Without this complete budget, only network
energy is reported. Partial budgets fail rather than fabricating a total.

Public API: load_results, summarize, pareto_front, select_scn, system_energy,
write_report. See their docstrings and report_paper_results.py --help.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


IDENTITY = ("experiment", "dataset", "protocol", "split", "variant", "model_id")
REQUIRED = ("experiment", "dataset", "protocol", "split", "variant", "seed", "macro_f1")
DEFAULT_METRICS = ("macro_f1", "accuracy", "params", "parameters", "energy_network_j", "energy_pe_norm")
DERIVED_METRICS = ("energy_frontend_j", "energy_total_j")
ENERGY_NUMBERS = ("duration_s", "adc_power_w", "sigma_delta_power_w")
ENERGY_FIELDS = (
    "frontend", "duration_s", "duration_source", "energy_network_source",
    "frontend_source", "adc_power_w", "adc_source",
    "sigma_delta_power_w", "sigma_delta_source",
)
PROVENANCE = (
    "run_id", "source_uri", "source", "source_file", "source_row", "source_sha256",
    "selected_model_id", "selected_variant", "selection_policy",
    "validation_source_file", "validation_source_row", "validation_source_sha256",
    "validation_run_id", "validation_source_uri", "validation_source", "validation_score",
)
RESERVED = set(PROVENANCE) - {"run_id", "source_uri", "source"}


def _number(value: Any, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: expected a finite number, got {value!r}") from None
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name}: expected a finite {'nonnegative ' if nonnegative else ''}number")
    return result


def _source(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: an explicit source/assumption is required")
    return value.strip()


def system_energy(
    network_energy_j: float, duration_s: float, *, input_mode: str,
    frontend: str, network_source: str, duration_source: str,
    frontend_source: str, adc_power_w: float | None = None,
    adc_source: str | None = None, sigma_delta_power_w: float | None = None,
    sigma_delta_source: str | None = None,
) -> dict[str, Any]:
    """Return network J + sum(front-end W) * duration s with explicit sources.

    This is an accounting boundary, not a device simulator: sensor, radio and
    host costs are excluded unless already included in the documented inputs.
    No total is inferred from incomplete components. ``none`` explicitly means
    bypass, not unknown/zero power. Analog/raw B1 must be supplied separately.
    """
    network = _number(network_energy_j, "network_energy_j")
    duration = _number(duration_s, "duration_s")
    if duration <= 0:
        raise ValueError("duration_s must be positive")
    topology = {
        "adc": {"adc"}, "sigma_delta": {"sigma_delta"},
        "adc+sigma_delta": {"adc", "sigma_delta"}, "none": set(),
    }
    if input_mode not in {"raw", "analog", "events"} or frontend not in topology:
        raise ValueError("Explicit input_mode raw/analog/events and frontend topology required")
    components = topology[frontend]
    if input_mode == "raw" and "adc" not in components:
        raise ValueError("raw input requires an explicit ADC budget")
    if input_mode == "events" and "sigma_delta" not in components:
        raise ValueError("events input requires an explicit Sigma-Delta budget")
    sources = {
        "network": _source(network_source, "network_source"),
        "duration": _source(duration_source, "duration_source"),
        "frontend": _source(frontend_source, "frontend_source"),
    }
    power = 0.0
    component_energies = {}
    for name, value, source in (
        ("adc", adc_power_w, adc_source),
        ("sigma_delta", sigma_delta_power_w, sigma_delta_source),
    ):
        if name in components:
            component_power = _number(value, f"{name}_power_w")
            sources[name] = _source(source, f"{name}_source")
            power += component_power
            component_energies[f"energy_{name}_j"] = component_power * duration
        elif value is not None or source is not None:
            raise ValueError(f"{name}: component supplied but absent from frontend topology")
    frontend_j = power * duration
    total = network + frontend_j
    if not math.isfinite(total):
        raise ValueError("System energy overflow")
    return {
        "input_mode": input_mode, "frontend": frontend,
        "energy_network_j": network, "energy_frontend_j": frontend_j,
        "energy_total_j": total, "components": component_energies, "sources": sources,
    }


def _metric_names(metrics: Sequence[str]) -> tuple[str, ...]:
    names = tuple(metrics)
    if "macro_f1" not in names or len(set(names)) != len(names):
        raise ValueError("metrics must be unique and include macro_f1")
    protected = set(IDENTITY + ("seed",) + PROVENANCE + ENERGY_FIELDS
                    + ("input_mode",) + DERIVED_METRICS)
    for name in names:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in protected:
            raise ValueError(f"Invalid/reserved metric name: {name!r}")
    return names


def load_results(
    paths: Iterable[str | Path], *, metrics: Sequence[str] = DEFAULT_METRICS,
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...], list[dict[str, Any]]]:
    """Read CSVs; return (rows, group_fields, metrics_present, source_manifest).

    Extra metadata dimensions are collected across all files and must be
    nonempty everywhere. Optional metrics may be wholly absent in a group;
    partially observed metrics make that group non-publicable.
    """
    names = _metric_names(metrics)
    rows: list[dict[str, Any]] = []
    manifest = []
    all_headers: set[str] = set()
    seen_paths = set()
    for path in paths:
        path = Path(path).resolve()
        if path in seen_paths:
            raise ValueError(f"Duplicate input file: {path}")
        seen_paths.add(path)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        headers = reader.fieldnames or []
        if len(set(headers)) != len(headers) or any(not h or h != h.strip() for h in headers):
            raise ValueError(f"{path}: duplicate, empty or untrimmed CSV headers")
        missing = set(REQUIRED) - set(headers)
        if missing:
            raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
        if set(headers) & (RESERVED | set(DERIVED_METRICS)):
            raise ValueError(f"{path}: reserved provenance/derived-energy column supplied")
        all_headers.update(headers)
        count = 0
        for row in reader:
            line = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line}: malformed CSV row")
            record: dict[str, Any] = {key: value.strip() for key, value in row.items()}
            for name in REQUIRED[:-1]:
                if not record[name]:
                    raise ValueError(f"{path}:{line}: empty {name}")
            if not re.fullmatch(r"[0-9]+", record["seed"]):
                raise ValueError(f"{path}:{line}: seed must be a nonnegative integer")
            record["seed"] = int(record["seed"])
            if record["split"] not in {"train", "val", "test"}:
                raise ValueError(f"{path}:{line}: split must be train, val or test")
            record.setdefault("model_id", record["variant"])
            record.update(source_file=str(path), source_row=line, source_sha256=digest)
            for metric in names:
                value = record.get(metric, "")
                record[metric] = None if value == "" else _number(
                    value, f"{path}:{line}:{metric}", nonnegative=metric in DEFAULT_METRICS,
                )
                if metric in {"macro_f1", "accuracy"} and record[metric] is not None:
                    if not 0 <= record[metric] <= 1:
                        raise ValueError(f"{path}:{line}:{metric}: expected a fraction in [0,1]")
            for field in ENERGY_NUMBERS:
                if field in record:
                    record[field] = None if record[field] == "" else _number(record[field], field)
            # A bare network source is useful for compute-only reports.
            has_budget = any(record.get(field) not in (None, "") for field in ENERGY_FIELDS
                             if field != "energy_network_source")
            if has_budget:
                budget = system_energy(
                    record.get("energy_network_j"), record.get("duration_s"),
                    input_mode=record.get("input_mode"), frontend=record.get("frontend"),
                    network_source=record.get("energy_network_source"),
                    duration_source=record.get("duration_source"),
                    frontend_source=record.get("frontend_source"),
                    adc_power_w=record.get("adc_power_w"), adc_source=record.get("adc_source") or None,
                    sigma_delta_power_w=record.get("sigma_delta_power_w"),
                    sigma_delta_source=record.get("sigma_delta_source") or None,
                )
                for metric in DERIVED_METRICS:
                    record[metric] = budget[metric]
            rows.append(record)
            count += 1
        if not count:
            raise ValueError(f"{path}: no result rows")
        manifest.append({"path": str(path), "sha256": digest, "rows": count})
    if not rows:
        raise ValueError("No input results")
    active_metrics = tuple(name for name in names if name in all_headers)
    if any("energy_total_j" in row for row in rows):
        active_metrics += DERIVED_METRICS
    extras = all_headers - set(IDENTITY + ("seed",) + names + PROVENANCE)
    group_fields = IDENTITY + tuple(sorted(extras))
    for row in rows:
        for field in group_fields:
            if row.get(field) in ("", None):
                # Optional front-end fields are absent for other explicit topologies.
                if field in ENERGY_FIELDS:
                    row[field] = None
                else:
                    raise ValueError(f"{row['source_file']}:{row['source_row']}: empty grouping field {field}")
        for metric in active_metrics:
            row.setdefault(metric, None)
    _groups(rows, group_fields)
    return rows, group_fields, active_metrics, manifest


def _groups(
    rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str],
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    if not set(IDENTITY).issubset(group_fields):
        raise ValueError(f"group_fields must include {IDENTITY}; protocols may not be pooled")
    if len(set(group_fields)) != len(group_fields) or "seed" in group_fields:
        raise ValueError("group_fields must be unique and must not include seed")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    seen = set()
    for row in rows:
        if any(row.get(field) in ("", None) for field in IDENTITY):
            raise ValueError("Every result needs nonempty identity/protocol fields")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Every result needs a nonnegative integer seed")
        key = tuple(row[field] for field in group_fields)
        identity = (*key, row["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate seed {row['seed']} in group {dict(zip(group_fields, key))}")
        seen.add(identity)
        groups[key].append(row)
    if not groups:
        raise ValueError("No result groups")
    return groups


def summarize(
    rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str],
    metrics: Sequence[str], *, descriptive: bool = False,
) -> list[dict[str, Any]]:
    """Sample SD (ddof=1); >=5 distinct seeds and complete observed metrics.

    Strict mode rejects incomplete groups. Descriptive mode still rejects
    duplicate seeds, always labels all output NON_PUBLISHABLE, and reports
    per-metric n; SD for n<2 is None, never zero.
    """
    if "macro_f1" not in metrics or len(set(metrics)) != len(metrics):
        raise ValueError("Summary metrics must be unique and include macro_f1")
    result = []
    for key, members in _groups(rows, group_fields).items():
        reasons = []
        if len(members) < 5:
            reasons.append(f"requires >=5 distinct seeds, got {len(members)}")
        record = dict(zip(group_fields, key))
        record["seeds"] = sorted(row["seed"] for row in members)
        record["n_seeds"] = len(members)
        for metric in metrics:
            values = [_number(row[metric], metric, nonnegative=metric in DEFAULT_METRICS + DERIVED_METRICS)
                      for row in members if row.get(metric) is not None]
            if metric in {"macro_f1", "accuracy"} and any(not 0 <= value <= 1 for value in values):
                raise ValueError(f"{metric}: expected a fraction in [0,1]")
            if (metric == "macro_f1" or values) and len(values) != len(members):
                reasons.append(f"{metric}: {len(values)}/{len(members)} observations")
            record[f"{metric}_n"] = len(values)
            record[f"{metric}_mean"] = statistics.mean(values) if values else None
            record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            record[f"{metric}_mean_pm_std"] = (
                f"{record[f'{metric}_mean']:.6g} +/- {record[f'{metric}_std']:.6g}"
                if len(values) > 1 else None
            )
        if reasons and not descriptive:
            raise ValueError(f"Incomplete group {dict(zip(group_fields, key))}: {'; '.join(reasons)}")
        record["status"] = "NON_PUBLISHABLE" if descriptive else "PUBLISHABLE"
        record["warnings"] = reasons
        record["origins"] = [{name: row[name] for name in PROVENANCE if name in row}
                             for row in members]
        result.append(record)
    return result


def pareto_front(
    points: Sequence[Mapping[str, Any]], *, score: str = "macro_f1",
    energy: str = "energy_network_j",
) -> list[bool | None]:
    """Maximize score, minimize J; equal points both survive, missing -> None.

    Caller must supply a single comparable protocol/split. This computes a
    descriptive frontier only; it is NOT a test-based model-selection policy.
    """
    if score not in {"accuracy", "macro_f1"}:
        raise ValueError("Pareto score must be accuracy or macro_f1")
    coordinates = []
    for point in points:
        x, y = point.get(energy), point.get(score)
        if x is None or y is None:
            coordinates.append(None)
        else:
            x, y = _number(x, energy), _number(y, score)
            if y > 1:
                raise ValueError(f"{score}: expected a fraction in [0,1]")
            coordinates.append((x, y))
    return [
        None if point is None else not any(
            other is not None and other[0] <= point[0] and other[1] >= point[1]
            and (other[0] < point[0] or other[1] > point[1])
            for other in coordinates
        ) for point in coordinates
    ]


def select_scn(
    rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str], *,
    variants: Sequence[str], score: str = "macro_f1",
    energy: str = "energy_network_j",
) -> list[dict[str, Any]]:
    """Per seed, select SCNacc=max val score and SCNminE=min val energy.

    Explicit candidate variants only. Ties use (variant, model_id), never test
    metrics. All candidate/seed validation and test pairs are required before
    any selection; no survivor bias from silently missing candidates. Output
    contains corresponding held-out test rows with validation provenance.
    SCNminE has no accuracy constraint. SCNsel is deliberately unsupported.
    """
    if score not in {"macro_f1", "accuracy"} or not variants:
        raise ValueError("Selection requires explicit variants and macro_f1/accuracy score")
    if energy not in {"energy_network_j", "energy_total_j", "energy_pe_norm"}:
        raise ValueError("Selection energy must be energy_network_j, energy_total_j or energy_pe_norm")
    if set(variants) & {"SCNacc", "SCNminE", "SCNsel"}:
        raise ValueError("Selection aliases (including SCNsel) cannot be candidate variants")
    _groups(rows, group_fields)
    contexts = tuple(field for field in group_fields if field not in {"split", "variant", "model_id"})
    pools: dict[tuple[Any, ...], dict[Any, dict[Any, dict[str, Mapping[str, Any]]]]] = {}
    present = {row["variant"] for row in rows}
    if set(variants) - present:
        raise ValueError(f"Unknown SCN candidate variants: {sorted(set(variants) - present)}")
    for row in rows:
        if row["variant"] not in variants or row["split"] not in {"val", "test"}:
            continue
        key = tuple(row[field] for field in contexts)
        candidate = (row["variant"], row["model_id"])
        pool = pools.setdefault(key, {})
        pool.setdefault(candidate, {}).setdefault(row["seed"], {})[row["split"]] = row
    if not pools:
        raise ValueError("SCN selection requires validation and test results")
    selected = []
    for key, candidates in pools.items():
        context_variants = {candidate[0] for candidate in candidates}
        if set(variants) - context_variants:
            raise ValueError(f"Incomplete SCN candidate pool in {dict(zip(contexts, key))}")
        seeds = set.union(*(set(candidate) for candidate in candidates.values()))
        for candidate, by_seed in candidates.items():
            if set(by_seed) != seeds:
                raise ValueError(f"Incomplete seed coverage for SCN candidate {candidate}")
            for seed, pair in by_seed.items():
                if set(pair) != {"val", "test"}:
                    raise ValueError(f"Missing val/test pair for SCN candidate {candidate}, seed {seed}")
                for metric in (score, energy):
                    if pair["val"].get(metric) is None:
                        raise ValueError(f"Missing validation {metric} for {candidate}, seed {seed}")
                    value = _number(pair["val"][metric], f"validation {metric}")
                    if metric == score and value > 1:
                        raise ValueError(f"validation {metric}: expected a fraction in [0,1]")
        for seed in sorted(seeds):
            for label, metric, sign in (("SCNacc", score, -1), ("SCNminE", energy, 1)):
                winner = min(candidates, key=lambda candidate: (
                    sign * _number(candidates[candidate][seed]["val"][metric], metric), candidate,
                ))
                pair = candidates[winner][seed]
                result = dict(pair["test"])
                result.update(
                    variant=label, model_id=label, selected_variant=winner[0],
                    selected_model_id=winner[1],
                    selection_policy=f"{'max' if sign == -1 else 'min'} val {metric}; ties=(variant,model_id)",
                    validation_score=pair["val"][metric],
                )
                for name in ("source_file", "source_row", "source_sha256", "run_id", "source_uri", "source"):
                    if name in pair["val"]:
                        result[f"validation_{name}"] = pair["val"][name]
                selected.append(result)
    return selected


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    columns = list(dict.fromkeys([*(fields or ()), *(key for row in rows for key in row)]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def write_report(
    paths: Iterable[str | Path], output_dir: str | Path, *,
    descriptive: bool = False, metrics: Sequence[str] = DEFAULT_METRICS,
    score: str = "macro_f1", energy: str = "energy_network_j",
    scn_variants: Sequence[str] = (),
) -> dict[str, Any]:
    """Write summary.csv/report.json/pareto.csv, per-context Agg PNGs and rows.

    Frontiers compare variants/model_ids only within an otherwise identical
    context (including split and all metadata). Test frontiers are explicitly
    descriptive, never used for selection. Selection, if requested, is a
    separate validation-only result set in selected_results.csv and summary.
    """
    rows, fields, active_metrics, sources = load_results(paths, metrics=metrics)
    if score not in {"macro_f1", "accuracy"}:
        raise ValueError("score must be macro_f1 or accuracy")
    if energy not in {"energy_network_j", "energy_total_j", "energy_pe_norm"}:
        raise ValueError("energy must be energy_network_j, energy_total_j or energy_pe_norm")
    if score not in active_metrics:
        raise ValueError(f"Requested score column is absent: {score}")
    if energy == "energy_total_j" and energy not in active_metrics:
        raise ValueError("Total energy requested but no explicit complete front-end budget is available")
    if energy == "energy_pe_norm" and energy not in active_metrics:
        raise ValueError("Normalized energy requested but energy_pe_norm is absent")
    selected = select_scn(rows, fields, variants=scn_variants, score=score, energy=energy) if scn_variants else []
    if selected and any(row["variant"] in {"SCNacc", "SCNminE"} for row in rows):
        raise ValueError("Input variants collide with generated selection aliases")
    summary = summarize([*rows, *selected], fields, active_metrics, descriptive=descriptive)
    compare_fields = tuple(field for field in fields if field not in {"variant", "model_id"})
    comparisons: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in summary:
        comparisons[tuple(record[field] for field in compare_fields)].append(record)
    pareto_rows = []
    plots = []
    figure_data = []
    energy_unit = "normalized_PE" if energy == "energy_pe_norm" else "J"
    for index, (context, records) in enumerate(comparisons.items(), start=1):
        points = [{score: row.get(f"{score}_mean"), energy: row.get(f"{energy}_mean")} for row in records]
        frontier = pareto_front(points, score=score, energy=energy)
        filename = f"pareto_{index:03d}.png"
        for record, point, is_front in zip(records, points, frontier):
            pareto_rows.append({
                **{field: record[field] for field in fields},
                "score_metric": score, "energy_metric": energy,
                "score_mean": point[score], "energy_mean": point[energy], "energy_unit": energy_unit,
                "energy_mean_j": point[energy] if energy_unit == "J" else None,
                "pareto": is_front, "purpose": "DESCRIPTIVE_NOT_MODEL_SELECTION",
                "status": record["status"], "origins": record["origins"],
            })
        if any(value is not None for value in frontier):
            figure_data.append((filename, context, records, points, frontier))
            plots.append({"file": filename, "context": dict(zip(compare_fields, context))})
    report = {
        "schema_version": 1, "status": "NON_PUBLISHABLE" if descriptive else "PUBLISHABLE",
        "sources": sources, "group_fields": fields, "metrics": active_metrics,
        "statistics": {"minimum_distinct_seeds": 5, "std_ddof": 1,
                       "missing": "null; no imputation; partially observed metrics fail strict mode"},
        "selection": {"candidate_variants": list(scn_variants), "score": score, "energy": energy,
                      "basis": "val only; per seed; SCNacc=max score; SCNminE=min energy; no accuracy floor",
                      "ties": "lexicographic (variant,model_id)", "SCNsel": "unsupported"},
        "pareto_purpose": "DESCRIPTIVE_NOT_MODEL_SELECTION",
        "pareto_axes": {"score": score, "energy": energy, "energy_unit": energy_unit},
        "energy_boundary": "network + documented ADC/Sigma-Delta W * duration s; no other system costs inferred",
        "summary": summary, "pareto": pareto_rows, "plots": plots,
    }
    # Validate all inputs/statistics before creating any report files.
    output = Path(output_dir)
    input_paths = {Path(item["path"]) for item in sources}
    if any((output / name).resolve() in input_paths for name in
           ("summary.csv", "report.json", "pareto.csv", "results.csv", "selected_results.csv")):
        raise ValueError("Output would overwrite an input CSV")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "results.csv", rows)
    _write_csv(output / "selected_results.csv", selected, list(dict.fromkeys(
        [*fields, "seed", *active_metrics, *PROVENANCE],
    )))
    _write_csv(output / "pareto.csv", pareto_rows)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    for filename, context, records, points, frontier in figure_data:
        figure = Figure(figsize=(8, 5), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for record, point, is_front in zip(records, points, frontier):
            if is_front is None:
                continue
            axis.errorbar(
                point[energy], point[score],
                xerr=record.get(f"{energy}_std"), yerr=record.get(f"{score}_std"),
                fmt="o" if is_front else "x", capsize=3,
                label=f"{record['variant']} / {record['model_id']}",
            )
        context_record = dict(zip(compare_fields, context))
        axis.set(xlabel=f"{energy} ({energy_unit}; lower is better)",
                 ylabel=f"{score} (higher is better)",
                 title=f"{report['status']} · descriptive frontier (not selection)")
        figure.suptitle(
            f"{context_record['dataset']} · {context_record['split']} · protocol "
            f"{context_record['protocol'][:20]}", fontsize="small",
        )
        axis.legend(fontsize="small")
        figure.savefig(output / filename, dpi=150, metadata={
            "Description": json.dumps({"context": context_record, "sources": sources}, ensure_ascii=False),
        })
        figure.clear()
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report
