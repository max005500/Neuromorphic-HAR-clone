"""Save auditable KFall window predictions and classification reports."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from preprocess.utils import json_ready

CLASS_NAMES = ("Fall", "Dynamic", "Static")


def _strict_values(value: Any) -> Any:
    value = json_ready(value)
    if isinstance(value, dict):
        return {str(key): _strict_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_strict_values(value), ensure_ascii=False,
                               allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _fall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    positive, predicted = y_true == 0, y_pred == 0
    tp = int(np.sum(positive & predicted))
    fn = int(np.sum(positive & ~predicted))
    fp = int(np.sum(~positive & predicted))
    tn = int(np.sum(~positive & ~predicted))
    return {
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "specificity": _ratio(tn, tn + fp),
        "false_positive_rate": _ratio(fp, tn + fp),
        "true_positive_windows": tp,
        "missed_fall_windows": fn,
        "false_fall_windows": fp,
        "true_negative_windows": tn,
    }


def _confusion_plot(path: Path, counts: np.ndarray, normalized: np.ndarray) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 4.8), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    try:
        for axis, values, title, is_normalized in zip(
            figure.subplots(1, 2), (counts, normalized),
            ("Conteos", "Normalizada por clase real"), (False, True),
        ):
            maximum = 1.0 if is_normalized else max(1, int(counts.max()))
            rendered = axis.imshow(values, cmap="Blues", vmin=0, vmax=maximum)
            axis.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES,
                     yticklabels=CLASS_NAMES, xlabel="Predicción", ylabel="Clase real", title=title)
            for row in range(3):
                for column in range(3):
                    value = values[row, column]
                    label = ("—" if counts[row].sum() == 0 else f"{value:.1%}") if is_normalized else str(int(value))
                    axis.text(column, row, label, ha="center", va="center",
                              color="white" if value > maximum / 2 else "#0f172a", fontsize=12)
            figure.colorbar(rendered, ax=axis, shrink=0.8)
        figure.suptitle("KFall · clasificación de ventanas de test", fontsize=14)
        figure.savefig(path, dpi=160)
    finally:
        figure.clear()


def save_evaluation_reports(
    output_dir: str | Path, *, y_true: np.ndarray, probabilities: np.ndarray,
    window_indices: np.ndarray, split: dict[str, np.ndarray],
    metadata: dict[str, Any], summary: dict[str, Any],
) -> dict[str, Any]:
    """Save full reports, preserving the original test-cache index for every row.

    Undefined metric denominators are represented by ``None`` in JSON. Macro
    averages omit undefined class values, following sklearn's NaN convention.
    All fall metrics describe individual windows, not continuous alerts.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    truth = np.asarray(y_true)
    indices = np.asarray(window_indices)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 1 or truth.dtype.kind not in "iu" or not np.isin(truth, range(3)).all():
        raise ValueError("y_true must be a one-dimensional integer array of KFall class IDs 0..2.")
    if not len(truth):
        raise ValueError("Cannot report an empty test selection.")
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != len(truth):
        raise ValueError("window_indices must contain one integer cache index per prediction.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("window_indices must not repeat the same cached window.")
    if probabilities.shape != (len(truth), 3) or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must have finite shape (N, 3).")
    if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-4):
        raise ValueError("Each probability row must contain values in [0, 1] summing to one.")
    cached_labels = np.asarray(split["y_h1"])
    if np.any(indices < 0) or np.any(indices >= len(cached_labels)):
        raise ValueError("window_indices refer to windows outside the test cache.")
    if not np.array_equal(truth, cached_labels[indices]):
        raise ValueError("Prediction targets do not match the selected test-cache windows.")

    truth = truth.astype(np.int64, copy=False)
    indices = indices.astype(np.int64, copy=False)
    predicted = probabilities.argmax(axis=1).astype(np.int64)
    subjects = np.asarray(split["subject"])[indices]
    records = metadata["recordings"]
    csv_rows = []
    for position, index in enumerate(indices):
        index = int(index)
        file_index = int(split["file_idx"][index])
        if not 0 <= file_index < len(records):
            raise ValueError(f"Test window {index} has an invalid recording index.")
        record = records[file_index]
        subject = str(subjects[position])
        if record.get("subject", subject) != subject:
            raise ValueError(f"Test window {index} does not match its source subject.")
        lead_time = None
        if truth[position] == 0 and "lead_time_ms" in split:
            lead_time = _strict_values(float(split["lead_time_ms"][index]))
        csv_rows.append({
            "window_index": index, "split": "test", "subject": subject,
            "file": str(record["file"]), "code": record["code"], "trial": record["trial"],
            "start": int(split["start"][index]), "end_frame": int(split["end_frame"][index]),
            "true_class_id": int(truth[position]), "true_class": CLASS_NAMES[truth[position]],
            "predicted_class_id": int(predicted[position]), "predicted_class": CLASS_NAMES[predicted[position]],
            "correct": int(truth[position] == predicted[position]),
            "confidence": float(probabilities[position, predicted[position]]),
            "prob_fall": float(probabilities[position, 0]),
            "prob_dynamic": float(probabilities[position, 1]),
            "prob_static": float(probabilities[position, 2]), "lead_time_ms": lead_time,
        })

    report_arguments = dict(labels=list(range(3)), target_names=list(CLASS_NAMES), zero_division=np.nan)
    report = classification_report(truth, predicted, output_dict=True, **report_arguments)
    counts = confusion_matrix(truth, predicted, labels=list(range(3)))
    support = counts.sum(axis=1)
    normalized = np.divide(counts, support[:, None], out=np.zeros_like(counts, dtype=float), where=support[:, None] != 0)
    recall = np.divide(np.diag(counts), support, out=np.zeros(3, dtype=float), where=support != 0)
    per_subject = {}
    for subject in sorted(np.unique(subjects).tolist()):
        selected = subjects == subject
        per_subject[str(subject)] = {
            "n": int(selected.sum()), "accuracy": float(np.mean(truth[selected] == predicted[selected])),
            "class_counts": {name: int(np.sum(truth[selected] == label)) for label, name in enumerate(CLASS_NAMES)},
            "fall_window_metrics": _fall_metrics(truth[selected], predicted[selected]),
        }
    metrics = {
        **summary, "n": len(truth), "n_samples": len(truth),
        "accuracy": float(np.mean(truth == predicted)),
        "balanced_accuracy": float(np.mean(recall[support > 0])),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "class_counts": {name: int(np.sum(truth == label)) for label, name in enumerate(CLASS_NAMES)},
        "predicted_class_counts": {name: int(np.sum(predicted == label)) for label, name in enumerate(CLASS_NAMES)},
        "fall_window_metrics": _fall_metrics(truth, predicted), "per_subject": per_subject,
        "evaluation_scope": "offline_test_window_classification",
        "timing_note": "lead_time_ms is the annotated time remaining before impact at a window endpoint; it is not measured warning time from a continuous alerting system.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = ("metrics.json", "classification_report.txt", "classification_report.json",
                 "confusion_matrix.npy", "confusion_matrix_norm.npy", "confusion_matrix.png",
                 "predictions.csv", "y_true.npy", "y_pred.npy", "probabilities.npy", "window_indices.npy")
    metrics["artifacts"] = {name: str(output_dir / name) for name in filenames}
    for name, array in (("y_true", truth), ("y_pred", predicted), ("probabilities", probabilities),
                        ("window_indices", indices), ("confusion_matrix", counts), ("confusion_matrix_norm", normalized)):
        np.save(output_dir / f"{name}.npy", array, allow_pickle=False)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    _save_json(output_dir / "classification_report.json", report)
    (output_dir / "classification_report.txt").write_text(
        classification_report(truth, predicted, digits=4, **report_arguments), encoding="utf-8")
    _confusion_plot(output_dir / "confusion_matrix.png", counts, normalized)
    metrics = _strict_values(metrics)
    _save_json(output_dir / "metrics.json", metrics)
    return metrics
