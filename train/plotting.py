"""Render training histories as accuracy and loss curves without a display."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _metric_values(history: list[dict[str, Any]], split: str, key: str) -> np.ndarray:
    values = []
    for row in history:
        metrics = row.get(split, {})
        value = _number(metrics.get(key))
        if key in {"acc_h2", "loss_h2"} and "n_h2" in metrics:
            if not _number(metrics["n_h2"]) > 0:
                value = float("nan")
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def save_training_curves(
    history: list[dict[str, Any]], metrics_dir: str | Path,
) -> dict[str, Path]:
    """Save train/validation PNGs, omitting unsupervised posture-head metrics.

    Accuracy values retain their original [0, 1] scale and are displayed as
    percentages. Missing and non-finite observations create gaps in the curves.
    """
    if not history:
        return {}

    # Import rendering code only when plots are requested, keeping preprocessing
    # independent of Matplotlib's backend and font initialization.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray([row.get("epoch", index) for index, row in enumerate(history, 1)])

    def has_values(key: str) -> bool:
        return any(np.isfinite(_metric_values(history, split, key)).any()
                   for split in ("train", "val"))

    head1_accuracy = "acc_h1" if has_values("acc_h1") else "acc"
    supervised_head2 = has_values("acc_h2")
    accuracy_panels = [(head1_accuracy, "Activity accuracy" if head1_accuracy == "acc_h1" else "Accuracy")]
    if supervised_head2:
        accuracy_panels.append(("acc_h2", "Posture accuracy (Head 2)"))
    loss_panels = [("loss", "Total loss")]
    if has_values("loss_h1"):
        loss_panels.append(("loss_h1", "Activity loss (Head 1)"))
    if supervised_head2 and has_values("loss_h2"):
        loss_panels.append(("loss_h2", "Posture loss (Head 2)"))

    paths = {}
    for name, panels in (("accuracy", accuracy_panels), ("loss", loss_panels)):
        figure = Figure(figsize=(6.5 * len(panels), 4.5), layout="constrained", facecolor="white")
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, len(panels), squeeze=False)[0]
        try:
            for axis, (key, title) in zip(axes, panels):
                for split, label, color, style in (
                    ("train", "Training", "#2563eb", "-"),
                    ("val", "Validation", "#ea580c", "--"),
                ):
                    axis.plot(epochs, _metric_values(history, split, key), label=label,
                              color=color, linestyle=style, linewidth=2, marker="o", markersize=4)
                axis.set(title=title, xlabel="Epoch", ylabel="Accuracy (%)" if name == "accuracy" else "Loss")
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                if len(epochs) == 1:
                    axis.set_xlim(float(epochs[0]) - 0.5, float(epochs[0]) + 0.5)
                if name == "accuracy":
                    axis.set_ylim(0, 1)
                    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
                axis.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.7)
                axis.set_axisbelow(True)
                axis.spines[["top", "right"]].set_visible(False)
                axis.legend(frameon=False)
            destination = metrics_dir / f"{name}.png"
            figure.savefig(destination, dpi=160, bbox_inches="tight")
            paths[name] = destination
        finally:
            figure.clear()
    return paths
