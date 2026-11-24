"""Small confusion accumulators for exact epoch-level macro F1."""

import numpy as np
import torch


def update_confusion(counts, truth, prediction):
    n_classes = len(counts)
    encoded = truth.detach().long() * n_classes + prediction.detach().long()
    counts += torch.bincount(encoded, minlength=n_classes**2).reshape(n_classes, n_classes).cpu().numpy()


def confusion_metrics(counts):
    counts = np.asarray(counts)
    true = counts.sum(axis=1)
    predicted = counts.sum(axis=0)
    diagonal = np.diag(counts)
    f1 = np.divide(2 * diagonal, true + predicted, out=np.zeros(len(counts), dtype=float),
                   where=(true + predicted) != 0)
    return {"accuracy": float(diagonal.sum() / counts.sum()) if counts.sum() else None,
            "macro_f1": float(f1.mean()) if counts.sum() else None,
            "n": int(counts.sum()), "confusion_matrix": counts.tolist()}
