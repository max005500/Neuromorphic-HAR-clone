"""Reports for root and conditional branches, using the saved test/val targets."""
from pathlib import Path
import json
import numpy as np


def save_hierarchy_reports(directory, truth, probabilities):
    from sklearn.metrics import classification_report, confusion_matrix
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(truth, dtype=np.int64)
    event = np.isin(truth, [0, 3])
    specifications = {
        'group': (np.ones(len(truth), dtype=bool), event.astype(int), ['Normal', 'Evento']),
        'normal': (~event, (truth == 2).astype(int), ['Dynamic', 'Static']),
        'event': (event, (truth == 3).astype(int), ['Fall', 'Postimpact']),
    }
    report = {'decision_rule': 'root argmax, then selected child argmax',
              'leaf_probability_rule': 'joint probability = root probability * conditional child probability',
              'note': 'Branch metrics condition on the true group; final four-class metrics include routing errors.'}
    for name, (mask, targets, labels) in specifications.items():
        probs = np.asarray(probabilities[name])
        if probs.shape != (len(truth), 2) or not np.isfinite(probs).all():
            raise ValueError('Hierarchy probabilities must have finite shape (N, 2).')
        prediction = probs.argmax(1)
        report[name] = {
            'n': int(mask.sum()),
            'accuracy': float(np.mean(prediction[mask] == targets[mask])) if mask.any() else None,
            'confusion_matrix': confusion_matrix(targets[mask], prediction[mask], labels=[0, 1]).tolist(),
            'classification_report': classification_report(targets[mask], prediction[mask], labels=[0, 1],
                target_names=labels, output_dict=True, zero_division=0) if mask.any() else None,
        }
        np.save(directory / f'{name}_probabilities.npy', probs)
        from evaluation.kfall_reports import _confusion_plot
        counts = np.asarray(report[name]['confusion_matrix'])
        denominator = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(counts, denominator, out=np.zeros_like(counts, dtype=float), where=denominator != 0)
        _confusion_plot(directory / f'{name}_confusion_matrix.png', counts, normalized, tuple(labels))
    group = np.asarray(probabilities['group']).argmax(1)
    prediction = np.where(group == 0, np.asarray(probabilities['normal']).argmax(1) + 1,
                          np.asarray(probabilities['event']).argmax(1) * 3)
    np.save(directory / 'routed_predictions.npy', prediction)
    report['routed_accuracy'] = float(np.mean(prediction == truth))
    (directory / 'hierarchy_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
