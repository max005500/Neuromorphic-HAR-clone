#!/usr/bin/env python3
"""Compare two runs on identical validation windows without consulting test."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from evaluation.kfall_data import load_evaluation_data


def read_validation(run):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    arrays, metadata, _, _ = load_evaluation_data(run, config, split_name='val')
    truth = np.load(run / 'metrics/y_true_val.npy', allow_pickle=False)
    predicted = np.load(run / 'metrics/y_pred_val.npy', allow_pickle=False)
    index_file = run / 'metrics/validation/window_indices.npy'
    indices = np.load(index_file, allow_pickle=False) if index_file.exists() else np.arange(len(truth))
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(indices) != len(truth)
            or np.any(indices < 0) or np.any(indices >= len(arrays['y_h1']))
            or len(np.unique(indices)) != len(indices)):
        raise ValueError('Validation indices must be unique, in-range one-dimensional integers.')
    if truth.ndim != 1 or not len(truth) or truth.shape != predicted.shape or not np.array_equal(truth, arrays['y_h1'][indices]):
        raise ValueError('Validation predictions do not match the saved validation cache.')
    if not np.isin(predicted, [0, 1, 2, 3]).all() or not np.any(truth == 0):
        raise ValueError('Four-class predictions with annotated Fall examples are required.')
    order = np.argsort(indices)
    indices, truth, predicted = indices[order], truth[order], predicted[order]
    cohort = np.column_stack((arrays['file_idx'][indices], arrays['end_frame'][indices], truth))
    return dict(run_dir=str(run.resolve()), config=config, metadata=metadata, cohort=cohort,
        cohort_sha256=hashlib.sha256(cohort.astype('<i8').tobytes()).hexdigest(),
        accuracy=float(accuracy_score(truth, predicted)),
        macro_f1=float(f1_score(truth, predicted, labels=[0, 1, 2, 3], average='macro', zero_division=0)),
        fall_recall=float(recall_score(truth == 0, predicted == 0, zero_division=0)))


def compare_validation(baseline, candidate, tolerance=0.):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative.')
    left, right = read_validation(baseline), read_validation(candidate)
    if not np.array_equal(left['cohort'], right['cohort']):
        raise ValueError('Validation endpoint/label cohorts differ; the comparison is not matched.')
    for key in ('source_signature', 'split_subjects', 'version', 'channels', 'units'):
        if left['metadata'][key] != right['metadata'][key]:
            raise ValueError(f'Validation provenance differs: {key}.')
    if left['metadata']['config'] != right['metadata']['config']:
        raise ValueError('Encoder/label configurations must match for an architecture comparison.')
    for key in ('batch_size', 'epochs', 'lr', 'wd', 'label_smoothing', 'amp', 'class_weight',
                'train_per_class', 'tau', 'thresh', 'hard_reset', 'merge_polarities', 'conv_ch',
                'kernels', 'strides', 'pool_kernels', 'pool_strides', 'pool_paddings',
                'offset_hidden', 'spiking_heads', 'head_rate_scale', 'p_drop', 'sequence_length', 'boundary_ignore_seconds',
                'patience', 'plateau_patience', 'lr_factor', 'min_lr', 'grad_clip'):
        if left['config'].get(key) != right['config'].get(key):
            raise ValueError(f'Training settings differ beyond the experimental architecture: {key}.')
    accepted = all(right[key] + tolerance >= left[key] for key in ('macro_f1', 'fall_recall'))
    return dict(split='val', test_consulted=False, non_degrading=accepted,
        criterion='candidate macro-F1 and Fall recall >= baseline minus tolerance', tolerance=tolerance,
        baseline={key: left[key] for key in ('run_dir', 'accuracy', 'macro_f1', 'fall_recall')},
        candidate={key: right[key] for key in ('run_dir', 'accuracy', 'macro_f1', 'fall_recall')},
        cohort_sha256=left['cohort_sha256'],
        note='Point estimate comparison; does not establish statistical superiority or automatically change the model.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=0.)
    parser.add_argument('--output', type=Path, default=Path('results/kfall_validation_comparison.json'))
    args = parser.parse_args(argv)
    try:
        result = compare_validation(args.baseline, args.candidate, args.tolerance)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
