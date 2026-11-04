#!/usr/bin/env python3
"""Plan/run isolated recurrent-SNN ablations on the same uniform KFall cohort."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

PRESETS = {
    'A': ('none', 0., 0., 'single'),
    'B': ('lif', 0., 0., 'single'),
    'C': ('lif', .1, 0., 'single'),
    'D': ('lif', .1, .1, 'single'),
    'F': ('lif', 0., 0., 'hierarchical'),
}


def commands(args, remaining):
    result = []
    for length in args.sequence_lengths:
        for key in args.experiments:
            memory, transition, boundary, head = PRESETS[key]
            if key in {'C', 'D'}:
                transition = args.transition_weight
            if key == 'D':
                boundary = args.boundary_weight
            directory = args.out_dir / f'{key}_K{length}'
            command = [sys.executable, str(Path(__file__).with_name('train_kfall_temporal.py')), *remaining,
                '--memory', memory, '--sequence-length', str(length), '--head-mode', head,
                '--lambda-transition', str(transition), '--lambda-boundary', str(boundary),
                '--out-dir', str(directory), '--run', f'kfall_temporal_{key}_K{length}']
            result.append(dict(experiment=key, sequence_length=length, command=command))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['plan', 'train'], default='plan')
    parser.add_argument('--experiments', nargs='+', choices=list(PRESETS), default=['A', 'B', 'C', 'D'])
    parser.add_argument('--sequence-lengths', type=int, nargs='+', default=[20, 32])
    parser.add_argument('--transition-weight', type=float, default=.1)
    parser.add_argument('--boundary-weight', type=float, default=.1)
    parser.add_argument('--out-dir', type=Path, default=Path('runs/kfall_recurrent_ablation'))
    parser.add_argument('--results-dir', type=Path, default=Path('results/kfall_recurrent_ablation'))
    args, remaining = parser.parse_known_args(argv)
    if any(k < 1 for k in args.sequence_lengths) or len(set(args.sequence_lengths)) != len(args.sequence_lengths):
        parser.error('Sequence lengths must be positive and distinct.')
    plan = commands(args, remaining)
    # Validate every proposed command before launching the first experiment.
    from train_kfall_temporal import get_args
    for row in plan:
        get_args(row['command'][2:])
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'[plan] {args.results_dir / "plan.json"}', flush=True)
    if args.mode == 'plan':
        return
    rows = []
    for item in plan:
        directory = Path(item['command'][item['command'].index('--out-dir') + 1])
        before = set(directory.glob('*'))
        subprocess.run(item['command'], check=True)
        created = [path for path in set(directory.glob('*')) - before if (path / 'metrics/validation/metrics.json').is_file()]
        if len(created) != 1:
            raise RuntimeError('Unable to identify the newly completed experiment.')
        run = created[0]
        val = json.loads((run / 'metrics/validation/metrics.json').read_text())
        report = dict(experiment=item['experiment'], sequence_length=item['sequence_length'], run_dir=str(run),
            validation_macro_f1=val['macro_f1'], validation_fall_recall=val['fall_window_metrics']['recall'],
            validation_accuracy=val['accuracy'], cohort_sha256=val['cohort_sha256'])
        test = run / 'metrics/test_best/metrics.json'
        if test.exists():
            measured = json.loads(test.read_text())
            report.update(test_accuracy=measured['accuracy'], test_macro_f1=measured['macro_f1'],
                          event_metrics=measured['event_metrics'])
        rows.append(report)
        if len({entry['cohort_sha256'] for entry in rows}) != 1:
            raise ValueError('Ablation runs do not use the same validation endpoint/label cohort.')
        (args.results_dir / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
    import csv
    fields = ['experiment', 'sequence_length', 'validation_accuracy', 'validation_macro_f1',
              'validation_fall_recall', 'test_accuracy', 'test_macro_f1', 'run_dir']
    with (args.results_dir / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    print(f'[saved] {args.results_dir / "comparison.csv"}')


if __name__ == '__main__':
    main()
