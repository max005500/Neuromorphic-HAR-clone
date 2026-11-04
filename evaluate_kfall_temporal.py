#!/usr/bin/env python3
"""Evaluate a temporal KFall checkpoint with memory reset at each recording."""
import argparse
from pathlib import Path
from evaluation.kfall_temporal import evaluate_temporal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_temporal(args.run_dir, split_name=args.split, device=args.device,
                                   cache_dir=args.cache_dir, output_dir=args.output_dir)
        print(f"[{args.split}] accuracy={report['accuracy']:.2%} macro_F1={report['macro_f1']:.4f}")
        print(f"[saved] {report['output_dir']}")
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
