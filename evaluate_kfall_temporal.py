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
    parser.add_argument('--fuse-bn', action='store_true', help='Fold BatchNorm into Conv/Linear for FP32 inference.')
    parser.add_argument('--weight-bits', type=int, choices=[8],
                        help='Simulate symmetric per-output-channel INT8 weights with FP32 computation/state.')
    args = parser.parse_args(argv)
    try:
        report = evaluate_temporal(args.run_dir, split_name=args.split, device=args.device,
                                   cache_dir=args.cache_dir, output_dir=args.output_dir,
                                   fuse_bn=args.fuse_bn, weight_bits=args.weight_bits)
        print(f"[{args.split}] accuracy={report['accuracy']:.2%} macro_F1={report['macro_f1']:.4f}")
        print(f"[saved] {report['output_dir']}")
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
