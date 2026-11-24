#!/usr/bin/env python3
"""Aggregate already evaluated runs; never trains or evaluates a checkpoint.

Example:
  python report_paper_results.py results.csv --output-dir results/paper
  python report_paper_results.py results.csv --output-dir results/exploratory --descriptive

CSV/API and energy-boundary details: evaluation.paper_statistics module docstring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.paper_statistics import DEFAULT_METRICS, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Required CSV: experiment,dataset,protocol,split,variant,seed,macro_f1.\n"
            "split=train/val/test; scores are fractions; seed is an integer.\n"
            "Optional: model_id,accuracy,params/parameters,energy_network_j,energy_pe_norm,\n"
            "run_id,source_uri,source. Normalized PE is NOT joules.\n"
            "All other columns are grouping metadata; do not omit protocol details.\n"
            "Strict: >=5 distinct seeds/group, sample SD; duplicates always fail.\n"
            "Missing metrics remain null; partially observed metrics fail strict mode.\n"
            "Outputs: summary.csv, report.json (input SHA256 + row origins), results.csv,\n"
            "selected_results.csv, pareto.csv, Agg PNGs when energy is available.\n"
            "SCN selection requires complete val/test pairs for every candidate/seed.\n"
            "SCNacc=max val score; SCNminE=min val energy (no accuracy floor).\n"
            "Ties: variant/model_id lexical order; SCNsel is not implemented.\n"
            "Test Pareto plots are descriptive, NEVER a model-selection policy.\n"
            "See evaluation.paper_statistics for explicit ADC/Sigma-Delta energy inputs."
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="CSV results (no implicit schema adapters)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--descriptive", action="store_true", help="Allow incomplete groups; label ALL output NON_PUBLISHABLE")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                        help="Numeric metric columns; must include macro_f1. Unknown columns otherwise group as metadata.")
    parser.add_argument("--score", choices=("macro_f1", "accuracy"), default="macro_f1")
    parser.add_argument("--energy", choices=("energy_network_j", "energy_total_j", "energy_pe_norm"),
                        default="energy_network_j", help="Pareto/SCNminE axis; normalized PE requires explicit choice")
    parser.add_argument("--scn-variants", nargs="+", default=(),
                        help="Explicit SCN candidate variant pool; enables validation-only SCNacc/SCNminE")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = write_report(
            args.inputs, args.output_dir, descriptive=args.descriptive, metrics=args.metrics,
            score=args.score, energy=args.energy, scn_variants=args.scn_variants,
        )
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"{report['status']}: {len(report['summary'])} groups -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
