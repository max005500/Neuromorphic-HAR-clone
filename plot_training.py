#!/usr/bin/env python3
"""Generate accuracy/loss curves from saved histories without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.plotting import save_training_curves


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Run directories, metrics directories, or history.json files; defaults to all saved runs.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"), help="Directory searched when no paths are provided.")
    args = parser.parse_args(argv)
    histories = []
    if args.paths:
        for path in args.paths:
            candidates = [path] if path.is_file() else [path / "history.json", path / "metrics" / "history.json"]
            source = next((candidate for candidate in candidates if candidate.is_file()), None)
            if source is None:
                parser.error(f"No history.json found at {path}.")
            histories.append(source)
    else:
        histories = sorted(args.runs_root.glob("*/metrics/history.json"))
        if not histories:
            parser.error(f"No training histories found under {args.runs_root}.")

    for source in dict.fromkeys(histories):
        try:
            history = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read {source}: {exc}")
        if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
            parser.error(f"{source} must contain a list of epoch metrics.")
        paths = save_training_curves(history, source.parent)
        if not paths:
            print(f"[skip] Empty training history: {source}")
        for name, destination in paths.items():
            print(f"[plots] {name}={destination}")


if __name__ == "__main__":
    main()
