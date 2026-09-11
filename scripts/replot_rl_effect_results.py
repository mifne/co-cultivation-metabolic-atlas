#!/usr/bin/env python3
"""Regenerate the RL effect figure from an existing validated result set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_until_effect_wcfs1 import plot_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    payload = json.loads(
        (result_dir / "rl_effect_results.json").read_text(encoding="utf-8")
    )
    with (result_dir / "ppo_action_trajectory.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        trajectory = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    plot_results(payload, trajectory, result_dir)


if __name__ == "__main__":
    main()
