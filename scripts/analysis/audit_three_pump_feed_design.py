#!/usr/bin/env python3
"""Screen feed-channel combinations for a three-feed-pump fermenter."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.audit_rl_problem_feasibility import YEAST_EXTRACT_COMPOSITION
from src.coexistence_audit import SharedMediumCommunityLP
from src.utils import get_initial_params, load_sbml_models, select_consortium_models


def merge_caps(
    medium: dict[str, float], dt: float, additions: list[dict[str, float]]
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for addition in additions:
        for metabolite, rate in addition.items():
            rates[metabolite] = rates.get(metabolite, 0.0) + float(rate)
    return {
        metabolite: medium.get(metabolite, 0.0) / dt + rate
        for metabolite, rate in rates.items()
    }


def run(args: argparse.Namespace) -> dict:
    models = select_consortium_models(load_sbml_models(args.sbml_dir))
    biomass, medium = get_initial_params(models)
    solver = SharedMediumCommunityLP(models, medium, biomass, dt=args.dt)
    yeast_rate = {
        metabolite: args.yeast_extract_rate * coefficient
        for metabolite, coefficient in YEAST_EXTRACT_COMPOSITION.items()
    }
    channels = {
        "OR16_maltotriose": {"mlttr_e": args.specific_rate},
        "NS21_putrescine": {"ptrc_e": args.specific_rate},
        "LP_mannitol": {"mnl_e": args.specific_rate},
        "yeast_extract": yeast_rate,
    }
    combinations = []
    names = list(channels)
    for size in range(len(names) + 1):
        for selected in itertools.combinations(names, size):
            caps = merge_caps(medium, args.dt, [channels[name] for name in selected])
            result = solver.solve(caps)
            combinations.append(
                {
                    "channels": list(selected),
                    "channel_count": len(selected),
                    "feasible": result.feasible,
                    "common_growth_per_h": result.common_growth_per_h,
                }
            )
    carbon_candidates = {
        "glucose": "glc__D_e",
        "mannitol": "mnl_e",
        "glutamate": "glu__L_e",
        "succinate": "succ_e",
        "glycerol": "glyc_e",
    }
    alternatives = []
    for name, metabolite in carbon_candidates.items():
        caps = merge_caps(
            medium,
            args.dt,
            [yeast_rate, {metabolite: args.specific_rate}],
        )
        result = solver.solve(caps)
        alternatives.append(
            {
                "carbon": name,
                "metabolite": metabolite,
                "feasible_with_yeast_extract": result.feasible,
                "common_growth_per_h": result.common_growth_per_h,
            }
        )
    return {
        "parameters": {
            "dt_h": args.dt,
            "specific_feed_rate_mmol_l_h": args.specific_rate,
            "yeast_extract_equivalent_rate_g_l_h": args.yeast_extract_rate,
        },
        "channel_combinations": combinations,
        "carbon_alternatives": alternatives,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sbml-dir",
        type=Path,
        default=ROOT / "models" / "sbml" / "final_consortium",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "three_pump_feed_design"
    )
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--specific-rate", type=float, default=0.5)
    parser.add_argument("--yeast-extract-rate", type=float, default=2.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = run(args)
    json_path = args.output_dir / "three_pump_feed_design.json"
    csv_path = args.output_dir / "three_pump_channel_combinations.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("channels", "channel_count", "feasible", "common_growth_per_h"),
        )
        writer.writeheader()
        for row in payload["channel_combinations"]:
            writer.writerow({**row, "channels": "+".join(row["channels"])})
    minimal = [
        item
        for item in payload["channel_combinations"]
        if item["feasible"]
        and item["channel_count"]
        == min(
            row["channel_count"]
            for row in payload["channel_combinations"]
            if row["feasible"]
        )
    ]
    print("Minimal feasible channel sets:")
    print(json.dumps(minimal, ensure_ascii=False, indent=2))
    print("Carbon alternatives with yeast extract:")
    print(json.dumps(payload["carbon_alternatives"], ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
