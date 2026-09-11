#!/usr/bin/env python3
"""Compare the iBT721, deposited 2022, and repaired 2022 WCFS1 GEMs.

The comparison uses one canonical extracellular namespace and a shared-medium
LP, so absent uptake bounds cannot be confused with absent nutrients.  It
reports attainable growth in the repository's starting medium and the minimum
additional net nutrient supplies required at a requested maintenance growth.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coexistence_audit import (  # noqa: E402
    SharedMediumCommunityLP,
    _single_external_metabolite,
    find_growth_reaction,
)
from src.utils import get_initial_params  # noqa: E402


MODEL_PATHS = {
    "WCFS1 iBT721": ROOT
    / "models/sbml/reference/Lactiplantibacillus_plantarum_WCFS1_iBT721.xml",
    "WCFS1 2022 deposit": ROOT
    / "models/sbml/reference/Lactiplantibacillus_plantarum_WCFS1_Koduru2022.xml",
    "WCFS1 2022 repaired": ROOT
    / "models/sbml/reference/Lactiplantibacillus_plantarum_WCFS1_Koduru2022_repaired.xml",
}


def _diagnostic_model(model, uptake_bound: float):
    local = model.copy()
    growth, coefficient = find_growth_reaction(local)
    local.objective = growth
    local.objective.direction = "max" if coefficient > 0 else "min"
    for exchange in local.exchanges:
        external = _single_external_metabolite(exchange)
        if external is None:
            continue
        _, stoich = external
        exchange.bounds = (
            min(float(exchange.lower_bound), -uptake_bound)
            if stoich < 0
            else min(float(exchange.lower_bound), 0.0),
            max(float(exchange.upper_bound), uptake_bound)
            if stoich > 0
            else max(float(exchange.upper_bound), 1000.0),
        )
    return local


def _mass_balance_counts(model) -> tuple[int, int]:
    assessed = 0
    imbalanced = 0
    boundaries = set(model.boundary)
    for reaction in model.reactions:
        if reaction in boundaries:
            continue
        assessed += 1
        if reaction.check_mass_balance():
            imbalanced += 1
    return assessed, imbalanced


def audit_model(label: str, path: Path, target_growth: float, dt: float) -> dict[str, Any]:
    source = read_sbml_model(str(path))
    diagnostic = _diagnostic_model(source, uptake_bound=20.0)
    biomass = {label: 0.1}
    _, medium = get_initial_params({label: diagnostic})
    solver = SharedMediumCommunityLP(
        {label: diagnostic},
        medium,
        biomass,
        dt=dt,
        growth_threshold=target_growth,
    )
    current = solver.solve()
    open_caps = {
        metabolite: 20.0 * biomass[label]
        for metabolite in solver.exchange_terms
    }
    fully_open = solver.solve(open_caps)
    rescue = solver.diagnose_required_supply(minimum_growth=target_growth)
    assessed, imbalanced = _mass_balance_counts(source)
    growth_reaction, _ = find_growth_reaction(source)
    gene_ids = [gene.id for gene in source.genes]
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "model_id": source.id,
        "model_name": source.name,
        "growth_reaction": growth_reaction.id,
        "reactions": len(source.reactions),
        "metabolites": len(source.metabolites),
        "genes": len(source.genes),
        "lp_locus_tag_fraction": (
            sum(gene.startswith("lp_") for gene in gene_ids) / max(1, len(gene_ids))
        ),
        "internal_reactions_assessed": assessed,
        "imbalanced_internal_reactions": imbalanced,
        "current_medium_growth_per_h": current.common_growth_per_h,
        "fully_open_growth_per_h": fully_open.common_growth_per_h,
        "target_growth_per_h": target_growth,
        "required_additional_supply": rescue,
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "label",
        "model_id",
        "growth_reaction",
        "reactions",
        "metabolites",
        "genes",
        "lp_locus_tag_fraction",
        "imbalanced_internal_reactions",
        "current_medium_growth_per_h",
        "fully_open_growth_per_h",
        "target_growth_per_h",
        "required_metabolites",
        "required_total_mmol_l_h",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            required = record["required_additional_supply"]
            writer.writerow(
                {
                    **{field: record.get(field, "") for field in fields},
                    "required_metabolites": ";".join(
                        item["metabolite"] for item in required
                    ),
                    "required_total_mmol_l_h": sum(
                        item["additional_supply_mmol_l_h"] for item in required
                    ),
                }
            )


def plot(records: list[dict[str, Any]], output_dir: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ["#0072B2", "#9A9A9A", "#009E73"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    labels = [record["label"] for record in records]
    x = list(range(len(records)))

    axes[0].bar(x, [record["genes"] for record in records], color=colors)
    axes[0].set_ylabel("Genes with GPR")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_title("a", loc="left", fontweight="bold")

    width = 0.35
    axes[1].bar(
        [value - width / 2 for value in x],
        [record["current_medium_growth_per_h"] for record in records],
        width,
        color="#56B4E9",
        label="Current medium",
    )
    axes[1].bar(
        [value + width / 2 for value in x],
        [record["fully_open_growth_per_h"] for record in records],
        width,
        color="#E69F00",
        label="All exchanges",
    )
    axes[1].set_ylabel(r"Max–min growth (h$^{-1}$)")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_title("b", loc="left", fontweight="bold")

    selected = records[-1]["required_additional_supply"][:10]
    metabolites = [item["metabolite"] for item in reversed(selected)]
    supplies = [item["additional_supply_mmol_l_h"] for item in reversed(selected)]
    axes[2].barh(metabolites, supplies, color="#009E73")
    axes[2].set_xlabel(r"Additional supply (mmol L$^{-1}$ h$^{-1}$)")
    axes[2].set_title("c", loc="left", fontweight="bold")
    axes[2].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"lplantarum_model_replacement_audit.{suffix}", dpi=400)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/lplantarum_model_replacement",
    )
    parser.add_argument("--target-growth", type=float, default=0.005)
    parser.add_argument("--dt", type=float, default=0.2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        audit_model(label, path, args.target_growth, args.dt)
        for label, path in MODEL_PATHS.items()
    ]
    (args.output_dir / "model_comparison.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "model_comparison.csv", records)
    plot(records, args.output_dir)
    print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
