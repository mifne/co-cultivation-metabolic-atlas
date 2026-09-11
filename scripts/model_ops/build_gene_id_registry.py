#!/usr/bin/env python3
"""Canonicalize consortium GPR gene IDs and build auditable alias tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cobra.io import read_sbml_model, write_sbml_model

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gene_id_registry import (  # noqa: E402
    DEFAULT_CONFIG,
    NON_RESOLVABLE_ALIAS_TYPES,
    build_model_registry,
    canonicalize_model_gene_ids,
    load_registry_config,
    registry_alias_index,
)


def _write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    config = load_registry_config(args.config)
    all_records = []
    model_audits = []

    for profile in config["organisms"]:
        path = args.model_dir / profile["model_file"]
        model = read_sbml_model(str(path))
        records_before, _ = build_model_registry(model, profile)
        if args.canonicalize_models:
            audit = canonicalize_model_gene_ids(model, profile)
            write_sbml_model(model, str(path))
            round_trip = read_sbml_model(str(path))
            if len(round_trip.genes) != len(model.genes):
                raise RuntimeError(f"Gene count changed during SBML round trip: {path}")
            records_after, _ = build_model_registry(round_trip, profile)
            by_project = {record.project_gene_id: record for record in records_before}
            for record in records_after:
                prior = by_project.get(record.project_gene_id)
                if prior:
                    prior.reaction_ids.update(record.reaction_ids)
                    prior.aliases.update(record.aliases)
            records = list(by_project.values())
        else:
            audit = {
                "organism_key": profile["organism_key"],
                "genes_before": len(model.genes),
                "genes_after": len(model.genes),
                "renamed_aliases": 0,
                "merged_duplicate_gene_objects": 0,
                "simplified_gpr_rules": 0,
                "retired_unresolved_gene_ids": [],
                "historical_quarantined_gene_ids": [
                    record.canonical_model_gene_id
                    for record in records_before
                    if record.mapping_status == "retired_unresolved"
                ],
                "unresolved_gene_ids": [
                    record.canonical_model_gene_id
                    for record in records_before
                    if record.mapping_status == "unresolved"
                ],
            }
            records = records_before
        all_records.extend(records)
        model_audits.append(audit)

    registry_alias_index(all_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    registry_rows = [record.as_tsv_row() for record in all_records]
    registry_rows.sort(key=lambda row: row["project_gene_id"])
    registry_fields = [
        "project_gene_id",
        "organism_key",
        "species_name",
        "assembly_accession",
        "canonical_locus_tag",
        "canonical_model_gene_id",
        "protein_accession",
        "gene_symbol",
        "product",
        "mapping_status",
        "mapping_method",
        "reaction_count",
        "reaction_ids",
        "evidence",
    ]
    _write_tsv(args.output_dir / "gene_id_registry.tsv", registry_rows, registry_fields)

    alias_rows = []
    for record in all_records:
        for alias, alias_type in sorted(record.aliases.items()):
            alias_rows.append(
                {
                    "organism_key": record.organism_key,
                    "alias": alias,
                    "alias_type": alias_type,
                    "resolvable": alias_type not in NON_RESOLVABLE_ALIAS_TYPES,
                    "project_gene_id": record.project_gene_id,
                    "canonical_locus_tag": record.canonical_locus_tag,
                }
            )
    alias_rows.sort(key=lambda row: (row["organism_key"], row["alias"], row["project_gene_id"]))
    _write_tsv(
        args.output_dir / "gene_id_aliases.tsv",
        alias_rows,
        [
            "organism_key",
            "alias",
            "alias_type",
            "resolvable",
            "project_gene_id",
            "canonical_locus_tag",
        ],
    )

    status_counts: dict[str, int] = {}
    for row in registry_rows:
        status_counts[row["mapping_status"]] = status_counts.get(row["mapping_status"], 0) + 1
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.relative_to(ROOT)),
        "canonicalized_models": bool(args.canonicalize_models),
        "canonical_policy": config["canonical_policy"],
        "registry_gene_count": len(registry_rows),
        "alias_count": len(alias_rows),
        "status_counts": status_counts,
        "model_audits": model_audits,
    }
    (args.output_dir / "gene_id_registry.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Gene ID registry audit",
        "",
        "GPR内では株別locus tagを使用し、`CCG_<organism>_<locus_tag>`をプロジェクト全体の一意キーとする。元のSBML/FASTA IDとNCBI protein accessionは別名表に保持する。",
        "",
        "| Organism | Before | After | Renamed aliases | Merged duplicates | Retired IDs | Simplified GPRs | Quarantined history | Unresolved |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for audit in model_audits:
        lines.append(
            f"| {audit['organism_key']} | {audit['genes_before']} | {audit['genes_after']} | "
            f"{audit['renamed_aliases']} | {audit['merged_duplicate_gene_objects']} | "
            f"{len(audit['retired_unresolved_gene_ids'])} | "
            f"{audit['simplified_gpr_rules']} | "
            f"{', '.join(audit['historical_quarantined_gene_ids']) or 'none'} | "
            f"{', '.join(audit['unresolved_gene_ids']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "未解決IDは自動推測せず、配列または原著の根拠を追加してから設定ファイルで解決する。",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "models/sbml/final_consortium"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "models/gene_registry"
    )
    parser.add_argument("--canonicalize-models", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
