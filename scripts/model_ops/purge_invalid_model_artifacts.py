#!/usr/bin/env python3
"""Remove ML artifacts generated with the invalid pre-WCFS1 consortium.

The target list is deliberately explicit.  Raw SBML sources, curation
provenance, current WCFS1 analyses, and source code are never selected.
Run without ``--apply`` to audit the deletion set.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/data_invalidation_2026-09-01.json"

INVALID_DIRECTORIES = (
    "models/fba_surrogate",
    "models/fba_surrogate_gpu_lp",
    "models/fba_surrogate_gpu_lp_2048",
    "models/fba_surrogate_gpu_lp_512",
    "models/fba_surrogate_pilot",
    "models/fba_surrogate_pilot_dict",
    "outputs",
    "results/corrected_feed_screen",
    "results/flux_nutrition_audit",
    "results/rl_behavior",
    "results/rl_behavior_100k",
    "results/rl_behavior_baseline",
    "results/rl_behavior_fix",
    "results/rl_eval",
    "results/rl_problem_feasibility",
    "results/rl_problem_feasibility_oracle",
    "results/rl_problem_feasibility_static",
    "results/rl_problem_feasibility_summary",
    "results/three_pump_feed_design",
    "results/coexistence_audit_24h",
    "results/coexistence_audit_48h",
    "results/coexistence_audit_final",
    "results/coexistence_audit_ph_control",
    "results/coexistence_audit_static",
    "results/coexistence_audit_static2",
    "results/coexistence_audit_static_v2",
)

INVALID_RESULT_PREFIXES = (
    "benchmark_",
    "compare_",
    "exact_fba_",
    "fba_",
    "gpu_",
    "hardware_",
    "parallel_",
    "ppo_",
    "profile_",
    "resource_",
    "rollout_",
    "scale_",
    "workstation_",
)

INVALID_PAPER_SUFFIXES = {".csv", ".pdf", ".png"}


def _assert_safe(path: Path) -> Path:
    resolved_root = ROOT.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root) or resolved == resolved_root:
        raise RuntimeError(f"Refusing unsafe deletion target: {resolved}")
    return resolved


def _files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return [item for item in path.rglob("*") if item.is_file()]
    return []


def collect_targets() -> tuple[list[Path], list[Path]]:
    directories = [ROOT / relative for relative in INVALID_DIRECTORIES]
    files = []
    results = ROOT / "results"
    if results.exists():
        files.extend(
            path
            for path in results.iterdir()
            if path.is_file() and path.name.startswith(INVALID_RESULT_PREFIXES)
        )
    paper = ROOT / "paper_figures"
    if paper.exists():
        files.extend(
            path
            for path in paper.iterdir()
            if path.is_file() and path.suffix.lower() in INVALID_PAPER_SUFFIXES
        )
    files.append(ROOT / "output/pdf/workstation_predicted_behavior.pdf")
    directories = sorted({_assert_safe(path) for path in directories if path.exists()})
    files = sorted({_assert_safe(path) for path in files if path.exists()})
    return directories, files


def run(apply: bool) -> dict:
    directories, files = collect_targets()
    if apply and not directories and not files and MANIFEST.exists():
        previous = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if previous.get("applied"):
            return previous
    all_files = []
    for path in directories:
        all_files.extend(_files_under(path))
    all_files.extend(files)
    unique_files = sorted(set(all_files))
    total_bytes = sum(path.stat().st_size for path in unique_files)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "applied": bool(apply),
        "reason": (
            "Artifacts were generated before activation of the repaired WCFS1 2022 GEM "
            "and/or explicitly contain Lactobacillus_plantarum_iNF517. Reaction order, "
            "stoichiometry, GPRs, observations, policies, and surrogate targets are not "
            "compatible with the current consortium."
        ),
        "deleted_file_count": len(unique_files) if apply else 0,
        "selected_file_count": len(unique_files),
        "selected_bytes": total_bytes,
        "selected_gib": total_bytes / 1024**3,
        "directory_targets": [str(path.relative_to(ROOT)) for path in directories],
        "file_targets": [str(path.relative_to(ROOT)) for path in files],
        "preserved": [
            "models/sbml/reference",
            "models/sbml/final_consortium",
            "models/gene_registry",
            "results/coexistence_audit_wcfs1_2022",
            "results/flux_nutrition_wcfs1_2022",
            "results/lplantarum_model_replacement*",
            "results/wcfs1_coexistence_strategy*",
            "results/program_*",
        ],
    }
    if apply:
        for path in files:
            if path.exists():
                path.unlink()
        for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            if path.exists():
                shutil.rmtree(path)
    if apply:
        MANIFEST.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
