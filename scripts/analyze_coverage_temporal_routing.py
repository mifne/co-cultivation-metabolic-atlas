"""Replay temporal compact-bank routing on an immutable held-out LP trace.

This is a proposal-ordering diagnostic.  It does not solve an LP, use the
stored CPU reference vectors as features/warm starts, change a trajectory, or
measure runtime speed.  Strict original-LP certificate coverage measured in a
separate GPU run is treated as the immutable boolean outcome table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_basis_bank_rollout import environment
from src.compact_training_data import checked_npz, sha256_file
from src.coverage_router import CoverageRouter, coverage_metrics
from src.coverage_router_binding import compact_feature_schema, compact_router_provenance
from src.cpu_repeated_lp import _problem
from src.fba_surrogate import model_fingerprint
from src.gpu_certified_basis import CommunityCoordinates
from src.lp_trace import problem_hash, problem_request
from src.selected_lp_features import SelectedLPFeatures


POLICIES = ("stateless_learned", "previous_first", "learned_first")


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_problem_without_reference(directory: Path, entry: dict):
    """Restore one original LP while deliberately not reading x/y references."""
    directory = directory.resolve()
    path = (directory / entry["filename"]).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("Untrusted LP trace path")
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("LP trace checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        required = {"a_data", "a_indices", "a_indptr", "a_shape", "rhs",
                    "lower", "upper", "c", "neq", "reference_x", "reference_y"}
        if set(data.files) != required:
            raise ValueError("Unexpected LP trace payload")
        # reference_x/reference_y are intentionally neither indexed nor loaded.
        a = csr_matrix((data["a_data"], data["a_indices"], data["a_indptr"]),
                       shape=tuple(data["a_shape"]))
        problem = (a, data["rhs"], data["lower"], data["upper"],
                   data["c"], int(data["neq"]))
    rebuilt = _problem(*problem_request(problem))
    if problem_hash(rebuilt) != entry["problem_sha256"]:
        raise ValueError("LP trace input changed during restoration")
    a, _, _, _, _, neq = rebuilt
    expected = (entry.get("rows"), entry.get("columns"),
                entry.get("nonzeros"), entry.get("equalities"))
    if expected != (a.shape[0], a.shape[1], a.nnz, neq):
        raise ValueError("LP trace dimensions disagree with the payload")
    return rebuilt


class _CpuCompactFeatureBuilder:
    """Exact CPU counterpart of CompactBank.prepare_host + selected encoding."""

    def __init__(self, root, variable_rows, feature_indices):
        root = csr_matrix(root, dtype=np.float64)
        variables = np.asarray(variable_rows)
        if (not root.has_canonical_format or not np.isfinite(root.data).all()
                or variables.ndim != 1 or variables.dtype.kind not in "iu"
                or len(np.unique(variables)) != len(variables)
                or np.any(variables < 0) or np.any(variables >= root.shape[0])):
            raise ValueError("Invalid compact root or variable rows")
        self.root = root
        self.variables = variables.astype(np.int64, copy=True)
        rows, columns = root.shape
        data_rows = np.repeat(np.arange(rows), np.diff(root.indptr))
        row_map = np.full(rows, -1, dtype=np.int64)
        row_map[self.variables] = np.arange(len(self.variables))
        local = row_map[data_rows]
        positions = np.flatnonzero(local >= 0)
        self.pattern = dict(
            fixed=np.flatnonzero(local < 0),
            positions=positions,
            local_rows=local[positions],
            columns=root.indices[positions].copy(),
        )
        self.selector = SelectedLPFeatures(feature_indices, dict(
            rhs=(rows,), lower=(columns,), upper=(columns,), c=(columns,),
            delta=(len(self.variables), columns), col_scale=(columns,),
            row_scale=(rows,)))

    def inputs(self, problem):
        root, pattern = self.root, self.pattern
        if problem.a.shape != root.shape:
            raise ValueError("Changed compact LP shape")
        if (not isinstance(problem.a, csr_matrix) or not problem.a.has_canonical_format
                or not np.array_equal(problem.a.indptr, root.indptr)
                or not np.array_equal(problem.a.indices, root.indices)):
            # Match the production fallback exactly when a variable row changes
            # sparse support.  Fixed-row changes remain forbidden.
            delta = (problem.a - root).tocsr()
            fixed_rows = np.ones(root.shape[0], dtype=bool)
            fixed_rows[self.variables] = False
            if (not np.isfinite(delta.data).all()
                    or np.max(np.abs(delta[fixed_rows].data), initial=0.) > 2e-12):
                raise ValueError("Changed unsupported compact matrix row")
            dense = delta[self.variables].toarray()
        else:
            change = problem.a.data - root.data
            if (not np.isfinite(change).all()
                    or np.max(np.abs(change[pattern["fixed"]]), initial=0.) > 2e-12):
                raise ValueError("Changed unsupported compact matrix coefficient")
            dense = np.zeros((len(self.variables), root.shape[1]), dtype=change.dtype)
            dense[pattern["local_rows"], pattern["columns"]] = change[pattern["positions"]]
        return dict(delta=dense, **{name:getattr(problem, name) for name in
            ("rhs", "lower", "upper", "c", "col_scale", "row_scale")})

    def features(self, problem):
        return self.selector.cpu_features(self.inputs(problem))


def _promote(base, previous, policy):
    base = np.asarray(base)
    if previous < 0 or policy == "stateless_learned":
        return base
    remainder = base[base != previous]
    if policy == "previous_first":
        return np.r_[previous, remainder]
    if policy == "learned_first":
        # Keep the learned top-1 prediction first.  The most recently certified
        # candidate gets the next available slot and may displace only the
        # lowest-ranked candidate after truncation.
        if base[0] == previous:
            return base
        return np.r_[base[0], previous, base[1:][base[1:] != previous]]
    raise ValueError(f"Unknown temporal policy: {policy}")


def replay_policy(coverage, learned_order, *, trajectories, steps, k, policy):
    """Causally update prior-candidate state within each fixed trajectory."""
    coverage = np.asarray(coverage)
    learned_order = np.asarray(learned_order)
    if (coverage.ndim != 2 or coverage.dtype != np.bool_
            or learned_order.shape != coverage.shape
            or learned_order.dtype.kind not in "iu"
            or trajectories * steps != len(coverage)
            or not 1 <= k <= coverage.shape[1]):
        raise ValueError("Invalid temporal coverage replay inputs")
    accepted = np.zeros(len(coverage), dtype=bool)
    selected = np.full(len(coverage), -1, dtype=np.int32)
    prior = np.full(len(coverage), -1, dtype=np.int32)
    promoted = np.zeros(len(coverage), dtype=bool)
    previous_currently_passes = np.zeros(len(coverage), dtype=bool)
    for trajectory in range(trajectories):
        previous = -1
        for step in range(steps):
            row = trajectory * steps + step
            prior[row] = previous
            if previous >= 0:
                promoted[row] = previous != learned_order[row, 0]
                previous_currently_passes[row] = coverage[row, previous]
            order = _promote(learned_order[row], previous, policy)[:k]
            hits = coverage[row, order]
            if hits.any():
                accepted[row] = True
                selected[row] = int(order[np.flatnonzero(hits)[0]])
            # The stateless control deliberately carries no candidate across
            # rows; this also keeps its temporal telemetry truthful.
            previous = (-1 if policy == "stateless_learned"
                        else int(selected[row]))
    return dict(accepted=accepted, selected=selected, prior=prior,
                promoted=promoted, previous_currently_passes=previous_currently_passes)


def _policy_summary(state, row_identity):
    accepted = state["accepted"]
    transitions = state["prior"] >= 0
    promoted = state["promoted"]
    prior_pass = state["previous_currently_passes"]
    by_trajectory = []
    for trajectory in sorted(set(row["trajectory_index"] for row in row_identity)):
        ids = np.array([row["trajectory_index"] == trajectory for row in row_identity])
        by_trajectory.append(dict(
            trajectory_index=trajectory,
            seed=row_identity[np.flatnonzero(ids)[0]]["seed"],
            accepted_rows=int(accepted[ids].sum()), rows=int(ids.sum())))
    return dict(
        accepted_rows=int(accepted.sum()),
        acceptance_rate=float(accepted.mean()),
        rows_with_previous=int(transitions.sum()),
        rows_with_promoted_non_top1_previous=int(promoted.sum()),
        previous_candidate_still_certifies=int((transitions & prior_pass).sum()),
        previous_candidate_no_longer_certifies=int((transitions & ~prior_pass).sum()),
        distinct_selected_candidates=int(len(np.unique(state["selected"][state["selected"] >= 0]))),
        by_trajectory=by_trajectory,
    )


def _paired(first, second, row_identity):
    a, b = first["accepted"], second["accepted"]
    first_only = np.flatnonzero(a & ~b)
    second_only = np.flatnonzero(~a & b)
    def labels(indices):
        return [row_identity[int(index)] for index in indices]
    return dict(
        both_accept=int((a & b).sum()),
        neither_accept=int((~a & ~b).sum()),
        previous_first_only=int(len(first_only)),
        learned_first_only=int(len(second_only)),
        selected_candidate_differs=int(np.sum(first["selected"] != second["selected"])),
        previous_first_only_rows=labels(first_only),
        learned_first_only_rows=labels(second_only),
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--router-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    trace_path = args.trace / "manifest.json"
    bank_path = args.bank / "manifest.json"
    trace = _load_json(trace_path)
    bank_manifest = _load_json(bank_path)
    if trace.get("status") != "completed" or trace.get("role") != "development_diagnostic_not_training":
        raise ValueError("A completed held-out diagnostic trace is required")
    if bank_manifest.get("status") != "completed":
        raise ValueError("A completed compact bank is required")
    seeds = trace.get("seeds")
    steps = trace.get("completed_steps")
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or not isinstance(steps, list) or len(steps) != len(seeds)
            or len(set(steps)) != 1 or steps[0] < 1):
        raise ValueError("Trace requires unique equal-length trajectories")
    if set(seeds) & set(bank_manifest.get("train_seeds", [])):
        raise ValueError("Held-out trace overlaps bank training seeds")

    with np.load(args.coverage, allow_pickle=False) as data:
        required = {"coverage", "candidate_indices", "routing_order", "routing_topk",
                    "batch_sizes", "candidate_chunk_size", "complete", "schema_version",
                    "certificate_thresholds", "timing", "failures", "metadata",
                    "routing_summary", "scope"}
        if set(data.files) != required:
            raise ValueError("Unexpected coverage artifact schema")
        coverage = data["coverage"].copy()
        candidate_indices = data["candidate_indices"].copy()
        coverage_metadata = json.loads(str(data["metadata"]))
        certificate_thresholds = json.loads(str(data["certificate_thresholds"]))
        coverage_complete = bool(data["complete"])
    rows = len(seeds) * steps[0]
    if (not coverage_complete or coverage.dtype != np.bool_ or coverage.shape[0] != rows
            or not np.array_equal(candidate_indices, np.arange(coverage.shape[1]))):
        raise ValueError("Incomplete or reordered held-out coverage")
    trace_sha = sha256_file(trace_path)
    bank_sha = sha256_file(bank_path)
    if (coverage_metadata.get("trace_sha256") != trace_sha
            or coverage_metadata.get("bank_sha256") != bank_sha
            or coverage_metadata.get("seeds") != seeds):
        raise ValueError("Coverage provenance disagrees with trace/bank")

    stages = [stage for stage in bank_manifest.get("stages", []) if stage.get("stage") == "maxmin"]
    if len(stages) != 1:
        raise ValueError("Exactly one maxmin bank stage is required")
    stage = stages[0]
    if len(stage.get("entries", [])) != coverage.shape[1]:
        raise ValueError("Coverage candidate count differs from bank")
    raw_root = checked_npz(args.bank / "maxmin", "root.npz", stage["root_sha256"])
    root = csr_matrix((raw_root["a_data"], raw_root["a_indices"], raw_root["a_indptr"]),
                      shape=tuple(raw_root["a_shape"]))
    nearest = checked_npz(args.bank / "maxmin", "router.npz", stage["router_sha256"])
    fake_bank = SimpleNamespace(host_a=root, variable_rows=raw_root["variable_rows"],
                                feature_indices=nearest["indices"])
    training_coverage_sha = sha256_file(args.bank / "coverage.npz")
    expected_provenance = compact_router_provenance(
        bank_sha, stage, fake_bank, training_coverage_sha,
        bank_manifest["model_fingerprints"])
    router = CoverageRouter.load(args.router, expected_sha256=args.router_sha256,
                                 expected_provenance=expected_provenance)
    for name, value in expected_provenance.items():
        if router.provenance.get(name) != value:
            raise ValueError(f"Router stage binding mismatch: {name}")

    sample, layout = environment(seeds[0])
    current_fingerprints = {name:model_fingerprint(model)
                            for name, model in sample.simulator.models.items()}
    if current_fingerprints != trace.get("model_fingerprints") or current_fingerprints != bank_manifest.get("model_fingerprints"):
        raise ValueError("Current GEM fingerprints disagree with immutable artifacts")
    metadata = dict(
        growth_terms=layout._growth_terms,
        exchange_terms=layout._exchange_terms,
        reaction_ids=[reaction.id for model in sample.simulator.models.values()
                      for reaction in model.reactions],
        reaction_species=[name for name, model in sample.simulator.models.items()
                          for reaction in model.reactions],
    )

    entries = {(entry["environment_id"], entry["step"]):entry
               for entry in trace.get("entries", []) if entry.get("stage") == "maxmin"}
    if len(entries) != rows:
        raise ValueError("Incomplete or duplicate maxmin trace entries")
    first = _load_problem_without_reference(args.trace, entries[(0, 1)])
    if stage.get("key") != ["maxmin", *first[0].shape, first[-1]]:
        raise ValueError("Trace LP family differs from maxmin bank")
    coordinates = CommunityCoordinates(metadata, first[0], first[-1])
    feature_builder = _CpuCompactFeatureBuilder(
        root, raw_root["variable_rows"], nearest["indices"])
    features = []
    row_identity = []
    for environment_id, seed in enumerate(seeds):
        for step in range(1, steps[0] + 1):
            original = first if (environment_id, step) == (0, 1) else \
                _load_problem_without_reference(args.trace, entries[(environment_id, step)])
            normalized = coordinates.normalize(*original)
            features.append(feature_builder.features(normalized))
            row_identity.append(dict(row=environment_id * steps[0] + step - 1,
                                     trajectory_index=environment_id,
                                     seed=seed, step=step))
    features = np.stack(features)
    learned_order = router.rank(features, k=coverage.shape[1])
    learned_metrics = coverage_metrics(coverage, learned_order, top_k=(1, 2, 4))

    results = {}
    for k in (1, 2, 4):
        states = {policy:replay_policy(coverage, learned_order,
            trajectories=len(seeds), steps=steps[0], k=k, policy=policy)
            for policy in POLICIES}
        results[str(k)] = dict(
            policies={policy:_policy_summary(state, row_identity)
                      for policy, state in states.items()},
            paired_previous_vs_learned_first=_paired(
                states["previous_first"], states["learned_first"], row_identity),
            paired_previous_vs_stateless=_paired(
                states["previous_first"], states["stateless_learned"], row_identity),
        )

    report = dict(
        status="completed",
        scope=("Offline causal routing-state replay over a fixed held-out LP trace. "
               "This is not closed-loop dFBA, an LP solve, or a speed benchmark."),
        policies=dict(
            stateless_learned="Use learned candidate order; retain no temporal state.",
            previous_first=("Production-equivalent promotion: put the candidate that certified "
                            "the preceding LP first, then the learned order without duplicates."),
            learned_first=("Keep learned top-1 first; put the candidate that certified the "
                           "preceding LP second when distinct, then the remaining learned order."),
        ),
        evidence_contract=dict(
            coverage="Immutable strict original-LP certificate booleans from the held-out GPU coverage artifact.",
            features="Original LP inputs only; stored CPU reference_x/reference_y arrays were not read.",
            temporal_state="Only a candidate certified on the immediately preceding fixed-trace row is retained.",
            cpu_lp_solves=0,
            gpu_lp_solves=0,
            training_or_refit=False,
            trajectory_effects="Routing cannot change the fixed LP inputs in this replay; closed-loop confirmation remains required."),
        provenance=dict(
            trace_manifest=str(trace_path.resolve()), trace_manifest_sha256=trace_sha,
            coverage=str(args.coverage.resolve()), coverage_sha256=sha256_file(args.coverage),
            bank_manifest=str(bank_path.resolve()), bank_manifest_sha256=bank_sha,
            bank_training_coverage_sha256=training_coverage_sha,
            router=str(args.router.resolve()), router_sha256=args.router_sha256,
            script_sha256=sha256_file(Path(__file__)), seeds=seeds,
            steps_per_trajectory=steps[0], row_order="trajectory-major then step",
            model_fingerprints=current_fingerprints,
            feature_schema_sha256=compact_feature_schema(fake_bank)[1],
            certificate_thresholds=certificate_thresholds),
        reproduction_check=dict(
            heldout_learned_metrics=learned_metrics,
            requirement="Must match the previously reported stateless learned K1/K2/K4 counts."),
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"status":"completed", "output":str(args.output),
                      "stateless":learned_metrics,
                      "counts":{k:{p:v["accepted_rows"] for p,v in row["policies"].items()}
                                for k,row in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
