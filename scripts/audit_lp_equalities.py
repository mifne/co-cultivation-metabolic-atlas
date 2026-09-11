"""Read-only structure audit for exact equality-preserving LP reduction.

The audit inspects original LP coefficients and configured bounds, never learns
constraints from reference flux values.  All reference solutions are checksum-
validated by the trace reader but are not used to define an elimination.  No
model, training set, solver, or neural architecture is modified here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lp_trace import load_trace_lp


STAGES = ("maxmin", "aggregate", "exchange")
SCOPE = ("Read-only mathematical structure audit. Training and diagnostic roles "
         "remain separate. Only original matrix/RHS/bound data determine structural "
         "statistics; reference x/y do not define constraints. No fitting, LP solving, "
         "GPU execution, biological change, or online speed claim.")
REFERENCES = [
    dict(title="DC3: A learning method for optimization with hard constraints",
         url="https://arxiv.org/html/2104.12225v1", year=2021,
         relevant_sections=["3.1 Equality completion", "3.2 Inequality correction"],
         interpretation=("Predict independent coordinates, complete dependent variables using "
                         "equalities, then correct inequalities along the equality manifold. "
                         "Completion requires an appropriate nonsingular dependent block; "
                         "the paper is not evidence that this GEM LP meets our KKT or speed targets.")),
    dict(title="Physics-Informed Neural Networks with Hard Linear Equality Constraints",
         url="https://arxiv.org/abs/2402.07251", year=2024,
         interpretation=("KKT-derived projection layers enforce linear equalities. This motivates "
                         "constraint-preserving output maps, not full LP optimality or a "
                         "guaranteed performance advantage for the present sparse problem.")),
    dict(title="Homogeneous Linear Inequality Constraints for Neural Network Activations",
         url="https://openaccess.thecvf.com/content_CVPRW_2020/html/w45/Frerix_Homogeneous_Linear_Inequality_Constraints_for_Neural_Network_Activations_CVPRW_2020_paper.html",
         year=2020,
         interpretation=("Feasible-set parameterization moves work to initialization. Its "
                         "homogeneous inequality setting differs from changing GEM bounds; "
                         "do not extrapolate its reported speed to this application.")),
]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equality_hash(problem, *, trim_trailing_zero_columns=False):
    """Hash exact equality values/indices/RHS; optionally ignore unused suffixes."""
    a, rhs, _, _, _, neq = problem
    equality = a[:neq].tocsr(copy=True)
    equality.sum_duplicates()
    equality.eliminate_zeros()
    equality.sort_indices()
    width = int(equality.indices.max()+1) if equality.nnz else 0
    shape = (neq, width if trim_trailing_zero_columns else a.shape[1])
    digest = hashlib.sha256()
    for name, values, dtype in (("shape", shape, "<i8"),
                                ("indptr", equality.indptr, "<i8"),
                                ("indices", equality.indices, "<i8"),
                                ("data", equality.data, "<f8"),
                                ("rhs", rhs[:neq], "<f8")):
        values = np.asarray(values, dtype=dtype)
        digest.update(name.encode())
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def coefficient_statistics(matrix):
    absolute = np.abs(matrix.data[matrix.data != 0])
    minimum = float(absolute.min()) if len(absolute) else None
    maximum = float(absolute.max()) if len(absolute) else None
    return dict(nonzeros=int(len(absolute)), abs_min_nonzero=minimum, abs_max=maximum,
                abs_max_over_min=maximum/minimum if minimum else None,
                dynamic_range_warning="Coefficient dynamic range is NOT a matrix condition number",
                abs_quantiles={str(q):float(np.quantile(absolute, q)) for q in (0, .5, .9, .99, 1)} if len(absolute) else {},
                numerical_condition_number=None,
                condition_note="Not computed: sparse equality rows can be dependent; no dense SVD or inverse was formed")


def low_arity_graph(problem):
    """A structural elimination lower bound, not a reduction implementation.

    A spanning tree of homogeneous two-entry rows determines all component
    variables from at most one coordinate. A homogeneous singleton or a zero
    bound fixes its whole component to zero. Extra cycle equations and rows
    with >=3 entries may remove further freedom, so the reported count is only
    a conservative lower bound. No division, nullspace, pivot or solution map
    is constructed and no graph component is discarded from the real LP.
    """
    a, rhs, _, _, _, neq = problem
    n = a.shape[1]
    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)
    singleton_columns, forest_edges, cycle_edges = set(), 0, 0
    def find(column):
        while parent[column] != column:
            parent[column] = parent[parent[column]]
            column = parent[column]
        return int(column)
    for row in range(neq):
        if rhs[row] != 0:
            continue
        start, end = a.indptr[row:row+2]
        columns = a.indices[start:end]
        if len(columns) == 1:
            singleton_columns.add(int(columns[0]))
        elif len(columns) == 2:
            first, second = map(find, columns)
            if first == second:
                cycle_edges += 1
            else:
                if size[first] < size[second]:
                    first, second = second, first
                parent[second] = first
                size[first] += size[second]
                forest_edges += 1
    roots = np.array([find(column) for column in range(n)], dtype=np.int64)
    base_anchors = set(roots[list(singleton_columns)])
    return dict(roots=roots, base_anchors=base_anchors,
                singleton_columns=sorted(singleton_columns), forest_edges=forest_edges,
                cycle_edges=cycle_edges, component_count=len(set(roots)),
                largest_component_size=int(np.bincount(roots, minlength=n).max(initial=0)))


def graph_elimination_bound(graph, fixed_zero_columns):
    roots = graph["roots"]
    columns = np.asarray(fixed_zero_columns, dtype=np.int64)
    if columns.ndim != 1 or np.any(columns < 0) or np.any(columns >= len(roots)):
        raise ValueError("Invalid fixed-zero columns")
    anchors = graph["base_anchors"] | set(roots[columns])
    zero_mask = np.isin(roots, list(anchors))
    eliminated = graph["forest_edges"]+len(anchors)
    return dict(two_entry_forest_independent_relations=int(graph["forest_edges"]),
                two_entry_cycle_rows_not_counted_as_extra_independent=int(graph["cycle_edges"]),
                graph_components=int(graph["component_count"]),
                largest_component_columns=graph["largest_component_size"],
                zero_anchored_components=len(anchors),
                directly_singleton_forced_columns=len(graph["singleton_columns"]),
                zero_columns_implied_by_small_rows_and_supplied_zero_bounds=int(zero_mask.sum()),
                eliminated_coordinates_lower_bound=int(eliminated),
                remaining_coordinates_upper_bound=int(len(roots)-eliminated),
                warning=("Only homogeneous original 1/2-entry equalities plus supplied exact zero bounds. "
                         "Not numerical rank, a constructed reduced LP, or a prediction of runtime. "
                         "Cycles/higher-arity rows can impose additional restrictions."))


def equality_details(problem):
    a, rhs, _, _, _, neq = problem
    equality = a[:neq]
    counts = np.diff(equality.indptr)
    homogeneous = rhs[:neq] == 0
    risky = []
    for row in np.flatnonzero(homogeneous & (counts == 2)):
        start, end = equality.indptr[row:row+2]
        columns, values = equality.indices[start:end], equality.data[start:end]
        absolute = np.abs(values)
        ratio = float(absolute.max()/absolute.min())
        if ratio >= 1e3 or absolute.min() <= 1e-4:
            pivot = int(np.argmax(absolute))
            other = 1-pivot
            risky.append(dict(row_index=int(row), columns=columns.tolist(), coefficients=values.tolist(),
                              rhs=float(rhs[row]), abs_coefficient_ratio=ratio,
                              locally_nonamplifying_eliminated_column=int(columns[pivot]),
                              remaining_column=int(columns[other]),
                              local_substitution_factor=float(-values[other]/values[pivot]),
                              warning="Local candidate only; shared dependencies/cycles require globally consistent elimination"))
    return dict(rows=int(neq), columns=int(a.shape[1]),
                used_column_max=int(equality.indices.max()) if equality.nnz else None,
                unused_trailing_columns=int(a.shape[1]-(equality.indices.max()+1)) if equality.nnz else a.shape[1],
                homogeneous_rows=int(homogeneous.sum()), nonhomogeneous_rows=int((~homogeneous).sum()),
                nnz=int(equality.nnz), row_nnz_histogram={str(k):int(v) for k,v in sorted(Counter(counts).items())},
                homogeneous_zero_entry_rows=int(np.sum(homogeneous & (counts == 0))),
                homogeneous_single_entry_rows=int(np.sum(homogeneous & (counts == 1))),
                homogeneous_two_entry_rows=int(np.sum(homogeneous & (counts == 2))),
                singleton_columns=sorted(set(equality.indices[equality.indptr[row]]
                                           for row in np.flatnonzero(homogeneous & (counts == 1)))),
                coefficient_statistics=coefficient_statistics(equality),
                high_ratio_or_small_coefficient_two_entry_rows=risky)


def selected_entries(manifest, steps=None):
    """Require each selected environment/time to contain all three stages."""
    seeds = manifest["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Unique non-empty trace seeds required")
    if manifest.get("status") != "completed":
        raise ValueError("Completed trace required")
    completed = manifest["completed_steps"]
    if len(completed) != len(seeds) or min(completed) < 1 or len(set(completed)) != 1:
        raise ValueError("Complete equal-length trajectories required")
    steps = list(range(1, completed[0]+1)) if steps is None else list(steps)
    if not steps or len(set(steps)) != len(steps) or any(type(step) is not int or step < 1 or step > completed[0] for step in steps):
        raise ValueError("Invalid selected steps")
    by_key = {}
    for entry in manifest["entries"]:
        if entry["step"] not in steps:
            continue
        key = entry["environment_id"], entry["step"], entry["stage"]
        if key in by_key:
            raise ValueError("Duplicate selected LP")
        by_key[key] = entry
    expected = {(environment, step, stage) for environment in range(len(seeds))
                for step in steps for stage in STAGES}
    if set(by_key) != expected:
        raise ValueError("Missing or unexpected selected LP stage/environment")
    return [by_key[(environment, step, stage)] for step in steps
            for environment in range(len(seeds)) for stage in STAGES]


def audit_trace(directory, *, expected_role, steps=None):
    directory = Path(directory)
    manifest_path = directory/"manifest.json"
    original_manifest_hash = _sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("role") != expected_role:
        raise ValueError("Trace role differs from declared audit role")
    entries = selected_entries(manifest, steps)
    groups, bounds, records = {}, {}, []
    for index, entry in enumerate(entries):
        problem, _, _ = load_trace_lp(directory, entry)
        a, rhs, lower, upper, _, neq = problem
        full_hash = equality_hash(problem)
        trimmed_hash = equality_hash(problem, trim_trailing_zero_columns=True)
        fixed_zero = (lower == 0) & (upper == 0)
        fixed = np.isfinite(lower) & (lower == upper)
        key = (entry["stage"], full_hash)
        if full_hash not in groups:
            groups[full_hash] = dict(trimmed_hash=trimmed_hash, details=equality_details(problem),
                                     graph=low_arity_graph(problem), appearances=0, stages=set())
        group = groups[full_hash]
        group["appearances"] += 1
        group["stages"].add(entry["stage"])
        if key not in bounds:
            bounds[key] = dict(intersection=fixed_zero.copy(), union=fixed_zero.copy(), counts=[],
                               fixed_nonzero_counts=[], patterns=Counter(), representative_problem=problem,
                               matrix_nnz=[], shapes=set(), elimination_counts=[])
        bound = bounds[key]
        bound["intersection"] &= fixed_zero
        bound["union"] |= fixed_zero
        bound["counts"].append(int(fixed_zero.sum()))
        bound["fixed_nonzero_counts"].append(int(np.sum(fixed & ~fixed_zero)))
        bound["patterns"][hashlib.sha256(np.packbits(fixed_zero).tobytes()).hexdigest()] += 1
        bound["matrix_nnz"].append(int(a.nnz))
        bound["shapes"].add(a.shape)
        elimination = graph_elimination_bound(group["graph"], np.flatnonzero(fixed_zero))
        bound["elimination_counts"].append(elimination["eliminated_coordinates_lower_bound"])
        records.append(dict(environment_id=entry["environment_id"], seed=manifest["seeds"][entry["environment_id"]],
                            step=entry["step"], stage=entry["stage"], filename=entry["filename"],
                            payload_sha256=entry["sha256"], problem_sha256=entry["problem_sha256"],
                            equality_full_hash=full_hash, equality_trimmed_hash=trimmed_hash,
                            rows=a.shape[0], columns=a.shape[1], nnz=int(a.nnz), equalities=neq,
                            bound_fixed_zero_count=int(fixed_zero.sum()),
                            bound_fixed_nonzero_count=int(np.sum(fixed & ~fixed_zero)),
                            small_row_eliminated_coordinates_lower_bound=elimination["eliminated_coordinates_lower_bound"]))
        if (index+1) % 240 == 0:
            print(f"{directory.name}: verified {index+1}/{len(entries)} LPs", flush=True)
    if original_manifest_hash != _sha(manifest_path):
        raise ValueError("Trace manifest changed during audit")
    equality_groups = {}
    for fingerprint, group in groups.items():
        graph = group["graph"]
        equality_groups[fingerprint] = dict(trimmed_hash=group["trimmed_hash"], appearances=group["appearances"],
            stages=sorted(group["stages"]), details=group["details"],
            equality_only_small_row_elimination=graph_elimination_bound(graph, []))
    stage_bounds = []
    for (stage, fingerprint), bound in bounds.items():
        graph = groups[fingerprint]["graph"]
        intersection = np.flatnonzero(bound["intersection"]).tolist()
        union = np.flatnonzero(bound["union"]).tolist()
        stage_bounds.append(dict(stage=stage, equality_full_hash=fingerprint,
            LP_count=len(bound["counts"]), full_matrix_shapes=[list(shape) for shape in sorted(bound["shapes"])],
            full_matrix_nnz_range=[min(bound["matrix_nnz"]), max(bound["matrix_nnz"])],
            representative_full_matrix_coefficients=coefficient_statistics(bound["representative_problem"][0]),
            bound_fixed_zero_count_range=[min(bound["counts"]), max(bound["counts"])],
            bound_fixed_nonzero_count_range=[min(bound["fixed_nonzero_counts"]), max(bound["fixed_nonzero_counts"])],
            bound_fixed_zero_pattern_count=len(bound["patterns"]),
            bound_fixed_zero_intersection_columns=intersection, bound_fixed_zero_union_columns=union,
            bounds_intersection_scope="Observed original configured bounds in THIS trace selection only; future LPs must revalidate bounds",
            small_row_elimination_with_zero_intersection=graph_elimination_bound(graph, intersection),
            per_LP_small_row_eliminated_coordinate_lower_bound_range=[min(bound["elimination_counts"]), max(bound["elimination_counts"])]))
    return dict(directory=str(directory), role=manifest["role"], manifest_sha256=original_manifest_hash,
                model_fingerprints=manifest["model_fingerprints"], seeds=manifest["seeds"],
                selected_steps=sorted(set(e["step"] for e in entries)), LP_count=len(entries),
                all_payloads_and_CPU_reference_certificates_validated=True,
                equality_full_hash_count=len(groups),
                equality_trimmed_hash_count=len(set(group["trimmed_hash"] for group in groups.values())),
                equality_groups=equality_groups, stage_bounds=stage_bounds, records=records)


def run_audit(training, diagnostic, diagnostic_steps):
    started = time.perf_counter()
    source_paths = ("scripts/audit_lp_equalities.py", "src/lp_trace.py", "src/cpu_repeated_lp.py")
    source_hashes = {name:_sha(ROOT/name) for name in source_paths}
    train = audit_trace(training, expected_role="training_reference")
    evaluation = audit_trace(diagnostic, expected_role="development_diagnostic_not_training", steps=diagnostic_steps)
    if train["model_fingerprints"] != evaluation["model_fingerprints"]:
        raise ValueError("Model identities differ across audited traces")
    if set(train["seeds"]) & set(evaluation["seeds"]):
        raise ValueError("Training and diagnostic seeds overlap")
    if source_hashes != {name:_sha(ROOT/name) for name in source_paths}:
        raise ValueError("Imported audit sources changed during execution")
    hashes = sorted({group["trimmed_hash"] for trace in (train, evaluation)
                     for group in trace["equality_groups"].values()})
    return dict(status="completed", scope=SCOPE, source_hashes=source_hashes,
                audit_seconds_including_trace_IO_and_reference_checks=time.perf_counter()-started,
                training=train, diagnostic=evaluation,
                cross_trace=dict(model_fingerprints_equal=True, seeds_disjoint=True,
                    equality_trimmed_hashes=hashes, equality_trimmed_hash_count=len(hashes),
                    equality_coefficients_and_RHS_identical_ignoring_unused_suffix_columns=len(hashes)==1,
                    interpretation="Exact byte hashes of canonical CSR values/column identities/RHS; no row/column permutation or tolerance used"),
                implementation_implications=[
                    "Construct a sparse exact completion only from the unchanged equality matrix, not PCA of reference solutions.",
                    "Use large-magnitude local pivots and inspect graph dependencies/cycles to avoid inverse amplification.",
                    "The small-row graph estimate is a conservative structural opportunity, not a validated reduced solver or speedup.",
                    "Bound constraints, varying inequalities/objectives and original primal/dual/KKT certification remain mandatory after equality reduction.",
                    "Observed equality invariance supports setup reuse; every future model/column/RHS mismatch must invalidate that setup.",
                    "A learned proposal in reduced coordinates must train/evaluate through the same completion and correction used online."],
                primary_references=REFERENCES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("results/pf_gru_training16x60_20260905"))
    parser.add_argument("--diagnostic", type=Path, default=Path("results/pf_lp_trace_dev4x60_20260905"))
    parser.add_argument("--diagnostic-steps", type=int, nargs="+", default=[1, 2, 41])
    parser.add_argument("--output", type=Path, default=Path("results/pf_lp_equality_audit_20260905.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run_audit(args.training, args.diagnostic, args.diagnostic_steps)
    contents = json.dumps(report, indent=2, allow_nan=False,
                          default=lambda value: int(value) if isinstance(value, np.integer) else value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(contents)
    print(json.dumps(report["cross_trace"], indent=2))
    for trace in (report["training"], report["diagnostic"]):
        for stage in trace["stage_bounds"]:
            estimate = stage["small_row_elimination_with_zero_intersection"]
            print(f'{trace["role"]} {stage["stage"]}: zero bounds {stage["bound_fixed_zero_count_range"]}, '
                  f'eliminate >= {estimate["eliminated_coordinates_lower_bound"]}, '
                  f'remain <= {estimate["remaining_coordinates_upper_bound"]}', flush=True)


if __name__ == "__main__":
    main()
