"""Training-only exact CPU basis collection and one-at-a-time compact projection.

Cached normalized LPs are restored to original coordinates by the validated
CompactTrainingSource. Each seed owns one persistent RepeatedCpuLP environment;
steps are processed in order. Only a newly solved ORIGINAL-LP certificate and
complete HiGHS statuses create an anchor. No inverse is constructed here.

Old compact entries lack full row statuses: they are mandatory, byte-identical
inputs and are never reverse-engineered or deduplicated against new statuses.
CPU anchor certification does not certify its compact projection. The caller
must measure GPU self-row and full-training coverage before selecting entries.
"""

from __future__ import annotations

import copy
import hashlib
from importlib.metadata import version as package_version
from pathlib import Path
import shutil
import time

import numpy as np

from .compact_training_data import CompactTrainingSource, checked_child, sha256_file, validate_compact_stage
from .cpu_repeated_lp import RepeatedCpuLP, _certificate
from .gpu_compact_basis import project_basis


class BasisRebuildError(RuntimeError):
    """Failure carrying observed offline work accounting for caller persistence."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _json_safe(value):
    """Unavailable nonfinite diagnostic values are null, never fabricated zero."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def basis_status_signature(col_status, row_status):
    """Hash complete new HiGHS statuses; preserve equality lower/upper signs."""
    digest = hashlib.sha256()
    for name, supplied in (("col_status", col_status), ("row_status", row_status)):
        value = np.asarray(supplied)
        if value.ndim != 1 or value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value > 4):
            raise ValueError("Complete integer HiGHS status vectors are required")
        value = np.ascontiguousarray(value, dtype=np.int8)
        digest.update(name.encode())
        digest.update(np.asarray([len(value)], dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def original_lp_request(problem, *, stage='maxmin'):
    """Create the selected stage's unmodified original-unit CPU request."""
    stage = validate_compact_stage(stage)
    a, rhs, lower, upper, c, neq = problem
    return c.copy(), dict(A_eq=a[:neq].copy(), b_eq=rhs[:neq].copy(),
        A_ub=a[neq:].copy(), b_ub=rhs[neq:].copy(), bounds=np.column_stack((lower, upper)),
        method="highs-ds", _stage=stage)


def anchor_from_highs(normalized, original_problem, solution, basis, diagnostics):
    """Capture a factorization-free anchor from a completed, certified solve."""
    import highspy as hp
    certificate = _certificate(*original_problem, solution)
    if diagnostics.get("success") is not True or diagnostics.get("certificate_passed") is not True or not certificate["certificate_passed"]:
        raise ValueError("Fresh original-LP solution failed certification")
    m, n = normalized.a.shape
    if not basis.valid:
        raise ValueError("HiGHS returned an invalid basis")
    columns = np.asarray([int(value) for value in basis.col_status], dtype=np.int8)
    rows = np.asarray([int(value) for value in basis.row_status], dtype=np.int8)
    if columns.shape != (n,) or rows.shape != (m,):
        raise ValueError("Incomplete HiGHS basis status dimensions")
    basic_code = int(hp.HighsBasisStatus.kBasic)
    lower_code, upper_code, zero_code = (int(hp.HighsBasisStatus.kLower),
        int(hp.HighsBasisStatus.kUpper), int(hp.HighsBasisStatus.kZero))
    allowed = [basic_code, lower_code, upper_code, zero_code]
    if not np.isin(columns, allowed).all() or not np.isin(rows, allowed).all():
        raise ValueError("Generic/nonbasic HiGHS statuses cannot define exact endpoints")
    basic = np.flatnonzero(columns == basic_code).astype(np.int64)
    active = np.flatnonzero(rows != basic_code).astype(np.int64)
    if not len(basic) or len(basic) != len(active):
        raise ValueError("Unsupported empty or nonsquare active basis")
    # Compact maps place every active row at RHS. An inequality has only an
    # upper row bound, and a zero-status row can do this only when RHS is zero.
    if np.any(rows[normalized.neq:] == lower_code):
        raise ValueError("An inequality row cannot be active at its absent lower bound")
    if np.any((rows == zero_code) & (normalized.rhs != 0.)):
        raise ValueError("A zero-status active row has nonzero RHS")
    if np.any((columns == lower_code) & ~np.isfinite(normalized.lower)) or np.any((columns == upper_code) & ~np.isfinite(normalized.upper)):
        raise ValueError("A nonbasic variable selected an infinite endpoint")
    kind = np.zeros(n, dtype=np.int8)
    row_kind = np.zeros(m, dtype=np.int8)
    for status, tag in ((lower_code, -1), (upper_code, 1), (basic_code, 2)):
        kind[columns == status] = tag
        row_kind[rows == status] = tag
    x = np.asarray(solution.col_value, dtype=np.float64).copy()
    return dict(lp=normalized, basic=basic, active=active, kind=kind, row_kind=row_kind,
        col_status=columns.copy(), row_status=rows.copy(),
        cpu_anchor_values=x, cpu_anchor_objective=float(original_problem[4] @ x),
        compilation_seconds=float(diagnostics.get("total_seconds", 0.)), offline_cpu_lp_calls=1,
        original_certificate=certificate,
        basis_status_sha256=basis_status_signature(columns, rows))


def _validate_source(source):
    if not isinstance(source, CompactTrainingSource):
        raise TypeError("Use a validated CompactTrainingSource")
    if source.provenance.get("role") != "training_only_dictionary_reconstruction":
        raise ValueError("Only an explicitly training-only source is allowed")
    stage = source.stage
    m, n = source.root['a'].shape
    key = [stage, m, n, source.root['neq']]
    if (source.provenance.get('stage',stage) != stage
            or source.stage_manifest.get('key',key) != key
            or source.provenance.get('stage_key',key) != key):
        raise ValueError('Selected stage, root and provenance must agree')
    if (not source.seeds or len(set(source.seeds)) != len(source.seeds)
            or isinstance(source.steps, bool) or not isinstance(source.steps, int) or source.steps < 1
            or source.size != len(source.seeds)*source.steps):
        raise ValueError("Source must have complete trajectory-major step ordering")


def collect_basis_candidates(source, *, workers=4, progress=None):
    """Re-solve all cached training rows, deduplicating ONLY new full statuses.

    Returns ``(candidates, report)``. Each candidate contains a factorization-
    free ``anchor``, its first source index, and every same-status occurrence.
    Full original solutions come solely from the newly solved CPU LPs; no GPU
    answer, validation rollout solution, or older cached primal is consulted.
    """
    _validate_source(source)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    started = time.perf_counter()
    report = dict(status="collecting", role="offline_training_basis_rebuild",
        stage=source.stage, source_provenance=copy.deepcopy(source.provenance), workers=workers,
        seed_environment_ids=list(source.seeds), steps=source.steps, expected_rows=source.size,
        mandatory_base_entries=copy.deepcopy(source.stage_manifest.get("entries", [])),
        mandatory_base_count=len(source.stage_manifest.get("entries", [])),
        mandatory_policy="Copy old compact entries byte-identically; no old status reconstruction/deduplication",
        submitted_cpu_requests=0, offline_cpu_lp_calls=0, offline_cpu_solver_runs=0,
        numerical_retry_count=0, unique_new_statuses=0, duplicate_new_statuses=0,
        rows=[], cpu_batch_seconds=0., reconstruction_seconds=0., anchor_extraction_seconds=0.,
        full_inverse_count=0, projected_candidates=0,
        required_next_gate="GPU compact self-row AND full-training original-LP coverage before selection",
        gpu_projection_certified=False)
    candidates, by_signature = [], {}
    cpu = RepeatedCpuLP(workers=workers, reuse_basis=True)
    report.update(highspy_version=package_version("highspy"),
        solver_options=cpu._options({}), presolve="HiGHS default, recorded in each solver attempt",
        retry_policy=dict(max_numerical_retries=cpu.max_numerical_retries,
            ordered_attempts=["strict_basis_refactor", "strict_cold_no_presolve"],
            mathematical_lp_unchanged=True),
        diagnostic_nonfinite_policy="Non-finite or unavailable diagnostics become JSON null, never zero")
    try:
        for step0 in range(source.steps):
            before = time.perf_counter()
            indices = [trajectory*source.steps+step0 for trajectory in range(len(source.seeds))]
            normalized = [source.normalized_problem(index) for index in indices]
            original = [source.original_problem(index) for index in indices]
            requests = [original_lp_request(problem, stage=source.stage) for problem in original]
            report["reconstruction_seconds"] += time.perf_counter()-before
            report["submitted_cpu_requests"] += len(requests)
            before = time.perf_counter()
            results = cpu.solve_batch(requests, environment_ids=list(source.seeds))
            report["cpu_batch_seconds"] += time.perf_counter()-before
            report["offline_cpu_lp_calls"] += sum(int(result.diagnostics["cpu_lp_calls"]) for result in results)
            report["offline_cpu_solver_runs"] += sum(int(result.diagnostics["cpu_solver_runs"]) for result in results)
            report["numerical_retry_count"] += sum(int(result.diagnostics["numerical_retry_count"]) for result in results)
            if "effective_initial_options" not in report and results:
                report["effective_initial_options"] = copy.deepcopy(results[0].diagnostics["solver_attempts"][0]["effective_options"])
            # solve_batch joined every worker. Read live statuses/solutions now,
            # before any seed advances and mutates its persistent HiGHS model.
            for seed, index, q, p, result in zip(source.seeds, indices, normalized, original, results):
                row = dict(source_index=index, seed=seed, step=step0+1, diagnostics=copy.deepcopy(result.diagnostics))
                report["rows"].append(row)
                if not result.success:
                    raise ValueError(f"Uncertified original training LP at seed={seed}, step={step0+1}")
                before = time.perf_counter()
                m, n = p[0].shape
                solver = cpu.models[(seed, source.stage, m, n, p[-1])]["solver"]
                solution, basis = solver.getSolution(), solver.getBasis()
                if not np.array_equal(np.asarray(solution.col_value), result.x):
                    raise ValueError("Live HiGHS solution changed after the joined CPU batch")
                anchor = anchor_from_highs(q, p, solution, basis, result.diagnostics)
                signature = anchor["basis_status_sha256"]
                occurrence = dict(source_index=index, seed=seed, step=step0+1)
                if signature in by_signature:
                    candidate_index = by_signature[signature]
                    candidates[candidate_index]["occurrences"].append(occurrence)
                    candidates[candidate_index]["source_indices"].append(index)
                    report["duplicate_new_statuses"] += 1
                else:
                    candidate_index = len(candidates)
                    by_signature[signature] = candidate_index
                    candidates.append(dict(anchor=anchor, signature=signature, **occurrence,
                                           source_indices=[index], occurrences=[occurrence], gpu_projection_certified=False))
                row.update(candidate_index=candidate_index, basis_status_sha256=signature,
                           original_certificate=anchor["original_certificate"])
                report["unique_new_statuses"] = len(candidates)
                report["anchor_extraction_seconds"] += time.perf_counter()-before
            if progress is not None:
                progress(dict(phase="cpu_basis_collection", step=step0+1,
                    completed_rows=len(report["rows"]), unique_candidates=len(candidates),
                    seconds=time.perf_counter()-started))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      accounting_complete=False, total_seconds=time.perf_counter()-started)
        raise BasisRebuildError("Offline basis collection failed; observed cost is attached in .report", _json_safe(report)) from error
    finally:
        cpu.close()
    report.update(status="completed", accounting_complete=True, total_seconds=time.perf_counter()-started)
    return candidates, _json_safe(report)


def project_basis_candidate(candidate, source, *, extra_costs=()):
    """Project one deduplicated candidate; never form a complete basis inverse."""
    _validate_source(source)
    started = time.perf_counter()
    anchor = candidate["anchor"]
    if "inverse" in anchor:
        raise ValueError("Rebuild candidates must not contain a full inverse")
    if not anchor.get("original_certificate", {}).get("certificate_passed", False):
        raise ValueError("Only newly certified original-LP anchors can be projected")
    signature = basis_status_signature(anchor["col_status"], anchor["row_status"])
    if signature != candidate["signature"]:
        raise ValueError("Candidate basis statuses changed after collection")
    offset = (anchor["lp"].a-source.root["a"])[source.variable_rows].toarray()[None]
    arrays = project_basis(anchor, source.data, offset, source.variable_rows, extra_costs=extra_costs)
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite compact projection: {name}")
    record = dict(source="new_training_cpu_basis", stage=source.stage, source_index=int(candidate["source_index"]),
        seed=candidate["seed"], step=int(candidate["step"]),
        basis_status_sha256=signature, occurrences=copy.deepcopy(candidate["occurrences"]),
        projection_seconds=time.perf_counter()-started,
        projected_bytes=sum(value.nbytes for value in arrays.values()),
        full_inverse_constructed=False, gpu_projection_certified=False,
        required_next_gate="GPU compact self-row AND full-training original-LP coverage before selection")
    return arrays, record


def copy_mandatory_entries(source, base_stage_directory, output_stage_directory):
    """Copy old projected files exactly; do not infer missing HiGHS row status."""
    _validate_source(source)
    entries = source.stage_manifest.get("entries", [])
    if not entries:
        raise ValueError("Mandatory base entries are missing")
    source_dir, target_dir = Path(base_stage_directory).resolve(), Path(output_stage_directory).resolve()
    planned = []
    names = set()
    for entry in entries:
        name, digest = entry.get("filename"), entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Mandatory base entry requires its SHA256")
        path = checked_child(source_dir, name, digest)
        target = (target_dir/name).resolve()
        if not target.is_relative_to(target_dir) or target in names:
            raise ValueError("Mandatory destination path is unsafe or duplicated")
        if target.exists():
            raise FileExistsError(target)
        names.add(target)
        planned.append((path, target, entry))
    copied = []
    for path, target, entry in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha256_file(target) != entry["sha256"]:
            raise ValueError("Mandatory byte-identical copy verification failed")
        copied.append(dict(copy.deepcopy(entry), mandatory=True, byte_identical_base=True))
    return copied
