"""Oracle compression diagnostics, never an online predictor or speed benchmark.

The *current CPU reference solution* is intentionally encoded and reconstructed
to isolate error caused by the fixed low-rank codec from neural prediction and
iterative correction.  It is the best Euclidean projection in the codec's
scaled target metric when its rows are orthonormal, not a lower bound on LP
certificate error and not evidence of causal online accuracy.  No model is fit
on this evaluation subset and no LP solver or GPU corrector is run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cpu_repeated_lp import _certificate
from src.lp_trace import load_trace_lp
from src.temporal_lp_data import TemporalLPCodec


SCOPE = ("Oracle CPU-reference encode/decode diagnostic only. Current reference x/y "
         "are used intentionally to isolate compression error, never as online "
         "warm starts. No LP solve, neural inference, GPU correction, model fit, "
         "speed comparison, or closed-loop dFBA evaluation is performed.")


def original_certificate(problem, x, y):
    a, _, _, _, c, _ = problem
    solution = SimpleNamespace(col_value=x, row_dual=y,
                               col_dual=c-a.T@y, value_valid=True, dual_valid=True)
    return _certificate(*problem, solution)


def error_metrics(reference, reconstructed, scale):
    reference = np.asarray(reference, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if reference.ndim != 1 or reconstructed.shape != reference.shape or scale.shape != reference.shape:
        raise ValueError("Error metric vectors must have matching one-dimensional shapes")
    if not (np.isfinite(reference).all() and np.isfinite(reconstructed).all()
            and np.isfinite(scale).all() and np.all(scale > 0)):
        raise ValueError("Finite references/reconstruction and positive scales required")
    error = reconstructed-reference
    absolute = np.abs(error)
    return dict(dimensions=len(reference), reference_l2=float(np.linalg.norm(reference)),
                reference_abs_max=float(np.max(np.abs(reference), initial=0)),
                reconstruction_abs_max=float(np.max(np.abs(reconstructed), initial=0)),
                error_abs_max=float(np.max(absolute, initial=0)),
                error_rmse=float(np.sqrt(np.mean(error**2))),
                error_l2=float(np.linalg.norm(error)),
                error_l2_over_max_1_reference_l2=float(np.linalg.norm(error)/max(1., np.linalg.norm(reference))),
                scaled_error_rmse=float(np.sqrt(np.mean((error/scale)**2))),
                scaled_error_abs_max=float(np.max(np.abs(error/scale), initial=0)),
                coordinate_error_above_1e_5=int(np.count_nonzero(absolute > 1e-5)),
                coordinate_error_above_1e_7=int(np.count_nonzero(absolute > 1e-7)))


def dual_extremes(problem, y, reconstructed_y, *, limit=8):
    a, rhs, _, _, c, neq = problem
    order = np.argsort(-np.abs(y), kind="stable")[:limit]
    rows = []
    for index in order:
        values = a.data[a.indptr[index]:a.indptr[index+1]]
        nonzero = np.abs(values[values != 0])
        rows.append(dict(row_index=int(index), row_kind="equality" if index < neq else "inequality",
                         reference_y=float(y[index]), reconstructed_y=float(reconstructed_y[index]),
                         reconstruction_error=float(reconstructed_y[index]-y[index]),
                         rhs=float(rhs[index]), row_nonzeros=int(len(nonzero)),
                         row_coefficient_abs_max=float(np.max(nonzero, initial=0)),
                         row_coefficient_abs_min_nonzero=float(nonzero.min()) if len(nonzero) else None))
    return dict(reference_abs_quantiles={str(q):float(np.quantile(np.abs(y), q))
                                         for q in (0, .5, .9, .99, 1)},
                reference_abs_above={str(threshold):int(np.count_nonzero(np.abs(y) > threshold))
                                     for threshold in (1e3, 1e6, 1e9, 1e12)},
                objective_coefficient_abs_max=float(np.max(np.abs(c), initial=0)),
                reference_reduced_cost_abs_max=float(np.max(np.abs(c-a.T@y), initial=0)),
                reconstruction_reduced_cost_abs_max=float(np.max(np.abs(c-a.T@reconstructed_y), initial=0)),
                dual_activity_error_abs_max=float(np.max(np.abs(a.T@(reconstructed_y-y)), initial=0)),
                top_rows=rows)


def subset_spectrum(matrix):
    """A small evaluation-matrix spectrum, NOT the original training spectrum."""
    values = np.asarray(matrix, dtype=np.float64)
    singular = np.linalg.svd(values, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    return dict(shape=list(values.shape), singular_values=singular.tolist(),
                squared_singular_energy=total,
                cumulative_energy_fraction=(np.cumsum(energy)/total).tolist() if total else [0.]*len(energy))


def diagnose_samples(codec, samples):
    """Compare unchanged original-LP certificates for labeled oracle samples.

    Each sample is ``(labels, (problem, reference_x, reference_y))``.  This uses
    the saved affine PCA subspace directly, independent of a later invertible
    latent normalization added for neural optimization.  No references are
    returned to any solver and neither the codec nor inputs are modified.
    """
    samples = list(samples)
    if not samples:
        raise ValueError("At least one evaluation sample is required")
    n, m, neq = codec.metadata["dimensions"]
    arrays = codec.arrays
    components = arrays["components"]
    identity_error = float(np.max(np.abs(components@components.T-np.eye(codec.rank)), initial=0))
    if identity_error > 1e-8:
        raise ValueError("Saved PCA components are not orthonormal; projection interpretation is invalid")
    rows, normalized, reconstructions, errors = [], [], [], []
    for labels, (problem, x, y) in samples:
        if problem[0].shape != (m, n) or problem[-1] != neq:
            raise ValueError("Evaluation LP shape/phase does not match codec")
        reference_certificate = original_certificate(problem, x, y)
        if not reference_certificate["certificate_passed"]:
            raise ValueError("CPU reference fails the original-LP certificate")
        target = np.concatenate((x, y))
        standardized = (target-arrays["target_mean"])/arrays["target_scale"]
        # These are oracle coordinates; never use them in an online benchmark.
        coordinates = standardized@components.T
        projected = coordinates@components
        reconstructed = projected*arrays["target_scale"]+arrays["target_mean"]
        projected_x, projected_y = reconstructed[:n], reconstructed[n:]
        if not np.isfinite(reconstructed).all():
            raise ValueError("Oracle reconstruction became non-finite")
        row = dict(labels=dict(labels), reference_certificate=reference_certificate,
                   reconstruction_certificate=original_certificate(problem, projected_x, projected_y),
                   reconstructed_x_reference_y_certificate=original_certificate(problem, projected_x, y),
                   reference_x_reconstructed_y_certificate=original_certificate(problem, x, projected_y),
                   primal_error=error_metrics(x, projected_x, arrays["target_scale"][:n]),
                   dual_error=error_metrics(y, projected_y, arrays["target_scale"][n:]),
                   reference_objective=float(problem[4]@x),
                   reconstruction_objective=float(problem[4]@projected_x),
                   objective_abs_error=float(abs(problem[4]@(projected_x-x))),
                   primal_activity_error_abs_max=float(np.max(np.abs(problem[0]@(projected_x-x)), initial=0)),
                   oracle_latent_abs_max=float(np.max(np.abs(coordinates), initial=0)),
                   dual_extremes=dual_extremes(problem, y, projected_y))
        rows.append(row)
        normalized.append(standardized)
        reconstructions.append(projected)
        errors.append(projected-standardized)
    normalized, reconstructions, errors = map(np.stack, (normalized, reconstructions, errors))
    reference_energy = float(np.sum(normalized**2))
    discarded_energy = float(np.sum(errors**2))
    return dict(scope=SCOPE, rank=codec.rank, dimensions=dict(primal=n, dual=m, equality_rows=neq),
                sample_count=len(rows), certificate_thresholds=dict(primal=1e-5, dual=1e-7, relative_kkt_gap=1e-7),
                rows=rows, summary=dict(
                    reference_accepted=sum(r["reference_certificate"]["certificate_passed"] for r in rows),
                    reconstruction_accepted=sum(r["reconstruction_certificate"]["certificate_passed"] for r in rows),
                    reconstructed_x_reference_y_accepted=sum(r["reconstructed_x_reference_y_certificate"]["certificate_passed"] for r in rows),
                    reference_x_reconstructed_y_accepted=sum(r["reference_x_reconstructed_y_certificate"]["certificate_passed"] for r in rows),
                    max_reconstruction_primal_residual=max(r["reconstruction_certificate"]["primal_residual"] for r in rows),
                    max_reconstruction_dual_violation=max(r["reconstruction_certificate"]["dual_violation"] for r in rows),
                    max_reconstruction_relative_kkt_gap=max(r["reconstruction_certificate"]["relative_kkt_gap"] for r in rows),
                    max_primal_coordinate_error=max(r["primal_error"]["error_abs_max"] for r in rows),
                    max_dual_coordinate_error=max(r["dual_error"]["error_abs_max"] for r in rows),
                    max_reference_dual_abs=max(r["dual_error"]["reference_abs_max"] for r in rows)),
                compression=dict(
                    projection_metric="Euclidean distance of (reference - training mean) / training target scale",
                    projection_warning="Metric-optimal oracle projection is not an optimum of LP residual, KKT gap or endpoint error",
                    components_orthonormality_max_abs_error=identity_error,
                    training_singular_values=None,
                    training_singular_values_note="Not saved in this codec; no training refit performed",
                    evaluation_spectrum_scope="Only selected evaluation rows, centered by saved TRAIN mean, not re-centered on evaluation",
                    reference_subset_spectrum=subset_spectrum(normalized),
                    reconstruction_subset_spectrum=subset_spectrum(reconstructions),
                    residual_subset_spectrum=subset_spectrum(errors),
                    reference_primal_subset_spectrum=subset_spectrum(normalized[:, :n]),
                    reference_dual_subset_spectrum=subset_spectrum(normalized[:, n:]),
                    reconstruction_primal_subset_spectrum=subset_spectrum(reconstructions[:, :n]),
                    reconstruction_dual_subset_spectrum=subset_spectrum(reconstructions[:, n:]),
                    residual_primal_subset_spectrum=subset_spectrum(errors[:, :n]),
                    residual_dual_subset_spectrum=subset_spectrum(errors[:, n:]),
                    scaled_reference_squared_energy=reference_energy,
                    scaled_discarded_squared_energy=discarded_energy,
                    scaled_discarded_energy_fraction=discarded_energy/reference_energy if reference_energy else None,
                    scaled_primal_error_rmse=float(np.sqrt(np.mean(errors[:, :n]**2))),
                    scaled_dual_error_rmse=float(np.sqrt(np.mean(errors[:, n:]**2)))))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_diagnostic(artifact, trace, steps, environments):
    artifact, trace = Path(artifact), Path(trace)
    if not steps or any(type(step) is not int or step < 1 for step in steps) or len(set(steps)) != len(steps):
        raise ValueError("Positive unique steps are required")
    if type(environments) is not int or environments < 1:
        raise ValueError("Positive environment count required")
    training = json.loads((artifact/"manifest.json").read_text())
    trace_manifest = json.loads((trace/"manifest.json").read_text())
    if training.get("status") != "completed" or trace_manifest.get("status") != "completed":
        raise ValueError("Completed artifact and trace required")
    codec_path = artifact/"codec.npz"
    if training.get("codec_sha256") != _sha(codec_path):
        raise ValueError("Codec checksum does not match artifact manifest")
    codec = TemporalLPCodec.load(codec_path)
    if trace_manifest["model_fingerprints"] != codec.metadata["model_fingerprints"]:
        raise ValueError("Evaluation model provenance does not match codec")
    seeds = trace_manifest["seeds"]
    if environments > len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Insufficient or duplicate evaluation environments")
    used_seeds = set(codec.metadata["train_seeds"]) | set(codec.metadata["development_seeds"])
    if used_seeds & set(seeds[:environments]):
        raise ValueError("Evaluation seeds overlap codec training/development")
    stage = codec.metadata["stage"]
    entries = {}
    for entry in trace_manifest["entries"]:
        if entry["stage"] == stage and entry["environment_id"] < environments and entry["step"] in steps:
            key = (entry["step"], entry["environment_id"])
            if key in entries:
                raise ValueError("Duplicate requested evaluation LP")
            entries[key] = entry
    if set(entries) != {(step, environment) for step in steps for environment in range(environments)}:
        raise ValueError("Incomplete requested evaluation LP cohort")
    samples = []
    for step in steps:
        for environment in range(environments):
            entry = entries[(step, environment)]
            labels = dict(step=step, environment_id=environment, seed=seeds[environment],
                          stage=stage, problem_sha256=entry["problem_sha256"], filename=entry["filename"])
            samples.append((labels, load_trace_lp(trace, entry)))
    report = diagnose_samples(codec, samples)
    report.update(status="completed", configuration=dict(artifact=str(artifact), trace=str(trace),
                   steps=steps, environments=environments),
                  artifact_manifest_sha256=_sha(artifact/"manifest.json"), codec_sha256=_sha(codec_path),
                  trace_manifest_sha256=_sha(trace/"manifest.json"), trace_role=trace_manifest.get("role"),
                  model_fingerprints=trace_manifest["model_fingerprints"],
                  train_seeds=codec.metadata["train_seeds"],
                  development_seeds=codec.metadata["development_seeds"],
                  evaluation_seeds=seeds[:environments],
                  source_hashes={name:_sha(ROOT/name) for name in (
                      "scripts/diagnose_temporal_compression.py", "src/temporal_lp_data.py",
                      "src/lp_trace.py", "src/cpu_repeated_lp.py")})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 41])
    parser.add_argument("--environments", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run_diagnostic(args.artifact, args.trace, args.steps, args.environments)
    # Deliberately reject non-finite JSON rather than silently making a failed
    # certificate look like a finite error; these real references are finite.
    contents = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(contents)
    print(json.dumps(report["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
