"""Strictly validate asymmetric CPU-pipeline/GPU-first batch reports.

This reader is intentionally specific to ``benchmark_compact_gpu.py`` runs
created with ``--gpu-first-batch``.  The CPU comparator and GPU trajectory use
different schedulers, so their internal timers must not be combined using the
older symmetric-pipeline accounting rules.  A summary is returned only after
the source snapshot, evaluation scope, original-LP certificates, counters,
timers, and endpoint gates have all been checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


STAGES = ("maxmin", "aggregate", "exchange")
CERTIFICATE_LIMITS = {
    "primal_residual": 1e-5,
    "dual_violation": 1e-7,
    "relative_kkt_gap": 1e-7,
}
ENDPOINT_LIMIT = 0.01
ENDPOINT_FIELDS = ("pha_relative", "biomass_g_l", "phv_fraction")
CPU_NONOVERLAP_FIELDS = (
    "maxmin_service_seconds",
    "main_thread_resume_seconds",
    "cpu_preparation_seconds",
    "cpu_wait_seconds",
    "scheduler_overhead_seconds",
)
GPU_FIRST_PHASE_FIELDS = (
    "parse_normalize_seconds",
    "stack_upload_seconds",
    "device_rank_seconds",
    "evaluation_submission_seconds",
    "completion_download_validation_seconds",
    "rejected_cpu_seconds",
)
REQUIRED_SOURCE_FILES = frozenset({
    "scripts/benchmark_compact_gpu.py",
    "scripts/pipelined_microbatch.py",
    "src/community_solver.py",
    "src/dfba_simulator.py",
    "src/cpu_dictionary_lp.py",
    "src/cpu_repeated_lp.py",
    "src/gpu_hybrid_lp.py",
    "src/gpu_first_batch_lp.py",
    "src/gpu_candidate_transfer.py",
    "src/gpu_heterogeneous_compact.py",
})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ROUTER_FORMAT = "multilabel_original_lp_coverage_router"
ROUTER_VERSION = 1
ROUTER_FEATURE_FIELDS = (
    "rhs", "lower", "upper", "c", "delta", "col_scale", "row_scale")


class GpuFirstBatchSummaryError(ValueError):
    """The report cannot support the requested strict summary."""


def _mapping(value, label):
    if not isinstance(value, dict):
        raise GpuFirstBatchSummaryError(f"{label} must be an object")
    return value


def _sequence(value, label, length=None):
    if not isinstance(value, list):
        raise GpuFirstBatchSummaryError(f"{label} must be an array")
    if length is not None and len(value) != length:
        raise GpuFirstBatchSummaryError(f"{label} must contain {length} entries")
    return value


def _integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GpuFirstBatchSummaryError(
            f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value, label):
    if not isinstance(value, bool):
        raise GpuFirstBatchSummaryError(f"{label} must be boolean")
    return value


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GpuFirstBatchSummaryError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise GpuFirstBatchSummaryError(f"{label} must be finite")
    return value


def _number(value, label, *, positive=False):
    value = _finite(value, label)
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise GpuFirstBatchSummaryError(f"{label} must be finite and {qualifier}")
    return value


def _digest(value, label):
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise GpuFirstBatchSummaryError(
            f"{label} must be a lowercase hexadecimal SHA256")
    return value


def _close(actual, expected, label, *, rel_tol=1e-6, abs_tol=1e-6):
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise GpuFirstBatchSummaryError(
            f"{label} is inconsistent ({actual!r} != {expected!r})")


def _source_snapshot(source_hashes, source_directory, *, has_coverage_routers):
    hashes = _mapping(source_hashes, "source_hashes")
    if source_directory is None:
        raise GpuFirstBatchSummaryError(
            "source_directory is required to verify source snapshot hashes")
    root = Path(source_directory).resolve()
    if not root.is_dir():
        raise GpuFirstBatchSummaryError("source snapshot directory does not exist")
    required = set(REQUIRED_SOURCE_FILES)
    if has_coverage_routers:
        required.update({"src/coverage_router.py", "src/coverage_router_binding.py"})
    missing = sorted(required - set(hashes))
    if missing:
        raise GpuFirstBatchSummaryError(
            "source snapshot is missing required files: " + ", ".join(missing))
    if not hashes:
        raise GpuFirstBatchSummaryError("source_hashes must not be empty")

    verified = {}
    for name, claimed in hashes.items():
        if not isinstance(name, str) or not name:
            raise GpuFirstBatchSummaryError("source snapshot names must be nonempty strings")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise GpuFirstBatchSummaryError(
                f"source snapshot path is unsafe: {name!r}")
        claimed = _digest(claimed, f"source_hashes[{name!r}]")
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise GpuFirstBatchSummaryError(
                f"source snapshot path escapes its directory: {name!r}") from error
        if not target.is_file():
            raise GpuFirstBatchSummaryError(
                f"source snapshot file is missing: {name}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != claimed:
            raise GpuFirstBatchSummaryError(
                f"source snapshot SHA256 mismatch: {name}")
        verified[name] = claimed
    manifest_bytes = json.dumps(
        verified, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "directory": str(root),
        "files_verified": len(verified),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def _model_and_bank_contract(report):
    fingerprints = _mapping(report.get("model_fingerprints"), "model_fingerprints")
    if not fingerprints:
        raise GpuFirstBatchSummaryError("model_fingerprints must not be empty")
    checked = {}
    for name, digest in fingerprints.items():
        if not isinstance(name, str) or not name:
            raise GpuFirstBatchSummaryError(
                "model fingerprint names must be nonempty strings")
        checked[name] = _digest(digest, f"model_fingerprints[{name!r}]")

    manifest = _mapping(report.get("offline_bank_manifest"),
                        "offline_bank_manifest")
    if manifest.get("status") != "completed":
        raise GpuFirstBatchSummaryError("offline bank status must be completed")
    if manifest.get("model_fingerprints") != checked:
        raise GpuFirstBatchSummaryError(
            "offline bank model fingerprints disagree with evaluated models")
    train_seeds = _sequence(manifest.get("train_seeds"),
                            "offline_bank_manifest.train_seeds")
    checked_train = []
    for index, seed in enumerate(train_seeds):
        checked_train.append(_integer(
            seed, f"offline_bank_manifest.train_seeds[{index}]"))
    if len(set(checked_train)) != len(checked_train):
        raise GpuFirstBatchSummaryError("offline bank training seeds are duplicated")

    stages = _sequence(manifest.get("stages"), "offline_bank_manifest.stages", 3)
    candidate_counts = {}
    stage_contracts = {}
    for index, (stage, expected) in enumerate(zip(stages, STAGES)):
        stage = _mapping(stage, f"offline_bank_manifest.stages[{index}]")
        if stage.get("stage") != expected:
            raise GpuFirstBatchSummaryError(
                "offline bank stages must be ordered maxmin/aggregate/exchange")
        entries = _sequence(stage.get("entries"),
                            f"offline bank {expected} entries")
        if not entries:
            raise GpuFirstBatchSummaryError(
                f"offline bank {expected} candidate list must not be empty")
        key = _sequence(stage.get("key"), f"offline bank {expected} key", 4)
        if key[0] != expected:
            raise GpuFirstBatchSummaryError(
                f"offline bank {expected} stage key has the wrong stage")
        dimensions = [_integer(value, f"offline bank {expected} key[{offset}]",
                               minimum=1 if offset < 2 else 0)
                      for offset, value in enumerate(key[1:])]
        rows, columns, neq = dimensions
        if neq > rows:
            raise GpuFirstBatchSummaryError(
                f"offline bank {expected} equality count exceeds row count")
        candidate_ids = []
        for entry_index, raw_entry in enumerate(entries):
            entry = _mapping(
                raw_entry, f"offline bank {expected} entry {entry_index}")
            filename = entry.get("filename")
            if (not isinstance(filename, str) or not filename
                    or Path(filename).name != filename):
                raise GpuFirstBatchSummaryError(
                    f"offline bank {expected} candidate filename must be a basename")
            digest = _digest(
                entry.get("sha256"),
                f"offline bank {expected} entry {entry_index} SHA256")
            candidate_ids.append(filename + ":" + digest)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise GpuFirstBatchSummaryError(
                f"offline bank {expected} candidate identities are duplicated")
        candidate_counts[expected] = len(entries)
        stage_contracts[expected] = {
            "key": [expected, rows, columns, neq],
            "candidate_ids": candidate_ids,
            "candidate_count": len(entries),
        }
    return checked, set(checked_train), candidate_counts, stage_contracts


def _router_specs(configuration):
    """Normalize the legacy maxmin flag and the new per-stage CLI records."""
    primary = configuration.get("coverage_router")
    primary_sha = configuration.get("coverage_router_sha256")
    if (primary is None) != (primary_sha is None):
        raise GpuFirstBatchSummaryError(
            "coverage-router path and SHA256 must be supplied together")
    specs = {}
    if primary is not None:
        if not isinstance(primary, str) or not primary:
            raise GpuFirstBatchSummaryError(
                "configuration.coverage_router must be a nonempty path string")
        specs["maxmin"] = {"path": primary,
                           "sha256": _digest(
                               primary_sha, "configuration.coverage_router_sha256")}
    additional = configuration.get("stage_coverage_router", [])
    additional = _sequence(additional, "configuration.stage_coverage_router")
    for index, raw in enumerate(additional):
        row = _sequence(raw, f"configuration.stage_coverage_router[{index}]", 3)
        stage, path, digest = row
        if stage not in STAGES or stage in specs:
            raise GpuFirstBatchSummaryError(
                "stage coverage routers must use unique original3 stages")
        if not isinstance(path, str) or not path:
            raise GpuFirstBatchSummaryError(
                f"configuration.stage_coverage_router[{index}] path is invalid")
        specs[stage] = {
            "path": path,
            "sha256": _digest(
                digest, f"configuration.stage_coverage_router[{index}] SHA256"),
        }
    return specs


def _shape(value, label, length):
    value = _sequence(value, label, length)
    return [_integer(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _feature_schema(binding, stage_contract, label):
    schema = _mapping(binding.get("feature_schema"), f"{label}.feature_schema")
    if schema.get("version") != 1 or schema.get("field_order") != list(ROUTER_FEATURE_FIELDS):
        raise GpuFirstBatchSummaryError(
            f"{label} feature schema version/field order is invalid")
    rows, columns = stage_contract["key"][1:3]
    shapes = _mapping(schema.get("field_shapes"), f"{label} feature field shapes")
    if set(shapes) != set(ROUTER_FEATURE_FIELDS):
        raise GpuFirstBatchSummaryError(
            f"{label} feature schema has missing or unknown fields")
    checked_shapes = {
        "rhs": _shape(shapes["rhs"], f"{label} rhs shape", 1),
        "lower": _shape(shapes["lower"], f"{label} lower shape", 1),
        "upper": _shape(shapes["upper"], f"{label} upper shape", 1),
        "c": _shape(shapes["c"], f"{label} c shape", 1),
        "delta": _shape(shapes["delta"], f"{label} delta shape", 2),
        "col_scale": _shape(shapes["col_scale"], f"{label} col_scale shape", 1),
        "row_scale": _shape(shapes["row_scale"], f"{label} row_scale shape", 1),
    }
    expected = {
        "rhs": [rows], "lower": [columns], "upper": [columns],
        "c": [columns], "col_scale": [columns], "row_scale": [rows],
    }
    if any(checked_shapes[name] != shape for name, shape in expected.items()):
        raise GpuFirstBatchSummaryError(
            f"{label} feature shapes disagree with the stage LP")
    if checked_shapes["delta"][1] != columns:
        raise GpuFirstBatchSummaryError(
            f"{label} delta feature width disagrees with the LP columns")
    width = sum(math.prod(checked_shapes[name]) for name in ROUTER_FEATURE_FIELDS)
    if _integer(schema.get("original_feature_width"),
                f"{label} original feature width", minimum=1) != width:
        raise GpuFirstBatchSummaryError(
            f"{label} original feature width is inconsistent")
    indices = _sequence(schema.get("selected_feature_indices"),
                        f"{label} selected feature indices")
    checked_indices = [_integer(value, f"{label} feature index {index}")
                       for index, value in enumerate(indices)]
    if (not checked_indices or len(set(checked_indices)) != len(checked_indices)
            or max(checked_indices) >= width):
        raise GpuFirstBatchSummaryError(
            f"{label} selected feature indices are invalid")
    if _integer(schema.get("selected_feature_width"),
                f"{label} selected feature width", minimum=1) != len(checked_indices):
        raise GpuFirstBatchSummaryError(
            f"{label} selected feature width is inconsistent")
    if (schema.get("encoding") !=
            "nan_to_num(0,+1e13,-1e13);sign(x)*log1p(abs(x));float32"
            or schema.get("flattening") !=
            "C-order within fields, then field_order concatenation"):
        raise GpuFirstBatchSummaryError(
            f"{label} feature encoding contract is invalid")
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if _digest(binding.get("feature_schema_sha256"),
               f"{label}.feature_schema_sha256") != digest:
        raise GpuFirstBatchSummaryError(
            f"{label} feature schema SHA256 is inconsistent")
    return schema, digest


def _router_npz(path, expected_sha, label):
    """Independently validate a pickle-free router artifact and its metadata."""
    import numpy as np

    if not isinstance(path, str) or not path:
        raise GpuFirstBatchSummaryError(f"{label}.path must be a nonempty string")
    artifact = Path(path)
    if not artifact.is_file():
        raise GpuFirstBatchSummaryError(f"{label} router artifact is missing: {path}")
    actual_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise GpuFirstBatchSummaryError(f"{label} router artifact SHA256 mismatch")

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")

    try:
        with np.load(artifact, allow_pickle=False) as data:
            if ("metadata" not in data or data["metadata"].shape != ()
                    or data["metadata"].dtype.kind != "U"):
                raise GpuFirstBatchSummaryError(
                    f"{label} router requires scalar Unicode JSON metadata")
            metadata = json.loads(str(data["metadata"]),
                                  parse_constant=reject_constant)
            metadata = _mapping(metadata, f"{label} router metadata")
            architecture = _mapping(metadata.get("architecture"),
                                    f"{label} router architecture")
            candidate_count = _integer(
                architecture.get("candidate_count"),
                f"{label} router candidate count", minimum=1)
            input_dim = _integer(
                architecture.get("input_dim"), f"{label} router input dim",
                minimum=1)
            hidden = _integer(
                architecture.get("hidden_dim"), f"{label} router hidden dim")
            selected = _integer(
                architecture.get("selected_feature_count"),
                f"{label} router selected feature count", minimum=1)
            expected_arrays = {"indices", "mean", "scale", "w1", "b1"}
            if hidden:
                expected_arrays.update({"w2", "b2"})
            if set(data.files) != expected_arrays | {"metadata"}:
                raise GpuFirstBatchSummaryError(
                    f"{label} router weight names disagree with architecture")
            indices = data["indices"]
            if (indices.dtype != np.int64 or indices.ndim != 1
                    or len(indices) != selected or not len(indices)
                    or np.any(indices < 0) or np.any(indices >= input_dim)
                    or np.any(np.diff(indices) <= 0)):
                raise GpuFirstBatchSummaryError(
                    f"{label} router feature indices are invalid")
            width = len(indices)
            middle = hidden or candidate_count
            shapes = {
                "mean": (width,), "scale": (width,),
                "w1": (width, middle), "b1": (middle,),
            }
            if hidden:
                shapes.update(w2=(hidden, candidate_count),
                              b2=(candidate_count,))
            for name, shape in shapes.items():
                value = data[name]
                if (value.dtype != np.float32 or value.shape != shape
                        or not np.isfinite(value).all()):
                    raise GpuFirstBatchSummaryError(
                        f"{label} router array {name} is invalid")
            if np.any(data["scale"] <= 0):
                raise GpuFirstBatchSummaryError(
                    f"{label} router feature scale must be positive")
            activation = "tanh" if hidden else "linear"
            if architecture.get("activation") != activation:
                raise GpuFirstBatchSummaryError(
                    f"{label} router activation is inconsistent")
    except GpuFirstBatchSummaryError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise GpuFirstBatchSummaryError(
            f"{label} router artifact is malformed: {error}") from error
    if metadata.get("format") != ROUTER_FORMAT or metadata.get("version") != ROUTER_VERSION:
        raise GpuFirstBatchSummaryError(
            f"{label} router format/version is unsupported")
    return metadata, architecture


def _validate_router_bindings(report, specs, stage_contracts, models, *, bank_path):
    """Bind every configured stage to its archived report and NPZ provenance."""
    plural = report.get("coverage_router_bindings")
    singular = report.get("coverage_router_binding")
    if plural is None:
        # Backward compatibility for reports written before per-stage routers.
        bindings = {} if singular is None else {"maxmin": singular}
    else:
        bindings = _mapping(plural, "coverage_router_bindings")
        if singular is not None:
            if "maxmin" not in bindings or singular != bindings["maxmin"]:
                raise GpuFirstBatchSummaryError(
                    "legacy maxmin binding alias disagrees with plural bindings")
    if set(bindings) != set(specs):
        raise GpuFirstBatchSummaryError(
            "coverage router binding stages disagree with configuration")
    if not specs:
        if report.get("coverage_router_comparator") is not None:
            raise GpuFirstBatchSummaryError(
                "coverage router comparator exists without a configured router")
        return {}, {"cpu": 0.0, "gpu": 0.0}

    manifest_path = Path(bank_path) / "manifest.json"
    if not manifest_path.is_file():
        raise GpuFirstBatchSummaryError(
            "configured compact-bank manifest is missing")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        on_disk_manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GpuFirstBatchSummaryError(
            f"configured compact-bank manifest is malformed: {error}") from error
    if on_disk_manifest != report.get("offline_bank_manifest"):
        raise GpuFirstBatchSummaryError(
            "configured compact-bank manifest disagrees with the archived report")

    comparator = _mapping(report.get("coverage_router_comparator"),
                          "coverage_router_comparator")
    for field in ("same_final_candidate_bank", "same_learned_proposer",
                  "cpu_persistent_warm_basis"):
        if comparator.get(field) is not True:
            raise GpuFirstBatchSummaryError(
                f"coverage_router_comparator.{field} must be true")

    verified = {}
    setup = {"cpu": 0.0, "gpu": 0.0}
    bank_sha = None
    for stage in STAGES:
        if stage not in specs:
            continue
        label = f"coverage_router_bindings.{stage}"
        binding = _mapping(bindings[stage], label)
        spec = specs[stage]
        if binding.get("stage") != stage or binding.get("sha256") != spec["sha256"]:
            raise GpuFirstBatchSummaryError(
                f"{label} stage/artifact SHA disagrees with configuration")
        binding_path = binding.get("path")
        if (not isinstance(binding_path, str) or not binding_path
                or Path(binding_path).resolve() != Path(spec["path"]).resolve()):
            raise GpuFirstBatchSummaryError(
                f"{label} artifact path disagrees with configuration")
        stage_contract = stage_contracts[stage]
        if binding.get("stage_key") != stage_contract["key"]:
            raise GpuFirstBatchSummaryError(
                f"{label} stage key disagrees with the bank")
        candidate_count = _integer(
            binding.get("candidate_count"), f"{label}.candidate_count", minimum=1)
        if candidate_count != stage_contract["candidate_count"]:
            raise GpuFirstBatchSummaryError(
                f"{label} candidate count disagrees with the bank")
        if binding.get("candidate_ids") != stage_contract["candidate_ids"]:
            raise GpuFirstBatchSummaryError(
                f"{label} candidate order disagrees with the bank")
        _, schema_sha = _feature_schema(binding, stage_contract, label)
        current_bank_sha = _digest(
            binding.get("bank_manifest_sha256"), f"{label}.bank_manifest_sha256")
        if current_bank_sha != manifest_sha:
            raise GpuFirstBatchSummaryError(
                f"{label} is not pinned to the configured compact-bank manifest")
        if bank_sha is None:
            bank_sha = current_bank_sha
        elif current_bank_sha != bank_sha:
            raise GpuFirstBatchSummaryError(
                "coverage routers do not bind the same compact bank manifest")
        coverage_sha = _digest(
            binding.get("coverage_sha256"), f"{label}.coverage_sha256")
        metadata, architecture = _router_npz(
            binding.get("path"), spec["sha256"], label)
        provenance = _mapping(metadata.get("provenance"),
                              f"{label} router provenance")
        expected_provenance = {
            "bank_sha256": current_bank_sha,
            "coverage_sha256": coverage_sha,
            "feature_schema_sha256": schema_sha,
            "model_fingerprints": models,
            "candidate_ids": stage_contract["candidate_ids"],
            "stage": stage,
            "stage_key": stage_contract["key"],
        }
        for field, expected in expected_provenance.items():
            if provenance.get(field) != expected:
                raise GpuFirstBatchSummaryError(
                    f"{label} router provenance mismatch: {field}")
        if binding.get("architecture") != architecture:
            raise GpuFirstBatchSummaryError(
                f"{label} architecture disagrees with the router artifact")
        if architecture["candidate_count"] != candidate_count:
            raise GpuFirstBatchSummaryError(
                f"{label} artifact candidate count disagrees with the bank")
        if architecture["input_dim"] != binding["feature_schema"]["selected_feature_width"]:
            raise GpuFirstBatchSummaryError(
                f"{label} artifact input dimension disagrees with feature schema")
        cpu_setup = _number(
            binding.get("cpu_binding_and_numpy_setup_seconds"),
            f"{label}.cpu_binding_and_numpy_setup_seconds")
        gpu_setup = _number(
            binding.get("gpu_binding_and_upload_setup_seconds"),
            f"{label}.gpu_binding_and_upload_setup_seconds")
        setup["cpu"] += cpu_setup
        setup["gpu"] += gpu_setup
        verified[stage] = {
            "sha256": spec["sha256"],
            "stage_key": stage_contract["key"],
            "bank_manifest_sha256": current_bank_sha,
            "coverage_sha256": coverage_sha,
            "feature_schema_sha256": schema_sha,
            "candidate_count": candidate_count,
            "artifact_path": str(Path(binding["path"]).resolve()),
        }
    return verified, setup


def _configuration(report):
    configuration = _mapping(report.get("configuration"), "configuration")
    required = {
        "gpu_first_batch": True,
        "pipeline_cpu_stages": True,
        "hybrid": True,
        "cpu_backend": "dictionary",
        "tie_policy": "original3",
        "candidate_oracle": False,
        "candidate_diagnostics": False,
        "speculative_cpu": False,
        "heterogeneous_candidates": True,
        "hybrid_rounds": 0,
        "cpu_basis_handoff": False,
    }
    for field, expected in required.items():
        if configuration.get(field) != expected:
            raise GpuFirstBatchSummaryError(
                f"configuration.{field} must be {expected!r}")
    if configuration.get("gpu_stages") != list(STAGES):
        raise GpuFirstBatchSummaryError(
            "configuration.gpu_stages must be maxmin/aggregate/exchange")
    fallback = configuration.get("gpu_cpu_fallback")
    if fallback not in {"exact", "reject"}:
        raise GpuFirstBatchSummaryError(
            "configuration.gpu_cpu_fallback must be exact or reject")
    steps = _integer(configuration.get("steps"), "configuration.steps", minimum=1)
    environments = _integer(
        configuration.get("environments"), "configuration.environments", minimum=1)
    repeats = _integer(
        configuration.get("repeats"), "configuration.repeats", minimum=1)
    seed = _integer(configuration.get("seed"), "configuration.seed")
    stride = _integer(
        configuration.get("seed_stride"), "configuration.seed_stride")
    execution_order = configuration.get("execution_order")
    if execution_order not in {"cpu-first", "gpu-first", "alternate"}:
        raise GpuFirstBatchSummaryError(
            "configuration.execution_order is invalid")
    candidate_limit = _integer(
        configuration.get("candidate_limit"), "configuration.candidate_limit")
    bank = configuration.get("bank")
    if not isinstance(bank, str) or not bank:
        raise GpuFirstBatchSummaryError(
            "configuration.bank must be a nonempty path string")
    router_specs = _router_specs(configuration)
    return {
        "steps": steps,
        "environments": environments,
        "repeats": repeats,
        "seed": seed,
        "seed_stride": stride,
        "execution_order": execution_order,
        "candidate_limit": candidate_limit,
        "gpu_cpu_fallback": fallback,
        "bank": bank,
        "coverage_router": configuration.get("coverage_router"),
        "coverage_router_specs": router_specs,
    }


def _expected_execution_order(configured, repeat):
    if configured == "alternate":
        return "cpu-first" if repeat % 2 == 0 else "gpu-first"
    return configured


def _pipeline_blocks(history, *, steps, environments, block_size, wall, label):
    rows = _sequence(history, f"{label}.cpu_pipeline_history", steps)
    blocks = []
    current = None
    total = 0.0
    for cycle, raw in enumerate(rows):
        row = _mapping(raw, f"{label}.cpu_pipeline_history[{cycle}]")
        if row.get("failed") is True:
            raise GpuFirstBatchSummaryError(
                f"{label} CPU pipeline cycle {cycle} is marked failed")
        if _integer(row.get("cycle"), f"{label} CPU pipeline cycle ID") != cycle:
            raise GpuFirstBatchSummaryError(
                f"{label} CPU pipeline cycle IDs are not sequential")
        if _integer(row.get("batch"), f"{label} CPU cycle batch", minimum=1) != environments:
            raise GpuFirstBatchSummaryError(
                f"{label} CPU pipeline cycle has the wrong batch size")
        parts = {field: _number(row.get(field), f"{label} cycle {cycle} {field}")
                 for field in CPU_NONOVERLAP_FIELDS}
        cycle_total = _number(
            row.get("total_seconds"), f"{label} cycle {cycle} total_seconds")
        _close(sum(parts.values()), cycle_total,
               f"{label} cycle {cycle} CPU nonoverlap timers")
        spans = _mapping(row.get("async_span_seconds"),
                         f"{label} cycle {cycle} async_span_seconds")
        if set(spans) != {"aggregate", "exchange"}:
            raise GpuFirstBatchSummaryError(
                f"{label} cycle {cycle} async spans must contain aggregate/exchange")
        for stage in ("aggregate", "exchange"):
            _number(spans[stage], f"{label} cycle {cycle} {stage} async span")

        block_index = cycle // block_size
        if current is None or current["block_index"] != block_index:
            current = {
                "block_index": block_index,
                "start_step": cycle + 1,
                "end_step": cycle + 1,
                "steps": 0,
                "total_seconds": 0.0,
                **{field: 0.0 for field in CPU_NONOVERLAP_FIELDS},
            }
            blocks.append(current)
        current["end_step"] = cycle + 1
        current["steps"] += 1
        current["total_seconds"] += cycle_total
        for field, value in parts.items():
            current[field] += value
        total += cycle_total

    tolerance = max(0.01, wall * 1e-3)
    if total > wall + tolerance:
        raise GpuFirstBatchSummaryError(
            f"{label} CPU pipeline cycle totals exceed CPU wall time")
    return blocks, total, max(0.0, wall - total), tolerance


def _certificate_vector(record, field, environments, label):
    values = _sequence(record.get(field), f"{label}.{field}", environments)
    checked = [_number(value, f"{label}.{field}[{index}]")
               for index, value in enumerate(values)]
    limit = CERTIFICATE_LIMITS[field]
    if max(checked, default=0.0) > limit:
        raise GpuFirstBatchSummaryError(
            f"{label}.{field} exceeds original-LP limit {limit:g}")
    return checked


def _dictionary_choice(row, *, stage, router_stages, candidate_counts, label):
    """Validate which fixed-bank proposer supplied a cold CPU basis."""
    reason = row.get("dictionary_reason")
    candidate = row.get("dictionary_candidate")
    cold_reason = ("learned_coverage" if stage in router_stages
                   else "nearest_centroid")
    if reason == "existing_cpu_basis":
        if candidate is not None:
            raise GpuFirstBatchSummaryError(
                f"{label} existing CPU basis must not name a dictionary candidate")
        return
    if reason != cold_reason:
        raise GpuFirstBatchSummaryError(
            f"{label} dictionary proposer disagrees with the stage router binding")
    candidate = _integer(candidate, f"{label}.dictionary_candidate")
    if candidate >= candidate_counts[stage]:
        raise GpuFirstBatchSummaryError(
            f"{label} dictionary candidate is outside the fixed bank")


def _cpu_row(row, *, stage, environment_id, label, router_stages,
             candidate_counts, result_used=False):
    row = _mapping(row, label)
    if row.get("stage") != stage:
        raise GpuFirstBatchSummaryError(f"{label} has the wrong LP stage")
    if _integer(row.get("environment_id"), f"{label}.environment_id") != environment_id:
        raise GpuFirstBatchSummaryError(f"{label} has the wrong environment ID")
    if row.get("success") is not True or row.get("certificate_passed") is not True:
        raise GpuFirstBatchSummaryError(
            f"{label} is not a successful, original-LP-certified CPU result")
    if result_used and row.get("cpu_result_used") is not True:
        raise GpuFirstBatchSummaryError(
            f"{label} must be marked as used by the final output")
    _dictionary_choice(
        row, stage=stage, router_stages=router_stages,
        candidate_counts=candidate_counts, label=label)
    calls = _integer(row.get("cpu_lp_calls"), f"{label}.cpu_lp_calls", minimum=1)
    if calls != 1:
        raise GpuFirstBatchSummaryError(f"{label} must represent one LP request")
    runs = _integer(row.get("cpu_solver_runs"), f"{label}.cpu_solver_runs", minimum=1)
    retries = _integer(row.get("numerical_retry_count"),
                       f"{label}.numerical_retry_count")
    if runs > 1 + retries:
        raise GpuFirstBatchSummaryError(
            f"{label} solver runs exceed its recorded attempts")
    attempts = row.get("solver_attempts")
    if attempts is not None:
        attempts = _sequence(attempts, f"{label}.solver_attempts", 1 + retries)
        counted = 0
        for index, attempt in enumerate(attempts):
            attempt = _mapping(attempt, f"{label}.solver_attempts[{index}]")
            counted += int(_boolean(
                attempt.get("solver_run"), f"{label}.solver_attempts[{index}].solver_run"))
        if counted != runs:
            raise GpuFirstBatchSummaryError(
                f"{label} solver attempts disagree with cpu_solver_runs")
    certificates = {}
    for field, limit in CERTIFICATE_LIMITS.items():
        value = _number(row.get(field), f"{label}.{field}")
        if value > limit:
            raise GpuFirstBatchSummaryError(
                f"{label}.{field} exceeds original-LP limit {limit:g}")
        certificates[field] = value
    return calls, runs, retries, certificates


def _cpu_rows_for_record(record, environments, label):
    direct = record.get("rows")
    if isinstance(direct, list):
        return _sequence(direct, f"{label}.rows", environments)
    groups = _sequence(record.get("groups"), f"{label}.groups")
    rows = []
    for group_index, raw in enumerate(groups):
        group = _mapping(raw, f"{label}.groups[{group_index}]")
        if group.get("route") != "cpu_fallback":
            raise GpuFirstBatchSummaryError(
                f"{label}.groups[{group_index}] has an invalid route")
        rows.extend(_sequence(group.get("rows"),
                              f"{label}.groups[{group_index}].rows"))
    return _sequence(rows, f"{label} CPU rows", environments)


def _empty_stage_counts():
    return {stage: {
        "lp_outputs": 0,
        "bank_accepts": 0,
        "cpu_lp_calls": 0,
        "cpu_solver_runs": 0,
        "numerical_retry_count": 0,
        "candidate_evaluations": 0,
    } for stage in STAGES}


def _validate_cpu_history(history, *, steps, environments, router_stages,
                          candidate_counts, label):
    records = _sequence(history, f"{label}.cpu_history", steps * len(STAGES))
    counts = _empty_stage_counts()
    maxima = {stage: {field: 0.0 for field in CERTIFICATE_LIMITS}
              for stage in STAGES}
    for index, raw in enumerate(records):
        record_label = f"{label}.cpu_history[{index}]"
        record = _mapping(raw, record_label)
        stage = STAGES[index % len(STAGES)]
        cycle = index // len(STAGES)
        if record.get("stage") != stage:
            raise GpuFirstBatchSummaryError(
                f"{label}.cpu_history is not ordered maxmin/aggregate/exchange")
        if "pipeline_cycle" in record and _integer(
                record["pipeline_cycle"], f"{record_label}.pipeline_cycle") != cycle:
            raise GpuFirstBatchSummaryError(
                f"{record_label} has the wrong pipeline cycle")
        if _integer(record.get("batch"), f"{record_label}.batch", minimum=1) != environments:
            raise GpuFirstBatchSummaryError(
                f"{record_label} has the wrong batch size")
        if _integer(record.get("bank_accepts"),
                    f"{record_label}.bank_accepts") != 0:
            raise GpuFirstBatchSummaryError(
                f"{record_label} CPU reference cannot contain bank accepts")
        accepted = _sequence(record.get("accepted"),
                             f"{record_label}.accepted", environments)
        if any(value is not True for value in accepted):
            raise GpuFirstBatchSummaryError(
                f"{record_label} contains an unaccepted CPU LP")
        calls = _integer(record.get("cpu_lp_calls"),
                         f"{record_label}.cpu_lp_calls")
        if calls != environments:
            raise GpuFirstBatchSummaryError(
                f"{record_label} must execute one CPU LP per environment")
        vectors = {field: _certificate_vector(record, field, environments, record_label)
                   for field in CERTIFICATE_LIMITS}
        rows = _cpu_rows_for_record(record, environments, record_label)
        solver_runs = retries = 0
        for environment_id, row in enumerate(rows):
            _, row_runs, row_retries, row_certificates = _cpu_row(
                row, stage=stage, environment_id=environment_id,
                label=f"{record_label} CPU row {environment_id}",
                router_stages=router_stages,
                candidate_counts=candidate_counts)
            solver_runs += row_runs
            retries += row_retries
            for field, value in row_certificates.items():
                _close(value, vectors[field][environment_id],
                       f"{record_label} row/vector {field}",
                       rel_tol=1e-9, abs_tol=1e-12)
        if "cpu_solver_runs" in record and _integer(
                record["cpu_solver_runs"], f"{record_label}.cpu_solver_runs") != solver_runs:
            raise GpuFirstBatchSummaryError(
                f"{record_label} CPU solver-run count disagrees with rows")
        if "numerical_retry_count" in record and _integer(
                record["numerical_retry_count"],
                f"{record_label}.numerical_retry_count") != retries:
            raise GpuFirstBatchSummaryError(
                f"{record_label} retry count disagrees with rows")
        counts[stage]["lp_outputs"] += environments
        counts[stage]["cpu_lp_calls"] += calls
        counts[stage]["cpu_solver_runs"] += solver_runs
        counts[stage]["numerical_retry_count"] += retries
        counts[stage]["candidate_evaluations"] += _integer(
            record.get("candidate_evaluations", 0),
            f"{record_label}.candidate_evaluations")
        for field, values in vectors.items():
            maxima[stage][field] = max(maxima[stage][field], max(values))
    return maxima, counts


def _gpu_cpu_group(record, *, stage, environments, label):
    groups = _sequence(record.get("groups"), f"{label}.groups", 1)
    group = _mapping(groups[0], f"{label}.groups[0]")
    if group.get("route") != "cpu_fallback":
        raise GpuFirstBatchSummaryError(f"{label} CPU group route is invalid")
    positions = _sequence(group.get("ids"), f"{label} CPU group positions")
    environment_ids = _sequence(
        group.get("environment_ids"), f"{label} CPU group environment IDs",
        len(positions))
    rows = _sequence(group.get("rows"), f"{label} CPU group rows", len(positions))
    checked_positions = []
    for index, position in enumerate(positions):
        position = _integer(position, f"{label} CPU group position {index}")
        if position >= environments:
            raise GpuFirstBatchSummaryError(
                f"{label} CPU group position is out of range")
        checked_positions.append(position)
    if len(set(checked_positions)) != len(checked_positions):
        raise GpuFirstBatchSummaryError(f"{label} CPU group positions are duplicated")
    if environment_ids != checked_positions:
        raise GpuFirstBatchSummaryError(
            f"{label} CPU group environment IDs do not match stable batch positions")
    return checked_positions, rows


def _validate_gpu_history(history, *, steps, environments, fallback,
                          candidate_limit, candidate_counts, router_stages,
                          wall, block_size, label):
    records = _sequence(history, f"{label}.gpu_history", steps * len(STAGES))
    counts = _empty_stage_counts()
    maxima = {stage: {field: 0.0 for field in CERTIFICATE_LIMITS}
              for stage in STAGES}
    phase_totals = {field: 0.0 for field in GPU_FIRST_PHASE_FIELDS}
    blocks = []
    current = None
    gpu_stage_seconds = 0.0
    for index, raw in enumerate(records):
        record_label = f"{label}.gpu_history[{index}]"
        record = _mapping(raw, record_label)
        stage = STAGES[index % len(STAGES)]
        step = index // len(STAGES)
        if record.get("stage") != stage:
            raise GpuFirstBatchSummaryError(
                f"{label}.gpu_history is not ordered maxmin/aggregate/exchange")
        if _integer(record.get("batch"), f"{record_label}.batch", minimum=1) != environments:
            raise GpuFirstBatchSummaryError(
                f"{record_label} has the wrong batch size")
        if record.get("environment_ids") != list(range(environments)):
            raise GpuFirstBatchSummaryError(
                f"{record_label} lacks the stable full-cohort environment IDs")
        expected_router = ("learned_coverage" if stage in router_stages
                           else "nearest_centroid")
        if record.get("candidate_router") != expected_router:
            raise GpuFirstBatchSummaryError(
                f"{record_label} candidate router disagrees with its bound artifact")
        required_flags = {
            "gpu_first_batch": True,
            "gpu_cpu_fallback": fallback,
            "candidate_engine": "gpu_first_heterogeneous",
            "certificate_only": True,
            "device_routing": True,
            "packed_result_transfer": True,
            "repair_policy": "gpu_first_no_legacy_repair",
        }
        for field, expected in required_flags.items():
            if record.get(field) != expected:
                raise GpuFirstBatchSummaryError(
                    f"{record_label}.{field} must be {expected!r}")
        repair = _sequence(record.get("repair_eligible"),
                           f"{record_label}.repair_eligible", environments)
        if any(value is not False for value in repair):
            raise GpuFirstBatchSummaryError(
                f"{record_label} must not perform legacy GPU repair")
        for field in ("candidate_count", "observable_dispersion"):
            values = _sequence(record.get(field), f"{record_label}.{field}", environments)
            if any(value is not None for value in values):
                raise GpuFirstBatchSummaryError(
                    f"{record_label}.{field} must remain unavailable in certificate-only mode")

        seconds = _number(record.get("seconds"), f"{record_label}.seconds", positive=True)
        phases = _mapping(record.get("gpu_first_phases"),
                          f"{record_label}.gpu_first_phases")
        if set(phases) != set(GPU_FIRST_PHASE_FIELDS):
            raise GpuFirstBatchSummaryError(
                f"{record_label}.gpu_first_phases has missing or unknown timers")
        checked_phases = {field: _number(phases[field], f"{record_label}.{field}")
                          for field in GPU_FIRST_PHASE_FIELDS}
        _close(sum(checked_phases.values()), seconds,
               f"{record_label} GPU-first phase sum")
        preparation = _number(record.get("preparation_seconds"),
                              f"{record_label}.preparation_seconds")
        bank_seconds = _number(record.get("bank_seconds"),
                               f"{record_label}.bank_seconds")
        _close(preparation,
               checked_phases["parse_normalize_seconds"]
               + checked_phases["stack_upload_seconds"],
               f"{record_label} preparation phase")
        _close(bank_seconds,
               checked_phases["device_rank_seconds"]
               + checked_phases["evaluation_submission_seconds"]
               + checked_phases["completion_download_validation_seconds"],
               f"{record_label} GPU bank phase")

        vectors = {field: _certificate_vector(record, field, environments, record_label)
                   for field in CERTIFICATE_LIMITS}
        accepted = _sequence(record.get("accepted"),
                             f"{record_label}.accepted", environments)
        if any(value is not True for value in accepted):
            raise GpuFirstBatchSummaryError(
                f"{record_label} contains an uncertified final LP output")
        routes = _sequence(record.get("routes"), f"{record_label}.routes", environments)
        if any(route not in {"gpu_dictionary_batch", "cpu_after_gpu_certificate"}
               for route in routes):
            raise GpuFirstBatchSummaryError(f"{record_label} contains an invalid route")
        bank = _integer(record.get("bank_accepts"), f"{record_label}.bank_accepts")
        cpu = _integer(record.get("cpu_lp_calls"), f"{record_label}.cpu_lp_calls")
        used = _integer(record.get("cpu_results_used"),
                        f"{record_label}.cpu_results_used")
        gpu_nonbank = _integer(record.get("gpu_nonbank_accepts"),
                               f"{record_label}.gpu_nonbank_accepts")
        unused = _integer(record.get("cpu_speculative_unused"),
                          f"{record_label}.cpu_speculative_unused")
        cancelled = _integer(record.get("cpu_speculative_cancelled"),
                             f"{record_label}.cpu_speculative_cancelled")
        if gpu_nonbank or unused or cancelled:
            raise GpuFirstBatchSummaryError(
                f"{record_label} contains unsupported repair/speculation counts")
        if routes.count("gpu_dictionary_batch") != bank:
            raise GpuFirstBatchSummaryError(
                f"{record_label} bank accept count disagrees with routes")
        if routes.count("cpu_after_gpu_certificate") != cpu or used != cpu:
            raise GpuFirstBatchSummaryError(
                f"{record_label} CPU fallback counts disagree with routes")
        if fallback == "exact":
            if bank + cpu != environments:
                raise GpuFirstBatchSummaryError(
                    f"{record_label} bank accepts plus CPU calls must equal batch")
        elif bank != environments or cpu != 0:
            raise GpuFirstBatchSummaryError(
                f"{record_label} completed reject mode must certify every row on GPU")

        positions, cpu_rows = _gpu_cpu_group(
            record, stage=stage, environments=environments, label=record_label)
        expected_positions = [i for i, route in enumerate(routes)
                              if route == "cpu_after_gpu_certificate"]
        if positions != expected_positions or len(cpu_rows) != cpu:
            raise GpuFirstBatchSummaryError(
                f"{record_label} CPU fallback group disagrees with routes")
        solver_runs = retries = 0
        for position, row in zip(positions, cpu_rows):
            _, row_runs, row_retries, row_certificates = _cpu_row(
                row, stage=stage, environment_id=position,
                label=f"{record_label} CPU row {position}",
                router_stages=router_stages,
                candidate_counts=candidate_counts, result_used=True)
            solver_runs += row_runs
            retries += row_retries
            for field, value in row_certificates.items():
                _close(value, vectors[field][position],
                       f"{record_label} CPU row/vector {field}",
                       rel_tol=1e-9, abs_tol=1e-12)
        if _integer(record.get("cpu_solver_runs"),
                    f"{record_label}.cpu_solver_runs") != solver_runs:
            raise GpuFirstBatchSummaryError(
                f"{record_label} CPU solver-run count disagrees with rows")
        if _integer(record.get("numerical_retry_count"),
                    f"{record_label}.numerical_retry_count") != retries:
            raise GpuFirstBatchSummaryError(
                f"{record_label} retry count disagrees with rows")
        if record.get("cpu_initial_basis_proposals") is not bool(cpu):
            raise GpuFirstBatchSummaryError(
                f"{record_label} CPU initial-basis proposal flag disagrees with fallback count")

        width = min(candidate_limit, candidate_counts[stage]) if candidate_limit else candidate_counts[stage]
        candidate_evaluations = _integer(
            record.get("candidate_evaluations"),
            f"{record_label}.candidate_evaluations", minimum=1)
        if candidate_evaluations != environments * width:
            raise GpuFirstBatchSummaryError(
                f"{record_label} candidate evaluation count disagrees with B x K")

        counts[stage]["lp_outputs"] += environments
        counts[stage]["bank_accepts"] += bank
        counts[stage]["cpu_lp_calls"] += cpu
        counts[stage]["cpu_solver_runs"] += solver_runs
        counts[stage]["numerical_retry_count"] += retries
        counts[stage]["candidate_evaluations"] += candidate_evaluations
        for field, values in vectors.items():
            maxima[stage][field] = max(maxima[stage][field], max(values))
        for field, value in checked_phases.items():
            phase_totals[field] += value
        gpu_stage_seconds += seconds

        block_index = step // block_size
        if current is None or current["block_index"] != block_index:
            current = {
                "block_index": block_index,
                "start_step": step + 1,
                "end_step": step + 1,
                "steps": 0,
                "stage_seconds": 0.0,
                **{field: 0.0 for field in GPU_FIRST_PHASE_FIELDS},
            }
            blocks.append(current)
        current["end_step"] = step + 1
        if stage == STAGES[0]:
            current["steps"] += 1
        current["stage_seconds"] += seconds
        for field, value in checked_phases.items():
            current[field] += value

    tolerance = max(0.01, wall * 1e-3)
    if gpu_stage_seconds > wall + tolerance:
        raise GpuFirstBatchSummaryError(
            f"{label} GPU-first stage totals exceed GPU wall time")
    return (maxima, counts, phase_totals, blocks, gpu_stage_seconds,
            max(0.0, wall - gpu_stage_seconds), tolerance)


def _endpoint_maxima(errors, cpu_rows, gpu_rows, environments, label):
    errors = _sequence(errors, f"{label}.errors", environments)
    cpu_rows = _sequence(cpu_rows, f"{label}.cpu_rows", environments)
    gpu_rows = _sequence(gpu_rows, f"{label}.gpu_rows", environments)
    maxima = {field: 0.0 for field in ENDPOINT_FIELDS}
    for environment_id, (raw_error, raw_cpu, raw_gpu) in enumerate(
            zip(errors, cpu_rows, gpu_rows)):
        prefix = f"{label} environment {environment_id}"
        error = _mapping(raw_error, f"{prefix} errors")
        cpu = _mapping(raw_cpu, f"{prefix} CPU endpoint")
        gpu = _mapping(raw_gpu, f"{prefix} GPU endpoint")
        cpu_pha = _number(cpu.get("pha"), f"{prefix} CPU PHA")
        gpu_pha = _number(gpu.get("pha"), f"{prefix} GPU PHA")
        cpu_phv = _number(cpu.get("phv_fraction"), f"{prefix} CPU PHV")
        gpu_phv = _number(gpu.get("phv_fraction"), f"{prefix} GPU PHV")
        if cpu_phv > 1 or gpu_phv > 1:
            raise GpuFirstBatchSummaryError(f"{prefix} PHV fraction exceeds one")
        cpu_biomass = _mapping(cpu.get("biomass"), f"{prefix} CPU biomass")
        gpu_biomass = _mapping(gpu.get("biomass"), f"{prefix} GPU biomass")
        if not cpu_biomass or set(cpu_biomass) != set(gpu_biomass):
            raise GpuFirstBatchSummaryError(
                f"{prefix} endpoint biomass species do not match")
        biomass_error = 0.0
        for species in cpu_biomass:
            cpu_value = _number(cpu_biomass[species], f"{prefix} CPU biomass {species}")
            gpu_value = _number(gpu_biomass[species], f"{prefix} GPU biomass {species}")
            biomass_error = max(biomass_error, abs(cpu_value - gpu_value))
        for side, endpoint in (("CPU", cpu), ("GPU", gpu)):
            metabolites = _mapping(endpoint.get("metabolites"),
                                   f"{prefix} {side} metabolites")
            if not metabolites:
                raise GpuFirstBatchSummaryError(
                    f"{prefix} {side} metabolites must not be empty")
            for metabolite, value in metabolites.items():
                if not isinstance(metabolite, str) or not metabolite:
                    raise GpuFirstBatchSummaryError(
                        f"{prefix} {side} has an invalid metabolite name")
                _finite(value, f"{prefix} {side} metabolite {metabolite}")
        # The benchmark's formal endpoint gate contains PHA, biomass, and PHV.
        # Sparse extracellular snapshots may omit a numerically-zero metabolite
        # on one side, so require finite mappings without inventing an extra,
        # unreported metabolome-equivalence criterion here.
        recomputed = {
            "pha_relative": abs(cpu_pha - gpu_pha) / max(abs(cpu_pha), 1e-9),
            "biomass_g_l": biomass_error,
            "phv_fraction": abs(cpu_phv - gpu_phv),
        }
        if set(error) != set(ENDPOINT_FIELDS):
            raise GpuFirstBatchSummaryError(
                f"{prefix} error fields must be exactly the three endpoint gates")
        for field in ENDPOINT_FIELDS:
            value = _number(error[field], f"{prefix} {field}")
            if value > ENDPOINT_LIMIT:
                raise GpuFirstBatchSummaryError(
                    f"{prefix} {field} exceeds {ENDPOINT_LIMIT:g}")
            _close(value, recomputed[field], f"{prefix} reported {field}",
                   rel_tol=1e-9, abs_tol=1e-12)
            maxima[field] = max(maxima[field], value)
    return maxima


def _sum_counts(run_summaries, side):
    result = _empty_stage_counts()
    source = "cpu_reference_history_counts" if side == "cpu" else "gpu_history_counts"
    for run in run_summaries:
        for stage in STAGES:
            for field in result[stage]:
                result[stage][field] += run[source][stage][field]
    return result


def _max_certificates(run_summaries):
    result = {side: {stage: {field: 0.0 for field in CERTIFICATE_LIMITS}
                     for stage in STAGES}
              for side in ("cpu", "gpu")}
    for run in run_summaries:
        for side in result:
            for stage in STAGES:
                for field in CERTIFICATE_LIMITS:
                    result[side][stage][field] = max(
                        result[side][stage][field],
                        run["original_lp_certificate_maxima"][side][stage][field])
    return result


def summarize_gpu_first_batch_report(report, *, source_directory, block_size=20):
    """Validate one parsed report and return a JSON-serializable summary.

    ``source_directory`` must point to the report's ``.sources`` snapshot.  It
    is mandatory because digest strings inside a JSON report cannot prove what
    source bytes were actually archived.
    """
    report = _mapping(report, "report")
    block_size = _integer(block_size, "block_size", minimum=1)
    if report.get("status") != "completed":
        raise GpuFirstBatchSummaryError(
            "report status must be completed; incomplete GPU-only diagnostics are not performance successes")
    config = _configuration(report)
    (models, training_seeds, candidate_counts,
     stage_contracts) = _model_and_bank_contract(report)
    router_bindings, router_setup = _validate_router_bindings(
        report, config["coverage_router_specs"], stage_contracts, models,
        bank_path=config["bank"])
    router_stages = frozenset(router_bindings)
    source_snapshot = _source_snapshot(
        report.get("source_hashes"), source_directory,
        has_coverage_routers=bool(router_stages))
    fallback = config["gpu_cpu_fallback"]
    policy = _mapping(report.get("gpu_first_batch_policy"),
                      "gpu_first_batch_policy")
    if (policy.get("cpu_schedule") != "asynchronous aggregate/exchange CPU pipeline"
            or policy.get("gpu_schedule") !=
            "full-cohort maxmin/aggregate/exchange stage barriers"
            or policy.get("cpu_fallback") != fallback
            or policy.get("isolated_hardware_comparison") is not False):
        raise GpuFirstBatchSummaryError(
            "gpu_first_batch_policy disagrees with the asymmetric driver contract")
    if fallback == "reject":
        if report.get("performance_comparison_valid") is not False:
            raise GpuFirstBatchSummaryError(
                "GPU-only reject mode must be marked diagnostic-only")
    elif report.get("performance_comparison_valid") is False:
        raise GpuFirstBatchSummaryError(
            "completed exact-fallback report is unexpectedly marked diagnostic-only")

    runs = _sequence(report.get("runs"), "runs", config["repeats"])
    summaries = []
    seen_seeds = set()
    wall_totals = {"cpu": 0.0, "gpu": 0.0}
    overall_endpoints = {field: 0.0 for field in ENDPOINT_FIELDS}
    maximum_timing_tolerance = 0.0

    for repeat, raw_run in enumerate(runs):
        label = f"run[{repeat}]"
        run = _mapping(raw_run, label)
        if _integer(run.get("repeat"), f"{label}.repeat") != repeat:
            raise GpuFirstBatchSummaryError("run repeat IDs are not sequential")
        if "failure" not in run or run["failure"] is not None:
            raise GpuFirstBatchSummaryError(
                f"{label} contains or omits a failure diagnostic")
        if run.get("all_endpoint_gates_passed") is not True:
            raise GpuFirstBatchSummaryError(f"{label} endpoint gate did not pass")
        if run.get("cpu_driver_kind") != "pipeline" or run.get("gpu_driver_kind") != "barrier":
            raise GpuFirstBatchSummaryError(
                f"{label} must record pipeline CPU and barrier GPU drivers")
        expected_order = _expected_execution_order(config["execution_order"], repeat)
        if run.get("execution_order") != expected_order:
            raise GpuFirstBatchSummaryError(
                f"{label}.execution_order disagrees with the configured schedule")
        cold = _boolean(run.get("cold_first_use_included"),
                        f"{label}.cold_first_use_included")
        if cold != (repeat == 0):
            raise GpuFirstBatchSummaryError(
                f"{label}.cold_first_use_included is inconsistent with repeat")
        seeds = _sequence(run.get("seeds"), f"{label}.seeds", config["environments"])
        checked_seeds = [_integer(value, f"{label}.seeds[{index}]")
                         for index, value in enumerate(seeds)]
        expected_first = config["seed"] + repeat * config["seed_stride"]
        if checked_seeds != list(range(expected_first,
                                       expected_first + config["environments"])):
            raise GpuFirstBatchSummaryError(
                f"{label}.seeds disagree with seed/stride configuration")
        if len(set(checked_seeds)) != len(checked_seeds) or seen_seeds.intersection(checked_seeds):
            raise GpuFirstBatchSummaryError(
                f"{label} evaluation seeds are duplicated across runs")
        if training_seeds.intersection(checked_seeds):
            raise GpuFirstBatchSummaryError(
                f"{label} evaluation seeds overlap offline training seeds")
        seen_seeds.update(checked_seeds)

        for side in ("cpu", "gpu"):
            completed = _sequence(run.get(f"{side}_completed_steps"),
                                  f"{label}.{side}_completed_steps",
                                  config["environments"])
            if any(_integer(value, f"{label}.{side}_completed_steps[{index}]")
                   != config["steps"] for index, value in enumerate(completed)):
                raise GpuFirstBatchSummaryError(
                    f"{label} {side} did not complete every configured step")
        if "gpu_pipeline_history" in run and run["gpu_pipeline_history"] not in (None, []):
            raise GpuFirstBatchSummaryError(
                f"{label} barrier GPU driver must not report pipeline accounting")

        cpu_wall = _number(run.get("cpu_seconds"), f"{label}.cpu_seconds", positive=True)
        gpu_wall = _number(run.get("gpu_seconds"), f"{label}.gpu_seconds", positive=True)
        cpu_blocks, cpu_cycles, cpu_outside, cpu_tolerance = _pipeline_blocks(
            run.get("cpu_pipeline_history"), steps=config["steps"],
            environments=config["environments"], block_size=block_size,
            wall=cpu_wall, label=label)
        cpu_certificates, cpu_counts = _validate_cpu_history(
            run.get("cpu_history"), steps=config["steps"],
            environments=config["environments"],
            router_stages=router_stages, candidate_counts=candidate_counts,
            label=label)
        (gpu_certificates, gpu_counts, gpu_phases, gpu_blocks,
         gpu_stage_seconds, gpu_outside, gpu_tolerance) = _validate_gpu_history(
            run.get("gpu_history"), steps=config["steps"],
            environments=config["environments"], fallback=fallback,
            candidate_limit=config["candidate_limit"],
            candidate_counts=candidate_counts, router_stages=router_stages,
            wall=gpu_wall,
            block_size=block_size, label=label)
        maximum_timing_tolerance = max(
            maximum_timing_tolerance, cpu_tolerance, gpu_tolerance)

        expected_reference_lps = config["steps"] * len(STAGES) * config["environments"]
        reference_calls = sum(value["cpu_lp_calls"] for value in cpu_counts.values())
        if reference_calls != expected_reference_lps:
            raise GpuFirstBatchSummaryError(
                f"{label} CPU reference LP count is incomplete")
        reference_solver_runs = sum(
            value["cpu_solver_runs"] for value in cpu_counts.values())
        if _integer(run.get("cpu_solver_runs"), f"{label}.cpu_solver_runs") != reference_solver_runs:
            raise GpuFirstBatchSummaryError(
                f"{label}.cpu_solver_runs disagrees with CPU history")
        online_calls = sum(value["cpu_lp_calls"] for value in gpu_counts.values())
        online_solver_runs = sum(value["cpu_solver_runs"] for value in gpu_counts.values())
        if _integer(run.get("online_cpu_lp_calls"),
                    f"{label}.online_cpu_lp_calls") != online_calls:
            raise GpuFirstBatchSummaryError(
                f"{label}.online_cpu_lp_calls disagrees with GPU history")
        if _integer(run.get("online_cpu_solver_runs"),
                    f"{label}.online_cpu_solver_runs") != online_solver_runs:
            raise GpuFirstBatchSummaryError(
                f"{label}.online_cpu_solver_runs disagrees with GPU history")
        if _integer(run.get("model_bridge_cpu_lp_stage_calls"),
                    f"{label}.model_bridge_cpu_lp_stage_calls") != 0:
            raise GpuFirstBatchSummaryError(
                f"{label} contains an unaccounted model-bridge CPU LP")

        endpoint_maxima = _endpoint_maxima(
            run.get("errors"), run.get("cpu_rows"), run.get("gpu_rows"),
            config["environments"], label)
        for field, value in endpoint_maxima.items():
            overall_endpoints[field] = max(overall_endpoints[field], value)

        ratio = cpu_wall / gpu_wall
        if fallback == "exact":
            reported_ratio = _number(run.get("cpu_over_gpu_ratio"),
                                     f"{label}.cpu_over_gpu_ratio", positive=True)
            _close(reported_ratio, ratio, f"{label} CPU/GPU ratio",
                   rel_tol=1e-9, abs_tol=1e-12)
            if run.get("performance_comparison_valid") is False:
                raise GpuFirstBatchSummaryError(
                    f"{label} exact comparison is marked invalid")
        else:
            if "cpu_over_gpu_ratio" in run or "cpu_over_gpu_ratio_including_coverage_router_setup" in run:
                raise GpuFirstBatchSummaryError(
                    f"{label} GPU-only reject diagnostic must omit speed ratios")
            if run.get("performance_comparison_valid") is not False:
                raise GpuFirstBatchSummaryError(
                    f"{label} GPU-only reject diagnostic must be marked invalid for performance")
            ratio = None

        if router_stages:
            cpu_setup = _number(run.get("cpu_coverage_router_setup_seconds"),
                                f"{label}.cpu_coverage_router_setup_seconds")
            gpu_setup = _number(run.get("gpu_coverage_router_setup_seconds"),
                                f"{label}.gpu_coverage_router_setup_seconds")
            _close(cpu_setup, router_setup["cpu"] if cold else 0.0,
                   f"{label} CPU router binding setup")
            _close(gpu_setup, router_setup["gpu"] if cold else 0.0,
                   f"{label} GPU router binding setup")
            cpu_including = _number(
                run.get("cpu_seconds_including_coverage_router_setup"),
                f"{label}.cpu_seconds_including_coverage_router_setup", positive=True)
            gpu_including = _number(
                run.get("gpu_seconds_including_coverage_router_setup"),
                f"{label}.gpu_seconds_including_coverage_router_setup", positive=True)
            _close(cpu_including, cpu_wall + cpu_setup,
                   f"{label} CPU time including router setup")
            _close(gpu_including, gpu_wall + gpu_setup,
                   f"{label} GPU time including router setup")
            if fallback == "exact":
                setup_ratio = _number(
                    run.get("cpu_over_gpu_ratio_including_coverage_router_setup"),
                    f"{label}.cpu_over_gpu_ratio_including_coverage_router_setup",
                    positive=True)
                _close(setup_ratio, cpu_including / gpu_including,
                       f"{label} setup-inclusive CPU/GPU ratio",
                       rel_tol=1e-9, abs_tol=1e-12)

        wall_totals["cpu"] += cpu_wall
        wall_totals["gpu"] += gpu_wall
        summaries.append({
            "repeat": repeat,
            "seeds": checked_seeds,
            "execution_order": expected_order,
            "cold_first_use_included": cold,
            "wall_seconds": {"cpu": cpu_wall, "gpu": gpu_wall},
            **({"cpu_over_gpu_ratio": ratio} if ratio is not None else {}),
            "endpoint_error_maxima": endpoint_maxima,
            "cpu_pipeline_blocks": cpu_blocks,
            "cpu_pipeline_cycle_seconds": cpu_cycles,
            "cpu_wall_outside_pipeline_cycles_seconds": cpu_outside,
            "gpu_first_blocks": gpu_blocks,
            "gpu_first_phase_seconds": gpu_phases,
            "gpu_first_stage_seconds": gpu_stage_seconds,
            "gpu_wall_outside_stage_records_seconds": gpu_outside,
            "cpu_reference_history_counts": cpu_counts,
            "gpu_history_counts": gpu_counts,
            "original_lp_certificate_maxima": {
                "cpu": cpu_certificates,
                "gpu": gpu_certificates,
            },
        })

    cpu_counts = _sum_counts(summaries, "cpu")
    gpu_counts = _sum_counts(summaries, "gpu")
    certificates = _max_certificates(summaries)
    overall = {
        "performance_comparison_valid": fallback == "exact",
        "wall_seconds": wall_totals,
        "endpoint_error_maxima": overall_endpoints,
        "cpu_reference_history_counts": cpu_counts,
        "gpu_history_counts": gpu_counts,
        "cpu_reference_total_lp_calls": sum(
            row["cpu_lp_calls"] for row in cpu_counts.values()),
        "cpu_reference_total_solver_runs": sum(
            row["cpu_solver_runs"] for row in cpu_counts.values()),
        "gpu_bank_accepts": sum(row["bank_accepts"] for row in gpu_counts.values()),
        "gpu_exact_cpu_lp_calls": sum(
            row["cpu_lp_calls"] for row in gpu_counts.values()),
        "gpu_exact_cpu_solver_runs": sum(
            row["cpu_solver_runs"] for row in gpu_counts.values()),
        "gpu_candidate_evaluations": sum(
            row["candidate_evaluations"] for row in gpu_counts.values()),
        "original_lp_certificate_maxima": certificates,
    }
    if fallback == "exact":
        overall["cpu_over_gpu_ratio"] = wall_totals["cpu"] / wall_totals["gpu"]
        overall["cpu_over_gpu_ratio_scope"] = (
            "Descriptive ratio of summed validated end-to-end wall times for a "
            "pipelined CPU driver and an all-stage barrier GPU-first driver; no "
            "isolated hardware speedup or statistical superiority is claimed.")
    else:
        overall["diagnostic_scope"] = (
            "GPU-only reject-on-uncertified mode; no CPU/GPU performance ratio is valid.")

    return {
        "status": ("validated_completed_gpu_first_batch" if fallback == "exact"
                   else "validated_completed_gpu_only_diagnostic"),
        "configuration": {
            **config,
            "gpu_stages": list(STAGES),
            "cpu_driver_kind": "pipeline",
            "gpu_driver_kind": "barrier",
            "block_size_steps": block_size,
        },
        "model_fingerprints": models,
        "bank_candidate_counts": candidate_counts,
        "coverage_router_bindings": router_bindings,
        "source_snapshot": source_snapshot,
        "count_semantics": {
            "cpu_reference_lp_calls": "One exact CPU request for every environment and original3 stage.",
            "gpu_bank_accepts": "Outputs passing the unchanged original-LP GPU certificate.",
            "gpu_exact_cpu_lp_calls": "Actually executed exact CPU fallbacks after GPU rejection.",
            "cpu_solver_runs": "Actual HiGHS run attempts, distinct from LP request count.",
            "gpu_candidate_evaluations": "Individual environment-candidate reconstructions (B x K).",
        },
        "timing_scope": {
            "cpu": "Only nonoverlapping CPU pipeline cycle fields are additive; asynchronous spans overlap.",
            "gpu": "gpu_first_phases are nonoverlapping within each stage record; stage records are serial barriers.",
            "comparison": "End-to-end scheduler comparison, not an isolated CPU-versus-GPU solver comparison.",
        },
        "validation_limits": {
            "endpoint_error_maximum": ENDPOINT_LIMIT,
            "original_lp_certificate": dict(CERTIFICATE_LIMITS),
            "timer_sum_relative_tolerance": 1e-6,
            "timer_sum_absolute_tolerance_seconds": 1e-6,
            "maximum_applied_stage_sum_vs_wall_tolerance_seconds": maximum_timing_tolerance,
        },
        "runs": summaries,
        "overall": overall,
    }


def summarize_gpu_first_batch_file(path, *, block_size=20):
    """Read a report and verify its sibling ``.sources`` snapshot."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return summarize_gpu_first_batch_report(
        report, source_directory=path.with_suffix(".sources"),
        block_size=block_size)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Strictly validate an asymmetric GPU-first batch benchmark JSON")
    parser.add_argument("report", type=Path)
    parser.add_argument("--block-size", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        summary = summarize_gpu_first_batch_file(
            args.report, block_size=args.block_size)
    except (OSError, json.JSONDecodeError, GpuFirstBatchSummaryError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
