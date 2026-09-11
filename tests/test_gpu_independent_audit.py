"""Synthetic audit fixtures only; no GPU scientific result is manufactured."""
import copy
import hashlib
import json
import pytest
from scripts.audit_gpu_qualification import audit, EXPECTED_SEEDS


@pytest.fixture
def archive(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("# Synthetic unit-test source\n")
    hashes = {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    config = dict(steps=120, method="tableau", host_presolve=True, rank_reduce=False,
        objective_scale=1., quadratic_regularization=0., reaction_support=None,
        tolerance=1e-8, time_limit=600.)
    trace = [dict(step=i, accepted=True, status="optimal;stage=parsimonious_exchange",
        pha=1., phv_fraction=.5, nh4=.05, biomass=dict(a=1., b=1., c=1.)) for i in range(120)]
    cpu = dict(seconds=1., steps=120, trajectory=trace, cpu_lp_stage_calls=360,
        gpu_lp_stage_calls=0, model_fingerprints=dict(a="a", b="b", c="c"))
    gpu = copy.deepcopy(cpu)
    gpu.update(cpu_lp_stage_calls=0, gpu_lp_stage_calls=360)
    history = dict(max_original_residual=1e-10, success=True, cpu_optimization_allowed=False,
        cpu_iteration_counts=dict(simplex_iteration_count=-1), total_seconds=.002,
        host_presolve_seconds=.0001, gpu=dict(success=True, status="optimal", cpu_lp_calls=0,
        total_seconds=.001, host_qr_crash_seconds=.0001, phase_iterations=[10, 2]))
    gpu["backend_history"] = [copy.deepcopy(history) for _ in range(360)]
    manifest = dict(status="five_seed_accuracy_pass", seeds=EXPECTED_SEEDS,
        implementation_sha256=hashes, supporting_source_sha256=hashes, config=config,
        gates=dict(pha_relative=.01, biomass_g_l=.01, phv_mole_fraction=.01,
                   steps=120, cpu_lp_calls=0, original_lp_feasibility_residual=1e-5), runs=[])
    for seed in EXPECTED_SEEDS:
        path = tmp_path / f"seed_{seed}.json"
        report = dict(implementation_sha256=hashes, config=config, runs=[dict(seed=seed,
            exact=cpu, gpu=gpu, errors=dict(pha_relative=0., biomass_max=0., phv_fraction=0.),
            error_scope="full_horizon_endpoint", passed=True)])
        path.write_text(json.dumps(report))
        manifest["runs"].append(dict(seed=seed, report=path.name,
            report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), passed=True))
    index = tmp_path / "qualification.json"
    index.write_text(json.dumps(manifest))
    return tmp_path, index


def test_synthetic_valid_archive(archive):
    root, index = archive
    result = audit(index, root)
    assert result["passed"]
    assert sum(r["gpu_logical_lp_stages"] for r in result["runs"]) == 1800


@pytest.mark.parametrize("mutation", ["endpoint", "residual", "cpu", "partial", "nan", "configuration"])
def test_rejects_even_if_pass_flag_and_hash_updated(archive, mutation):
    root, index = archive
    manifest = json.loads(index.read_text())
    path = root / manifest["runs"][0]["report"]
    report = json.loads(path.read_text())
    gpu = report["runs"][0]["gpu"]
    if mutation == "endpoint": gpu["trajectory"][-1]["biomass"]["a"] = 1.03
    elif mutation == "residual": gpu["backend_history"][0]["max_original_residual"] = 2e-5
    elif mutation == "cpu": gpu["backend_history"][0]["cpu_iteration_counts"]["simplex_iteration_count"] = 1
    elif mutation == "partial": gpu["trajectory"].pop()
    elif mutation == "nan": gpu["trajectory"][9]["pha"] = float("nan")
    else: report["config"]["tolerance"] = 1e-3
    path.write_text(json.dumps(report))
    manifest["runs"][0]["report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    index.write_text(json.dumps(manifest))
    result = audit(index, root)
    assert not result["passed"]
    assert not result["runs"][0]["passed"]


def test_requires_all_five_unique_seeds(archive):
    root, index = archive
    manifest = json.loads(index.read_text())
    manifest["runs"] = manifest["runs"][:4]
    index.write_text(json.dumps(manifest))
    assert not audit(index, root)["passed"]


def test_source_integrity(archive):
    root, index = archive
    (root / "source.py").write_text("# Changed synthetic unit-test source\n")
    result = audit(index, root)
    assert not result["passed"]
    assert any("source differs" in item for item in result["problems"])
