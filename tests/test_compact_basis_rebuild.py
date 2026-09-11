"""Tiny CPU-only rebuild tests; no actual 480-row training job or GPU use."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import src.compact_basis_rebuild as module
from src.compact_training_data import CompactTrainingSource


def _source(stage='maxmin'):
    # Two trajectories with two steps. Normalized x+y<=rhs; changing cost
    # changes which variable is basic. Scaling is deliberately not identity.
    data = dict(rhs=np.array([[1.], [2.], [1.5], [2.5]]),
        lower=np.zeros((4, 2)), upper=np.full((4, 2), 10.),
        c=np.array([[-1., -.25], [-.25, -1.], [-1., -.25], [-.25, -1.]]),
        delta=np.zeros((4, 1, 2)), col_scale=np.array([[2., 3.], [4., 5.], [3., 2.], [5., 4.]]),
        row_scale=np.array([[5.], [6.], [4.], [3.]]))
    return CompactTrainingSource(root=dict(a=csr_matrix([[1., 1.]]), neq=0),
        variable_rows=np.array([0]), data=data, feature_indices=np.array([0]),
        feature_scale=np.ones(1), seeds=(101, 202), steps=2,
        bank_manifest=dict(status="completed"), stage_manifest=dict(stage=stage, entries=[]),
        provenance=dict(role="training_only_dictionary_reconstruction", model_fingerprints={"toy":"a"*64}))


@pytest.mark.parametrize('stage',['maxmin','aggregate','exchange'])
def test_collector_reuses_seed_models_in_step_order_and_never_constructs_inverse(monkeypatch,stage):
    source = _source(stage)
    calls, progress = [], []
    original = module.RepeatedCpuLP.solve_batch
    def tracked(self, requests, *, environment_ids):
        calls.append((list(environment_ids), [(c.copy(), copy.deepcopy(kwargs)) for c, kwargs in requests]))
        return original(self, requests, environment_ids=environment_ids)
    monkeypatch.setattr(module.RepeatedCpuLP, "solve_batch", tracked)
    import src.gpu_certified_basis as compiled
    def forbidden(*args, **kwargs):
        raise AssertionError("Full inverse/factor_compiled_basis must not be used")
    monkeypatch.setattr(compiled, "factor_compiled_basis", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    candidates, report = module.collect_basis_candidates(source, workers=2, progress=progress.append)
    assert report["status"] == "completed"
    assert report["offline_cpu_lp_calls"] == 4
    assert report["offline_cpu_solver_runs"] >= 4
    assert report["full_inverse_count"] == 0
    assert report["unique_new_statuses"] == 2
    assert report["duplicate_new_statuses"] == 2
    assert [entry[0] for entry in calls] == [[101, 202], [101, 202]]
    assert report['stage'] == stage
    assert all(kwargs['_stage'] == stage for _,requests in calls for _,kwargs in requests)
    assert [row["source_index"] for row in report["rows"]] == [0, 2, 1, 3]
    assert sorted(candidate["source_indices"] for candidate in candidates) == [[0, 2], [1, 3]]
    # The worker gets ORIGINAL cost/bounds/RHS, not normalized coordinates.
    np.testing.assert_array_equal(calls[0][1][0][0], [-2., -.75])
    np.testing.assert_allclose(calls[0][1][0][1]["b_ub"], [.2])
    assert all("inverse" not in candidate["anchor"] for candidate in candidates)
    assert all(row["original_certificate"]["certificate_passed"] for row in report["rows"])
    assert report["solver_options"]["threads"] == 1
    assert report["solver_options"]["solver"] == "simplex"
    assert "presolve" in report["effective_initial_options"]
    assert report["highspy_version"]
    assert [event["completed_rows"] for event in progress] == [2, 4]
    assert not report["gpu_projection_certified"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('stage',['maxmin','aggregate','exchange'])
def test_project_one_candidate_matches_its_own_original_lp_without_full_inverse(monkeypatch,stage):
    source = _source(stage)
    candidates, _ = module.collect_basis_candidates(source, workers=2)
    def forbidden(*args, **kwargs):
        raise AssertionError("No complete inverse")
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    from src.gpu_compact_basis import CompactEvaluator
    bank = SimpleNamespace(cp=np, a=source.root["a"], at=source.root["a"].T.tocsr(),
        var=source.variable_rows, neq=source.root["neq"], math=None)
    for candidate in candidates:
        arrays, record = module.project_basis_candidate(candidate, source)
        assert "inverse" not in arrays
        assert not record["full_inverse_constructed"]
        assert not record["gpu_projection_certified"]
        assert record['stage'] == stage
        # NumPy tiny-operator test only: the real GPU self-row/480-state gate
        # is deliberately still outstanding and must be run by the builder.
        evaluator = CompactEvaluator(bank, arrays)
        index = candidate["source_index"]
        inputs = {key:value[index:index+1] for key, value in source.data.items()}
        result = evaluator.solve_device(**inputs)
        assert result["accepted"].tolist() == [True]
        np.testing.assert_allclose(result["values"][0], candidate["anchor"]["cpu_anchor_values"], atol=1e-12)


def test_new_full_status_signatures_preserve_row_status_and_dimensions():
    assert module.basis_status_signature(np.array([1, 0]), np.array([0])) != module.basis_status_signature(np.array([1, 0]), np.array([2]))
    assert module.basis_status_signature(np.array([1]), np.array([0, 1])) != module.basis_status_signature(np.array([1, 0]), np.array([1]))
    with pytest.raises(ValueError):
        module.basis_status_signature(np.array([1.]), np.array([0]))


def test_existing_compact_files_are_mandatory_byte_identical_not_status_deduplicated(tmp_path):
    source = _source()
    base, target = tmp_path/"base", tmp_path/"new"
    base.mkdir()
    entries = []
    # Identical bytes are intentionally kept twice; old missing row_kind is
    # never guessed in order to deduplicate against new full statuses.
    for index in range(2):
        path = base/f"basis_{index:04d}.npz"
        path.write_bytes(b"old compact bytes without row statuses")
        entries.append(dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    source.stage_manifest["entries"] = entries
    copied = module.copy_mandatory_entries(source, base, target)
    assert len(copied) == 2
    for original, row in zip(entries, copied):
        assert row["mandatory"] and row["byte_identical_base"]
        assert row["sha256"] == original["sha256"]
        assert (base/row["filename"]).read_bytes() == (target/row["filename"]).read_bytes()
    with pytest.raises(FileExistsError):
        module.copy_mandatory_entries(source, base, target)


def test_bad_mandatory_hash_is_rejected_before_any_output_file(tmp_path):
    source = _source()
    base, target = tmp_path/"base", tmp_path/"out"
    base.mkdir()
    (base/"basis.npz").write_bytes(b"changed")
    source.stage_manifest["entries"] = [dict(filename="basis.npz", sha256="0"*64)]
    with pytest.raises(ValueError, match="SHA"):
        module.copy_mandatory_entries(source, base, target)
    assert not target.exists()


def test_uncertified_new_cpu_result_aborts_without_projecting_and_closes_workers(monkeypatch):
    source = _source()
    original = module.RepeatedCpuLP.solve_batch
    instances = []
    def failed(self, requests, *, environment_ids):
        instances.append(self)
        results = original(self, requests, environment_ids=environment_ids)
        results[0].success = False
        results[0].diagnostics.update(success=False, certificate_passed=False, primal_residual=np.inf)
        return results
    monkeypatch.setattr(module.RepeatedCpuLP, "solve_batch", failed)
    with pytest.raises(module.BasisRebuildError) as caught:
        module.collect_basis_candidates(source, workers=2)
    report = caught.value.report
    assert report["status"] == "failed"
    assert report["offline_cpu_lp_calls"] == 2  # both joined jobs really ran
    assert report["rows"][0]["diagnostics"]["primal_residual"] is None
    assert report["projected_candidates"] == 0
    assert all(instance.closed for instance in instances)
    json.dumps(report, allow_nan=False)


def test_diagnostic_scope_is_never_accepted_as_training():
    source = _source()
    source.provenance["role"] = "diagnostic"
    with pytest.raises(ValueError, match="training-only"):
        module.collect_basis_candidates(source)


def test_projection_rejects_status_mutation_or_missing_certificate():
    source = _source()
    candidates, _ = module.collect_basis_candidates(source, workers=2)
    candidate = candidates[0]
    candidate["signature"] = "0"*64
    with pytest.raises(ValueError, match="statuses changed"):
        module.project_basis_candidate(candidate, source)
    candidate["anchor"]["original_certificate"]["certificate_passed"] = False
    with pytest.raises(ValueError, match="newly certified"):
        module.project_basis_candidate(candidate, source)


@pytest.mark.parametrize('stage',[None,'exchange_tie','../maxmin',1])
def test_invalid_stage_cannot_reach_cpu(stage):
    with pytest.raises(ValueError,match='Stage must'):
        module.collect_basis_candidates(_source(stage))
    with pytest.raises(ValueError,match='Stage must'):
        module.original_lp_request(_source().original_problem(0),stage=stage)


def test_stage_provenance_disagreement_is_rejected():
    source=_source('aggregate')
    source.provenance['stage']='maxmin'
    with pytest.raises(ValueError,match='provenance'):
        module.collect_basis_candidates(source)


@pytest.mark.parametrize('stage',['aggregate','exchange'])
def test_downstream_different_shape_cost_and_dynamic_row_are_not_replaced(stage):
    source=_source(stage)
    source.root=dict(a=csr_matrix([[1.,1.,1.],[1.,2.,1.]]),neq=1)
    source.variable_rows=np.array([1])
    source.data=dict(rhs=np.tile([2.,3.],(4,1)),lower=np.zeros((4,3)),upper=np.full((4,3),10.),
        c=np.array([[1.,-1.,.1],[.1,1.,-1.],[1.,-1.,.1],[.1,1.,-1.]]),
        delta=np.zeros((4,1,3)),col_scale=np.tile([2.,3.,4.],(4,1)),row_scale=np.tile([3.,4.],(4,1)))
    source.data['delta'][:,0,1]=[0.,.1,.2,.3]
    source.stage_manifest['key']=[stage,2,3,1]
    source.provenance.update(stage=stage,stage_key=[stage,2,3,1])
    candidates,report=module.collect_basis_candidates(source,workers=2)
    assert report['offline_cpu_lp_calls']==4
    assert all(row['original_certificate']['certificate_passed'] for row in report['rows'])
    from src.gpu_compact_basis import CompactEvaluator
    bank=SimpleNamespace(cp=np,a=source.root['a'],at=source.root['a'].T.tocsr(),
        var=source.variable_rows,neq=1,math=None)
    for candidate in candidates:
        index=candidate['source_index'];anchor=candidate['anchor']
        assert anchor['lp'].a.shape==(2,3) and anchor['lp'].neq==1
        np.testing.assert_array_equal(anchor['lp'].c,source.data['c'][index])
        assert anchor['lp'].a[1,1]==2.+source.data['delta'][index,0,1]
        arrays,_=module.project_basis_candidate(candidate,source)
        answer=CompactEvaluator(bank,arrays).solve_device(**{key:value[index:index+1] for key,value in source.data.items()})
        assert answer['accepted'].tolist()==[True]
        np.testing.assert_allclose(answer['values'][0],anchor['cpu_anchor_values'],atol=1e-12)
