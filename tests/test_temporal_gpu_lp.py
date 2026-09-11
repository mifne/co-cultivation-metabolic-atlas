"""Light CUDA integration tests for temporal warm starts and GPU correction."""

import hashlib

import numpy as np
import pytest
import torch

from src.temporal_gpu_lp import TemporalGpuLPBackend
from src.temporal_lp_data import TemporalLPCodec
from src.temporal_lp_model import (
    TemporalLPConfig,
    TemporalLPModel,
    temporal_checkpoint,
)


def _require_cuda():
    cp = pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is unavailable")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime is unavailable")
    return cp


def _codec(stage="maxmin"):
    # One primal and one zero inequality row.  The exact optimum of the test
    # LP is not encoded in this fixture; cold PDHG must move x from 0 to 1.
    arrays = {
        "matrix_keys": np.empty(0, dtype=np.int64),
        "feature_indices": np.array([0], dtype=np.int64),
        "feature_mean": np.zeros(1, dtype=np.float64),
        "feature_scale": np.ones(1, dtype=np.float64),
        "target_mean": np.zeros(2, dtype=np.float64),
        "target_scale": np.ones(2, dtype=np.float64),
        "components": np.eye(2, dtype=np.float64),
        "latent_scale": np.ones(2, dtype=np.float64),
    }
    return TemporalLPCodec(
        arrays,
        {
            "stage": stage,
            "dimensions": [1, 1, 0],
            "fixture": "one_variable_zero_row_v1",
        },
    )


def _write_artifact(path, *, kind="gru", codec=None, checkpoint_sha=None,
                    feature_dim=None, latent_dim=None):
    path.mkdir()
    codec = _codec() if codec is None else codec
    codec_path = path / "codec.npz"
    codec.save(codec_path)
    digest = hashlib.sha256(codec_path.read_bytes()).hexdigest()
    config = TemporalLPConfig(
        kind=kind,
        feature_dim=codec.feature_dim if feature_dim is None else feature_dim,
        latent_dim=codec.rank if latent_dim is None else latent_dim,
        hidden_dim=4,
        num_layers=1,
        residual=True,
    )
    torch.manual_seed(941)
    model = TemporalLPModel(config).eval()
    payload = temporal_checkpoint(
        model,
        metadata={"codec_sha256": digest if checkpoint_sha is None else checkpoint_sha},
    )
    torch.save(payload, path / f"{kind}.pt")
    return path


def _request(*, upper=1.0, stage="maxmin"):
    return np.array([-1.0]), {
        "A_ub": np.zeros((1, 1), dtype=float),
        "b_ub": np.zeros(1, dtype=float),
        "bounds": [(0.0, upper)],
        "method": "highs-ds",
        "_stage": stage,
    }


def test_checkpoint_codec_hash_and_architecture_mismatches_are_rejected(tmp_path):
    _require_cuda()
    wrong_hash = _write_artifact(
        tmp_path / "wrong_hash", checkpoint_sha="0" * 64
    )
    with pytest.raises(ValueError, match="Checkpoint/codec mismatch"):
        TemporalGpuLPBackend(wrong_hash, mode="gru")

    wrong_architecture = _write_artifact(
        tmp_path / "wrong_architecture", feature_dim=_codec().feature_dim + 1
    )
    with pytest.raises(ValueError, match="Model architecture/codec mismatch"):
        TemporalGpuLPBackend(wrong_architecture, mode="gru")


def test_stage_and_shape_mismatches_fail_before_correction(tmp_path):
    _require_cuda()
    artifact = _write_artifact(tmp_path / "artifact")
    backend = TemporalGpuLPBackend(
        artifact, mode="cold", iterations=1, check_interval=1
    )
    with pytest.raises(ValueError, match="separate model"):
        backend.solve_batch([_request(stage="aggregate")])
    with pytest.raises(ValueError, match="shape/phase"):
        backend.solve_batch(
            [(np.array([-1.0, 0.0]), {"bounds": [(0.0, 1.0)] * 2,
                                      "_stage": "maxmin"})]
        )
    assert backend.history == []
    assert backend.last_features == {}
    assert backend.last_certified == {}


def test_gpu_correction_succeeds_when_cpu_optimizers_are_forbidden(
    tmp_path, monkeypatch
):
    _require_cuda()
    artifact = _write_artifact(tmp_path / "artifact")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU optimizer was called")

    import highspy
    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "linprog", forbidden)
    monkeypatch.setattr(highspy.Highs, "run", forbidden)
    backend = TemporalGpuLPBackend(
        artifact, mode="cold", iterations=2, check_interval=1
    )
    result = backend.solve_batch([_request()], environment_ids=[17])[0]
    assert result.success
    np.testing.assert_allclose(result.x, [1.0], rtol=0.0, atol=1e-12)
    assert result.fun == pytest.approx(-1.0, abs=1e-12)
    assert result.diagnostics["certificate_passed"]
    assert result.diagnostics["accepted_iteration"] == 1
    assert result.diagnostics["cpu_lp_calls"] == 0
    assert backend.history[-1]["cpu_lp_calls"] == 0
    assert backend.history[-1]["iterations_run"] == 1


def test_failed_correction_clears_previous_certified_cache(tmp_path):
    _require_cuda()
    artifact = _write_artifact(tmp_path / "artifact")
    backend = TemporalGpuLPBackend(
        artifact, mode="previous", iterations=1, check_interval=1
    )
    accepted = backend.solve_batch([_request()], environment_ids=[23])[0]
    assert accepted.success and 23 in backend.last_certified

    # Removing the optimizing upper bound makes min(-x) unbounded.  A finite
    # previous iterate is not a certificate, and the failed value must replace
    # neither the cache nor the original-LP acceptance gate.
    rejected = backend.solve_batch(
        [_request(upper=None)], environment_ids=[23]
    )[0]
    assert not rejected.success and rejected.x is None and rejected.fun is None
    assert not rejected.diagnostics["certificate_passed"]
    assert rejected.diagnostics["accepted_iteration"] == -1
    assert 23 not in backend.last_certified
    assert 23 in backend.last_features
    assert backend.history[-1]["accepted"] == 0


def test_selective_and_global_reset_clear_all_environment_state(tmp_path):
    _require_cuda()
    artifact = _write_artifact(tmp_path / "artifact")
    backend = TemporalGpuLPBackend(
        artifact, mode="gru", iterations=2, check_interval=1
    )
    rows = backend.solve_batch(
        [_request(), _request()], environment_ids=[11, 29]
    )
    assert all(row.success for row in rows)
    assert set(backend.last_certified) == {11, 29}
    assert set(backend.last_features) == {11, 29}
    assert set(backend.session.environment_ids) == {11, 29}

    # A one-use iterator must reach both host caches and the GRU session.
    backend.reset(env for env in [11, "absent"])
    assert set(backend.last_certified) == {29}
    assert set(backend.last_features) == {29}
    assert set(backend.session.environment_ids) == {29}

    backend.reset()
    assert backend.last_certified == {}
    assert backend.last_features == {}
    assert backend.session.environment_ids == ()


def test_correction_exception_does_not_commit_recurrent_history(tmp_path, monkeypatch):
    _require_cuda()
    from src.gpu_pdhg_corrector import GpuPdhgCorrector
    artifact = _write_artifact(tmp_path / "artifact")
    backend = TemporalGpuLPBackend(artifact,mode="gru",iterations=2,check_interval=1)
    original = GpuPdhgCorrector.solve
    def fail(*_args, **_kwargs):
        raise RuntimeError('simulated GPU correction failure')
    monkeypatch.setattr(GpuPdhgCorrector,'solve',fail)
    with pytest.raises(RuntimeError,match='simulated'):
        backend.solve_batch([_request()],environment_ids=[3])
    assert backend.session.environment_ids == ()
    assert backend.last_certified == backend.last_features == {}
    assert backend.history == []
    monkeypatch.setattr(GpuPdhgCorrector,'solve',original)
    result = backend.solve_batch([_request()],environment_ids=[3])[0]
    assert result.success
    assert backend.session.environment_ids == (3,)


def test_equality_reduction_opt_in_and_plan_reuse(tmp_path,monkeypatch):
    _require_cuda()
    import highspy
    import scipy.optimize
    def forbidden(*_args,**_kwargs):
        raise AssertionError('No CPU optimizer allowed in GPU service')
    monkeypatch.setattr(highspy.Highs,'run',forbidden)
    monkeypatch.setattr(scipy.optimize,'linprog',forbidden)
    artifact = _write_artifact(tmp_path / 'artifact')
    backend = TemporalGpuLPBackend(artifact,mode='gru',iterations=2,check_interval=1,
                                  equality_reduction=True)
    assert backend.solve_batch([_request()])[0].success
    plan = backend.reduction_plan
    assert plan is not None
    assert backend.solve_batch([_request(upper=.5)])[0].success
    assert backend.reduction_plan is plan
    assert backend.history[-1]['equality_reduction']
    assert backend.history[-1]['setup']['equality_plan_build_seconds'] == 0.


@pytest.mark.parametrize('mode',['cold','previous','mean'])
def test_nonneural_modes_do_not_evaluate_latent_projection(tmp_path,mode,monkeypatch):
    _require_cuda()
    artifact = _write_artifact(tmp_path / 'artifact')
    backend = TemporalGpuLPBackend(artifact,mode=mode,iterations=2,check_interval=1)
    # These baselines must not pay for an unused dense neural encoder.
    def forbidden(*_args,**_kwargs):
        raise AssertionError('Unused neural matrix product in a nonneural baseline')
    monkeypatch.setattr(torch.Tensor,'__matmul__',forbidden)
    assert backend.solve_batch([_request()])[0].success


@pytest.mark.parametrize("execution", ["graph", "loop"])
def test_resident_backend_reuses_service_and_rebuilds_after_id_or_pattern_change(tmp_path, execution):
    _require_cuda()
    artifact = _write_artifact(tmp_path / "artifact")
    backend = TemporalGpuLPBackend(artifact, mode="cold", iterations=8, check_interval=4,
                                  equality_reduction=True, resident_execution=execution, graph_chunk=4)
    first = backend.solve_batch([_request(), _request()], environment_ids=[11, 29])
    assert all(row.success for row in first)
    service = backend._resident_corrector
    graph = service.workspace.graph
    second = backend.solve_batch([_request(upper=.5), _request(upper=.7)], environment_ids=[11, 29])
    assert all(row.success for row in second)
    assert backend._resident_corrector is service
    assert service.workspace.graph is graph
    assert backend.history[-1]["setup"]["reused"]
    np.testing.assert_allclose([row.x[0] for row in second], [.5, .7])

    reordered = backend.solve_batch([_request(), _request()], environment_ids=[29, 11])
    assert all(row.success for row in reordered)
    assert backend._resident_corrector is not service
    assert backend.history[-1]["setup"]["rebuild_reason"] == "batch_or_environment_order_changed"
    service = backend._resident_corrector
    changed = _request()
    changed[1]["A_ub"] = np.ones((1, 1))
    changed[1]["b_ub"] = np.array([10.])
    rows = backend.solve_batch([changed, changed], environment_ids=[29, 11])
    assert all(row.success for row in rows)
    assert backend._resident_corrector is not service
    assert backend.history[-1]["setup"]["rebuild_reason"] == "original_csr_pattern_changed"
    assert all(row.diagnostics["cpu_lp_calls"] == 0 for row in rows)


def test_resident_gpu_exception_does_not_commit_gru_state_or_caches(tmp_path, monkeypatch):
    _require_cuda()
    from src.gpu_resident_reduced_pdhg import GpuResidentReducedPdhg
    artifact = _write_artifact(tmp_path / "artifact")
    backend = TemporalGpuLPBackend(artifact, mode="gru", iterations=4, check_interval=4,
                                  equality_reduction=True, resident_execution="graph", graph_chunk=4)
    solve = GpuResidentReducedPdhg.solve
    def fail(*args, **kwargs):
        raise RuntimeError("resident GPU correction interrupted")
    monkeypatch.setattr(GpuResidentReducedPdhg, "solve", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        backend.solve_batch([_request()], environment_ids=[7])
    assert backend.session.environment_ids == ()
    assert backend.last_features == backend.last_certified == {}
    assert backend.history == []
    monkeypatch.setattr(GpuResidentReducedPdhg, "solve", solve)
    assert backend.solve_batch([_request()], environment_ids=[7])[0].success
    assert backend.session.environment_ids == (7,)


def test_resident_failed_new_problem_clears_previous_certified_cache(tmp_path):
    _require_cuda()
    artifact = _write_artifact(tmp_path / "artifact")
    backend = TemporalGpuLPBackend(artifact, mode="previous", iterations=4, check_interval=4,
                                  equality_reduction=True, resident_execution="graph", graph_chunk=4)
    assert backend.solve_batch([_request()], environment_ids=[8])[0].success
    service = backend._resident_corrector
    row = backend.solve_batch([_request(upper=None)], environment_ids=[8])[0]
    assert backend._resident_corrector is service
    assert not row.success
    assert row.x is None and row.fun is None
    assert row.diagnostics["accepted_iteration"] == -1
    assert 8 not in backend.last_certified


@pytest.mark.parametrize("kwargs", [
    {"resident_execution": "graph"},
    {"resident_execution": "bad", "equality_reduction": True},
    {"resident_execution": "graph", "equality_reduction": True, "graph_chunk": 0},
])
def test_resident_backend_configuration_is_explicit_and_validated(tmp_path, kwargs):
    _require_cuda()
    artifact = _write_artifact(tmp_path / "artifact")
    with pytest.raises(ValueError):
        TemporalGpuLPBackend(artifact, mode="cold", **kwargs)
