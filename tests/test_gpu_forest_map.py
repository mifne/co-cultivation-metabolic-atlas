import threading

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_forest_map import GpuForestMap, _prepare_host_maps
from src.lp_equality_reduction import HomogeneousEqualityReduction
from src.lp_trace import problem_hash


def _problem(negative=False, unbounded=False, identity=False):
    return (csr_matrix([[1., 1. if negative else -1., 0.], [0., 0., 1.]]),
            np.array([1. if identity else 0., 1.]),
            np.array([-np.inf if unbounded else 0., -2. if negative else 0., 0.]),
            np.array([np.inf if unbounded else 2., 0. if negative else 2., 2.]),
            np.array([-1., 0. if negative else -1., 1.]), 1)


def _plans(problems):
    plans = [HomogeneousEqualityReduction.from_problem(p) for p in problems]
    return plans, [plan.reduce(p) for plan, p in zip(plans, problems)]


def _host_expand(host, z, y):
    """Independent NumPy execution of the assembled maps (never production)."""
    xx = (host.transform @ z.ravel()).reshape(host.batch, host.original_n)
    yy = np.zeros((host.batch, host.original_m))
    batch = np.arange(host.batch)[:, None]
    yy[batch, host.kept_rows] = y
    q = host.objective.ravel() - host.original_at @ yy.ravel()
    cost = (host.transform.T @ q).reshape(host.batch, host.reduced_n)
    witness = np.where(cost >= 0, host.lower_witness, host.upper_witness)
    witness = np.where(witness >= 0, witness, host.fallback_witness)
    normal = np.zeros_like(xx)
    normal[batch, witness] = cost / host.weights[batch, witness]
    if host.eliminated_rows.shape[1]:
        yy[batch, host.eliminated_rows] = (host.dual_lift @ (q-normal.ravel())).reshape(host.batch, -1)
    return xx, yy


@pytest.mark.parametrize('negative', [False, True])
@pytest.mark.parametrize('identity', [False, True])
def test_cpu_host_maps_match_independent_plan_postsolve(negative, identity):
    inputs = [_problem(identity=identity), _problem(negative, identity=identity)]
    plans, reductions = _plans(inputs)
    hashes = [problem_hash(p) for p in inputs]
    host = _prepare_host_maps(plans, reductions)
    rng = np.random.default_rng(391)
    z = rng.uniform(-.5, .5, (2, host.reduced_n))
    y = rng.uniform(-.5, .5, (2, host.reduced_m))
    xx, yy = _host_expand(host, z, y)
    for i, reduction in enumerate(reductions):
        cost = reduction.problem[4] - reduction.problem[0].T @ y[i]
        np.testing.assert_allclose(xx[i], reduction.expand_primal(z[i]), atol=1e-13)
        np.testing.assert_allclose(yy[i], reduction.lift_dual(y[i], cost), atol=1e-13)
    np.testing.assert_allclose((host.compression @ xx.ravel()).reshape(z.shape), z, atol=1e-13)
    assert hashes == [problem_hash(p) for p in inputs]


def test_cpu_setup_copies_arrays_and_rejects_corrupted_maps_before_cuda():
    plans, reductions = _plans([_problem()])
    host = _prepare_host_maps(plans, reductions)
    saved = host.transform.data.copy()
    plans[0].T.data[:] *= 2
    np.testing.assert_array_equal(host.transform.data, saved)
    with pytest.raises(ValueError, match='plan map'):
        GpuForestMap(plans, reductions, cp=object())
    for value in (host.transform.data, host.compression.indptr, host.objective, host.weights):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 19


@pytest.mark.parametrize('field', ['lower_witness', 'upper_witness', 'fallback_witness'])
def test_cpu_invalid_witness_rejected_before_cuda(field):
    plans, reductions = _plans([_problem()])
    getattr(reductions[0], field)[0] = 99
    with pytest.raises(ValueError, match='witness'):
        GpuForestMap(plans, reductions, cp=object())


def test_cpu_dynamic_objective_and_bounds_not_stale_plan_data():
    p = _problem()
    plans, _ = _plans([p])
    updated = (p[0], p[1], p[2], np.array([1., 2., 3.]), np.array([2., 3., 5.]), p[-1])
    reduction = plans[0].reduce(updated)
    host = _prepare_host_maps(plans, [reduction])
    np.testing.assert_array_equal(host.objective[0], updated[4])
    assert host.upper_witness[0, 0] == 0
    reduction.problem[4][0] += 1
    with pytest.raises(ValueError, match='Reduced LP'):
        GpuForestMap(plans, [reduction], cp=object())


def test_cpu_batch_and_plan_pairing_guards_before_cuda():
    plans, reductions = _plans([_problem(), _problem(True)])
    for a, b in (([], []), (plans, reductions[:1]), (plans, reductions[::-1])):
        with pytest.raises(ValueError):
            GpuForestMap(a, b, cp=object())
    plans, reductions = _plans([_problem(), _problem(identity=True)])
    with pytest.raises(ValueError, match='Regroup'):
        GpuForestMap(plans, reductions, cp=object())


def test_cpu_missing_endpoints_use_fallback_witness():
    p = (csr_matrix([[1., 1., 0.], [0., 0., 1.]]), np.array([0., 1.]),
         np.array([-np.inf, -np.inf, 0.]), np.array([np.inf, np.inf, 2.]),
         np.array([-1., 0., 1.]), 1)
    plans, reductions = _plans([p])
    host = _prepare_host_maps(plans, reductions)
    assert host.lower_witness[0, 0] == host.upper_witness[0, 0] == -1
    z, y = np.array([[2., 0.]]), np.zeros((1, 1))
    xx, yy = _host_expand(host, z, y)
    np.testing.assert_allclose(yy[0], reductions[0].lift_dual(y[0], reductions[0].problem[4]))
    np.testing.assert_allclose(xx[0], reductions[0].expand_primal(z[0]))


def test_gpu_maps_match_cpu_and_own_output():
    import cupy as cp
    plans, reductions = _plans([_problem(), _problem(True)])
    mapper = GpuForestMap(plans, reductions)
    z = cp.asarray([[2., 0.], [2., 0.]], dtype=cp.float64)
    y = cp.zeros((2, 1), dtype=cp.float64)
    xx, yy = mapper.expand(z, y)
    expected = _host_expand(mapper._host, z.get(), y.get())
    np.testing.assert_allclose(xx.get(), expected[0], atol=1e-13)
    np.testing.assert_allclose(yy.get(), expected[1], atol=1e-13)
    zz, kept_y = mapper.compress(xx, yy)
    np.testing.assert_allclose(zz.get(), z.get(), atol=1e-13)
    np.testing.assert_allclose(kept_y.get(), y.get(), atol=1e-13)
    xx[:] = 19
    yy[:] = 19
    np.testing.assert_allclose(zz.get(), z.get(), atol=1e-13)
    np.testing.assert_allclose(kept_y.get(), y.get(), atol=1e-13)
    plans[0].T.data[:] = 100
    new_x, _ = mapper.expand(z, y)
    np.testing.assert_allclose(new_x.get(), expected[0], atol=1e-13)


def test_gpu_shape_dtype_and_stream_guards_and_nonfinite_propagation():
    import cupy as cp
    plans, reductions = _plans([_problem()])
    mapper = GpuForestMap(plans, reductions, cp=cp)
    z, y = cp.ones((1, 2), dtype=cp.float64), cp.zeros((1, 1), dtype=cp.float64)
    for bad in (z[0], z.astype(cp.float32), z.get()):
        with pytest.raises(ValueError):
            mapper.expand(bad, y)
    with cp.cuda.Stream(non_blocking=True):
        with pytest.raises(RuntimeError, match='stream'):
            mapper.expand(z, y)
    old_thread = mapper.thread
    mapper.thread = threading.get_ident() + 1
    with pytest.raises(RuntimeError, match='thread'):
        mapper.expand(z, y)
    mapper.thread = old_thread
    z[0, 0] = cp.nan
    xx, _ = mapper.expand(z, y)
    assert not np.isfinite(xx.get()).all()


def test_gpu_identity_forest_and_no_retained_rows():
    import cupy as cp
    for p in (_problem(identity=True),
              (csr_matrix([[1., -1.]]), np.zeros(1), np.zeros(2),
               np.ones(2), np.array([-1., -1.]), 1)):
        plans, reductions = _plans([p])
        mapper = GpuForestMap(plans, reductions)
        z = cp.ones((1, mapper.reduced_n), dtype=cp.float64)
        y = cp.zeros((1, mapper.reduced_m), dtype=cp.float64)
        xx, yy = mapper.expand(z, y)
        expected = _host_expand(mapper._host, z.get(), y.get())
        np.testing.assert_allclose(xx.get(), expected[0], atol=1e-13)
        np.testing.assert_allclose(yy.get(), expected[1], atol=1e-13)
        back_z, back_y = mapper.compress(xx, yy)
        np.testing.assert_allclose(back_z.get(), z.get(), atol=1e-13)
        assert back_y.shape == y.shape
