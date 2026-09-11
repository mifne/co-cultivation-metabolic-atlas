from types import SimpleNamespace
import numpy as np
import pytest
from src.cpu_repeated_lp import _problem, _certificate
from src.gpu_block_lp import assemble_blocks, certify_blocks_device, GpuBlockLP, box_face_feasibility
from src.lp_trace import problem_request


def problems():
    return [_problem([-2., -1.], dict(A_ub=[[1., 1.]], b_ub=[limit],
            bounds=[(0., 3.), (0., 4.)])) for limit in (5., 4.)]


def gpu():
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('No CUDA GPU')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('No CUDA GPU')
    return cp


def test_block_independence():
    ps = problems(); a, rl, ru, lo, hi, c = assemble_blocks(ps)
    np.testing.assert_array_equal(a.toarray(), [[1., 1., 0., 0.], [0., 0., 1., 1.]])
    np.testing.assert_array_equal(ru, [5., 4.])
    assert np.isneginf(rl).all()
    np.testing.assert_array_equal(c, [-2., -1., -2., -1.])


def test_reject_incompatible_dimensions():
    with pytest.raises(ValueError):
        assemble_blocks([])
    with pytest.raises(ValueError):
        assemble_blocks([*problems(), _problem([1.], {})])


def test_gpu_cert_matches_original_per_block():
    cp = gpu(); ps = problems(); packed = assemble_blocks(ps)
    x, y = np.array([3., 2., 3., 1.]), np.array([-1., -1.])
    rows = certify_blocks_device(ps, packed, x, y, cp=cp)
    for i, p in enumerate(ps):
        xx, yy = x[2*i:2*i+2], y[i:i+1]
        ref = _certificate(*p, SimpleNamespace(col_value=xx, row_dual=yy,
            col_dual=p[4]-p[0].T@yy, value_valid=True, dual_valid=True))
        for k in ('primal_residual','dual_violation','relative_kkt_gap','certificate_passed'):
            assert rows[i][k] == ref[k]
    assert all(row['certificate_passed'] for row in rows)
    x[2] += 1e-3
    changed = certify_blocks_device(ps, packed, x, y, cp=cp)
    assert changed[0]['certificate_passed'] and not changed[1]['certificate_passed']


def test_block_cert_does_not_hide_small_objective_error():
    cp = gpu()
    ps = [_problem([scale], dict(bounds=[(0., 1.)])) for scale in (1e10, 1.)]
    rows = certify_blocks_device(ps, assemble_blocks(ps), np.array([0., 1e-3]), np.empty(0), cp=cp)
    assert rows[0]['certificate_passed'] and not rows[1]['certificate_passed']


def test_nonfinite_rejected():
    cp = gpu(); ps = problems()
    rows = certify_blocks_device(ps, assemble_blocks(ps), np.full(4, np.nan), np.full(2, -1.), cp=cp)
    assert not any(r['certificate_passed'] for r in rows)


def test_actual_cuopt_blocks():
    gpu(); pytest.importorskip('cuopt')
    backend = GpuBlockLP(time_limit=5.)
    results = backend.solve_batch([problem_request(p) for p in problems()])
    assert all(r.success for r in results), backend.history
    np.testing.assert_allclose(results[0].x, [3., 2.], atol=1e-6)
    np.testing.assert_allclose(results[1].x, [3., 1.], atol=1e-6)
    assert backend.history[-1]['cpu_lp_calls'] == 0


def test_public_gpu_entry_validates_before_solve():
    gpu(); pytest.importorskip('cuopt')
    backend = GpuBlockLP(time_limit=5.)
    with pytest.raises(ValueError):
        backend.solve_batch([(np.array([np.nan]), {})])
    assert not backend.history


def test_box_face_keeps_original_inputs_unchanged():
    p = _problem([-1.], dict(A_ub=[[1.]], b_ub=[10.], bounds=[(0., 3.)]))
    q = box_face_feasibility(p)
    assert p[2][0] == 0. and p[4][0] == -1.
    assert q[2][0] == q[3][0] == 3. and q[4][0] == 0.
    assert q[0] is p[0] and q[1] is p[1]
    with pytest.raises(ValueError):
        box_face_feasibility(problems()[0])


def test_box_face_gpu_certificate_and_infeasible_proposal():
    gpu(); pytest.importorskip('cuopt')
    backend = GpuBlockLP(time_limit=5., tolerance=1e-6, box_face=True, box_dual_certificate=True, presolve=0)
    good = _problem([-1.], dict(A_ub=[[1.]], b_ub=[10.], bounds=[(0., 3.)]))
    result = backend.solve_batch([problem_request(good)])[0]
    assert result.success and abs(result.fun+3.) <= 1e-7
    # The original LP is feasible with optimum x=2, but the guessed box
    # face x=3 is infeasible. Do not return a solution or call it optimal.
    bad_face = _problem([-1.], dict(A_ub=[[1.]], b_ub=[2.], bounds=[(0., 3.)]))
    assert not backend.solve_batch([problem_request(bad_face)])[0].success


def test_box_certificate_uses_valid_dual_not_noisy_solver_dual():
    cp = gpu()
    p = _problem([-1.], dict(A_ub=[[1.]], b_ub=[10.], bounds=[(0., .005)]))
    raw = certify_blocks_device([p], assemble_blocks([p]), np.array([.005]), np.array([1.]), cp=cp)
    assert not raw[0]['certificate_passed']
    rows = certify_blocks_device([p], assemble_blocks([p]), np.array([.005]), np.array([1.]), cp=cp,
                                allow_box_dual=True)
    assert rows[0]['certificate_passed']
    assert rows[0]['certificate_source'] == 'analytic_box_bound'
    original = _certificate(*p, SimpleNamespace(col_value=[.005], row_dual=[0.],
        col_dual=[-1.], value_valid=True, dual_valid=True))
    assert original['certificate_passed']


@pytest.mark.parametrize('rhs,x,upper', [(0., .005, .005), (10., .003, .005), (10., .005, None)])
def test_box_certificate_cannot_hide_primal_gap_or_unboundedness(rhs, x, upper):
    cp = gpu()
    p = _problem([-1.], dict(A_ub=[[1.]], b_ub=[rhs], bounds=[(0., upper)]))
    rows = certify_blocks_device([p], assemble_blocks([p]), np.array([x]), np.array([0.]), cp=cp,
                                allow_box_dual=True)
    assert not rows[0]['certificate_passed']
