import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts.probe_downstream_gpu_coverage import (
    _json_safe, _load_problem_without_reference, hull_oracle,
    coefficient_range, original_candidates, paired_certificate,
    scale_reduced_problem, select_entries,
)
from src.compact_training_data import sha256_file
from src.lp_trace import problem_hash


def problem():
    return (csr_matrix([[1., 1.]]), np.array([1.]), np.zeros(2),
            np.ones(2), np.array([1., 0.]), 1)


def test_original_scaling_preserves_both_primal_and_dual_conventions():
    x, y = original_candidates(np.array([[6., 20.]]), np.array([[7., 11.]]),
                               np.array([2., 4.]), np.array([3., 5.]))
    np.testing.assert_array_equal(x, [[3., 5.]])
    np.testing.assert_array_equal(y, [[21., 55.]])


@pytest.mark.parametrize('scale', [np.array([0., 4.]), np.array([np.nan, 4.]), np.array([2.])])
def test_coordinate_guard_rejects_invalid_scale(scale):
    with pytest.raises(ValueError):
        original_candidates(np.ones((2, 2)), np.ones((2, 1)), scale, np.ones(1))


def test_convex_hull_can_repair_infeasible_candidates_and_fully_certify():
    proposed = np.array([[-1., 2.], [2., -1.]])
    dual = np.zeros((2, 1))
    assert not any(paired_certificate(problem(), x, y)['certificate_passed'] for x, y in zip(proposed, dual))
    outcome = hull_oracle(problem(), proposed, dual)
    assert outcome['solver_success']
    assert outcome['original_primal_feasible']
    assert outcome['full_original_certificate_passed']
    assert outcome['hull_membership_valid'] and outcome['hull_certified_solution_found']
    assert outcome['original_objective'] == pytest.approx(0.)
    assert outcome['alpha_sum'] == pytest.approx(1.)
    assert all(value >= 0 for value in outcome['alpha'])


def test_convex_failure_does_not_preclude_affine_solution():
    proposed = np.array([[2., -1.], [3., -2.]])
    dual = np.zeros((2, 1))
    convex = hull_oracle(problem(), proposed, dual)
    affine = hull_oracle(problem(), proposed, dual, affine=True)
    assert not convex['solver_success']
    assert affine['full_original_certificate_passed']
    assert min(affine['alpha']) < 0
    assert affine['alpha_l1'] > 1


def test_feasibility_and_solver_success_do_not_imply_optimality():
    outcome = hull_oracle(problem(), np.array([[.5, .5]]), np.zeros((1, 1)))
    assert outcome['solver_success'] and outcome['original_primal_feasible']
    assert not outcome['full_original_certificate_passed']
    assert outcome['best_complete_pair']['certificate']['relative_kkt_gap'] == pytest.approx(.5)


def test_bad_convex_weights_cannot_prove_hull_membership_even_for_certified_x(monkeypatch):
    monkeypatch.setattr('scripts.probe_downstream_gpu_coverage.linprog',
                        lambda *args, **kwargs: SimpleNamespace(success=True, status=0,
                            message='test', x=np.array([2.])))
    p = (csr_matrix([[1.]]), np.array([2.]), np.array([0.]),
         np.array([3.]), np.array([0.]), 1)
    outcome = hull_oracle(p, np.array([[1.]]), np.zeros((1, 1)))
    assert outcome['full_original_certificate_passed']
    assert not outcome['hull_membership_valid']
    assert not outcome['hull_certified_solution_found']


def test_invalid_family_and_nonfinite_candidates_are_excluded():
    outcome = hull_oracle(problem(), np.array([[0., 1.], [1., 0.], [np.nan, 0.]]),
                         np.zeros((3, 1)), candidate_ids=np.array([3, 5, 8]),
                         family_valid=np.array([False, True, True]))
    assert outcome['eligible_candidate_ids'] == [5]
    assert not outcome['full_original_certificate_passed']
    empty = hull_oracle(problem(), np.array([[0., 1.]]), np.zeros((1, 1)),
                        family_valid=np.array([False]))
    assert empty['status'] == 'no_finite_family_valid_candidates'
    assert 'solver_status' not in empty


def test_certificate_is_a_single_whole_pair_not_independent_minima():
    outcome = hull_oracle(problem(), np.array([[0., 1.], [0., 1.]]),
                         np.array([[-.2], [-.3]]))
    assert all(not row['certificate']['certificate_passed'] for row in outcome['paired_certificates'])
    assert not outcome['full_original_certificate_passed']
    assert outcome['best_complete_pair'] in outcome['paired_certificates']


def test_reconstructed_original_constraints_override_reduced_solver_success(monkeypatch):
    # A bogus solver-success flag with inconsistent weights must not pass.
    monkeypatch.setattr('scripts.probe_downstream_gpu_coverage.linprog',
                        lambda *args, **kwargs: SimpleNamespace(success=True, status=0,
                            message='test', x=np.array([0.])))
    outcome = hull_oracle(problem(), np.array([[0., 1.]]), np.zeros((1, 1)))
    assert outcome['solver_success']
    assert not outcome['original_primal_feasible']
    assert not outcome['full_original_certificate_passed']


def test_input_loader_does_not_load_reference_vectors(tmp_path):
    p = problem()
    a, rhs, lower, upper, cost, neq = p
    path = tmp_path/'input.npz'
    # Object arrays cannot be loaded with allow_pickle=False, but the input
    # loader must not index these deliberately unusable validation references.
    np.savez(path, a_data=a.data, a_indices=a.indices, a_indptr=a.indptr,
             a_shape=a.shape, rhs=rhs, lower=lower, upper=upper, c=cost, neq=neq,
             reference_x=np.array([object()], dtype=object),
             reference_y=np.array([object()], dtype=object))
    entry = dict(filename=path.name, sha256=sha256_file(path), problem_sha256=problem_hash(p),
                 rows=1, columns=2, nonzeros=2, equalities=1)
    restored = _load_problem_without_reference(tmp_path, entry)
    assert problem_hash(restored) == problem_hash(p)


def trace_manifest():
    return dict(status='completed', role='development_diagnostic_not_training',
                seeds=[10, 11], completed_steps=[8, 8], entries=[
                    dict(stage='aggregate', environment_id=0, step=step,
                         filename=f'lp_{step:03d}_1_000.npz') for step in [1, 2, 4, 8]])


def test_selects_requested_steps_without_flattening_trajectories():
    assert [e['step'] for e in select_entries(trace_manifest(), 'aggregate', 0, [1, 2, 4, 8])] == [1, 2, 4, 8]


@pytest.mark.parametrize('kind', ['role', 'duplicate', 'label', 'missing'])
def test_trace_selection_fails_closed(kind):
    manifest = trace_manifest()
    if kind == 'role':
        manifest['role'] = 'training_reference'
    elif kind == 'duplicate':
        manifest['entries'].append(manifest['entries'][0])
    elif kind == 'label':
        manifest['entries'][0]['filename'] = '../x.npz'
    else:
        manifest['entries'] = manifest['entries'][1:]
    with pytest.raises(ValueError):
        select_entries(manifest, 'aggregate', 0, [1, 2, 4, 8])


def test_nonfinite_json_encoding_never_changes_false_certificate_flag():
    safe = _json_safe(dict(metric=np.inf, certificate_passed=np.bool_(False), nested=[np.nan]))
    assert json.loads(json.dumps(safe, allow_nan=False)) == dict(metric=None, certificate_passed=False, nested=[None])


def test_runtime_failure_is_saved_and_registered_resources_are_closed(tmp_path, monkeypatch):
    from scripts.probe_downstream_gpu_coverage import main
    trace_dir, bank_dir = tmp_path/'trace', tmp_path/'bank'
    trace_dir.mkdir(); bank_dir.mkdir()
    trace = trace_manifest()
    trace['model_fingerprints'] = {'test': 'a'}
    trace['entries'] += [dict(stage=stage, environment_id=0, step=step,
                             filename=f'lp_{step:03d}_{index}_000.npz')
                         for stage, index in [('maxmin', 0), ('exchange', 2)]
                         for step in [1, 2, 4, 8]]
    (trace_dir/'manifest.json').write_text(json.dumps(trace))
    (bank_dir/'manifest.json').write_text(json.dumps(dict(status='completed',
        model_fingerprints={'test': 'a'}, train_seeds=[90])))
    output = tmp_path/'report.json'
    monkeypatch.setattr('scripts.probe_downstream_gpu_coverage._load_problem_without_reference',
                        lambda *args: problem())
    closed = []
    def failed_run(*args):
        args[-1].callback(lambda: closed.append(True))
        raise RuntimeError('injected diagnostic failure')
    monkeypatch.setattr('scripts.probe_downstream_gpu_coverage._run_gpu_diagnostic', failed_run)
    monkeypatch.setattr('sys.argv', ['probe', '--trace', str(trace_dir), '--bank', str(bank_dir),
        '--trace-sha256', sha256_file(trace_dir/'manifest.json'),
        '--bank-sha256', sha256_file(bank_dir/'manifest.json'), '--output', str(output)])
    with pytest.raises(RuntimeError, match='injected'):
        main()
    report = json.loads(output.read_text())
    assert report['status'] == 'failed' and report['error_type'] == 'RuntimeError'
    assert closed == [True]


def test_positive_rescaling_preserves_rows_signs_and_has_no_large_coefficients():
    eq, eq_rhs = np.array([[1e20, -2e20], [0., 0.]]), np.array([3e20, 1.])
    ub, ub_rhs = np.array([[-4e22, 5e22]]), np.array([-2e22])
    cost = np.array([1e30, -1e30])
    original = [array.copy() for array in (eq, eq_rhs, ub, ub_rhs, cost)]
    scaled, metadata = scale_reduced_problem(eq, eq_rhs, ub, ub_rhs, cost)
    for result, source in zip(scaled, original):
        assert result.shape == source.shape
        assert np.max(np.abs(result), initial=0.) <= 1.
        np.testing.assert_array_equal(np.sign(result), np.sign(source))
    for unchanged, source in zip((eq, eq_rhs, ub, ub_rhs, cost), original):
        np.testing.assert_array_equal(unchanged, source)
    assert metadata['rows_removed'] == 0
    assert metadata['original_certificate_thresholds_changed'] is False
    assert scaled[1][1] == 1.  # An inconsistent zero=1 row is not removed.


def test_scaled_oracle_retains_full_original_certificate():
    candidates, duals = np.array([[-1., 2.], [2., -1.]]), np.zeros((2, 1))
    result = hull_oracle(problem(), candidates, duals, scale_reduced=True)
    assert result['hull_certified_solution_found']
    assert result['positive_reduced_scaling_enabled']
    assert result['original_objective'] == pytest.approx(0.)


def test_magnitude_spread_is_labeled_not_as_condition_number():
    stats = coefficient_range(np.array([0., -1e15, 1e-8, np.inf]))
    assert stats['nonfinite_elements'] == 1
    assert stats['minimum_nonzero_absolute'] == 1e-8
    assert stats['maximum_absolute'] == 1e15
    assert stats['magnitude_dynamic_range_not_condition_number'] == pytest.approx(1e23)
