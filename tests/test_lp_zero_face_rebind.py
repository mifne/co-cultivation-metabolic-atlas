"""Current-input zero-face proof replay, without LP solves, QR or CUDA."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.lp_trace import problem_hash
from src.lp_zero_face import InfeasibleZeroRow, ZeroFaceReduction


def problem(a, rhs, lo, hi, c=None, neq=1):
    return (csr_matrix(a, dtype=float), np.asarray(rhs, dtype=float),
        np.asarray(lo, dtype=float), np.asarray(hi, dtype=float),
        np.ones(len(lo)) if c is None else np.asarray(c, dtype=float), neq)


def changed(p, **kwargs):
    values = [a.copy() if hasattr(a, 'copy') else a for a in p]
    for name, value in kwargs.items():
        i = ('a', 'rhs', 'lo', 'hi', 'c', 'neq').index(name)
        values[i] = csr_matrix(value, dtype=float) if i == 0 else (
            int(value) if i == 5 else np.asarray(value, dtype=float))
    return tuple(values)


def assert_fresh_equivalent(plan, current):
    result = plan.rebind(current)
    fresh = ZeroFaceReduction(current, remove_duplicate_equalities=plan.remove_duplicate_equalities,
        fix_singleton_equalities=plan.fix_singleton_equalities,
        singleton_min_coefficient=plan.singleton_min_coefficient)
    assert result is not plan
    assert result.original_hash == fresh.original_hash
    assert result.reduced_hash == fresh.reduced_hash
    assert result.objective_offset == fresh.objective_offset
    for name in ('rows', 'columns', 'fixed_columns', 'fixed_values', 'removed_zero_rows',
                 'removed_duplicate_rows', 'explicit_fixed_columns', 'forced_zero_columns'):
        np.testing.assert_array_equal(getattr(result, name), getattr(fresh, name))
        assert not getattr(result, name).flags.writeable
        assert not np.shares_memory(getattr(result, name), getattr(plan, name))
    result.validate_integrity()
    plan.validate_integrity()
    return result


def test_numeric_inequality_support_rhs_cost_and_bounds_use_current_arrays():
    p = problem([[1, 2, 0, 0], [0, 0, 1, 0]], [0, 2], [0, 0, -2, 0], [3, 4, 8, 9])
    plan = ZeroFaceReduction(p)
    current = changed(p, a=[[1, 2, 0, 0], [2, 0, 3, -1]], rhs=[0, 7],
                      lo=[0, 0, -4, 0], hi=[5, 7, 12, 15], c=[-8, 5, -2, 4])
    hashes = problem_hash(p), problem_hash(current)
    result = assert_fresh_equivalent(plan, current)
    assert result.original_hash != plan.original_hash
    assert result.reduced[0].toarray().tolist() == [[3., -1.]]
    assert result.reduced[1].tolist() == [7.]
    assert result.reduced[4].tolist() == [-2., 4.]
    assert (problem_hash(p), problem_hash(current)) == hashes
    current[0].data[-1] = 20.
    current[4][-1] = 99.
    result.validate_integrity()
    assert result.reduced[4][-1] == 4.


@pytest.mark.parametrize('deduplicate', [True, False])
def test_nonzero_explicit_fixed_substitution_duplicate_rows_and_offset(deduplicate):
    p = problem([[1, 1, 0], [-1, -1, 0], [2, 0, 0], [0, 0, 1]],
                [5, -5, 6, 4], [3, -1, 0], [3, 9, 8], [7, 2, -1], neq=2)
    current = changed(p, a=[[1, 1, 0], [-1, -1, 0], [4, 0, 0], [1, 0, 2]],
                      rhs=[5, -5, 12, 11], c=[11, 4, -3])
    result = assert_fresh_equivalent(ZeroFaceReduction(p,
        remove_duplicate_equalities=deduplicate), current)
    assert result.objective_offset == 33.
    assert result.reduced[1][-1] == 8.


@pytest.mark.parametrize('row,lo,hi', [([1, 2, 0], [0, 0, -1], [3, 4, 9]),
    ([-1, -2, 0], [0, 0, -1], [3, 4, 9]),
    ([1, -2, 0], [-3, 0, -1], [0, 4, 9])])
def test_signed_min_max_proofs_survive_nonbinding_endpoint_change(row, lo, hi):
    p = problem([row, [0, 0, 1]], [0, 4], lo, hi)
    current = changed(p, hi=[value * 2 for value in hi], lo=[value * 2 for value in lo])
    assert_fresh_equivalent(ZeroFaceReduction(p), current)


def test_cascading_and_interior_singleton_witnesses_replay_forward():
    p = problem([[1, -1, 0], [1, 0, 0], [0, 0, 1]], [0, 0, 5],
                [-3, -4, -1], [3, 4, 8], neq=2)
    plan = ZeroFaceReduction(p, fix_singleton_equalities=True)
    assert [w.row for w in plan.witnesses] == [1, 0]
    assert [w.orientation for w in plan.witnesses] == ['equality', 'equality']
    current = changed(p, lo=[-7, -8, -2], hi=[7, 8, 9], rhs=[0, 0, 6])
    result = assert_fresh_equivalent(plan, current)
    assert result.witnesses == plan.witnesses


@pytest.mark.parametrize('mutation,match', [
    (dict(a=[[1, 3, 0], [0, 0, 1]]), 'equality'),
    (dict(rhs=[1e-300, 5]), 'equality'),
    (dict(a=[[1, 0, 0], [0, 0, 1]]), 'equality'),
    (dict(neq=0), 'dimensions'),
    (dict(lo=[-1, 0, 0]), 'witness'),
    (dict(hi=[0, 4, 8]), 'explicit fixed'),
    (dict(hi=[3, 4, 0]), 'explicit fixed'),
    (dict(lo=[0, 0, -np.inf]), 'finite-bound'),
    (dict(hi=[3, 4, np.inf]), 'finite-bound'),
    (dict(a=[[1, 2, 0], [0, 0, 0]]), 'zero-row structure'),
    (dict(a=[[1, 2, 0], [0, 0, 0]], rhs=[0, -1]), 'impossible zero'),
])
def test_changed_structure_or_invalid_saved_proof_rejects_without_mutation(mutation, match):
    p = problem([[1, 2, 0], [0, 0, 1]], [0, 5], [0, 0, 0], [3, 4, 8])
    plan = ZeroFaceReduction(p)
    current = changed(p, **mutation)
    hashes = plan.original_hash, plan.reduced_hash, problem_hash(current)
    with pytest.raises(ValueError, match=match):
        plan.rebind(current)
    assert (plan.original_hash, plan.reduced_hash, problem_hash(current)) == hashes
    plan.validate_integrity()


def test_explicit_fixed_value_change_rejects():
    p = problem([[1, 0], [0, 1]], [3, 4], [3, 0], [3, 8])
    with pytest.raises(ValueError, match='explicit fixed'):
        ZeroFaceReduction(p).rebind(changed(p, lo=[4, 0], hi=[4, 8]))


def test_new_implicit_zero_face_rejects_even_when_finite_masks_unchanged():
    p = problem([[1, 1, 0], [0, 0, 1]], [0, 5], [-1, -1, 0], [3, 4, 8])
    with pytest.raises(ValueError, match='Additional zero-face'):
        ZeroFaceReduction(p).rebind(changed(p, lo=[0, 0, 0]))


def test_old_removed_inequality_becomes_nonzero_or_infeasible_rejects():
    p = problem([[1, 2, 0], [1, 0, 0], [0, 0, 1]], [0, 0, 5],
                [0, 0, 0], [3, 4, 8])
    plan = ZeroFaceReduction(p)
    with pytest.raises(ValueError, match='zero-row structure'):
        plan.rebind(changed(p, a=[[1, 2, 0], [1, 0, 1], [0, 0, 1]]))
    with pytest.raises(InfeasibleZeroRow):
        plan.rebind(changed(p, rhs=[0, -1e-300, 5]))
    assert_fresh_equivalent(plan, changed(p, rhs=[0, 7, 8]))


def test_interior_singleton_requires_zero_inside_current_bounds():
    p = problem([[1, 0], [0, 1]], [0, 5], [-1, 0], [3, 8])
    plan = ZeroFaceReduction(p, fix_singleton_equalities=True)
    with pytest.raises(InfeasibleZeroRow):
        plan.rebind(changed(p, lo=[1, 0]))


def test_tiny_singleton_remains_uneliminated_and_all_fixed_case_works():
    p = problem([[1e-15, 0], [0, 1]], [0, 5], [-1, 0], [3, 8])
    plan = ZeroFaceReduction(p, fix_singleton_equalities=True)
    assert not plan.witnesses
    assert_fresh_equivalent(plan, changed(p, lo=[-2, 0], hi=[4, 9]))
    zero = problem([[1, 2]], [0], [0, 0], [2, 3])
    result = assert_fresh_equivalent(ZeroFaceReduction(zero), changed(zero, c=[-1, 2]))
    assert result.reduced[0].shape == (0, 0)


@pytest.mark.parametrize('target', ['rows', 'witnesses', 'config'])
def test_corrupted_saved_map_proof_or_configuration_rejects(target):
    p = problem([[1, 2, 0], [0, 0, 1]], [0, 5], [0, 0, 0], [3, 4, 8])
    plan = ZeroFaceReduction(p)
    if target == 'rows':
        plan.rows = np.array([0])
    elif target == 'witnesses':
        plan.witnesses = (replace(plan.witnesses[0], orientation='max'),)
    else:
        plan.fix_singleton_equalities = True
    with pytest.raises(ValueError, match='proof or coordinate'):
        plan.rebind(p)


def test_rebind_does_not_call_constructor_solver_or_rank_discovery(monkeypatch):
    import scipy.linalg
    import scipy.optimize
    p = problem([[1, 2, 0], [0, 0, 1]], [0, 5], [0, 0, 0], [3, 4, 8])
    plan = ZeroFaceReduction(p)
    def forbidden(*args, **kwargs):
        raise AssertionError('Rebind must only replay already proved sparse maps')
    monkeypatch.setattr(ZeroFaceReduction, '__init__', forbidden)
    monkeypatch.setattr(scipy.optimize, 'linprog', forbidden)
    monkeypatch.setattr(scipy.linalg, 'qr', forbidden)
    result = plan.rebind(changed(p, rhs=[0, 7]))
    assert result.reduced[1].tolist() == [7.]
