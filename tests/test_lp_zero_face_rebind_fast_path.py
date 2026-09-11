"""Proof-equivalent zero-sign shortcut; no LP solver, CUDA or timings."""
from dataclasses import replace
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import src.lp_zero_face as zero_face
from src.lp_trace import problem_hash
from src.lp_zero_face import InfeasibleZeroRow, ZeroFaceReduction


def problem(a, rhs, lo, hi, c=None, neq=1):
    return (csr_matrix(a, dtype=np.float64), np.asarray(rhs, dtype=np.float64),
            np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64),
            np.ones(len(lo)) if c is None else np.asarray(c, dtype=np.float64), neq)


def changed(p, **updates):
    values = [v.copy() if hasattr(v, 'copy') else v for v in p]
    for name, value in updates.items():
        index = ('a', 'rhs', 'lo', 'hi', 'c', 'neq').index(name)
        values[index] = (csr_matrix(value, dtype=np.float64) if index == 0 else
                         int(value) if index == 5 else np.asarray(value, dtype=np.float64))
    return tuple(values)


def outcome(plan, current):
    try:
        return plan.rebind(current)
    except ValueError as error:
        return type(error), str(error)


def assert_same_numeric_plan(first, second):
    assert first.original_hash == second.original_hash
    assert first.reduced_hash == second.reduced_hash
    assert first.objective_offset == second.objective_offset
    for name in ('columns', 'rows', 'fixed_columns', 'fixed_values',
                 'explicit_fixed_columns', 'forced_zero_columns',
                 'removed_duplicate_rows', 'removed_zero_rows'):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    first.validate_integrity()
    second.validate_integrity()


def assert_full_replay_equivalent(monkeypatch, plan, current):
    old_hashes = problem_hash(plan.original), problem_hash(plan.reduced), problem_hash(current)
    optimized = outcome(plan, current)
    with monkeypatch.context() as patch:
        # Force the unchanged row-by-row witness and complete fixed-point
        # replay path. No fresh constructor is used for this reference.
        patch.setattr(zero_face, '_same_bound_zero_signatures', lambda *args: False)
        replay = outcome(plan, current)
    assert old_hashes == (problem_hash(plan.original), problem_hash(plan.reduced), problem_hash(current))
    if isinstance(optimized, tuple):
        assert optimized == replay
        return False
    assert isinstance(replay, ZeroFaceReduction)
    assert_same_numeric_plan(optimized, replay)
    assert optimized.witnesses == replay.witnesses
    assert optimized.duplicate_equalities == replay.duplicate_equalities
    assert optimized._proof_fingerprint == replay._proof_fingerprint
    fresh = ZeroFaceReduction(current,
        remove_duplicate_equalities=plan.remove_duplicate_equalities,
        fix_singleton_equalities=plan.fix_singleton_equalities,
        singleton_min_coefficient=plan.singleton_min_coefficient)
    assert_same_numeric_plan(optimized, fresh)
    # A different discovery order may yield another valid witness sequence
    # after changed signs. Both postsolves still implement the same LP maps.
    reduced_x = np.linspace(-1., 1., len(optimized.columns))
    reduced_y = np.linspace(-2., 2., len(optimized.rows))
    np.testing.assert_array_equal(optimized.lift_primal(reduced_x), fresh.lift_primal(reduced_x))
    np.testing.assert_array_equal(optimized.lift_dual(reduced_y), replay.lift_dual(reduced_y))
    for name in ('columns', 'fixed_values', 'rows'):
        assert not np.shares_memory(getattr(optimized, name), getattr(plan, name))
        assert not getattr(optimized, name).flags.writeable
    return True


def test_sign_guard_includes_zero_crossing_unbounded_and_signed_zero():
    old_lo = np.array([-np.inf, -3., -0., 0., 1.])
    old_hi = np.array([-1., 0., 3., np.inf, 4.])
    assert zero_face._same_bound_zero_signatures(
        [-np.inf, -9., 0., -0., 2.], [-7., -0., 9., np.inf, 8.], old_lo, old_hi)
    for column, new_value in ((0, 0.), (1, 0.), (2, -1.), (3, 1.), (4, -1.)):
        current = old_lo.copy()
        current[column] = new_value
        assert not zero_face._same_bound_zero_signatures(current, old_hi, old_lo, old_hi)


def count_row_proof_calls(plan, current):
    calls = []
    old_profile = sys.getprofile()
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code.co_name == 'row_proof':
            calls.append(frame.f_locals['row'])
    try:
        sys.setprofile(profile)
        result = outcome(plan, current)
    finally:
        sys.setprofile(old_profile)
    return result, calls


def test_shortcut_really_skips_all_row_proof_calls(monkeypatch):
    p = problem([[1, -1, 0], [1, 0, 0], [0, 0, 1]], [0, 0, 4],
                [-2, -3, -1], [2, 3, 7], neq=2)
    plan = ZeroFaceReduction(p, fix_singleton_equalities=True)
    assert [w.row for w in plan.witnesses] == [1, 0]
    current = changed(p, lo=[-4, -9, -2], hi=[4, 9, 10], c=[3, -4, 2])
    result, calls = count_row_proof_calls(plan, current)
    assert isinstance(result, ZeroFaceReduction)
    assert calls == []
    with monkeypatch.context() as patch:
        patch.setattr(zero_face, '_same_bound_zero_signatures', lambda *args: False)
        reference, calls = count_row_proof_calls(plan, current)
    assert calls == [1, 0, 0, 1]
    assert_same_numeric_plan(result, reference)
    assert_full_replay_equivalent(monkeypatch, plan, current)


@pytest.mark.parametrize('singletons', [False, True])
def test_sign_change_detects_a_new_fixed_cascade(monkeypatch, singletons):
    p = problem([[1, 1, 0], [0, 0, 1]], [0, 4], [-1, -1, 0], [3, 4, 8])
    plan = ZeroFaceReduction(p, fix_singleton_equalities=singletons)
    current = changed(p, lo=[0, 0, 0])
    result, calls = count_row_proof_calls(plan, current)
    assert result[0] is ValueError
    assert 'Additional zero-face' in result[1]
    assert calls
    assert not assert_full_replay_equivalent(monkeypatch, plan, current)


def test_harmless_sign_change_still_takes_full_replay(monkeypatch):
    p = problem([[1, -1, 0], [0, 0, 1]], [0, 4], [-3, -2, -1], [4, 5, 8])
    current = changed(p, lo=[0, -2, -1])
    result, calls = count_row_proof_calls(ZeroFaceReduction(p), current)
    assert isinstance(result, ZeroFaceReduction)
    assert calls == [0]
    assert_full_replay_equivalent(monkeypatch, ZeroFaceReduction(p), current)


@pytest.mark.parametrize('coefficient', [1e-15, -1e-15, 1e-12, -1e-12, 2e-12, -2e-12])
def test_singleton_floor_is_exactly_preserved(monkeypatch, coefficient):
    p = problem([[coefficient, 0], [0, 1]], [0, 5], [-1, 0], [3, 8])
    plan = ZeroFaceReduction(p, fix_singleton_equalities=True)
    assert bool(plan.witnesses) == (abs(coefficient) >= 1e-12)
    assert_full_replay_equivalent(monkeypatch, plan, changed(p, lo=[-2, 0], hi=[7, 9]))
    # Crossing zero must trigger the old out-of-bounds singleton rejection
    # only if the coefficient is actually eligible for a singleton proof.
    assert_full_replay_equivalent(monkeypatch, plan, changed(p, lo=[1, 0]))


@pytest.mark.parametrize('deduplicate', [False, True])
def test_duplicate_reproof_is_implied_by_complete_equality_checks(monkeypatch, deduplicate):
    p = problem([[1, 1, 0], [-1, -1, 0], [2, 0, 0], [0, 0, 1]],
                [5, -5, 6, 4], [3, -1, 0], [3, 9, 8], [7, 2, -1], neq=2)
    plan = ZeroFaceReduction(p, remove_duplicate_equalities=deduplicate)
    current = changed(p, a=[[1, 1, 0], [-1, -1, 0], [4, 0, 0], [1, 0, 2]],
                      rhs=[5, -5, 12, 11], c=[11, 4, -3])
    assert_full_replay_equivalent(monkeypatch, plan, current)
    # Ensure the removed pairwise getrow loop has not merely been relocated.
    with monkeypatch.context() as patch:
        patch.setattr(csr_matrix, 'getrow', lambda *args: pytest.fail('No duplicate row slicing'))
        rebound = plan.rebind(current)
    assert rebound.objective_offset == 33.
    for mutation in (dict(rhs=[5, -4, 6, 4]),
                     dict(a=[[1, 1, 0], [-1, -2, 0], [2, 0, 0], [0, 0, 1]]),
                     dict(lo=[4, -1, 0], hi=[4, 9, 8])):
        assert not assert_full_replay_equivalent(monkeypatch, plan, changed(p, **mutation))


@pytest.mark.parametrize('neq', [0, 2])
def test_csr_prefix_handles_empty_or_all_equality_blocks(monkeypatch, neq):
    p = problem([[1, -1, 0], [0, 1, -1]], [0, 0], [-2]*3, [4]*3, neq=neq)
    assert_full_replay_equivalent(monkeypatch, ZeroFaceReduction(p), changed(p, c=[2, -1, 7]))
    candidate = changed(p, a=[[1, -2, 0], [0, 1, -1]])
    accepted = assert_full_replay_equivalent(monkeypatch, ZeroFaceReduction(p), candidate)
    assert accepted == (neq == 0)


@pytest.mark.parametrize('updates', [dict(lo=[0, 0]), dict(hi=[3, np.inf]),
    dict(lo=[0, -np.inf]), dict(rhs=[np.nan, 4]), dict(c=[1, np.inf]),
    dict(lo=[np.inf, 0], hi=[np.inf, 8]), dict(lo=[0, -np.inf], hi=[3, -np.inf]),
    dict(lo=[np.nan, 0]), dict(a=[[np.inf, 0], [0, 1]])])
def test_fast_signs_never_bypass_fixed_topology_or_nonfinite_checks(monkeypatch, updates):
    p = problem([[1, 0], [0, 1]], [0, 4], [0, 1], [3, 8])
    plan = ZeroFaceReduction(p)
    candidate = changed(p, **updates)
    if updates == dict(lo=[0, 0]):
        # Merely changing the sign of an unforced variable is legal here.
        assert_full_replay_equivalent(monkeypatch, plan, candidate)
    else:
        assert not assert_full_replay_equivalent(monkeypatch, plan, candidate)


@pytest.mark.parametrize('target', ['original', 'reduced', 'witness', 'map', 'configuration'])
def test_sign_shortcut_cannot_bypass_source_integrity(monkeypatch, target):
    p = problem([[1, 1, 0], [0, 0, 1]], [0, 4], [0, 0, 0], [3, 4, 8])
    plan = ZeroFaceReduction(p)
    if target in ('original', 'reduced'):
        lp = getattr(plan, target)
        lp[4].flags.writeable = True
        lp[4][0] += 1.
    elif target == 'witness':
        plan.witnesses = (replace(plan.witnesses[0], orientation='max'),)
    elif target == 'map':
        plan.columns = np.array([0])
    else:
        plan.singleton_min_coefficient = 1e-14
    with monkeypatch.context() as patch:
        patch.setattr(zero_face, '_same_bound_zero_signatures',
                      lambda *args: pytest.fail('Integrity must run before shortcut'))
        with pytest.raises(ValueError, match='modified'):
            plan.rebind(p)


@pytest.mark.parametrize('seed', [17, 29, 71, 113])
def test_randomized_replay_and_full_rebuild_property(monkeypatch, seed):
    rng = np.random.default_rng(seed)
    bounds = np.array([[-np.inf, np.inf], [-np.inf, 0], [0, np.inf], [-3, 5],
        [0, 4], [-4, 0], [0, 0], [2, 2], [-2, -2], [1, 5], [-5, -1]], dtype=float)
    coefficients = np.array([0, 0, 0, -2, -1, 1, 2, 1e-15, -1e-15, 1e-12, -2e-12])
    accepted = rejected = stable_sign_cases = 0
    for trial in range(100):
        n = int(rng.integers(2, 9))
        neq = int(rng.integers(0, 7))
        m = neq + int(rng.integers(1, 4))
        a = rng.choice(coefficients, size=(m, n))
        rhs = np.r_[np.zeros(neq), np.full(m-neq, 100.)]
        # Periodically inject a literal/sign-negated equality duplicate.
        if neq >= 2 and trial % 3 == 0:
            a[1] = (-1 if trial % 2 else 1) * a[0]
        selected = bounds[rng.integers(0, len(bounds), size=n)]
        p = problem(a, rhs, selected[:, 0], selected[:, 1], rng.normal(size=n), neq)
        try:
            plan = ZeroFaceReduction(p, remove_duplicate_equalities=bool(trial % 2),
                fix_singleton_equalities=bool(trial % 3), singleton_min_coefficient=1e-12)
        except InfeasibleZeroRow:
            continue
        lo, hi = p[2].copy(), p[3].copy()
        fixed = lo == hi
        scale = rng.uniform(.25, 4., size=n)
        lo[~fixed] *= scale[~fixed]
        hi[~fixed] *= scale[~fixed]
        if trial % 2:
            # Include sign/finite/fixed topology changes, not just cases
            # satisfying the shortcut's hypotheses. They must take replay or
            # the unchanged earlier rejection gates, never stale acceptance.
            column = int(rng.integers(n))
            lo[column], hi[column] = bounds[int(rng.integers(len(bounds)))]
        current_a = a.copy()
        current_a[neq:] *= rng.uniform(.5, 2., size=(m-neq, n))
        current = changed(p, a=current_a, lo=lo, hi=hi, c=rng.normal(size=n),
                          rhs=np.r_[np.zeros(neq), np.full(m-neq, 200.)])
        stable_sign_cases += int(zero_face._same_bound_zero_signatures(lo, hi, p[2], p[3]))
        if assert_full_replay_equivalent(monkeypatch, plan, current):
            accepted += 1
        else:
            rejected += 1
    assert accepted >= 10
    assert rejected >= 2
    assert stable_sign_cases >= 10
