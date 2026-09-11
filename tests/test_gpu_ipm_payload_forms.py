"""Per-call constraint-form reuse; no CUDA or timing benchmarks."""

import numpy as np
import pytest

import src.gpu_ipm_numeric_update as update
from src.lp_trace import problem_hash
from tests.test_gpu_ipm_numeric_update import (
    _assert_snapshot, _device_snapshot, _mock_solver, _problems,
)


@pytest.mark.parametrize('exact', [False, True])
def test_rebind_validates_old_and_new_forms_once_each_without_persistent_cache(monkeypatch, exact):
    solver = _mock_solver(exact_equalities=exact)
    old, new = update.prepare_host_rebind(solver, _problems(1))
    expected_hashes = [problem_hash(p) for state in (old, new) for p in state.problems]
    calls = []
    original = update.constraint_form
    def counted(problem):
        calls.append(problem_hash(problem))
        return original(problem)
    monkeypatch.setattr(update, 'constraint_form', counted)
    update.rebind_forest_ipm(solver, _problems(1))
    assert calls == expected_hashes
    assert len(calls) == 2*solver.batch
    assert '_forms' not in vars(solver) and '_old_forms' not in vars(solver)
    calls.clear()
    update.rebind_forest_ipm(solver, _problems(2))
    assert len(calls) == 2*solver.batch  # A later update must validate afresh.


@pytest.mark.parametrize('state_index', [0, 1])
def test_private_reused_forms_payloads_match_unchanged_two_argument_api(state_index):
    solver = _mock_solver()
    states = update.prepare_host_rebind(solver, _problems(1))
    state = states[state_index]
    hashes = [problem_hash(p) for p in (*states[0].problems, *states[1].problems)]
    expected = update._payloads(solver, state)
    old_forms = tuple(update.constraint_form(p) for p in states[0].problems)
    forms = tuple(update.constraint_form(p) for p in state.problems)
    actual = update._payloads(solver, state, _forms=forms, _old_forms=old_forms)
    assert actual[0].keys() == expected[0].keys() and actual[1].keys() == expected[1].keys()
    for path, wanted in expected[0].items():
        got = actual[0][path]
        assert got.shape == wanted.shape
        for attr in ('data', 'indices', 'indptr'):
            np.testing.assert_array_equal(getattr(got, attr), getattr(wanted, attr))
    for path, wanted in expected[1].items():
        np.testing.assert_array_equal(actual[1][path], wanted)
    assert hashes == [problem_hash(p) for p in (*states[0].problems, *states[1].problems)]


@pytest.mark.parametrize('mask_index', [4, 5, 6])
def test_reused_forms_preserve_exact_old_new_coordinate_mask_rejection(mask_index):
    solver = _mock_solver()
    old, new = update.prepare_host_rebind(solver, _problems(1))
    old_forms = tuple(update.constraint_form(p) for p in old.problems)
    forms = [list(update.constraint_form(p)) for p in new.problems]
    changed = forms[0][mask_index].copy()
    if mask_index == 4: changed[0] = ~changed[0]
    else: changed[0] += 1
    forms[0][mask_index] = changed
    with pytest.raises(update.NumericRebindRejected, match='coordinate masks changed'):
        update._payloads(solver, new, _forms=forms, _old_forms=old_forms)


def test_new_constraint_form_validation_failure_remains_precommit_and_nonmutating(monkeypatch):
    solver = _mock_solver()
    snapshot = _device_snapshot(solver)
    prior_state, prior_hashes = solver._last_internal_state, solver.problem_hashes
    calls = 0
    original = update.constraint_form
    def validate(problem):
        nonlocal calls
        calls += 1
        if calls == solver.batch+1: raise ValueError('New current LP validation failed')
        return original(problem)
    monkeypatch.setattr(update, 'constraint_form', validate)
    with pytest.raises(update.NumericRebindRejected, match='New current LP validation failed'):
        update.rebind_forest_ipm(solver, _problems(1))
    _assert_snapshot(solver, snapshot)
    assert solver._last_internal_state is prior_state
    assert solver.problem_hashes == prior_hashes
    assert solver.factor.factored and not solver.factor.failed
