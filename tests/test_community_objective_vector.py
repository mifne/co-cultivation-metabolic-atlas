"""CPU-only equivalence tests for shared CPU/GPU objective extraction."""
import numpy as np
import pytest
from cobra import Model, Reaction

import src.community_solver as module


def _model(name):
    model = Model(name)
    model.add_reactions([Reaction(name + '_' + str(index)) for index in range(3)])
    return model


def _legacy(models, offsets, size):
    total = np.zeros(size, dtype=np.float64); per_species = {}
    for species, (start, end) in offsets.items():
        coefficients = np.asarray([float(reaction.objective_coefficient)
            for reaction in models[species].reactions], dtype=np.float64)
        per_species[species] = coefficients.copy()
        if models[species].objective.direction == 'min': coefficients = -coefficients
        total[start:end] = coefficients
    return total, per_species


def _assert_equal(models, offsets, size):
    expected = _legacy(models, offsets, size)
    actual = module.CooperativeCommunityFbaSolver._objective_vector(models, offsets, size)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[0].dtype == np.float64
    for species in models:
        np.testing.assert_array_equal(actual[1][species], expected[1][species])
        assert actual[1][species].dtype == np.float64
    return actual


@pytest.mark.parametrize('direction', ['min', 'max'])
def test_bulk_objective_matches_legacy_positive_negative_zero_and_offsets(direction):
    models = {name: _model(name) for name in ('a', 'b')}
    models['a'].objective = {models['a'].reactions[0]: 1.25, models['a'].reactions[2]: -.75}
    models['a'].objective.direction = direction
    models['b'].objective = {}
    models['b'].objective.direction = 'min' if direction == 'max' else 'max'
    actual = _assert_equal(models, {'a': (0, 3), 'b': (5, 8)}, 10)
    np.testing.assert_array_equal(actual[0][[3, 4, 8, 9]], np.zeros(4))


def test_bulk_objective_reads_live_replacements_and_inplace_edits_without_aliasing():
    model = _model('live'); models = {'live': model}; offsets = {'live': (0, 3)}
    model.objective = {model.reactions[0]: 2.}
    before = _assert_equal(models, offsets, 3)
    model.reactions[0].objective_coefficient = -.25
    model.reactions[1].objective_coefficient = .5
    _assert_equal(models, offsets, 3)
    model.objective = {model.reactions[2]: 4.}
    model.objective.direction = 'min'
    after = _assert_equal(models, offsets, 3)
    np.testing.assert_array_equal(before[0], [2., 0., 0.])
    np.testing.assert_array_equal(after[0], [0., 0., -4.])
    np.testing.assert_array_equal(after[1]['live'], [0., 0., 4.])
    after[1]['live'][2] = 99.
    assert after[0][2] == -4.


def test_bulk_objective_calls_official_helper_once_per_species(monkeypatch):
    models = {name: _model(name) for name in ('a', 'b')}
    for model in models.values(): model.objective = model.reactions[0]
    calls = []; original = module.linear_reaction_coefficients
    def counted(model, **kwargs):
        calls.append(model.id)
        return original(model, **kwargs)
    monkeypatch.setattr(module, 'linear_reaction_coefficients', counted)
    _assert_equal(models, {'a': (0, 3), 'b': (3, 6)}, 6)
    assert calls == ['a', 'b']


def test_sparse_selection_skips_zero_reactions_and_empty_objective(monkeypatch):
    model = _model('sparse')
    model.objective = model.reactions[1]
    original = module.linear_reaction_coefficients
    visited = []
    def counted(model, reactions=None):
        visited.append([reaction.id for reaction in reactions])
        return original(model, reactions=reactions)
    monkeypatch.setattr(module, 'linear_reaction_coefficients', counted)
    _assert_equal({'s':model}, {'s':(0,3)}, 3)
    assert visited == [['sparse_1']]
    model.objective = {}
    _assert_equal({'s':model}, {'s':(0,3)}, 3)
    assert visited == [['sparse_1']]


def test_sparse_selection_matches_asymmetric_reverse_only_and_auxiliary_terms():
    model = _model('unusual')
    a,b,c = model.reactions
    auxiliary = model.problem.Variable('auxiliary')
    model.add_cons_vars(auxiliary)
    for expression in (3*a.forward_variable-2*a.reverse_variable,
                       2*b.reverse_variable,
                       a.flux_expression-4*c.flux_expression+2*auxiliary+7):
        model.objective = model.problem.Objective(expression, direction='max')
        _assert_equal({'m':model}, {'m':(0,3)}, 3)


def test_sparse_selection_observes_rename_add_remove_and_context_restore():
    model = _model('structure')
    model.objective = model.reactions[0]
    before = _assert_equal({'m':model}, {'m':(0,3)}, 3)
    model.reactions[0].id = 'renamed'
    _assert_equal({'m':model}, {'m':(0,3)}, 3)
    with model:
        extra = Reaction('extra')
        model.add_reactions([extra])
        model.objective = {extra:3., model.reactions[1]:-2.}
        _assert_equal({'m':model}, {'m':(0,4)}, 4)
    after = _assert_equal({'m':model}, {'m':(0,3)}, 3)
    np.testing.assert_array_equal(before[0], after[0])


def test_sparse_selection_deepcopy_uses_copied_solver_variables():
    import copy
    model = _model('copy')
    model.objective = {model.reactions[0]:.25,model.reactions[2]:.5}
    duplicate = copy.deepcopy(model)
    duplicate.reactions[0].objective_coefficient = 4.
    _assert_equal({'m':duplicate}, {'m':(0,3)}, 3)
    _assert_equal({'m':model}, {'m':(0,3)}, 3)
