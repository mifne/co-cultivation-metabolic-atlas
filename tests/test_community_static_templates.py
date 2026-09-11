"""Exact and non-mutating checks for reusable cooperative LP structure."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.community_solver import (
    CooperativeCommunityFbaSolver,
    _assemble_parsimonious_exchange_lp,
    _build_parsimonious_exchange_template,
)
from tests.test_community_exchange_assembly import _legacy_assembly
from tests.test_community_solver import _crossfeeding_models


def _same_csr(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('optimize,aggregate', [(True, 3.5), (False, 3.5), (True, 0.0)])
def test_cached_exchange_template_is_exactly_the_legacy_lp(seed, optimize, aggregate):
    rng = np.random.default_rng(seed)
    n_fluxes = 19
    ub = rng.normal(size=(7, n_fluxes + 1))
    eq = rng.normal(size=(11, n_fluxes + 1))
    ub[rng.random(ub.shape) < .72] = 0.
    eq[rng.random(eq.shape) < .79] = 0.
    a_ub, a_eq = csr_matrix(ub), csr_matrix(eq)
    b_ub = rng.normal(size=len(ub))
    objective = rng.normal(size=n_fluxes)
    objective[rng.random(n_fluxes) < .65] = 0.
    exchanges = (
        ('a', 18, np.float64(-1.)), ('b', 0, 2.),
        ('a', 5, np.float32(0.)), ('c', 16, -.25),
        ('b', 0, -2.), ('d', 7, 1.),
    )
    biomass = {'a': 1., 'b': 0., 'c': .031, 'd': -1.}
    template = _build_parsimonious_exchange_template(a_eq, n_fluxes, exchanges)
    args = (a_ub, b_ub, a_eq, n_fluxes, exchanges, biomass,
            objective, aggregate, optimize, .973)
    expected = _legacy_assembly(*args)
    actual = _assemble_parsimonious_exchange_lp(*args, _template=template)
    np.testing.assert_array_equal(actual[0], expected[0])
    _same_csr(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    _same_csr(actual[3], expected[3])
    # The solver-owned immutable arrays are the actual shared objects.
    assert actual[0] is template['objective']
    assert actual[3] is template['extended_eq']
    with pytest.raises(ValueError):
        actual[0][0] = 9.
    with pytest.raises(ValueError):
        actual[3].data[0] = 9.
    # Dynamic matrices and RHS remain owned by this particular invocation.
    actual[1].data[:] = 7.; actual[2][:] = 8.
    again = _assemble_parsimonious_exchange_lp(*args, _template=template)
    _same_csr(again[1], expected[1])
    np.testing.assert_array_equal(again[2], expected[2])


def test_template_rejects_different_order_index_stoichiometry_and_scalar_type():
    a_eq = csr_matrix(np.eye(3, 5))
    exchanges = (('a', 0, 1.), ('b', 2, -1.))
    template = _build_parsimonious_exchange_template(a_eq, 4, exchanges)
    base = (csr_matrix((2, 5)), np.zeros(2), a_eq, 4)
    tail = ({'a': 1., 'b': 2.}, np.ones(4), 1., True, .99)
    edits = (
        tuple(reversed(exchanges)),
        (('a', 1, 1.), ('b', 2, -1.)),
        (('a', 0, 1.), ('b', 2, -2.)),
        (('a', 0, np.float32(1.)), ('b', 2, -1.)),
    )
    for changed in edits:
        with pytest.raises(ValueError, match='metadata changed'):
            _assemble_parsimonious_exchange_lp(
                *base, changed, *tail, _template=template
            )


def test_solver_reuses_static_equalities_and_objectives_with_live_inputs(monkeypatch):
    models = _crossfeeding_models()
    original_bounds = {
        species: {reaction.id: reaction.bounds for reaction in model.exchanges}
        for species, model in models.items()
    }
    solver = CooperativeCommunityFbaSolver(
        models,
        original_bounds,
        optimize_live_objectives=True,
        parsimonious_exchange=True,
        highs_method='highs-ds',
    )
    calls = []
    original_solve = solver._solve_linear_program

    def capture(objective, **kwargs):
        calls.append({
            'objective': objective,
            'A_eq': kwargs['A_eq'],
            'A_ub': kwargs['A_ub'],
            'b_ub': np.asarray(kwargs['b_ub']).copy(),
            'bounds': tuple(kwargs['bounds']),
        })
        return original_solve(objective, **kwargs)

    monkeypatch.setattr(solver, '_solve_linear_program', capture)
    states = (
        ({'producer': 1., 'consumer': .7}, {'carbon_e': 10., 'factor_e': 0.}),
        ({'producer': .23, 'consumer': 1.9}, {'carbon_e': 2., 'factor_e': .4}),
    )
    results = []
    for biomass, medium in states:
        results.append(solver.solve(models, biomass, medium, dt=.2))
    assert all(solution is not None for result in results for solution in result.values())
    assert len(calls) == 6
    # stage 1 and 2 use one equality object both within and between time steps.
    assert len({id(calls[index]['A_eq']) for index in (0, 1, 3, 4)}) == 1
    assert calls[0]['A_eq'] is solver._stage_a_eq
    # The separately widened equality structure for stage 3 is also stable.
    assert calls[2]['A_eq'] is calls[5]['A_eq']
    assert calls[2]['A_eq'] is solver._parsimonious_exchange_template['extended_eq']
    assert calls[2]['A_eq'] is not solver._stage_a_eq
    assert calls[0]['objective'] is calls[3]['objective'] is solver._stage1_objective
    assert calls[2]['objective'] is calls[5]['objective']
    assert not solver._stage1_objective.flags.writeable
    assert not solver._stage_a_eq.data.flags.writeable
    assert not calls[2]['objective'].flags.writeable
    assert not calls[2]['A_eq'].data.flags.writeable
    # State-dependent supplies, biomass coefficients and t bounds still change.
    assert not np.array_equal(calls[0]['b_ub'], calls[3]['b_ub'])
    assert not np.array_equal(calls[0]['A_ub'].data, calls[3]['A_ub'].data)
    assert calls[1]['bounds'][-1] != calls[4]['bounds'][-1] or (
        results[0]['producer'].fluxes['Growth'] != results[1]['producer'].fluxes['Growth']
    )


def test_solver_rebuilds_stage_three_template_after_private_metadata_edit():
    models = _crossfeeding_models()
    solver = CooperativeCommunityFbaSolver(models)
    old = solver._parsimonious_exchange_template
    metabolite = next(iter(solver._exchange_terms))
    species, index, stoich, reaction = solver._exchange_terms[metabolite][0]
    solver._exchange_terms[metabolite][0] = (
        species, index, np.float32(stoich), reaction
    )
    variables, new = solver._stage_three_template()
    assert new is solver._parsimonious_exchange_template
    assert new is not old
    assert len(variables) == sum(map(len, solver._exchange_terms.values()))

