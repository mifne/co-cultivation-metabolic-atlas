"""Exact stage-three LP equivalence against the original row-by-row assembly."""

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack

from src.community_solver import _assemble_parsimonious_exchange_lp


def _legacy_assembly(
    a_ub, b_ub, a_eq, n_fluxes, exchange_variables, biomass_g_l,
    global_objective, aggregate, optimize_live_objectives, objective_fraction,
):
    """The previous implementation, kept only as a regression oracle."""
    n_aux = len(exchange_variables)
    extended_ub = hstack(
        [a_ub, csr_matrix((a_ub.shape[0], n_aux))], format="csr"
    )
    extra_rows = []
    extra_rhs = []
    if optimize_live_objectives and aggregate > 1e-9:
        performance = np.zeros(n_fluxes + 1 + n_aux)
        performance[:n_fluxes] = -global_objective
        extra_rows.append(csr_matrix(performance.reshape(1, -1)))
        extra_rhs.append(-aggregate * objective_fraction)
    for aux, (species, index, stoich) in enumerate(exchange_variables):
        scale = max(1e-12, float(biomass_g_l[species])) * stoich
        for sign in (1.0, -1.0):
            row = np.zeros(n_fluxes + 1 + n_aux)
            row[index] = sign * scale
            row[n_fluxes + 1 + aux] = -1.0
            extra_rows.append(csr_matrix(row.reshape(1, -1)))
            extra_rhs.append(0.0)
    extended_ub = vstack([extended_ub, *extra_rows], format="csr")
    extended_rhs = np.concatenate((b_ub, np.asarray(extra_rhs)))
    extended_eq = hstack(
        [a_eq, csr_matrix((a_eq.shape[0], n_aux))], format="csr"
    )
    objective = np.zeros(n_fluxes + 1 + n_aux)
    objective[n_fluxes + 1:] = 1.0
    return objective, extended_ub, extended_rhs, extended_eq


def _assert_same_csr(actual, expected):
    assert actual.format == expected.format == "csr"
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "optimize,aggregate", [(True, 1.5), (False, 1.5), (True, 0.0), (True, 1e-9)]
)
def test_vectorized_exchange_assembly_matches_legacy_exactly(seed, optimize, aggregate):
    rng = np.random.default_rng(seed)
    n_fluxes = 17
    ub = rng.normal(size=(5, n_fluxes + 1))
    ub[rng.random(ub.shape) < 0.7] = 0.0
    eq = rng.normal(size=(7, n_fluxes + 1))
    eq[rng.random(eq.shape) < 0.8] = 0.0
    a_ub, a_eq = csr_matrix(ub), csr_matrix(eq)
    b_ub = rng.normal(size=5)
    objective = rng.normal(size=n_fluxes)
    objective[::3] = 0.0
    # Arbitrary exchange ordering, repeated reaction index, zero coefficient,
    # and the exact original biomass floor semantics must all be preserved.
    exchanges = [
        ("a", 13, -1.0), ("b", 0, 2.0), ("a", 5, 0.0),
        ("c", 16, -0.25), ("b", 0, -2.0), ("d", 7, 1.0),
    ]
    args = (
        a_ub, b_ub, a_eq, n_fluxes, exchanges,
        {"a": 1.0, "b": 0.0, "c": 0.031, "d": -1.0},
        objective, aggregate, optimize, 0.97,
    )
    actual = _assemble_parsimonious_exchange_lp(*args)
    expected = _legacy_assembly(*args)
    np.testing.assert_array_equal(actual[0], expected[0])
    _assert_same_csr(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    _assert_same_csr(actual[3], expected[3])
    np.testing.assert_array_equal(a_ub.toarray(), ub)
    np.testing.assert_array_equal(a_eq.toarray(), eq)
    # Downstream consumers must not be able to mutate the original matrices.
    actual[1].data[:] = 0.0
    actual[3].data[:] = 0.0
    np.testing.assert_array_equal(a_ub.toarray(), ub)
    np.testing.assert_array_equal(a_eq.toarray(), eq)


def test_vectorized_exchange_assembly_same_highs_solution():
    n_fluxes = 2
    a_ub = csr_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b_ub = np.array([4.0, 3.0])
    a_eq = csr_matrix([[1.0, -1.0, 0.0]])
    exchanges = [("a", 0, -1.0), ("b", 1, 1.0)]
    args = (
        a_ub, b_ub, a_eq, n_fluxes, exchanges, {"a": 1.0, "b": 0.5},
        np.array([1.0, 1.0]), 4.0, True, 0.95,
    )
    # Bounds, equality RHS, and objectives are unchanged by the caller.
    bounds = [(0.0, 4.0), (0.0, 3.0), (0.0, 0.0)] + [(0.0, None)] * 2
    results = []
    for assemble in (_legacy_assembly, _assemble_parsimonious_exchange_lp):
        objective, extended_ub, extended_rhs, extended_eq = assemble(*args)
        result = linprog(
            objective, A_ub=extended_ub, b_ub=extended_rhs,
            A_eq=extended_eq, b_eq=np.zeros(1), bounds=bounds, method="highs",
        )
        assert result.success
        results.append(result)
    np.testing.assert_array_equal(results[0].x, results[1].x)
    assert results[0].fun == results[1].fun
