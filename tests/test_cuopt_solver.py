"""GPU correctness tests for the rank-reduced cuOpt FBA adapter."""

from __future__ import annotations

import numpy as np
import pytest
from cobra import Metabolite, Model, Reaction
from cobra.util.array import create_stoichiometric_matrix

from src.cuopt_solver import CuOptConfig, CuOptFbaSolver, cuopt_available
from src.dfba_simulator import dFBASimulator


def _cuda_available() -> bool:
    if not cuopt_available():
        return False
    try:
        from cupy.cuda import runtime

        return runtime.getDeviceCount() > 0
    except Exception:
        return False


def _rank_deficient_fba_model() -> Model:
    model = Model("rank_deficient_test")
    a = Metabolite("a_c")
    b = Metabolite("b_c")
    duplicate_b = Metabolite("duplicate_b_c")

    source = Reaction("SOURCE_A", lower_bound=0.0, upper_bound=10.0)
    source.add_metabolites({a: 1.0})
    conversion = Reaction("CONVERT", lower_bound=0.0, upper_bound=10.0)
    conversion.add_metabolites({a: -1.0, b: 1.0, duplicate_b: 1.0})
    product = Reaction("PRODUCT", lower_bound=0.0, upper_bound=10.0)
    product.add_metabolites({b: -1.0, duplicate_b: -1.0})
    model.add_reactions([source, conversion, product])
    model.objective = product
    return model


def test_safe_lower_bound_preserves_fixed_reaction() -> None:
    simulator = object.__new__(dFBASimulator)
    assert simulator._safe_lb(0.0, 0.0) == 0.0


@pytest.mark.skipif(not _cuda_available(), reason="cuOpt CUDA device unavailable")
def test_cuopt_removes_dependent_rows_and_preserves_full_residual() -> None:
    model = _rank_deficient_fba_model()
    matrix = create_stoichiometric_matrix(
        model, array_type="lil", dtype=np.float64
    ).tocsr()
    assert np.linalg.matrix_rank(matrix.toarray()) < matrix.shape[0]

    solver = CuOptFbaSolver(
        model,
        CuOptConfig(
            method="pdlp",
            time_limit=3.0,
            crossover=True,
            optimality_tolerance=1e-5,
            validation_tolerance=1e-7,
        ),
    )
    solution = solver.solve(model)

    assert solution is not None
    assert solution.objective_value == pytest.approx(10.0, abs=1e-6)
    assert solver.stats.dependent_rows_removed >= 1
    assert solver.stats.max_equality_residual <= 1e-7
    assert solver.stats.max_bound_violation <= 1e-7
