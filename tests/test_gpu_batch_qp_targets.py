import numpy as np
import pytest

from src.gpu_batch_qp import BatchedCooperativeQpProjector


def test_mass_weighted_phbv_target_interpolates_both_repeat_units():
    pytest.importorskip("torch")
    projector = BatchedCooperativeQpProjector(
        ["NS21"], ["unused_e"], [], device="cpu",
        parsimonious_reference=False,
    )
    result = projector.project(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.zeros(2), np.ones(2), np.ones(1), np.ones(1),
        target_reaction_indices=[0, 1],
        target_reaction_weights=[0.08609, 0.10012],
        target_flux=[(0.08609 + 0.10012) / 2.0],
    )
    assert result.feasible.all()
    assert result.fluxes[0] == pytest.approx([0.5, 0.5], abs=1e-5)


def test_fast_path_reports_blended_residual_and_correct_shared_index():
    pytest.importorskip("torch")
    projector = BatchedCooperativeQpProjector(
        ["toy"], ["first_e", "second_e"], [(1, 0, 0, 1.0)],
        device="cpu", parsimonious_reference=False,
    )
    result = projector.project(
        [[0.0], [0.0002]], [0.0], [1.0], [1.0], [0.0, 0.0],
        target_reaction_index=0, target_flux=[0.0001],
    )
    assert result.feasible.all()
    assert result.fluxes[0, 0] == pytest.approx(0.0001)
    assert result.max_shared_violation[0] == pytest.approx(0.0001)
    assert result.max_shared_violation_index[0] == 1


def test_target_projection_uses_complementary_infeasible_anchors():
    # Only [0,0] is individually feasible. The two infeasible anchors can
    # nevertheless produce [0.8,0.8], unreachable by blending feasible anchors.
    projector = BatchedCooperativeQpProjector(
        ["toy"], ["unused_e"], [], device="cpu",
        parsimonious_reference=False, maximum_iterations=4000,
    )
    result = projector.project(
        [[0, 0], [2, 0], [0, 2]], [0, 0], [1, 1], [1], [1],
        target_reaction_indices=[0, 1], target_reaction_weights=[0.5, 0.5],
        target_flux=[0.8], enforce_target_in_hull=True,
    )
    assert result.feasible.all()
    assert result.target_projection_attempted
    assert result.target_projection_success.all()
    assert result.fluxes.shape == (1, 2)
    assert result.fluxes.mean() == pytest.approx(0.8, abs=0.003)
    assert np.max(result.fluxes) <= 1.0002


def test_unattainable_neural_target_keeps_physical_gpu_result():
    projector = BatchedCooperativeQpProjector(
        ["toy"], ["unused_e"], [], device="cpu", maximum_iterations=100,
    )
    result = projector.project(
        [[0, 0], [2, 2]], [0, 0], [0.5, 0.5], [1], [1],
        target_reaction_index=0, target_flux=[1.0], enforce_target_in_hull=True,
    )
    assert result.feasible.all()
    assert result.target_projection_attempted
    assert not result.target_projection_success.any()
    assert np.max(result.fluxes) <= 0.5002


def test_redundant_internal_columns_do_not_prevent_target_convergence():
    projector = BatchedCooperativeQpProjector(
        ["toy"], ["unused_e"], [], device="cpu", maximum_iterations=4000,
    )
    rng = np.random.default_rng(5)
    fluxes = np.concatenate(([[0, 0], [2, 0], [0, 2]],
                            rng.uniform(-1000, 1000, (3, 1000))), axis=1)
    result = projector.project(
        fluxes, np.r_[0, 0, np.full(1000, -1000)],
        np.r_[1, 1, np.full(1000, 1000)], [1], [1],
        target_reaction_indices=[0, 1], target_reaction_weights=[0.5, 0.5],
        target_flux=[0.8], enforce_target_in_hull=True,
    )
    assert result.feasible.all() and result.target_projection_success.all()
    assert result.fluxes[0, :2].mean() == pytest.approx(0.8, abs=0.003)
    assert np.abs(result.fluxes[0, 2:]).max() <= 1000


def test_batched_admm_merges_target_success_and_physical_fallback_separately():
    projector = BatchedCooperativeQpProjector(
        ["toy"], ["unused_e"], [], device="cpu", maximum_iterations=2000,
    )
    result = projector.project(
        np.asarray([[[0, 0], [2, 0], [0, 2]], [[0, 0], [2, 2], [2, 2]]]),
        np.zeros((2, 2)), np.asarray([[1, 1], [0.5, 0.5]]), np.ones((2, 1)),
        np.ones((2, 1)), target_reaction_indices=[0, 1],
        target_reaction_weights=[0.5, 0.5], target_flux=[0.8, 1.0],
        enforce_target_in_hull=True,
    )
    assert result.feasible.all()
    assert result.target_projection_success.tolist() == [True, False]
    assert result.fluxes[0].mean() == pytest.approx(0.8, abs=0.003)
    assert result.fluxes[1].max() <= 0.5002
