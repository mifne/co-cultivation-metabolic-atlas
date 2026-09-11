import numpy as np
import pytest

from src.gpu_batch_qp import BatchedCooperativeQpProjector
from src.gpu_multioutput_qp import MultiOutputCooperativeQpProjector
from src.gpu_multioutput_qp import structural_face_mask


def projector(device="cpu", **kwargs):
    pytest.importorskip("torch")
    return MultiOutputCooperativeQpProjector(BatchedCooperativeQpProjector(
        ["toy"], ["carbon_e"], [(0, 0, 0, 1.0)], device=device,
        parsimonious_reference=False, maximum_iterations=2000, **kwargs), strength=1000)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_joint_output_batch_improves_and_obeys_shared_supply(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    result = projector(device).project(
        np.asarray([[[0, 0], [2, 0], [0, 2]]]*2),
        np.zeros((2, 2)), np.ones((2, 2)), np.ones((2, 1)), [[0.5], [1.0]],
        decision_indices=[0, 1], decision_targets=[[0.8, 0.8], [0.2, 0.9]],
        decision_scales=[1, 1], decision_weights=[1, 1])
    assert result.feasible.all()
    assert result.multioutput_improved.all()
    assert np.all(result.multioutput_loss_after < result.multioutput_loss_before)
    assert result.fluxes[0] == pytest.approx([0.5, 0.8], abs=0.01)
    assert result.fluxes[1] == pytest.approx([0.2, 0.9], abs=0.01)
    assert result.weights.sum(axis=1) == pytest.approx([1, 1], abs=1e-6)
    assert result.weights.min() >= 0


def test_unattainable_and_closed_reaction_targets_remain_soft():
    result = projector().project([[0, 0], [0.2, 0]], [0, 0], [1, 0], [1], [1],
        decision_indices=[0, 1], decision_targets=[1, 100], decision_scales=[1, 1])
    assert result.feasible.all()
    assert result.fluxes[0] == pytest.approx([0.2, 0], abs=0.005)


def test_affine_projection_extends_hull_but_checks_previously_redundant_bounds():
    from scipy.sparse import csr_matrix
    base = BatchedCooperativeQpProjector(["toy"], ["unused_e"], [], device="cpu",
        parsimonious_reference=False, maximum_iterations=2000)
    affine = MultiOutputCooperativeQpProjector(base, 1000, signed_weights=True,
                                             stoichiometry=csr_matrix([[1., -1.]]))
    result = affine.project([[0., 0.], [.2, .2]], [0, 0], [.6, 1], [1], [1],
        decision_indices=[0, 1], decision_targets=[.8, 1], decision_scales=[1, 1])
    assert result.feasible.all()
    np.testing.assert_allclose(result.fluxes[0], [.6, .6], atol=.02)
    assert result.fluxes.max() <= .6002
    assert result.weights.min() < 0


def test_affine_projection_requires_an_explicit_mass_balance_audit():
    with pytest.raises(ValueError, match="stoichiometry"):
        MultiOutputCooperativeQpProjector(projector().base, signed_weights=True)


def test_independent_mixing_still_couples_shared_resources():
    base = BatchedCooperativeQpProjector(["A", "B"], ["carbon_e"],
        [(0, 0, 0, 1.), (0, 1, 1, 1.)], species_offsets=[(0, 1), (1, 2)],
        device="cpu", maximum_iterations=2000, block_composition_candidates=1)
    result = MultiOutputCooperativeQpProjector(base, 1000, independent_species=True).project(
        [[0., 1.], [1., 0.]], [0, 0], [1, 1], [1, 1], [1],
        decision_indices=[0, 1], decision_targets=[.8, .8], decision_scales=[1, 1])
    assert result.feasible.all()
    assert result.fluxes[0].sum() <= 1+base.shared_tolerance
    np.testing.assert_allclose(result.fluxes[0], [.5, .5], atol=.01)


def test_structural_face_mask_preserves_cancellation_and_handles_empty_group():
    torch = pytest.importorskip("torch")
    flux = torch.tensor([[[0., 0.], [1., 1.], [0., 2.]], [[-1., 0.], [1., 1.], [0., 2.]]])
    mask = structural_face_mask(flux, torch.tensor([[-1., 0.], [-1., 0.]]),
                                torch.tensor([[0., 2.], [0., 2.]]))
    assert mask.tolist() == [[True, False, True], [True, True, True]]
    empty = structural_face_mask(torch.tensor([[[1.], [2.], [0.], [0.]]]),
                                 torch.tensor([[0.]]), torch.tensor([[0.]]), groups=2)
    assert empty.all()


def test_soft_objective_matches_analytic_two_anchor_solution():
    result = projector().project([[0], [1]], [0], [1], [1], [1],
        decision_indices=[0], decision_targets=[0.8], decision_scales=[1])
    # alpha=1: minimize alpha*w^2 + 0.5*1000*(w-0.8)^2.
    assert result.fluxes[0, 0] == pytest.approx(1000*0.8/1002, abs=0.001)


def test_unconverged_iterate_is_recovered_without_relaxing_physics():
    base = BatchedCooperativeQpProjector(["toy"], ["unused_e"], [],
        device="cpu", maximum_iterations=1, parsimonious_reference=False)
    result = MultiOutputCooperativeQpProjector(base, 1000).project(
        [[0, 0], [2, 2]], [0, 0], [0.05, 0.8], [1], [1],
        decision_indices=[0, 1], decision_targets=[1, 1], decision_scales=[1, 1])
    assert result.feasible.all()
    assert result.multioutput_step_fraction[0] < 1
    assert result.max_bound_violation[0] <= base.bound_tolerance
    assert result.multioutput_loss_after[0] <= result.multioutput_loss_before[0]


def test_species_composition_preserves_layout():
    base = BatchedCooperativeQpProjector(["A", "B"], ["unused_e"], [],
        species_offsets=[(0, 1), (1, 2)], device="cpu", maximum_iterations=2000)
    result = MultiOutputCooperativeQpProjector(base, 1000).project(
        [[0, 1], [1, 0]], [0, 0], [1, 1], [1, 1], [1],
        decision_indices=[0, 1], decision_targets=[0.8, 0.8], decision_scales=[1, 1])
    assert result.feasible.all()
    assert result.weights.shape == (1, 4)
    assert result.fluxes[0] == pytest.approx([0.8, 0.8], abs=0.01)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_independent_species_simplexes_extend_hull_and_remain_batched(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    base = BatchedCooperativeQpProjector(["A", "B"], ["unused_e"], [],
        species_offsets=[(0, 1), (1, 2)], device=device, maximum_iterations=2000,
        block_composition_candidates=1, parsimonious_reference=False)
    result = MultiOutputCooperativeQpProjector(base, 1000, independent_species=True).project(
        np.asarray([[[0, 1], [1, 0]]]*2), np.zeros((2, 2)), np.ones((2, 2)),
        np.ones((2, 2)), np.ones((2, 1)), decision_indices=[0, 1],
        decision_targets=[[.8, .8], [.3, .3]], decision_scales=[1, 1])
    assert result.feasible.all()
    np.testing.assert_allclose(result.fluxes, [[.8, .8], [.3, .3]], atol=.01)
    np.testing.assert_allclose(result.weights.reshape(2, 2, 2).sum(axis=2), .5, atol=1e-6)


@pytest.mark.parametrize("target,scale", [([np.nan, 1], [1, 1]), ([1, 1], [0, 1])])
def test_invalid_neural_inputs_rejected(target, scale):
    with pytest.raises(ValueError):
        projector().project([[0, 0], [1, 1]], [0, 0], [1, 1], [1], [1],
            decision_indices=[0, 1], decision_targets=target, decision_scales=scale)


def test_offline_independent_species_oracle_does_not_share_simplex_weights():
    from scripts.audit_multioutput_hull_capacity import independent_operators
    from scipy.optimize import linprog
    from scipy import sparse
    base = BatchedCooperativeQpProjector(["A", "B"], ["unused_e"], [],
        species_offsets=[(0, 1), (1, 2)], device="cpu")
    matrix, rhs, decision, equality = independent_operators(
        np.asarray([[0., 1.], [1., 0.]]), base, [1, 1], np.zeros(2), np.ones(2), np.ones(1), [0, 1])
    result = linprog(np.zeros(4), A_ub=matrix, b_ub=rhs,
        A_eq=sparse.vstack((equality, sparse.csr_matrix(decision))),
        b_eq=[1, 1, .8, .8], bounds=(0, None), method="highs")
    assert result.success  # impossible with the joint simplex (v_A + v_B = 1)
    assert decision @ result.x == pytest.approx([.8, .8])
