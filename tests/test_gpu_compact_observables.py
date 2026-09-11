"""Candidate spread is a GPU diagnostic, independent of LP acceptance."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_compact_basis import CompactBank


class _Candidate:
    def __init__(self, cp, physical, valid, accepted):
        self.cp = cp
        self.physical = cp.asarray(physical)
        self.valid = cp.asarray(valid)
        self.accepted = cp.asarray(accepted)
        self.calls = 0

    def evaluate(self, inputs):
        cp = self.cp
        self.calls += 1
        ids = inputs['rhs'][:, 0].astype(cp.int64)
        raw = self.physical[ids] * inputs['col_scale']
        accepted = self.accepted[ids]
        zeros = cp.zeros(len(ids))
        return dict(
            accepted=accepted, input_family_valid=self.valid[ids],
            raw_values=raw, values=cp.where(accepted[:, None], self.physical[ids], cp.nan),
            raw_reduced=cp.zeros_like(raw), raw_y=cp.zeros_like(inputs['rhs']),
            activity=cp.zeros_like(inputs['rhs']), objective=zeros,
            primal_residual=cp.ones(len(ids)), dual_violation=zeros,
            relative_kkt_gap=zeros, basis_dual_violation=zeros,
        )


def _bank(full_batch, physical, valid=None, accepted=None):
    cp = pytest.importorskip('cupy')
    candidates, batch, columns = physical.shape
    bank = CompactBank(
        dict(a=csr_matrix(np.eye(columns)), neq=0), [], [],
        np.zeros((candidates, 1)), np.array([0]), np.ones(1),
        full_batch_candidates=full_batch,
    )
    if valid is None:
        valid = np.ones((candidates, batch), dtype=bool)
    if accepted is None:
        accepted = np.zeros((candidates, batch), dtype=bool)
    bank.evaluators = [
        _Candidate(cp, physical[i], valid[i], accepted[i]) for i in range(candidates)
    ]
    inputs = dict(
        lower=cp.zeros((batch, columns)),
        rhs=cp.asarray(np.column_stack((np.arange(batch), np.zeros(batch)))),
        col_scale=cp.asarray(np.broadcast_to(np.arange(columns) + 2., (batch, columns))),
    )
    return bank, inputs


@pytest.mark.parametrize('full_batch', [False, True])
def test_observable_spread_matches_original_flux_sample_std(full_batch):
    rng = np.random.default_rng(2213)
    physical = rng.normal(size=(4, 3, 3))
    valid = np.ones((4, 3), dtype=bool)
    valid[1, 1] = False
    physical[2, 1, 2] = np.nan
    valid[1:, 2] = False
    bank, inputs = _bank(full_batch, physical, valid)
    projection = csr_matrix([[1., 0., 1.], [0., 2., 0.]])
    scales = np.array([2., 0.5])
    bank.configure_observables(projection, scales)
    order = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [2, 3, 0, 1]])
    output = bank.evaluate_device(inputs, order=order)
    np.testing.assert_array_equal(output['candidate_count'].get(), [4, 2, 1])
    expected = []
    for row in range(3):
        mask = valid[:, row] & np.isfinite(physical[:, row]).all(axis=1)
        projected = (projection @ physical[mask, row].T).T
        expected.append(
            np.max(np.std(projected, axis=0, ddof=1) / scales)
            if len(projected) >= 2 else np.inf
        )
    np.testing.assert_allclose(output['observable_dispersion'].get(), expected, rtol=1e-13)
    assert not output['accepted'].get().any()
    if full_batch:
        assert [e.calls for e in bank.evaluators] == [1, 1, 1, 1]


@pytest.mark.parametrize('full_batch', [False, True])
def test_repeated_candidate_order_never_double_counts(full_batch):
    physical = np.array([[[1., 2.]], [[5., 6.]]])
    bank, inputs = _bank(full_batch, physical)
    bank.configure_observables(csr_matrix([[1., 0.]]), [1.])
    output = bank.evaluate_device(inputs, order=np.array([[0, 0, 1, 1]]))
    assert output['candidate_count'].get().tolist() == [2]
    np.testing.assert_allclose(output['observable_dispersion'].get(), [np.sqrt(8.)])


@pytest.mark.parametrize('full_batch', [False, True])
def test_observables_do_not_change_accepted_solution_or_ranking(full_batch):
    physical = np.array([[[1., 2.], [3., 4.]], [[8., 9.], [10., 11.]]])
    accepted = np.array([[True, False], [False, True]])
    bank, inputs = _bank(full_batch, physical, accepted=accepted)
    order = np.array([[0, 1], [0, 1]])
    baseline = bank.evaluate_device(inputs, order=order)
    baseline_order = bank.rank().get()
    assert 'candidate_count' not in baseline
    assert 'observable_dispersion' not in baseline
    bank.configure_observables(csr_matrix([[1., 1.]]), [1.])
    diagnostic = bank.evaluate_device(inputs, order=order)
    for key in ('accepted', 'candidate_index', 'raw_values', 'values', 'input_family_valid'):
        np.testing.assert_array_equal(diagnostic[key].get(), baseline[key].get())
    np.testing.assert_array_equal(bank.rank().get(), baseline_order)
    assert diagnostic['candidate_count'].get().tolist() == ([2, 2] if full_batch else [1, 2])


@pytest.mark.parametrize(
    'matrix,scales',
    [
        (csr_matrix((0, 2)), []),
        (csr_matrix([[1., 0., 0.]]), [1.]),
        (csr_matrix([[np.nan, 0.]]), [1.]),
        (csr_matrix([[1., 0.]]), [[1.]]),
        (csr_matrix([[1., 0.]]), [1., 2.]),
        (csr_matrix([[1., 0.]]), [0.]),
        (csr_matrix([[1., 0.]]), [-1.]),
        (csr_matrix([[1., 0.]]), [np.inf]),
    ],
)
def test_observable_configuration_validation(matrix, scales):
    bank, _ = _bank(False, np.zeros((1, 1, 2)))
    with pytest.raises(ValueError):
        bank.configure_observables(matrix, scales)
