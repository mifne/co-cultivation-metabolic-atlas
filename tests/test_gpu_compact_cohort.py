"""Fixed-cohort screening must certify every environment, not just neighbors."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_basis_bank import GpuBasisBank
from src.gpu_certified_basis import compile_basis
from src.gpu_compact_basis import CompactBank, project_basis
from tests.test_gpu_revised_basis import cpu, problem
from tests.test_gpu_compact_observables import _bank


def _real_bank(capture):
    pytest.importorskip('cupy')
    p, q = problem(), problem(rhs=(0.1, 3.))
    anchors = [compile_basis(v.a, v.rhs, v.lower, v.upper, v.c, v.neq) for v in (p, q)]
    full = GpuBasisBank([anchors[0]], [0])
    training = {key: value.get() for key, value in full.prepare_host([p, q]).items()}
    entries = [project_basis(anchor, training, np.zeros((1, 1, 2)), [0]) for anchor in anchors]
    bank = CompactBank(
        dict(a=p.a, neq=0), [0], entries + [entries[0]],
        np.zeros((3, 1)), np.array([0]), np.ones(1),
        capture=capture, full_batch_candidates=True,
    )
    return bank, p, q


@pytest.mark.parametrize('capture', [False, True])
def test_cohort_certifies_non_neighbor_candidates_without_cpu_lp(capture, monkeypatch):
    import highspy
    import scipy.optimize
    bank, p, q = _real_bank(capture)
    expected = [cpu(p).fun, cpu(q).fun]

    def forbidden(*args, **kwargs):
        raise AssertionError('Online CPU LP forbidden')

    monkeypatch.setattr(highspy.Highs, 'run', forbidden)
    monkeypatch.setattr(scipy.optimize, 'linprog', forbidden)
    try:
        inputs = bank.prepare_host([p, q])
        # Each environment's supplied nearest candidate has the wrong active set.
        near = bank.evaluate_device(inputs, order=np.array([[1], [0]]))
        assert near['accepted'].get().tolist() == [False, False]
        output = bank.evaluate_cohort(inputs, [0, 1])
        assert output['accepted'].get().tolist() == [True, True]
        assert output['candidate_index'].get().tolist() == [0, 1]
        np.testing.assert_allclose(output['objective'].get(), expected, atol=1e-7)
        assert output['cpu_lp_calls'] == 0
        assert output['candidate_evaluations'] == 4
        assert len(bank.last_candidates) == 2
        for _, ids, certificate in bank.last_candidates:
            np.testing.assert_array_equal(ids, [0, 1])
            assert certificate['accepted'].shape == (2,)
        # Fixed batch shape replays with changed inputs, not stale certificates.
        reverse = bank.evaluate_cohort(bank.prepare_host([q, p]), [0, 1])
        assert reverse['candidate_index'].get().tolist() == [1, 0]
        np.testing.assert_allclose(reverse['objective'].get(), expected[::-1], atol=1e-7)
        if capture:
            assert len(bank.graph_cache) == 2
    finally:
        bank.clear_graph_cache()


@pytest.mark.parametrize('capture', [False, True])
def test_cohort_keeps_first_certificate_and_counts_all_candidates(capture):
    bank, p, q = _real_bank(capture)
    bank.configure_observables(csr_matrix([[1., 0.], [0., 1.]]), [1., 1.])
    try:
        output = bank.evaluate_cohort(bank.prepare_host([p, q]), [2, 0, 1])
        assert output['accepted'].get().all()
        # Candidate 0 also solves row 0 but cannot replace the earlier candidate 2.
        assert output['candidate_index'].get().tolist() == [2, 1]
        assert output['best_candidate_index'].get().tolist() == [2, 1]
        assert output['candidate_count'].get().tolist() == [3, 3]
        assert output['candidate_evaluations'] == 6
        assert np.isfinite(output['observable_dispersion'].get()).all()
        assert np.isfinite(bank.last_candidate_scores.get()).all()
        assert [item[0] for item in bank.last_candidates] == [2, 0, 1]
    finally:
        bank.clear_graph_cache()


@pytest.mark.parametrize('ranking', ['residual', 'count', 'count_only'])
def test_uncertified_cohort_preserves_best_repair_proposal(ranking):
    physical = np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]])
    bank, inputs = _bank(False, physical)
    cp = bank.cp
    bank.candidate_ranking = ranking
    for index, evaluator in enumerate(bank.evaluators):
        evaluate = evaluator.evaluate

        def with_score(query, evaluate=evaluate, index=index):
            output = evaluate(query)
            output['primal_residual'] = cp.full(1, [1e-4, 1e-3, 5e-4][index])
            output['primal_violation_count'] = cp.full(1, [1., 3., 2.][index])
            output['primal_violation_l1'] = cp.zeros(1)
            output['basis_dual_violation'] = cp.full(1, 1. if index == 0 else 0.)
            return output

        evaluator.evaluate = with_score
    output = bank.evaluate_cohort(inputs, [0, 1, 2])
    best = 0 if ranking == 'count_only' else 2
    assert not output['accepted'].get()[0]
    assert output['candidate_index'].get().tolist() == [-1]
    assert output['best_candidate_index'].get().tolist() == [best]
    assert bank.rank().get()[0, 0] == best
    np.testing.assert_array_equal(output['raw_values'].get(), physical[best] * inputs['col_scale'].get())
    assert np.isnan(output['values'].get()).all()


@pytest.mark.parametrize('indices', [[0, 0], [-1], [2], [[0]], [0.5]])
def test_cohort_rejects_invalid_host_candidate_sets(indices):
    bank, inputs = _bank(False, np.zeros((2, 1, 2)))
    with pytest.raises(ValueError):
        bank.evaluate_cohort(inputs, indices)


def test_empty_cohort_fails_closed_without_inventing_solution():
    bank, inputs = _bank(False, np.zeros((1, 1, 2)))
    bank.configure_observables(csr_matrix([[1., 0.]]), [1.])
    output = bank.evaluate_cohort(inputs, [])
    assert not output['accepted'].get().any()
    assert output['candidate_evaluations'] == 0
    assert output['candidate_count'].get().tolist() == [0]
    assert np.isinf(output['observable_dispersion'].get()).all()
