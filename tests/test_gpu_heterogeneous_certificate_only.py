"""GPU tests for the opt-in strict-certificate-only heterogeneous path."""

import numpy as np
import pytest

from src.gpu_heterogeneous_compact import HeterogeneousCompactBank
from tests.test_gpu_heterogeneous_compact import _fixture, _snapshot, _assert_snapshot
from tests.test_gpu_revised_basis import problem


REQUIRED = {
    "accepted", "candidate_index", "values", "objective",
    "primal_residual", "dual_violation", "relative_kkt_gap",
    "candidate_evaluations",
}


def _assert_required_equal(compact, full):
    assert set(compact) == REQUIRED
    assert compact["candidate_evaluations"] == full["candidate_evaluations"]
    for key in REQUIRED - {"candidate_evaluations"}:
        np.testing.assert_allclose(
            compact[key].get(), full[key].get(), atol=1e-11, rtol=1e-11,
            equal_nan=True,
        )


def test_certificate_only_finite_results_and_first_accepted_slot_match_full_path():
    cp, bank, full_service, training = _fixture()
    certificate_service = HeterogeneousCompactBank(bank, certificate_only=True)
    inputs = bank.prepare_host(training[:4])
    order = cp.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=cp.int32)
    try:
        full = full_service.evaluate(inputs, order)
        compact = certificate_service.evaluate(inputs, order)
        _assert_required_equal(compact, full)
        assert compact["accepted"].get().all()
        assert compact["candidate_index"].get().tolist() == [0, 1, 0, 1]
        assert certificate_service.last_candidate_scores is None
        assert certificate_service.last_candidate_dual is None
        assert certificate_service.last_candidates is None
    finally:
        certificate_service.close(); full_service.close(); bank.clear_graph_cache()


def test_certificate_only_keeps_all_family_and_original_certificate_rejections():
    cp, bank, full_service, training = _fixture()
    certificate_service = HeterogeneousCompactBank(bank, certificate_only=True)
    queries = [
        training[0],                       # invalid first ID, valid second ID
        training[0],                       # nonpositive coordinate scale
        training[0],                       # inconsistent variable bounds
        training[0],                       # unsupported matrix direction
        problem(cost=(2., 1.)),            # changed sign/dual certificate
        problem(rhs=(-1., 3.)),            # infeasible original constraints
    ]
    inputs = bank.prepare_host(queries)
    inputs["row_scale"][1, 0] = 0.
    inputs["lower"][2, 0] = 2.
    inputs["upper"][2, 0] = 1.
    inputs["delta"][3, 0, 1] = .5
    order = cp.asarray([[-1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                       dtype=cp.int32)
    try:
        full = full_service.evaluate(inputs, order)
        compact = certificate_service.evaluate(inputs, order)
        np.testing.assert_array_equal(compact["accepted"].get(), full["accepted"].get())
        np.testing.assert_array_equal(compact["candidate_index"].get(), full["candidate_index"].get())
        assert compact["accepted"].get()[0]
        assert compact["candidate_index"].get()[0] == 0
        assert not compact["accepted"].get()[1:4].any()
        # The final two rows exercise the unchanged strict certificate.  Their
        # exact acceptance is compared to the full path rather than assumed.
        rejected = ~compact["accepted"].get()
        assert np.isnan(compact["values"].get()[rejected]).all()
        assert (compact["candidate_index"].get()[rejected] == -1).all()
    finally:
        certificate_service.close(); full_service.close(); bank.clear_graph_cache()


def test_certificate_only_replay_changes_with_inputs_and_order_but_matches_eager():
    cp, bank, full_service, training = _fixture()
    service = HeterogeneousCompactBank(bank, certificate_only=True)
    first = bank.prepare_host(training[:2])
    changed = bank.prepare_host([training[3], training[2]])
    invalid = bank.prepare_host([training[0], training[1]])
    invalid["lower"][0, 0] = 3.
    invalid["upper"][0, 0] = 2.
    pairs = [
        (first, [[0, 1], [1, 0]]),
        (changed, [[1, 0], [0, 1]]),
        (invalid, [[0, 1], [-1, 99]]),
        (first, [[0, 1], [1, 0]]),
    ]
    try:
        for inputs, raw_order in pairs:
            order = cp.asarray(raw_order, dtype=cp.int32)
            expected = _snapshot(service.evaluate(inputs, order))
            replayed = service.evaluate_replay(inputs, order)
            _assert_snapshot(replayed, expected)
            assert set(replayed) == REQUIRED
            assert len(service.graph_cache) == 1
            assert service.last_candidate_scores is None
            assert service.last_candidate_dual is None
            assert service.last_candidates is None
        assert service.graph_compilation_seconds > 0.
    finally:
        service.close(); full_service.close(); bank.clear_graph_cache()


def test_certificate_only_graph_cache_key_separates_diagnostic_mode():
    cp, bank, full_service, training = _fixture()
    service = HeterogeneousCompactBank(bank, certificate_only=True)
    inputs = bank.prepare_host(training[:2])
    order = cp.asarray([[0, 1], [1, 0]], dtype=cp.int32)
    try:
        compact = service.evaluate_replay(inputs, order)
        assert set(compact) == REQUIRED and len(service.graph_cache) == 1
        service.certificate_only = False
        diagnostic = service.evaluate_replay(inputs, order)
        assert "candidate_metrics" in diagnostic
        assert len(service.graph_cache) == 2
        service.certificate_only = True
        replayed = service.evaluate_replay(inputs, order)
        assert set(replayed) == REQUIRED and len(service.graph_cache) == 2
    finally:
        service.close(); full_service.close(); bank.clear_graph_cache()


def test_certificate_only_constructor_requires_boolean():
    cp, bank, full_service, _ = _fixture()
    try:
        with pytest.raises(TypeError, match="boolean"):
            HeterogeneousCompactBank(bank, certificate_only=1)
    finally:
        full_service.close(); bank.clear_graph_cache()
