from types import SimpleNamespace

import numpy as np
import pytest

from src.compact_bank_coverage import (
    measure_compact_bank_coverage,
    save_compact_coverage_npz,
    summarize_routing_coverage,
)


class FakeEvaluator:
    def __init__(self, candidate, mode='normal'):
        self.candidate = candidate
        self.mode = mode

    def solve_device(self, **inputs):
        batch, columns = inputs['lower'].shape
        rows = inputs['rhs'].shape[1]
        state = inputs['rhs'][:, 0].astype(int)
        accepted = state % 3 == self.candidate
        family = np.ones(batch, dtype=bool)
        raw = np.zeros((batch, columns))
        if self.mode == 'family':
            family[accepted] = False
        elif self.mode == 'nonfinite':
            raw[accepted, 0] = np.nan
        primal = np.where(accepted, 0., 1.)
        if self.mode == 'gate':
            primal[accepted] = 2e-5
        elif self.mode == 'negative_primal':
            primal[accepted] = -1e-12
        dual = np.zeros(batch)
        gap = np.zeros(batch)
        if self.mode == 'negative_dual':
            dual[accepted] = -1e-12
        elif self.mode == 'negative_gap':
            gap[accepted] = -1e-12
        result = dict(
            accepted=accepted.copy(), input_family_valid=family,
            primal_residual=primal,
            dual_violation=dual, relative_kkt_gap=gap,
            objective=np.zeros(batch), raw_values=raw,
            raw_reduced=np.zeros((batch, columns)), raw_y=np.zeros((batch, rows)),
            activity=np.zeros((batch, rows)),
        )
        if self.mode == 'accepted_int':
            result['accepted'] = np.where(result['accepted'], -1, 0).astype(np.int8)
        elif self.mode == 'family_int':
            result['input_family_valid'] = np.where(
                result['input_family_valid'], -1, 0).astype(np.int8)
        return result


class MissingFamilyEvaluator(FakeEvaluator):
    def solve_device(self, **inputs):
        result = super().solve_device(**inputs)
        result.pop('input_family_valid')
        return result


class FakeBank:
    cp = np

    def __init__(self, evaluators=None):
        self.host_a = SimpleNamespace(shape=(2, 3))
        self.variable_rows = np.array([1])
        self.evaluators = evaluators or [FakeEvaluator(i) for i in range(3)]
        self.centers = np.array([[0.], [1.], [2.]])
        self.feature_scale = np.ones(1)

    def proposal_features(self, inputs):
        return inputs['rhs'][:, :1]


def inputs(states):
    states = np.asarray(states, dtype=float)
    batch = len(states)
    return dict(
        rhs=np.column_stack([states, np.zeros(batch)]),
        lower=np.zeros((batch, 3)), upper=np.ones((batch, 3)),
        c=np.zeros((batch, 3)), delta=np.zeros((batch, 1, 3)),
        col_scale=np.ones((batch, 3)), row_scale=np.ones((batch, 2)),
    )


def test_streamed_coverage_and_stateless_routing_summary():
    result = measure_compact_bank_coverage(FakeBank(),
        [inputs([0, 1]), inputs([2, 0])], candidate_chunk_size=2,
        routing_topk=(1, 2))
    assert result.complete
    assert result.batch_sizes == (2, 2)
    assert result.coverage.tolist() == [
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [True, False, False],
    ]
    assert result.routing_order.tolist() == [[0, 1], [1, 0], [2, 1], [0, 1]]
    summary = summarize_routing_coverage(result)
    assert summary['oracle_covered_rows'] == 4
    assert summary['topk']['1'] == dict(
        accepted_rows=4, acceptance_rate=1., ranking_miss_rows=0)


def test_family_full_vector_finiteness_and_visible_gate_fail_closed():
    bank = FakeBank([FakeEvaluator(0, 'family'), FakeEvaluator(1, 'nonfinite'),
                     FakeEvaluator(2, 'gate')])
    result = measure_compact_bank_coverage(bank, [inputs([0, 1, 2])], routing_topk=(1, 2))
    assert result.complete
    assert not result.coverage.any()


@pytest.mark.parametrize('metric', ['primal', 'dual', 'gap'])
def test_negative_certificate_metrics_fail_closed(metric):
    bank = FakeBank([FakeEvaluator(0, f'negative_{metric}')])
    bank.centers = np.array([[0.]])
    result = measure_compact_bank_coverage(bank, [inputs([0])], routing_topk=(1,))
    assert result.complete
    assert result.coverage.tolist() == [[False]]


@pytest.mark.parametrize('mode', ['accepted_int', 'family_int'])
def test_nonboolean_certificate_flags_are_recorded_as_failure(mode):
    bank = FakeBank([FakeEvaluator(0, mode)])
    bank.centers = np.array([[0.]])
    result = measure_compact_bank_coverage(bank, [inputs([0])], routing_topk=(1,))
    assert not result.complete
    assert result.coverage.tolist() == [[False]]
    assert result.candidate_failures[0]['error_type'] == 'ValueError'
    assert 'must be boolean' in result.candidate_failures[0]['error']


def test_missing_certificate_field_is_recorded_and_cannot_be_summarized():
    bank = FakeBank([MissingFamilyEvaluator(0)])
    bank.centers = np.array([[0.]])
    result = measure_compact_bank_coverage(bank, [inputs([0])], routing_topk=(1,))
    assert not result.complete
    assert result.coverage.tolist() == [[False]]
    assert result.candidate_failures[0]['candidate_index'] == 0
    with pytest.raises(ValueError, match='Incomplete'):
        summarize_routing_coverage(result)


def test_subset_requires_all_routed_candidates_for_summary():
    result = measure_compact_bank_coverage(FakeBank(), [inputs([0, 1])],
        candidate_indices=[0], routing_topk=(1, 2))
    with pytest.raises(ValueError, match='not measured'):
        summarize_routing_coverage(result)


def test_input_and_index_validation():
    bank = FakeBank()
    with pytest.raises(ValueError, match='exact compact LP fields'):
        measure_compact_bank_coverage(bank, [dict(inputs([0]), extra=np.zeros(1))],
            routing_topk=(1,))
    with pytest.raises(ValueError, match='integer sequence'):
        measure_compact_bank_coverage(bank, [inputs([0])], candidate_indices=[True],
            routing_topk=(1,))
    with pytest.raises(ValueError, match='positive integer'):
        measure_compact_bank_coverage(bank, [inputs([0])], candidate_chunk_size=0,
            routing_topk=(1,))


def test_nonpickle_atomic_npz_contains_boolean_coverage(tmp_path):
    result = measure_compact_bank_coverage(FakeBank(), [inputs([0, 1, 2])],
        routing_topk=(1, 2))
    path = tmp_path/'coverage.npz'
    save_compact_coverage_npz(path, result, dict(source='training-only'))
    with np.load(path, allow_pickle=False) as saved:
        assert saved['coverage'].dtype == np.bool_
        assert bool(saved['complete'])
        assert str(saved['metadata']) == '{"source": "training-only"}'
        assert 'original-LP certificate' in str(saved['scope'])
    with pytest.raises(FileExistsError):
        save_compact_coverage_npz(path, result, {})


def test_real_cupy_compact_bank_streams_strict_original_certificates_without_cpu_lp(monkeypatch):
    pytest.importorskip('cupy')
    import highspy

    from src.gpu_basis_bank import GpuBasisBank
    from src.gpu_certified_basis import compile_basis
    from src.gpu_compact_basis import CompactBank, project_basis
    from tests.test_gpu_revised_basis import problem

    anchor_problem = problem()
    rhs_update = problem(rhs=(6., 3.))
    unsupported_matrix_update = problem(coef=1.1)
    anchor = compile_basis(anchor_problem.a, anchor_problem.rhs,
        anchor_problem.lower, anchor_problem.upper, anchor_problem.c,
        anchor_problem.neq)
    full = GpuBasisBank([anchor], [0])
    prepared = full.prepare_host([anchor_problem, rhs_update])
    training = {name:value.get() for name, value in prepared.items()}
    projected = project_basis(anchor, training, np.zeros((1, 1, 2)), [0])
    bank = CompactBank(dict(a=anchor_problem.a, neq=0), [0], [projected],
        np.zeros((1, 1)), np.array([0]), np.ones(1))

    def forbidden(*args, **kwargs):
        raise AssertionError('Online CPU LP is forbidden during GPU coverage')

    monkeypatch.setattr(highspy.Highs, 'run', forbidden)
    queries = [anchor_problem, rhs_update, unsupported_matrix_update]
    compact_inputs = bank.prepare_host(queries)
    direct = bank.evaluators[0].solve_device(**compact_inputs)
    result = measure_compact_bank_coverage(bank, [compact_inputs],
        candidate_indices=[0], candidate_chunk_size=1, routing_topk=(1,))

    assert result.complete
    assert result.coverage[:, 0].tolist() == direct['accepted'].get().tolist()
    assert result.coverage[:, 0].tolist() == [True, True, False]
    assert result.candidate_failures == ()
