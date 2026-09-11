"""The CPU handoff is an initialization proposal, never an accepted flux."""
from copy import deepcopy

import numpy as np
import pytest

from src.gpu_hybrid_lp import HybridCertifiedBackend, cpu_fallback_requests
from tests.test_gpu_hybrid_backend import tiny_hybrid_factory, _request


def test_fallback_preserves_current_initializer_and_uses_prior_certified_basis():
    requests = [([1.], dict(bounds=(0., 2.))), ([-1.], dict(bounds=(0., 3.)))]
    before = deepcopy(requests)
    templates = [dict(col_status=[1], row_status=[0]),
                 dict(col_status=[0], row_status=[1])]
    result = cpu_fallback_requests(requests, templates, [1, 0], [0, 1])
    assert requests == before
    assert result[0][1]['_initial_basis'] is templates[1]
    assert result[0][1]['_basis_override'] == dict(
        basis=templates[0], source='previous_certified_gpu', age=1)
    assert result[1][1]['_basis_override']['basis'] is templates[1]
    assert result[0][0] is requests[0][0]


def test_missing_or_out_of_range_candidates_never_select_last_template():
    requests = [([1.], {}) for _ in range(3)]
    templates = [dict(col_status=[0], row_status=[1])]
    result = cpu_fallback_requests(requests, templates, [-1, 1, 0], [-1, 8, -1])
    assert result[0][1] == {} and result[1][1] == {}
    assert '_initial_basis' in result[2][1]
    assert all('_basis_override' not in kwargs for _, kwargs in result)


def test_disabled_handoff_preserves_default_proposal_only_behavior():
    template = dict(col_status=np.array([0]), row_status=np.array([1]))
    result = cpu_fallback_requests([([1.], {})], [template], [0])
    assert result[0][1] == {'_initial_basis': template}


@pytest.mark.parametrize('current,previous', [([], None), ([0], []), ([0, 0], [0])])
def test_candidate_lengths_are_checked(current, previous):
    with pytest.raises(ValueError, match='candidate per CPU request'):
        cpu_fallback_requests([([1.], {})], [], current, previous)


@pytest.mark.parametrize('options', [
    dict(cpu_basis_proposals=False, repair_rounds=0, gpu_stages=('maxmin',)),
    dict(cpu_basis_proposals=True, repair_rounds=1, gpu_stages=('maxmin',)),
    dict(cpu_basis_proposals=True, repair_rounds=0, gpu_stages=('exchange',)),
    dict(cpu_basis_proposals=True, repair_rounds=0, gpu_stages=('maxmin',), candidate_oracle=True),
])
def test_constructor_limits_handoff_to_the_tested_execution_policy(options):
    with pytest.raises(ValueError, match='CPU basis handoff requires'):
        HybridCertifiedBackend(None, {}, cpu_basis_handoff=True, **options)


def test_real_gpu_acceptance_hands_basis_to_existing_exact_cpu(tiny_hybrid_factory, monkeypatch):
    import src.gpu_hybrid_lp as module
    service, bank, p, q = tiny_hybrid_factory(
        cpu_basis_proposals=True, repair_rounds=0, gpu_stages=('maxmin',),
        cpu_basis_handoff=True)
    # Reuse the tiny two-variable fixture as the maxmin service stage. The
    # real CPU key still follows its two-column geometry; stable IDs and the
    # unchanged LP certificate are exercised rather than a biological model.
    old_key = next(iter(service.banks))
    key = ('maxmin', *old_key[1:])
    service.banks = {key: bank}
    service.host_bases = {key: service.host_bases[old_key]}
    monkeypatch.setattr(module, 'stage_key', lambda *args: key)
    assert service.solve_batch([_request(q)])[0].success  # establish CPU basis
    assert service.solve_batch([_request(p)])[0].success  # certified GPU only
    assert service.history[-1]['routes'] == ['gpu_dictionary']
    result = service.solve_batch([_request(q)])[0]
    assert result.success
    diagnostic = service.cpu.history[-1]['rows'][0]
    assert diagnostic['certificate_passed']
    assert diagnostic['basis_override_used']
    assert diagnostic['basis_override_source'] == 'previous_certified_gpu'
    assert not diagnostic['basis_reused']
    with pytest.raises(ValueError, match='fixed full-cohort'):
        service.solve_batch([_request(q), _request(q)])
    service.reset_trajectory()
    assert service._handoff_batch_size is None and not service.previous_candidates
