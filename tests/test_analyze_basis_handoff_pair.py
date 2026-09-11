from copy import deepcopy

import pytest

from scripts.analyze_basis_handoff_pair import _paired_summary
from tests.test_summarize_pipeline_run import complete_report


def pair():
    off = complete_report(steps=2)
    off['configuration'].update(output='off.json', cpu_basis_handoff=False, hybrid_rounds=0)
    for field in ('source_hashes', 'model_fingerprints', 'offline_bank_manifest',
                  'repair_operator_manifest'):
        off[field] = {'synthetic': 'same'}
    for history in off['runs'][0]['gpu_history']:
        history['routes'] = ['gpu_dictionary'] * history['bank_accepts'] + ['cpu_fallback'] * history['cpu_lp_calls']
        for group in history['groups']:
            group['ids'] = list(range(history['bank_accepts'], history['batch']))
            for row in group['rows']:
                row.update(simplex_iterations=10, setup_seconds=.01, solve_seconds=.02,
                           certificate_passed=True)
    on = deepcopy(off)
    on['configuration'].update(output='on.json', cpu_basis_handoff=True)
    row = on['runs'][0]['gpu_history'][0]['groups'][0]['rows'][0]
    row.update(basis_override_used=True, basis_override_requested=True,
               basis_override_rejected=False, basis_override_source='previous_certified_gpu',
               basis_override_age=1, simplex_iterations=5)
    return off, on


def test_analysis_pairs_overrides_and_keeps_wall_separate():
    result = _paired_summary(*pair())
    assert result['aggregate_applied_override']['count'] == 1
    assert result['aggregate_applied_override']['simplex_iterations']['delta'] == -5
    assert result['hybrid_wall_seconds']['delta'] == 0


@pytest.mark.parametrize('change', [
    lambda on: on.update(status='benchmark'),
    lambda on: on['runs'][0]['gpu_rows'][0].update(pha=float('nan')),
    lambda on: on['runs'][0].update(all_endpoint_gates_passed=False),
    lambda on: on['source_hashes'].update(synthetic='changed'),
    lambda on: on['runs'][0]['gpu_history'][0]['groups'][0].update(ids=[]),
])
def test_bad_completion_accuracy_provenance_or_row_mapping_is_rejected(change):
    off, on = pair()
    change(on)
    with pytest.raises(ValueError):
        _paired_summary(off, on)
