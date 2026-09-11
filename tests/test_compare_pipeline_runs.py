"""Audit gates for matched serial/pipelined development comparisons."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.compare_pipeline_runs import compare


def _endpoint(offset=0.0):
    return {
        'pha': 2.0 + offset,
        'phv_fraction': .4 + offset,
        'biomass': {'a': .7 + offset, 'b': 1.2 - offset},
        'metabolites': {'carbon_e': 3.0 - offset},
    }


def _report(*, pipeline, repeats=2):
    configuration = {
        'output': 'pipeline.json' if pipeline else 'serial.json',
        'pipeline_cpu_stages': pipeline,
        'tie_policy': 'original3',
        'cpu_backend': 'dictionary',
        'environments': 2,
        'steps': 3,
        'seed': 100,
        'seed_stride': 10,
        'repeats': repeats,
        'execution_order': 'alternate',
        'workers': 4,
        'bank': 'same-bank',
    }
    runs = []
    for repeat in range(repeats):
        rows = [_endpoint(0.), _endpoint(.001)]
        runs.append({
            'repeat': repeat,
            'seeds': [100 + 10 * repeat, 101 + 10 * repeat],
            'execution_order': 'cpu-first' if repeat % 2 == 0 else 'gpu-first',
            'cpu_seconds': 12.0 + repeat - (.5 if pipeline else 0.),
            'gpu_seconds': 9.0 + repeat - (.25 if pipeline else 0.),
            'cpu_rows': deepcopy(rows),
            'gpu_rows': deepcopy(rows),
            'cpu_completed_steps': [3, 3],
            'gpu_completed_steps': [3, 3],
            'failure': None,
            'all_endpoint_gates_passed': True,
            'errors': [
                {'pha_relative': 0., 'biomass_g_l': 0., 'phv_fraction': 0.},
                {'pha_relative': .001, 'biomass_g_l': .001, 'phv_fraction': .001},
            ],
            'online_cpu_lp_calls': 14 - repeat,
        })
    return {
        'status': 'completed',
        'configuration': configuration,
        'runs': runs,
        'source_hashes': {
            'src/community_solver.py': 'a' * 64,
            'src/gpu_hybrid_lp.py': 'b' * 64,
            **({'scripts/pipelined_microbatch.py': 'c' * 64} if pipeline else {}),
        },
        'model_fingerprints': {'a': 'model-a', 'b': 'model-b'},
        'simulated_hours_per_environment': .6,
    }


def _pair():
    return _report(pipeline=False), _report(pipeline=True)


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _recursive_keys(child)


def test_valid_two_group_comparison_reports_only_matched_pair_ratios_without_ci():
    serial, pipeline = _pair()
    result = compare(serial, pipeline)
    assert result['environments'] == 2
    assert result['steps'] == 3
    assert len(result['pairs']) == 2
    assert 'No CI or general superiority claim' in result['scope']
    for old, new, row in zip(serial['runs'], pipeline['runs'], result['pairs']):
        assert row['seeds'] == old['seeds'] == new['seeds']
        assert row['cpu_scheduler_ratio'] == old['cpu_seconds'] / new['cpu_seconds']
        assert row['hybrid_scheduler_ratio'] == old['gpu_seconds'] / new['gpu_seconds']
        assert row['pipeline_cpu_over_hybrid'] == new['cpu_seconds'] / new['gpu_seconds']
    forbidden = {'ci', 'confidence_interval', 'mean', 'median', 'std', 'sem',
                 'variance', 'p_value', 'significance', 'superior'}
    assert forbidden.isdisjoint(set(_recursive_keys(result)))


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(status='running'),
    lambda r: r.update(runs=[]),
    lambda r: r['runs'][0].update(failure={'error': 'failed'}),
    lambda r: r['runs'][0].update(all_endpoint_gates_passed=False),
    lambda r: r['runs'][0].update(cpu_completed_steps=[3, 2]),
    lambda r: r['runs'][0].update(gpu_completed_steps=[3]),
])
def test_incomplete_or_failed_trajectory_is_rejected(mutation):
    serial, pipeline = _pair()
    mutation(pipeline)
    with pytest.raises(ValueError):
        compare(serial, pipeline)


@pytest.mark.parametrize('field', ['pha_relative', 'biomass_g_l', 'phv_fraction'])
@pytest.mark.parametrize('value', [.010000001, np.nan, np.inf])
def test_original_cpu_gpu_accuracy_evidence_must_itself_be_complete_and_pass(field, value):
    """A stored True flag cannot replace validation of the original gate rows."""
    serial, pipeline = _pair()
    pipeline['runs'][0]['errors'][1][field] = value
    # Keep the summary flag True deliberately: compare() must inspect evidence.
    with pytest.raises(ValueError):
        compare(serial, pipeline)


@pytest.mark.parametrize('length', [0, 1, 3])
def test_original_accuracy_rows_require_one_entry_per_environment(length):
    serial, pipeline = _pair()
    pipeline['runs'][0]['errors'] = pipeline['runs'][0]['errors'][:length]
    if length == 3:
        pipeline['runs'][0]['errors'].append(
            {'pha_relative': 0., 'biomass_g_l': 0., 'phv_fraction': 0.}
        )
    with pytest.raises(ValueError):
        compare(serial, pipeline)


@pytest.mark.parametrize('side,field,value', [
    ('cpu', 'pha', np.nan), ('cpu', 'pha', np.inf),
    ('gpu', 'phv_fraction', np.nan), ('gpu', 'phv_fraction', np.inf),
])
def test_nonfinite_scheduler_endpoint_is_rejected(side, field, value):
    serial, pipeline = _pair()
    pipeline['runs'][0][side + '_rows'][0][field] = value
    with pytest.raises(ValueError):
        compare(serial, pipeline)


@pytest.mark.parametrize('side,metric', [('cpu', 'pha'), ('cpu', 'biomass'), ('gpu', 'phv')])
def test_scheduler_endpoint_change_strictly_above_one_percent_is_rejected(side, metric):
    serial, pipeline = _pair()
    endpoint = pipeline['runs'][0][side + '_rows'][0]
    if metric == 'pha':
        endpoint['pha'] *= 1.01000001
    elif metric == 'biomass':
        endpoint['biomass']['a'] += .01000001
    else:
        endpoint['phv_fraction'] += .01000001
    with pytest.raises(ValueError):
        compare(serial, pipeline)


@pytest.mark.parametrize('side,length', [('cpu', 0), ('cpu', 1), ('gpu', 0), ('gpu', 3)])
def test_scheduler_endpoint_length_must_equal_environment_count(side, length):
    serial, pipeline = _pair()
    values = pipeline['runs'][0][side + '_rows']
    pipeline['runs'][0][side + '_rows'] = (values * 2)[:length]
    with pytest.raises(ValueError):
        compare(serial, pipeline)


@pytest.mark.parametrize('key,value', [
    ('cpu_seconds', np.nan), ('cpu_seconds', np.inf), ('cpu_seconds', 0.),
    ('gpu_seconds', -1.),
])
def test_every_wall_time_must_be_positive_and_finite(key, value):
    serial, pipeline = _pair()
    pipeline['runs'][0][key] = value
    with pytest.raises(ValueError):
        compare(serial, pipeline)


@pytest.mark.parametrize('change', ['configuration', 'model', 'source_value', 'source_missing'])
def test_only_scheduler_may_differ_between_ablation_reports(change):
    serial, pipeline = _pair()
    if change == 'configuration':
        pipeline['configuration']['workers'] = 8
    elif change == 'model':
        pipeline['model_fingerprints']['a'] = 'changed'
    elif change == 'source_value':
        pipeline['source_hashes']['src/community_solver.py'] = 'c' * 64
    else:
        del pipeline['source_hashes']['src/community_solver.py']
    with pytest.raises(ValueError):
        compare(serial, pipeline)


@pytest.mark.parametrize('change', ['seed_values', 'seed_count', 'order', 'repeat'])
def test_each_run_is_strictly_paired_by_repeat_seed_count_values_and_order(change):
    serial, pipeline = _pair()
    row = pipeline['runs'][1]
    if change == 'seed_values':
        row['seeds'] = [777, 778]
    elif change == 'seed_count':
        serial['runs'][1]['seeds'] = row['seeds'] = [110]
    elif change == 'order':
        row['execution_order'] = 'cpu-first'
    else:
        row['repeat'] = 77
    with pytest.raises(ValueError):
        compare(serial, pipeline)


@pytest.mark.parametrize('side,change', [
    ('cpu', 'missing_species'), ('cpu', 'extra_species'),
    ('gpu', 'nonfinite_biomass'), ('gpu', 'nonfinite_metabolite'),
])
def test_endpoint_schema_and_all_numeric_state_are_finite(side, change):
    serial, pipeline = _pair()
    endpoint = pipeline['runs'][0][side + '_rows'][0]
    if change == 'missing_species':
        del endpoint['biomass']['a']
    elif change == 'extra_species':
        endpoint['biomass']['extra'] = 1.
    elif change == 'nonfinite_biomass':
        endpoint['biomass']['a'] = np.nan
    else:
        endpoint['metabolites']['carbon_e'] = np.inf
    with pytest.raises(ValueError):
        compare(serial, pipeline)


def test_serial_and_pipeline_roles_cannot_be_reversed_or_both_equal():
    serial, pipeline = _pair()
    with pytest.raises(ValueError):
        compare(pipeline, serial)
    serial['configuration']['pipeline_cpu_stages'] = True
    with pytest.raises(ValueError):
        compare(serial, pipeline)
