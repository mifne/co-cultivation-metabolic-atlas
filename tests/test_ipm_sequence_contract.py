"""CPU-only provenance/timing-scope contracts; no LP solve or GPU allocation."""
import ast
from copy import deepcopy
import hashlib
import inspect
import json

import pytest

from scripts.benchmark_ipm_sequence import main, validate_sequence_provenance


def _fixture(steps=(2, 3, 4), stage='maxmin', batch=2, seeds=(101, 202, 303)):
    entries = []
    for step in steps:
        for environment in range(batch):
            entries.append(dict(environment_id=environment, step=step, stage=stage,
                filename=f'lp_{step:03d}_{("maxmin", "aggregate", "exchange").index(stage)}_{environment:03d}.npz',
                problem_sha256=hashlib.sha256(f'{stage}:{step}:{environment}'.encode()).hexdigest(),
                sha256=hashlib.sha256(f'npz:{stage}:{step}:{environment}'.encode()).hexdigest()))
    manifest = dict(status='completed', role='development_diagnostic_not_training',
        seeds=list(seeds), completed_steps=[max(steps, default=0)]*len(seeds), entries=entries)
    raw = json.dumps(manifest, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    provenances = []
    for step in steps:
        selected = [deepcopy(entry) for entry in entries if entry['step'] == step]
        provenances.append(dict(input_manifest_sha256=digest,
            trace_directory='/immutable/development-trace', entries=selected,
            problem_sha256=[entry['problem_sha256'] for entry in selected]))
    return provenances, list(steps), stage, batch, raw


def _replace_manifest(arguments, mutate):
    provenances, steps, stage, batch, raw = arguments
    manifest = json.loads(raw)
    mutate(manifest)
    raw = json.dumps(manifest, sort_keys=True).encode()
    for provenance in provenances:
        provenance['input_manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    return provenances, steps, stage, batch, raw


@pytest.mark.parametrize('stage', ['maxmin', 'aggregate', 'exchange'])
def test_consistent_sequence_records_ordered_trace_identity_without_mutating_inputs(stage):
    arguments = _fixture(stage=stage)
    before = deepcopy(arguments)
    identity = validate_sequence_provenance(*arguments)
    assert identity == dict(input_manifest_sha256=hashlib.sha256(arguments[-1]).hexdigest(),
        ordered_environment_ids=[0, 1], ordered_environment_seeds=[101, 202],
        trace_directory='/immutable/development-trace')
    assert arguments == before
    identity['ordered_environment_ids'].reverse()
    identity['ordered_environment_seeds'][0] = 999
    assert arguments == before


def test_single_requested_step_is_a_valid_initial_sequence():
    assert validate_sequence_provenance(*_fixture(steps=(4,)))['ordered_environment_ids'] == [0, 1]


@pytest.mark.parametrize('position', [0, 1, 2])
def test_mixed_manifest_digests_are_rejected(position):
    arguments = _fixture()
    arguments[0][position]['input_manifest_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='Mixed trace manifests'):
        validate_sequence_provenance(*arguments)


def test_manifest_raw_bytes_must_match_validated_input_hashes():
    provenances, steps, stage, batch, raw = _fixture()
    with pytest.raises(ValueError, match='Mixed trace manifests'):
        validate_sequence_provenance(provenances, steps, stage, batch, raw+b'\n')


@pytest.mark.parametrize('position', [0, 1, 2])
def test_distinct_trace_directories_cannot_be_spliced(position):
    arguments = _fixture()
    arguments[0][position]['trace_directory'] = '/different/development-trace'
    with pytest.raises(ValueError, match='directories'):
        validate_sequence_provenance(*arguments)


@pytest.mark.parametrize('field,value', [('environment_id', 99), ('environment_id', 0),
    ('step', 99), ('stage', 'exchange'), ('problem_sha256', 'f'*64)])
def test_mislabeled_environment_stage_step_or_hash_is_rejected(field, value):
    arguments = _fixture()
    arguments[0][1]['entries'][1][field] = value
    with pytest.raises(ValueError, match='mismatch'):
        validate_sequence_provenance(*arguments)


def test_swapped_environment_entries_are_not_silently_reordered():
    arguments = _fixture()
    arguments[0][1]['entries'].reverse()
    arguments[0][1]['problem_sha256'].reverse()
    with pytest.raises(ValueError, match='mismatch'):
        validate_sequence_provenance(*arguments)


def test_provenance_hash_vector_must_match_the_current_entry_pair():
    arguments = _fixture()
    arguments[0][1]['problem_sha256'][0] = 'e'*64
    with pytest.raises(ValueError, match='mismatch'):
        validate_sequence_provenance(*arguments)


@pytest.mark.parametrize('field', ['entries', 'problem_sha256'])
@pytest.mark.parametrize('change', ['remove', 'append'])
def test_per_step_ordered_provenance_must_cover_exactly_the_batch(field, change):
    arguments = _fixture()
    values = arguments[0][1][field]
    if change == 'remove':
        values.pop()
    else:
        values.append(deepcopy(values[-1]))
    with pytest.raises(ValueError, match='Incomplete ordered input provenance'):
        validate_sequence_provenance(*arguments)


@pytest.mark.parametrize('steps', [[], [2, 4], [4, 3, 2], [2, 2, 3]])
def test_nonconsecutive_or_duplicate_requested_steps_fail_closed(steps):
    provenances, _, stage, batch, raw = _fixture()
    with pytest.raises(ValueError, match='Consecutive steps'):
        validate_sequence_provenance(provenances, steps, stage, batch, raw)


@pytest.mark.parametrize('change', ['remove', 'append'])
def test_missing_or_extra_whole_step_provenance_is_rejected(change):
    arguments = _fixture()
    if change == 'remove':
        arguments[0].pop()
    else:
        arguments[0].append(deepcopy(arguments[0][-1]))
    with pytest.raises(ValueError, match='Consecutive steps'):
        validate_sequence_provenance(*arguments)


@pytest.mark.parametrize('seeds', [[], [101], [101, 101], [202, 202, 303]])
def test_insufficient_or_duplicate_selected_environment_seeds_are_rejected(seeds):
    arguments = _replace_manifest(_fixture(), lambda manifest: manifest.update(seeds=seeds))
    with pytest.raises(ValueError, match='distinct recorded environments'):
        validate_sequence_provenance(*arguments)


def test_cpu_dimension_discovery_input_loads_precede_the_sequence_timer():
    tree = ast.parse(inspect.getsource(main))
    starts = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'sequence_start'
                for target in node.targets)]
    assert len(starts) == 1
    input_loads = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'load_inputs']
    assert input_loads and all(line < starts[0] for line in input_loads), (
        'CPU-only trace/flux-dimension input I/O must not inflate measured sequence execution')


def test_cpu_maxmin_dimension_discovery_reuses_inputs_already_loaded():
    tree = ast.parse(inspect.getsource(main))
    assert any(isinstance(node, ast.IfExp)
        and isinstance(node.body, ast.Subscript)
        and isinstance(node.body.value, ast.Name) and node.body.value.id == 'inputs'
        and isinstance(node.body.slice, ast.Constant) and node.body.slice.value == 0
        and isinstance(node.orelse, ast.Call) and isinstance(node.orelse.func, ast.Name)
        and node.orelse.func.id == 'load_inputs' for node in ast.walk(tree)), (
            'Maxmin CPU dimension discovery must use its already SHA-verified first input batch')
