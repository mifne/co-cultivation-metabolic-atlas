import copy
import json
import shutil

import numpy as np
import pytest

from scripts.rebind_unchanged_stage_router import rebind_router
from src.compact_training_data import sha256_file
from tests.test_coverage_router_binding import _bank, _models, _router


@pytest.fixture(params=['maxmin','aggregate','exchange'])
def files(tmp_path, request):
    stage_name = request.param
    old = tmp_path/'old'; stage_dir = old/stage_name; stage_dir.mkdir(parents=True)
    np.savez(stage_dir/'root.npz', a_shape=np.array([2, 3]), variable_rows=np.array([1]))
    np.savez(stage_dir/'router.npz', indices=np.array([0, 3]))
    entries = []
    for i in range(2):
        file = stage_dir/f'basis_{i:04d}.npz'
        np.savez(file, arbitrary_offline_data=np.array([i]))
        entries.append(dict(filename=file.name, sha256=sha256_file(file)))
    stage = dict(stage=stage_name, key=[stage_name, 2, 3, 1], entries=entries,
        root_sha256=sha256_file(stage_dir/'root.npz'), router_sha256=sha256_file(stage_dir/'router.npz'))
    manifest = dict(status='completed', stages=[stage], model_fingerprints=_models())
    (old/'manifest.json').write_text(json.dumps(manifest))
    new = tmp_path/'new'; shutil.copytree(old, new)
    updated = copy.deepcopy(manifest)
    other = 'exchange' if stage_name != 'exchange' else 'aggregate'
    updated['stages'].append(dict(stage=other, key=[other, 1, 2, 0], entries=[]))
    (new/'manifest.json').write_text(json.dumps(updated))
    router = _router(_bank(), stage, sha256_file(old/'manifest.json'))
    router_path = tmp_path/'old_router.npz'; saved = router.save(router_path)
    return old, new, router_path, saved['sha256'], tmp_path/'new_router.npz'


def run(files):
    old, new, path, digest, output = files
    return rebind_router(path, digest, old, sha256_file(old/'manifest.json'),
        new, sha256_file(new/'manifest.json'), output,
        stage_name=json.loads((old/'manifest.json').read_text())['stages'][0]['stage'])


def test_other_stage_change_rebinds_without_changing_any_weights(files):
    report = run(files)
    assert report['arrays_bitwise_unchanged'] and not report['training_performed']
    with np.load(files[2], allow_pickle=False) as old, np.load(files[4], allow_pickle=False) as new:
        for name in old.files:
            if name != 'metadata':
                np.testing.assert_array_equal(old[name], new[name])
        metadata = json.loads(str(new['metadata']))
        assert metadata['unchanged_stage_rebindings'][0]['all_stage_files_sha_verified']
    with pytest.raises(FileExistsError):
        run(files)


@pytest.mark.parametrize('change', ['stage', 'models', 'status', 'artifact'])
def test_changed_stage_or_tampered_inputs_are_rejected(files, change):
    _, new, _, _, output = files
    manifest = json.loads((new/'manifest.json').read_text())
    if change == 'stage':
        manifest['stages'][0]['entries'].reverse()
    elif change == 'models':
        manifest['model_fingerprints']['NS21'] = 'f'*64
    elif change == 'status':
        manifest['status'] = 'building'
    else:
        with (new/manifest['stages'][0]['stage']/'root.npz').open('ab') as stream:
            stream.write(b'tamper')
    (new/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        run(files)
    assert not output.exists()
