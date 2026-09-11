import numpy as np
import pytest
from src.cpu_repeated_lp import _problem
from src.lp_trace import problem_hash, problem_request, write_trace_lp, load_trace_lp


def example():
    return _problem(np.array([1., -2.]), dict(A_eq=[[1., 2.]], b_eq=[3.],
        A_ub=[[-1., 0.]], b_ub=[0.], bounds=[(None, None), (0., 4.)]))


def test_request_roundtrip():
    p = example()
    assert problem_hash(p) == problem_hash(_problem(*problem_request(p, stage='exchange')))


def test_disk_roundtrip(tmp_path):
    p = example()
    entry = write_trace_lp(tmp_path/'one.npz', p, reference_x=[0., 1.5], reference_y=[-1., -2.])
    restored, x, y = load_trace_lp(tmp_path, entry)
    assert problem_hash(restored) == problem_hash(p)
    np.testing.assert_array_equal(x, [0., 1.5])
    np.testing.assert_array_equal(y, [-1., -2.])
    with pytest.raises(FileExistsError):
        write_trace_lp(tmp_path/'one.npz', p, reference_x=x, reference_y=y)


def test_checksum_and_path_rejection(tmp_path):
    entry = write_trace_lp(tmp_path/'one.npz', example(), reference_x=[0., 1.5], reference_y=[-1., -2.])
    with pytest.raises(ValueError):
        load_trace_lp(tmp_path, dict(entry, sha256='invalid'))
    with pytest.raises(ValueError):
        load_trace_lp(tmp_path, dict(entry, filename='../one.npz'))
    with pytest.raises(ValueError):
        load_trace_lp(tmp_path, dict(entry, problem_sha256='invalid'))
    with pytest.raises(ValueError):
        load_trace_lp(tmp_path, dict(entry, rows=7))
    with pytest.raises(ValueError):
        load_trace_lp(tmp_path, dict(entry, step=1, stage='exchange', environment_id=0))


def test_hash_detects_current_bounds():
    p = example()
    changed = list(p)
    changed[2] = p[2].copy(); changed[2][0] = -4.
    assert problem_hash(p) != problem_hash(changed)


def test_reference_validation(tmp_path):
    with pytest.raises(ValueError):
        write_trace_lp(tmp_path/'bad.npz', example(), reference_x=[np.nan, 1.], reference_y=[0., 0.])
    entry = write_trace_lp(tmp_path/'badref.npz', example(), reference_x=[1., 1.], reference_y=[0., 0.])
    with pytest.raises(ValueError, match='does not certify'):
        load_trace_lp(tmp_path, entry)
