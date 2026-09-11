"""Lossless original-LP traces with purpose assigned by the capture manifest.

Diagnostic references remain validation-only; a separately captured explicit
training_reference trace may supply offline labels. A replay
must obtain its own solutions and advance each environment in recorded order.
This module restores LP inputs only; it does not restore a dFBA environment.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from .cpu_repeated_lp import _problem, _certificate


def problem_hash(problem):
    a, rhs, lower, upper, c, neq = problem
    digest = hashlib.sha256()
    for name, value in (('shape', a.shape), ('data', a.data),
                        ('indices', a.indices), ('indptr', a.indptr),
                        ('rhs', rhs), ('lower', lower), ('upper', upper),
                        ('c', c), ('neq', neq)):
        array = np.asarray(value, dtype='<i8' if name in
                           {'shape', 'indices', 'indptr', 'neq'} else '<f8')
        digest.update(name.encode())
        digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def problem_request(problem, *, stage=None):
    a, rhs, lower, upper, c, neq = problem
    request = dict(A_eq=a[:neq], b_eq=rhs[:neq], A_ub=a[neq:],
                   b_ub=rhs[neq:], bounds=np.column_stack((lower, upper)),
                   method='highs-ds', options=dict(threads=1, parallel=False))
    if stage is not None:
        request['_stage'] = stage
    return c, request


def write_trace_lp(path, problem, *, reference_x, reference_y):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    a, rhs, lower, upper, c, neq = problem
    x, y = np.asarray(reference_x), np.asarray(reference_y)
    if x.shape != c.shape or y.shape != rhs.shape or not (
            np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Invalid reference solution')
    np.savez_compressed(path, a_data=a.data, a_indices=a.indices,
        a_indptr=a.indptr, a_shape=a.shape, rhs=rhs, lower=lower,
        upper=upper, c=c, neq=neq, reference_x=x, reference_y=y)
    return dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                problem_sha256=problem_hash(problem), rows=a.shape[0],
                columns=a.shape[1], nonzeros=a.nnz, equalities=int(neq))


def load_trace_lp(directory, entry):
    directory = Path(directory).resolve()
    path = (directory / entry['filename']).resolve()
    if not path.is_relative_to(directory) or hashlib.sha256(
            path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Untrusted LP trace path or checksum')
    with np.load(path, allow_pickle=False) as data:
        a = csr_matrix((data['a_data'], data['a_indices'], data['a_indptr']),
                       shape=tuple(data['a_shape']))
        problem = (a, data['rhs'], data['lower'], data['upper'],
                   data['c'], int(data['neq']))
        x, y = data['reference_x'], data['reference_y']
    # Validate through the same original-LP parser, including normalization.
    rebuilt = _problem(*problem_request(problem))
    if problem_hash(rebuilt) != entry['problem_sha256']:
        raise ValueError('LP trace input changed during restoration')
    a, rhs, _, _, c, neq = rebuilt
    if (entry.get('rows'), entry.get('columns'), entry.get('nonzeros'), entry.get('equalities')) != (
            a.shape[0], a.shape[1], a.nnz, neq):
        raise ValueError('LP trace dimension metadata does not match its payload')
    if any(k in entry for k in ('step', 'stage', 'environment_id')):
        stage = entry.get('stage')
        if stage not in ('maxmin','aggregate','exchange') or not (
                isinstance(entry.get('step'), int) and entry['step'] >= 1 and
                isinstance(entry.get('environment_id'), int) and entry['environment_id'] >= 0):
            raise ValueError('Invalid LP trace labels')
        expected = f'lp_{entry["step"]:03d}_{("maxmin","aggregate","exchange").index(stage)}_{entry["environment_id"]:03d}.npz'
        if entry['filename'] != expected:
            raise ValueError('LP trace labels disagree with the recorded filename')
    if x.shape != rebuilt[4].shape or y.shape != rebuilt[1].shape or not (
            np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Invalid reference solution')
    reference = SimpleNamespace(col_value=x, row_dual=y, col_dual=c-a.T@y,
                                value_valid=True, dual_valid=True)
    if not _certificate(*rebuilt, reference)['certificate_passed']:
        raise ValueError('Reference solution does not certify the original LP')
    return rebuilt, x, y
