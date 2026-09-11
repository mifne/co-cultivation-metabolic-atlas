"""Exact old/new coordinate and CSR-preparation equivalence, without LP solves."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from src.gpu_certified_basis import CommunityCoordinates, NormalizedLP
from src.gpu_compact_basis import CompactBank


def _coordinates():
    coordinates = CommunityCoordinates.__new__(CommunityCoordinates)
    coordinates.n_fluxes = 4
    coordinates.column_species = np.array([0, 0, 1, 1])
    coordinates.row_species = np.array([0, 1])
    coordinates.probes = [(4, 1, -1.), (5, 3, -1.)]
    return coordinates


def _arrays(dtype=np.float64, biomass=(2., 3.)):
    matrix = csr_matrix(np.array([
        [2., -3., 0., 0., 0.], [0., 0., 1., -2., 0.],
        [-1., 0., 0., 0., 1.], [0., 0., -1., 0., 1.],
        [0., -biomass[0], 0., 0., 0.], [0., 0., 0., -biomass[1], 0.],
    ], dtype=dtype))
    return matrix, np.arange(6.), np.array([0., -np.inf, -1., 0., 0.]), np.array([2., np.inf, 4., 5., 1.]), np.array([-1., .25, -2., 0., 0.]), 2


def _legacy_normalize(coordinates, a, rhs, lower, upper, c, neq):
    a = csr_matrix(a, dtype=float)
    biomass = np.array([float(a[row, col])/stoich for row, col, stoich in coordinates.probes])
    if not np.isfinite(biomass).all() or np.any(biomass <= 0):
        raise ValueError('Biomass coordinate scaling must be positive and finite')
    col_scale = np.ones(a.shape[1]); col_scale[:coordinates.n_fluxes] = biomass[coordinates.column_species]
    row_scale = np.ones(a.shape[0]); valid = coordinates.row_species >= 0
    row_scale[np.flatnonzero(valid)] = biomass[coordinates.row_species[valid]]
    row_scale[neq:neq+len(biomass)] = biomass
    normalized = (diags(row_scale) @ a @ diags(1/col_scale)).tocsr()
    return NormalizedLP(normalized, np.asarray(rhs)*row_scale, np.asarray(lower)*col_scale,
        np.asarray(upper)*col_scale, np.asarray(c)/col_scale, neq, col_scale, row_scale)


def _assert_same(actual, expected):
    assert actual.a.shape == expected.a.shape and actual.a.dtype == expected.a.dtype
    for attribute in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(actual.a, attribute), getattr(expected.a, attribute))
    for name in ('rhs', 'lower', 'upper', 'c', 'col_scale', 'row_scale'):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.neq == expected.neq


@pytest.mark.parametrize('dtype,biomass', [
    (np.float64, (2., 3.)), (np.float32, (2., 3.)), (np.int32, (2., 3.)),
    (np.float64, (0.013, 6.28)), (np.float32, (0.013, 6.28)),
])
def test_direct_coordinate_scaling_is_bitwise_equal_and_nonmutating(dtype, biomass):
    coordinates = _coordinates(); arrays = _arrays(dtype, biomass)
    before = arrays[0].copy()
    expected = _legacy_normalize(coordinates, *arrays)
    actual = coordinates.normalize(*arrays)
    _assert_same(actual, expected)
    actual.a.data[:] = 17.; actual.a.indices[:] = 0; actual.a.indptr[:] = 0
    for field in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(arrays[0], field), getattr(before, field))


@pytest.mark.parametrize('variant', ['explicit_zero', 'unsorted', 'duplicate', 'underflow', 'nonfinite'])
def test_coordinate_edge_cases_preserve_legacy_sparse_fallback(variant):
    coordinates = _coordinates(); arrays = list(_arrays())
    a = arrays[0].copy()
    if variant == 'explicit_zero':
        a.data[0] = 0.
    elif variant == 'unsorted':
        a.data[:2] = a.data[:2][::-1]; a.indices[:2] = a.indices[:2][::-1]
        a.has_sorted_indices = False; a.has_canonical_format = False
    elif variant == 'duplicate':
        a = csr_matrix((np.insert(a.data, 1, 0.5), np.insert(a.indices, 1, 0),
            a.indptr+np.r_[0, np.ones(a.shape[0], dtype=int)]), shape=a.shape)
        assert not a.has_canonical_format
    elif variant == 'underflow':
        a.data[0] = 1e-320
        a[4, 1] = -1e-10
    else:
        a.data[0] = np.nan
    arrays[0] = a
    _assert_same(coordinates.normalize(*arrays), _legacy_normalize(coordinates, *arrays))


@pytest.mark.parametrize('biomass', [0., -1., np.nan, np.inf])
def test_invalid_coordinate_scale_still_rejected(biomass):
    coordinates = _coordinates(); arrays = _arrays(biomass=(biomass, 1.))
    for normalize in (coordinates.normalize, lambda *args: _legacy_normalize(coordinates, *args)):
        with pytest.raises(ValueError, match='positive and finite'):
            normalize(*arrays)


def _bank(root, variables, cp=np):
    bank = CompactBank.__new__(CompactBank)
    bank.cp = cp; bank.host_a = root.a; bank.neq = root.neq
    bank.variable_rows = np.asarray(variables, dtype=int)
    return bank


def _legacy_prepare(bank, problems):
    deltas = []
    for problem in problems:
        if problem.a.shape != bank.host_a.shape or problem.neq != bank.neq:
            raise ValueError('Changed compact LP shape')
        delta = (problem.a-bank.host_a).tocsr()
        fixed = np.ones(delta.shape[0], dtype=bool); fixed[bank.variable_rows] = False
        if np.max(np.abs(delta[fixed].data), initial=0) > 2e-12:
            raise ValueError('Changed unsupported matrix row')
        deltas.append(delta[bank.variable_rows].toarray())
    return dict(delta=np.stack(deltas), **{key: np.stack([getattr(problem, key) for problem in problems])
        for key in ('rhs', 'lower', 'upper', 'c', 'col_scale', 'row_scale')})


@pytest.mark.parametrize('device', [False, True])
def test_compact_preparation_same_pattern_structural_fallback_and_all_fields(device):
    cp = pytest.importorskip('cupy') if device else np
    coordinates = _coordinates()
    root = coordinates.normalize(*_arrays())
    bank = _bank(root, [3, 2], cp)
    changed = coordinates.normalize(*_arrays(biomass=(2.2, 3.3)))
    changed = replace(changed, rhs=changed.rhs+0.7, lower=changed.lower-0.1,
        upper=changed.upper+0.2, c=changed.c-0.3)
    structural = changed.a.tolil(); structural[3, 1] = .13
    structural = replace(changed, a=structural.tocsr())
    for problems in ([root, changed], [structural, changed], [changed, root]):
        expected = _legacy_prepare(bank, problems)
        actual = bank.prepare_host(problems)
        for name, values in expected.items():
            measured = actual[name].get() if device else actual[name]
            np.testing.assert_array_equal(measured, values)
    # Replacing the root's CSR structure refreshes cached coefficient locations.
    bank.host_a = structural.a
    expected = _legacy_prepare(bank, [structural])
    actual = bank.prepare_host([structural])
    for name in expected:
        np.testing.assert_array_equal(actual[name].get() if device else actual[name], expected[name])


@pytest.mark.parametrize('variant', ['fixed_value', 'fixed_new_column', 'shape', 'neq'])
def test_preparation_cannot_hide_unsupported_row_or_shape_changes(variant):
    root = _coordinates().normalize(*_arrays()); bank = _bank(root, [2, 3])
    bank.prepare_host([root])  # Prime the fast-path layout cache.
    changed = root.a.tolil()
    if variant == 'fixed_value':
        changed[0, 0] += .01
    elif variant == 'fixed_new_column':
        changed[0, 4] = .01
    elif variant == 'shape':
        changed.resize((changed.shape[0]+1, changed.shape[1]))
    query = replace(root, a=changed.tocsr(), neq=root.neq+(variant == 'neq'))
    for prepare in (bank.prepare_host, lambda problems: _legacy_prepare(bank, problems)):
        with pytest.raises(ValueError):
            prepare([query])


def test_preparation_noncanonical_and_duplicate_variable_rows_use_legacy_path():
    root = _coordinates().normalize(*_arrays())
    bank = _bank(root, [2, 2, 3])
    a = root.a.copy(); start, stop = a.indptr[2:4]
    a.data[start:stop] = a.data[start:stop][::-1]; a.indices[start:stop] = a.indices[start:stop][::-1]
    a.has_sorted_indices = False; a.has_canonical_format = False
    query = replace(root, a=a)
    expected = _legacy_prepare(bank, [query])
    actual = bank.prepare_host([query])
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('changed_pattern', [False, True])
def test_nonfinite_fixed_coefficients_are_rejected_on_both_paths(bad, changed_pattern):
    root = _coordinates().normalize(*_arrays()); bank = _bank(root, [2, 3])
    bank.prepare_host([root])
    changed = root.a.tolil()
    changed[0, 0] = bad
    if changed_pattern:
        changed[2, 1] = .13  # Force the general sparse-difference path.
    with pytest.raises(ValueError, match='Nonfinite'):
        bank.prepare_host([replace(root, a=changed.tocsr())])


def test_nonfinite_reference_and_overflowed_change_fail_closed():
    root = _coordinates().normalize(*_arrays()); bank = _bank(root, [2, 3])
    bank.host_a = root.a.copy(); bank.host_a.data[0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        bank.prepare_host([root])
    bank.host_a = root.a.copy(); bank.host_a.data[0] = -1e308
    changed = root.a.copy(); changed.data[0] = 1e308
    with np.errstate(over='ignore'), pytest.raises(ValueError, match='Nonfinite'):
        bank.prepare_host([replace(root, a=changed)])
