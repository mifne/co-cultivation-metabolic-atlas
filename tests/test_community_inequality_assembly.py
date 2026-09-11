"""Exact legacy equivalence for shared CPU/GPU sparse medium-row assembly."""
import numpy as np
from scipy.sparse import csr_matrix, vstack

from src.community_solver import CooperativeCommunityFbaSolver


def _solver():
    solver = CooperativeCommunityFbaSolver.__new__(CooperativeCommunityFbaSolver)
    solver.n_fluxes = 5
    solver._growth_terms = {'b': (3, .75), 'a': (0, 1.)}
    solver._exchange_terms = {
        'z': [('a', 2, 1e16, 'r1'), ('b', 4, -.25, 'r2'),
              ('a', 2, -1e16, 'r3'), ('a', 2, 1., 'r4')],
        'h_e': [('a', 1, -1., 'h1')],
        'empty': [],
        'glucose': [('a', 0, -.1, 'g1'), ('b', 3, -.3, 'g2')],
        'cancels': [('a', 1, .5, 'c1'), ('a', 1, -.5, 'c2'), ('b', 4, 0., 'c3')],
        'h2o_e': [('a', 2, -1., 'w1')],
        'oh1_e': [('b', 4, -1., 'o1')],
    }
    return solver


def _legacy(solver, biomass, medium, dt):
    rows, rhs = [], []
    for species, (index, coefficient) in solver._growth_terms.items():
        row = np.zeros(solver.n_fluxes+1, dtype=np.float64)
        row[index] = -coefficient; row[-1] = 1.
        rows.append(csr_matrix(row.reshape(1, -1))); rhs.append(0.)
    for metabolite, terms in sorted(solver._exchange_terms.items()):
        row = np.zeros(solver.n_fluxes+1, dtype=np.float64)
        for species, index, stoich, _ in terms:
            row[index] += max(1e-12, float(biomass[species])) * stoich
        supply = (1e6 if metabolite in {'h2o_e', 'h_e', 'oh1_e'} else
                  max(0., float(medium.get(metabolite, 0.))) / dt)
        rows.append(csr_matrix(row.reshape(1, -1))); rhs.append(supply)
    return vstack(rows, format='csr'), np.asarray(rhs, dtype=np.float64)


def _assert_exact(solver, biomass, medium, dt=.2):
    expected, expected_rhs = _legacy(solver, biomass, medium, dt)
    actual, actual_rhs = solver._community_inequalities(biomass, medium, dt)
    assert actual.shape == expected.shape
    assert actual.has_sorted_indices and actual.has_canonical_format
    assert actual.data.dtype == expected.data.dtype
    assert actual.indices.dtype == expected.indices.dtype
    assert actual.indptr.dtype == expected.indptr.dtype
    for field in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    np.testing.assert_array_equal(actual_rhs, expected_rhs)
    return actual, actual_rhs


def test_sparse_assembly_matches_legacy_order_duplicate_accumulation_and_zeros():
    solver = _solver()
    for biomass in ({'a': 1., 'b': .7}, {'a': 0., 'b': -3.}, {'a': .123456789, 'b': 1.2345}):
        matrix, rhs = _assert_exact(solver, biomass, {'glucose': .3, 'z': -4., 'h_e': 0.})
        row = len(solver._growth_terms)+sorted(solver._exchange_terms).index('cancels')
        assert matrix[row].nnz == 0
    # This adversarial sum must stay (1e16 - 1e16) + 1, not a reordered sum.
    matrix, _ = _assert_exact(solver, {'a': 1., 'b': .7}, {})
    assert matrix[-1, 2] == 1.


def test_template_reused_for_states_but_invalidated_by_inplace_layout_changes():
    solver = _solver(); biomass = {'a': 1., 'b': 2.}
    _assert_exact(solver, biomass, {})
    template = solver._community_inequality_template
    _assert_exact(solver, {'a': .2, 'b': .3}, {'glucose': .8}, dt=.4)
    assert solver._community_inequality_template is template
    edits = [
        lambda: solver._exchange_terms['glucose'].append(('a', 0, .2, 'extra')),
        lambda: solver._exchange_terms['z'].reverse(),
        lambda: solver._growth_terms.update(a=(1, -.5)),
        lambda: solver._exchange_terms.update(new_metabolite=[('b', 2, -2., 'new')]),
        lambda: setattr(solver, 'n_fluxes', 6),
    ]
    for edit in edits:
        template = solver._community_inequality_template
        edit(); _assert_exact(solver, biomass, {})
        assert solver._community_inequality_template is not template


def test_returned_matrix_mutation_does_not_corrupt_template_or_later_results():
    solver = _solver(); biomass = {'a': .4, 'b': .8}
    matrix, rhs = _assert_exact(solver, biomass, {'glucose': 1.})
    matrix.data[:] = 99.; matrix.indices[:] = 0; rhs[:] = -9.
    _assert_exact(solver, biomass, {'glucose': 1.})


def test_scalar_dtype_changes_and_negative_index_aliases_keep_legacy_semantics():
    solver = _solver(); solver._exchange_terms = {'same': [
        ('a', -1, np.float32(.3), 'one'), ('a', 5, np.float32(.2), 'two')]}
    solver._growth_terms = {'a': (-1, .7)}
    _assert_exact(solver, {'a': .123456789}, {})
    template = solver._community_inequality_template
    # Numerically equal scalar metadata can have different multiplication
    # promotion rules; changing its type must invalidate the cached products.
    solver._exchange_terms['same'][0] = ('a', -1, float(np.float32(.3)), 'one')
    _assert_exact(solver, {'a': .123456789}, {})
    assert solver._community_inequality_template is not template
