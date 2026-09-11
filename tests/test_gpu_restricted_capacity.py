"""CPU-only checks for the storage bound; no CuPy import or device work."""
import itertools

import numpy as np

from src.gpu_restricted_basis import restricted_basis_capacity


def test_legacy_capacity_formula_is_unchanged():
    for k0, m, n, prior, width, pivots in itertools.product(
            (0, 1, 3), (3, 17), (4, 20), (0, 1, 4), (1, 8, 64), (1, 4, 256)):
        previous = k0+min(prior, m, n)
        expected = min(k0+m, k0+n, 1 << (max(1, previous+width)-1).bit_length())
        assert restricted_basis_capacity(previous, k0, m, n, width, pivots) == expected


def test_pivot_aware_capacity_keeps_previous_rank_at_bucket_boundaries():
    assert restricted_basis_capacity(3, 3, 7000, 8000, 64, 4) == 128
    assert restricted_basis_capacity(3, 3, 7000, 8000, 64, 4, True) == 8
    assert restricted_basis_capacity(7, 3, 7000, 8000, 64, 4, True) == 16
    assert restricted_basis_capacity(8, 3, 7000, 8000, 1, 4, True) == 16
    assert restricted_basis_capacity(9, 3, 6, 8, 64, 4, True) == 9
    assert restricted_basis_capacity(8, 0, 8, 10, 1, 1, True) == 8
    for k0 in (0, 1, 3):
        for m, n in ((1, 2), (5, 3), (8, 12)):
            for changed in range(min(m, n)+1):
                previous = k0+changed
                for width, pivots in itertools.product((1, 2, 64), (1, 4, 256)):
                    capacity = restricted_basis_capacity(previous, k0, m, n, width, pivots, True)
                    assert previous <= capacity <= k0+min(m, n)
                    assert min(min(m, n), changed+min(width, pivots)) <= capacity-k0


def test_replacement_bound_under_pivots_flips_and_changed_column_sets():
    # The exact reachable set consists of the current M basics plus the K
    # selected nonbasics. Swaps may revisit a slot or undo an earlier round;
    # a flip never changes membership. The selected set changes every round.
    rng = np.random.default_rng(94221)
    for m, n in ((3, 5), (5, 3), (8, 8)):
        root = set(range(m))
        basic = list(range(m))
        for _ in range(60):
            available = np.array(sorted(set(range(m+n))-set(basic)))
            width = int(rng.integers(1, n+1))
            selected = rng.choice(available, width, replace=False).tolist()
            previous_missing = len(root-set(basic))
            pivots = int(rng.integers(1, 7))
            upper = min(m, n, previous_missing+min(width, pivots))
            for _ in range(pivots):
                if rng.integers(3):
                    row, slot = int(rng.integers(m)), int(rng.integers(width))
                    basic[row], selected[slot] = selected[slot], basic[row]
                assert len(root-set(basic)) <= upper
                assert len(set(basic)) == m
            for k0 in (0, 3):
                capacity = restricted_basis_capacity(k0+previous_missing, k0, m, n, width, pivots, True)
                assert k0+len(root-set(basic)) <= capacity
