"""Exact, conservative zero-face reduction with original-LP dual postsolve.

Only zero-RHS *equalities* and existing bound signs prove extra fixed zeros.
Optional singleton propagation additionally proves x=0 from a*x=0 even when
zero is inside its bounds, with a signed equality-dual postsolve.
Exact equal/sign-negated equality duplicates may additionally be removed.
There is no optimizer, feasibility tolerance, numerical rank test, reference solution or
model edit. This host-side setup transform is not a GPU solver. Returned LPs
may have zero remaining columns; a caller must handle that trivial case.
The optional singleton coefficient floor only excludes rows from reduction;
it never rounds, drops, or relaxes a constraint.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .gpu_pdhg_corrector import _validated_problem
from .lp_trace import problem_hash


@dataclass(frozen=True)
class ZeroFaceWitness:
    """Original row and the columns first proved zero by this equality.

    ``min``/``max`` witnesses use a one-sided bound-face dual repair.
    ``equality`` has exactly one newly fixed column and uses an unrestricted
    signed multiplier so its reduced cost becomes zero at an interior bound.
    """
    row: int
    orientation: str
    newly_fixed_columns: tuple
    newly_fixed_coefficients: tuple


@dataclass(frozen=True)
class EqualityDuplicate:
    """Exact equality relation after fixed-column substitution.

    Original ``row`` equals ``sign`` times ``representative_row`` in the
    substituted LP. Postsolve may set its initial row multiplier to zero.
    """
    row: int
    representative_row: int
    sign: int


class InfeasibleZeroRow(ValueError):
    """Fixed-column substitution leaves a provably impossible zero row."""


def _readonly(value):
    value = np.array(value, copy=True)
    value.flags.writeable = False
    return value


def _freeze_problem(problem):
    a, rhs, lo, hi, c, neq = problem
    a = a.copy()
    for value in (a.data, a.indices, a.indptr):
        value.flags.writeable = False
    return a, *(_readonly(v) for v in (rhs, lo, hi, c)), int(neq)


def _same_bound_zero_signatures(lower, upper, old_lower, old_upper):
    """Compare only exact zero-relative signs, after input validation.

    Signed zero is deliberately equivalent: all zero-face predicates use
    numeric equality with zero. Infinite endpoints retain their proper sign.
    This predicate alone is insufficient; rebind also checks explicit fixed
    coordinates/values, retained finite masks, and the complete equality LP.
    """
    return (np.array_equal(np.sign(lower), np.sign(old_lower))
            and np.array_equal(np.sign(upper), np.sign(old_upper)))


class ZeroFaceReduction:
    """Equivalent bounded LP and reversible original-index mapping.

    ``columns`` and ``rows`` map reduced indices to original indices.
    ``fixed_columns`` / ``fixed_values`` reconstruct eliminated primals.
    ``objective_offset`` must be added to the reduced objective for reporting.
    ``witnesses`` are forward proof order; dual lifting uses reverse order.
    ``duplicate_equalities`` records exact removed equality representatives;
    their initial lifted dual is zero. Duplicate removal defaults on and can
    be disabled with ``remove_duplicate_equalities=False``. Inequality rows
    are never deduplicated, and surviving original row order is retained.
    All owned snapshots and mappings are read-only, and lift rejects mutation
    of either LP snapshot. Caller-owned source arrays are never changed.

    GPU composition needs only these arrays, original CSR and the witnesses:
    scatter kept primals/duals, then in reverse witness order shift one original
    equality multiplier and update its incident reduced costs. That postsolve
    consists only of device scatter, reduction and multiply/add operations.
    Original-unit LP certification remains mandatory after any solve/lift.
    """

    def __init__(self, problem, *, remove_duplicate_equalities=True,
                 fix_singleton_equalities=False, singleton_min_coefficient=1e-12):
        if type(remove_duplicate_equalities) is not bool:
            raise ValueError('remove_duplicate_equalities must be boolean')
        if type(fix_singleton_equalities) is not bool:
            raise ValueError('fix_singleton_equalities must be boolean')
        if (isinstance(singleton_min_coefficient, (bool, np.bool_))
                or not np.isfinite(singleton_min_coefficient)
                or singleton_min_coefficient <= 0.):
            raise ValueError('singleton_min_coefficient must be positive and finite')
        self.fix_singleton_equalities = fix_singleton_equalities
        self.remove_duplicate_equalities = remove_duplicate_equalities
        self.singleton_min_coefficient = float(singleton_min_coefficient)
        a, rhs, lo, hi, c, neq = _validated_problem(problem)
        if np.isposinf(lo).any() or np.isneginf(hi).any():
            raise ValueError('Impossible infinite fixed/bound value')
        self.original = _freeze_problem((a, rhs, lo, hi, c, neq))
        self.original_hash = problem_hash(self.original)
        lower, upper = lo.copy(), hi.copy()
        explicit_fixed = np.isfinite(lower) & (lower == upper)
        witnesses = []
        skipped_small_singleton_rows = set()
        while True:
            changed = False
            for row in range(neq):
                # An exact tiny nonzero RHS is not a zero-face certificate.
                if rhs[row] != 0.:
                    continue
                begin, end = a.indptr[row:row+2]
                columns, coefficients = a.indices[begin:end], a.data[begin:end]
                if not len(columns):
                    continue
                # _validated_problem removed all explicitly zero coefficients.
                minimum_bounds = np.where(coefficients > 0., lower[columns], upper[columns])
                maximum_bounds = np.where(coefficients > 0., upper[columns], lower[columns])
                newly = lower[columns] != upper[columns]
                if np.all(minimum_bounds == 0.):
                    orientation = 'min'
                elif np.all(maximum_bounds == 0.):
                    orientation = 'max'
                elif fix_singleton_equalities and np.count_nonzero(newly) == 1:
                    # Do not subtract products of nonzero fixed values or
                    # divide a nonzero RHS: neither is an exact zero proof.
                    if not np.all(lower[columns[~newly]] == 0.):
                        continue
                    singleton_column = int(columns[newly][0])
                    singleton_coefficient = float(coefficients[newly][0])
                    if abs(singleton_coefficient) < self.singleton_min_coefficient:
                        # Retain this equality verbatim. Tiny relations can
                        # arise from rounded forest algebra; do not use them
                        # to tighten an original residual-tolerant problem.
                        skipped_small_singleton_rows.add(int(row))
                        continue
                    if not lower[singleton_column] <= 0. <= upper[singleton_column]:
                        raise InfeasibleZeroRow(
                            f'Original singleton equality row {row} requires zero outside column {singleton_column} bounds')
                    orientation = 'equality'
                else:
                    continue
                if not np.any(newly):
                    continue
                newly_columns = columns[newly]
                witnesses.append(ZeroFaceWitness(int(row), orientation,
                    tuple(map(int, newly_columns)), tuple(map(float, coefficients[newly]))))
                lower[newly_columns] = 0.
                upper[newly_columns] = 0.
                changed = True
            if not changed:
                break
        fixed = np.isfinite(lower) & (lower == upper)
        self.columns = _readonly(np.flatnonzero(~fixed))
        self.fixed_columns = _readonly(np.flatnonzero(fixed))
        self.fixed_values = _readonly(lower[fixed])
        self.explicit_fixed_columns = _readonly(np.flatnonzero(explicit_fixed))
        self.forced_zero_columns = _readonly(np.flatnonzero(fixed & ~explicit_fixed))
        self.witnesses = tuple(witnesses)
        self.skipped_small_singleton_rows = _readonly(
            np.asarray(sorted(skipped_small_singleton_rows), dtype=np.int64))
        reduced_a = a[:, self.columns].tocsr()
        reduced_rhs = rhs - a[:, self.fixed_columns] @ self.fixed_values
        if not np.isfinite(reduced_rhs).all():
            raise ValueError('Fixed-column substitution overflowed')
        zero = np.diff(reduced_a.indptr) == 0
        invalid = zero & np.r_[reduced_rhs[:neq] != 0., reduced_rhs[neq:] < 0.]
        if np.any(invalid):
            row = int(np.flatnonzero(invalid)[0])
            raise InfeasibleZeroRow(f'Original row {row} is zero with impossible RHS {reduced_rhs[row]}')
        duplicate = np.zeros(len(rhs), dtype=bool)
        duplicate_relations = []
        if remove_duplicate_equalities:
            seen = {}
            for row in range(neq):
                if zero[row]:
                    continue
                begin, end = reduced_a.indptr[row:row+2]
                indices = reduced_a.indices[begin:end]
                coefficients = reduced_a.data[begin:end]
                sign = 1 if coefficients[0] > 0. else -1
                # Only identical or sign-negated rows/RHS. No division by
                # pivot, proportional matching, rounding or rank threshold.
                key = (indices.tobytes(), (sign*coefficients).tobytes(),
                       float(sign*reduced_rhs[row]))
                if key in seen:
                    representative, representative_sign = seen[key]
                    duplicate[row] = True
                    duplicate_relations.append(EqualityDuplicate(int(row),
                        representative, sign*representative_sign))
                else:
                    seen[key] = (int(row), sign)
        self.duplicate_equalities = tuple(duplicate_relations)
        self.removed_duplicate_rows = _readonly(np.flatnonzero(duplicate))
        self.rows = _readonly(np.flatnonzero(~(zero | duplicate)))
        self.removed_zero_rows = _readonly(np.flatnonzero(zero))
        reduced_neq = int(np.sum(self.rows < neq))
        self.reduced = _freeze_problem((reduced_a[self.rows].tocsr(), reduced_rhs[self.rows],
            lo[self.columns], hi[self.columns], c[self.columns], reduced_neq))
        self.reduced_hash = problem_hash(self.reduced)
        self.objective_offset = float(c[self.fixed_columns] @ self.fixed_values)
        if not np.isfinite(self.objective_offset):
            raise ValueError('Fixed-column objective offset overflowed')
        self._proof_fingerprint = self._map_fingerprint()

    def _map_fingerprint(self):
        """Detect reassignment as well as mutation of a saved proof/map."""
        digest = hashlib.sha256()
        for name in ('columns', 'rows', 'fixed_columns', 'fixed_values',
                     'explicit_fixed_columns', 'forced_zero_columns',
                     'skipped_small_singleton_rows', 'removed_duplicate_rows', 'removed_zero_rows'):
            value = np.asarray(getattr(self, name), dtype='<f8' if name == 'fixed_values' else '<i8')
            digest.update(name.encode('ascii'))
            digest.update(np.asarray(value.shape, dtype='<i8').tobytes())
            digest.update(value.tobytes())
        # These frozen records contain only scalar/tuple fields. Copying every
        # immutable leaf recursively with asdict dominates hot integrity scans.
        # Preserve the exact historical JSON/hash bytes; use the general path
        # for unexpected subclasses so their extra fields cannot be ignored.
        witnesses = [dict(row=w.row, orientation=w.orientation,
            newly_fixed_columns=w.newly_fixed_columns,
            newly_fixed_coefficients=w.newly_fixed_coefficients)
            if type(w) is ZeroFaceWitness else asdict(w) for w in self.witnesses]
        duplicates = [dict(row=d.row, representative_row=d.representative_row, sign=d.sign)
            if type(d) is EqualityDuplicate else asdict(d) for d in self.duplicate_equalities]
        digest.update(json.dumps(dict(witnesses=witnesses,
            duplicates=duplicates,
            singleton=self.fix_singleton_equalities, floor=self.singleton_min_coefficient,
            deduplicate=self.remove_duplicate_equalities), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode())
        return digest.hexdigest()

    def validate_integrity(self):
        if (problem_hash(self.original) != self.original_hash
                or problem_hash(self.reduced) != self.reduced_hash):
            raise ValueError('Zero-face LP snapshot was modified')
        if self._map_fingerprint() != self._proof_fingerprint:
            raise ValueError('Zero-face proof or coordinate map was modified')

    def rebind(self, problem):
        """Re-prove this coordinate map for new numeric input, without discovery.

        Equality A/RHS and explicit fixed values must match exactly. Saved
        witnesses are re-proved against CURRENT bounds; newly provable fixed
        columns, changed finite-bound topology or changed zero-row placement
        require a fresh plan. Inequality values/support and retained finite
        endpoints may otherwise change. Every numeric array is rebuilt from
        the new input. Neither this plan nor caller-owned input is mutated.

        This host sparse-algebra update is not an LP solve, a rank test or a
        certificate of optimality. GPU callers must update their complete
        original-LP certificate buffers transactionally before using it.
        """
        self.validate_integrity()
        a, rhs, lo, hi, c, neq = _validated_problem(problem)
        old_a, old_rhs, old_lo, old_hi, _old_c, old_neq = self.original
        if a.shape != old_a.shape or neq != old_neq:
            raise ValueError('Zero-face rebind dimensions/equality block changed; rebuild required')
        # A CSR initial row block is an exact prefix; avoid allocating two
        # sparse equality matrices merely to compare these same arrays.
        equality_end, old_equality_end = int(a.indptr[neq]), int(old_a.indptr[neq])
        if (not np.array_equal(a.indptr[:neq+1], old_a.indptr[:neq+1])
                or not np.array_equal(a.indices[:equality_end], old_a.indices[:old_equality_end])
                or not np.array_equal(a.data[:equality_end], old_a.data[:old_equality_end])
                or not np.array_equal(rhs[:neq], old_rhs[:neq])):
            raise ValueError('Zero-face rebind equality A/RHS changed; rebuild required')
        if np.isposinf(lo).any() or np.isneginf(hi).any():
            raise ValueError('Impossible infinite fixed/bound value')
        explicit = np.flatnonzero(np.isfinite(lo) & (lo == hi))
        if (not np.array_equal(explicit, self.explicit_fixed_columns)
                or not np.array_equal(lo[explicit], old_lo[explicit])):
            raise ValueError('Zero-face rebind explicit fixed columns/values changed; rebuild required')
        for current, old in ((lo, old_lo), (hi, old_hi)):
            if not np.array_equal(np.isfinite(current[self.columns]), np.isfinite(old[self.columns])):
                raise ValueError('Zero-face rebind retained finite-bound topology changed; rebuild required')
        same_zero_signs = _same_bound_zero_signatures(lo, hi, old_lo, old_hi)
        lower, upper = lo.copy(), hi.copy()

        def row_proof(row):
            begin, end = a.indptr[row:row+2]
            columns, coefficients = a.indices[begin:end], a.data[begin:end]
            newly = lower[columns] != upper[columns]
            if rhs[row] != 0. or not len(columns) or not np.any(newly):
                return None, columns, coefficients, newly
            minimum = np.where(coefficients > 0., lower[columns], upper[columns])
            maximum = np.where(coefficients > 0., upper[columns], lower[columns])
            if np.all(minimum == 0.):
                return 'min', columns, coefficients, newly
            if np.all(maximum == 0.):
                return 'max', columns, coefficients, newly
            if self.fix_singleton_equalities and np.count_nonzero(newly) == 1:
                value = float(coefficients[newly][0])
                if (np.all(lower[columns[~newly]] == 0.)
                        and abs(value) >= self.singleton_min_coefficient):
                    column = int(columns[newly][0])
                    if not lower[column] <= 0. <= upper[column]:
                        raise InfeasibleZeroRow('Current singleton witness requires zero outside bounds')
                    return 'equality', columns, coefficients, newly
            return None, columns, coefficients, newly

        if same_zero_signs:
            # Equality A/RHS and explicit fixed coordinates/values already
            # match exactly. Every row_proof predicate depends only on that
            # fixed mask and the bounds' zero-relative signs: ==0 for faces,
            # and <=0<= for interior singletons. Therefore its first result
            # is identical. Setting each newly proved coordinate to zero
            # preserves those predicates, so induction gives the same entire
            # witness order, orientation, coefficients and final fixed point.
            # No new cascade can start there: its first applicable row would
            # also have been applicable in the saved, exhausted fixed point.
            # This proof does not use a tolerance, cached bound values, or a
            # solution. A sign change takes the full replay/check path below.
            lower[self.forced_zero_columns] = 0.
            upper[self.forced_zero_columns] = 0.
        else:
            for witness in self.witnesses:
                orientation, columns, coefficients, newly = row_proof(witness.row)
                if (orientation != witness.orientation
                        or tuple(map(int, columns[newly])) != witness.newly_fixed_columns
                        or tuple(map(float, coefficients[newly])) != witness.newly_fixed_coefficients):
                    raise ValueError('Zero-face witness is not valid for current bounds; rebuild required')
                lower[columns[newly]] = 0.
                upper[columns[newly]] = 0.
        fixed = np.flatnonzero(np.isfinite(lower) & (lower == upper))
        if (not np.array_equal(fixed, self.fixed_columns)
                or not np.array_equal(lower[fixed], self.fixed_values)):
            raise ValueError('Zero-face replay changed the fixed coordinates; rebuild required')
        if not same_zero_signs:
            # Any new cascade must have a first applicable row at this fixed
            # point. One full pass detects it after changed bound signs.
            for row in range(neq):
                if row_proof(row)[0] is not None:
                    raise ValueError('Additional zero-face fixed columns are now provable; rebuild required')

        reduced_a = a[:, self.columns].tocsr()
        reduced_rhs = rhs - a[:, self.fixed_columns] @ self.fixed_values
        if not np.isfinite(reduced_rhs).all():
            raise ValueError('Fixed-column substitution overflowed')
        zero = np.diff(reduced_a.indptr) == 0
        invalid = zero & np.r_[reduced_rhs[:neq] != 0., reduced_rhs[neq:] < 0.]
        if np.any(invalid):
            raise InfeasibleZeroRow('Current fixed-column substitution leaves an impossible zero row')
        if not np.array_equal(np.flatnonzero(zero), self.removed_zero_rows):
            raise ValueError('Zero-face removed/retained zero-row structure changed; rebuild required')
        # The complete equality A/RHS and substituted fixed coordinates/values
        # are unchanged, hence the reduced equalities (including RHS) are
        # unchanged. validate_integrity already authenticated their duplicate
        # proofs. Re-slicing each pair cannot establish any additional fact.
        # Inequalities may change; their current zero-row checks above remain.

        # Allocate a new owned snapshot only after every structural check.
        result = object.__new__(type(self))
        for name in ('fix_singleton_equalities', 'remove_duplicate_equalities', 'singleton_min_coefficient'):
            setattr(result, name, getattr(self, name))
        for name in ('columns', 'rows', 'fixed_columns', 'fixed_values',
                     'explicit_fixed_columns', 'forced_zero_columns',
                     'skipped_small_singleton_rows', 'removed_duplicate_rows', 'removed_zero_rows'):
            setattr(result, name, _readonly(getattr(self, name)))
        result.witnesses = tuple(self.witnesses)
        result.duplicate_equalities = tuple(self.duplicate_equalities)
        result.original = _freeze_problem((a, rhs, lo, hi, c, neq))
        result.original_hash = problem_hash(result.original)
        result.reduced = _freeze_problem((reduced_a[result.rows].tocsr(), reduced_rhs[result.rows],
            lo[result.columns], hi[result.columns], c[result.columns], int(np.sum(result.rows < neq))))
        result.reduced_hash = problem_hash(result.reduced)
        result.objective_offset = float(c[result.fixed_columns] @ result.fixed_values)
        if not np.isfinite(result.objective_offset):
            raise ValueError('Fixed-column objective offset overflowed')
        result._proof_fingerprint = result._map_fingerprint()
        return result

    @staticmethod
    def _vector(supplied, length):
        result = np.asarray(supplied, dtype=np.float64)
        if result.shape != (length,) or not np.isfinite(result).all():
            raise ValueError('Finite vector with the exact reduced shape required')
        return result

    def lift_primal(self, reduced_x):
        self.validate_integrity()
        reduced_x = self._vector(reduced_x, len(self.columns))
        result = np.empty(len(self.original[4]), dtype=np.float64)
        result[self.columns] = reduced_x
        result[self.fixed_columns] = self.fixed_values
        return result

    def lift_dual(self, reduced_y):
        """Reconstruct original row duals; no optimizer or dual guess oracle.

        For a min-face row, new zero columns need q_j/a_j >= 0. Shifting
        y_row by -max(0,max(-q_j/a_j)) enforces that condition. A max-face
        row analogously needs q_j/a_j <= 0. An interior singleton equality
        instead needs q_j=0, so its multiplier shift is the signed q_j/a_j.
        Previously fixed dependencies
        may change and are repaired by earlier witnesses in reverse order.
        Original fixed variables need no reduced-cost sign restriction.
        """
        self.validate_integrity()
        reduced_y = self._vector(reduced_y, len(self.rows))
        a, rhs, _lo, _hi, c, _neq = self.original
        result = np.zeros(len(rhs), dtype=np.float64)
        result[self.rows] = reduced_y
        q = c - a.T @ result
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            try:
                for witness in reversed(self.witnesses):
                    columns = np.asarray(witness.newly_fixed_columns, dtype=np.int64)
                    coefficients = np.asarray(witness.newly_fixed_coefficients)
                    ratios = q[columns] / coefficients
                    if witness.orientation == 'min':
                        delta = -max(0., float(np.max(-ratios)))
                    elif witness.orientation == 'max':
                        delta = max(0., float(np.max(ratios)))
                    elif witness.orientation == 'equality':
                        if len(columns) != 1:
                            raise ValueError('Singleton equality witness must fix exactly one column')
                        delta = float(ratios[0])
                    else:
                        raise ValueError('Unknown zero-face witness orientation')
                    result[witness.row] += delta
                    begin, end = a.indptr[witness.row:witness.row+2]
                    q[a.indices[begin:end]] -= delta*a.data[begin:end]
            except FloatingPointError as error:
                raise ValueError('Zero-face dual postsolve overflowed') from error
        if not np.isfinite(result).all() or not np.isfinite(q).all():
            raise ValueError('Nonfinite zero-face dual postsolve')
        return result

    def lift(self, reduced_x, reduced_y):
        return self.lift_primal(reduced_x), self.lift_dual(reduced_y)
