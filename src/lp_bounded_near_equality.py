"""Explicit one-row working relaxation with an exact current-box defect bound.

This is NOT an exact-equality reduction: a small nonzero stored-data defect can
change the exact feasible set and optimum. The original LP is retained without
alteration. Every caller must expand the row dual (zero at the omitted row),
recompute the complete original-LP certificate, and inspect the direct dual
objective/equality residual terms. The static bound below alone never accepts
an LP solution or establishes PPO accuracy.

Only supplied candidate support is inspected; no QR, LP optimizer, reference
solution, GPU operation, or biological-model edit is used. Each constructor
rechecks coefficients, RHS AND current finite bounds using Fraction arithmetic.
"""

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math

import numpy as np
from scipy.sparse import csr_matrix

from .gpu_pdhg_corrector import _validated_problem
from .lp_trace import problem_hash
from .lp_zero_face import _freeze_problem, _readonly


def _integer(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('Integer row indices and integer/Fraction coefficients required')
    return int(value)


def _record(value):
    return dict(numerator=value.numerator, denominator=value.denominator,
                binary64_approximation=float(value))


@dataclass(frozen=True)
class NearEqualityCandidate:
    """Untrusted relation proposal a[row] ~= sum(coefficients*a[basis_rows]).

    Indices refer to the LP supplied to BoundedNearEqualityReduction, not to
    an upstream untransformed LP. Coefficients are exact Fraction/int values;
    silently rounded floating-point proposal weights are deliberately refused.
    """
    row: int
    basis_rows: tuple
    coefficients: tuple

    def __post_init__(self):
        row = _integer(self.row)
        try:
            basis = tuple(_integer(value) for value in self.basis_rows)
            weights = tuple(value if isinstance(value, Fraction) else Fraction(_integer(value))
                            for value in self.coefficients)
        except TypeError as error:
            raise ValueError('Iterable candidate support and coefficients required') from error
        if (row < 0 or not 1 <= len(basis) <= 256 or len(basis) != len(weights)
                or any(value < 0 for value in basis) or row in basis
                or len(set(basis)) != len(basis) or any(value == 0 for value in weights)):
            raise ValueError('One target and 1..256 distinct nonself support rows with nonzero weights required')
        object.__setattr__(self, 'row', row)
        object.__setattr__(self, 'basis_rows', basis)
        object.__setattr__(self, 'coefficients', weights)

    def as_dict(self):
        return dict(row=self.row, basis_rows=list(self.basis_rows),
                    coefficients=[dict(numerator=w.numerator, denominator=w.denominator)
                                  for w in self.coefficients])


def _exact_defect(a, rhs, candidate):
    residual = {}
    terms = [(candidate.row, Fraction(1)),
             *zip(candidate.basis_rows, (-w for w in candidate.coefficients))]
    for row, weight in terms:
        begin, end = a.indptr[row:row+2]
        for column, value in zip(a.indices[begin:end], a.data[begin:end]):
            column = int(column)
            residual[column] = residual.get(column, Fraction(0)) + weight*Fraction.from_float(float(value))
    defect = tuple(sorted((column, value) for column, value in residual.items() if value))
    rhs_defect = Fraction.from_float(float(rhs[candidate.row])) - sum(
        (weight*Fraction.from_float(float(rhs[row]))
         for row, weight in zip(candidate.basis_rows, candidate.coefficients)), Fraction(0))
    return defect, rhs_defect


class BoundedNearEqualityReduction:
    """Immutable copied working LP omitting exactly one verified candidate row.

    Let r=a_target-sum(w_i*a_i), rho=b_target-sum(w_i*b_i). We compute the
    exact interval of r*x-rho over the current box and require its absolute
    upper bound <= max_defect <= 1e-12. Relevant infinite bounds fail closed.

    The omitted residual is sum(w_i*retained_residual_i)+r*x-rho, so retained
    residuals and floating-point evaluation error MUST still be checked. The
    bound is in this input LP's row units, not a relative percent tolerance.

    rows maps working to input rows; columns, objective, bounds and inequality
    rows are unchanged. dual_compression is ONLY a selector. Dropping an old
    multiplier does not preserve A.T*y; no approximate redistribution is used.
    A newly solved working dual lifts with omitted multiplier zero.
    """

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('BoundedNearEqualityReduction is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, problem, candidate, *, max_defect=1e-12):
        if not isinstance(candidate, NearEqualityCandidate):
            raise ValueError('An explicit NearEqualityCandidate is required; no implicit row deletion')
        # Reconstruct even a previously supplied frozen candidate. It is a
        # proposal, not an authority to reuse an old input's proof or bounds.
        candidate = NearEqualityCandidate(candidate.row, candidate.basis_rows, candidate.coefficients)
        if (isinstance(max_defect, (bool, np.bool_)) or not isinstance(max_defect, (int, float, np.integer, np.floating))
                or not math.isfinite(max_defect) or not 0. < max_defect <= 1e-12):
            raise ValueError('Positive finite max_defect no larger than 1e-12 required')
        if isinstance(problem, (tuple, list)) and len(problem) == 6:
            source_a = problem[0].data if hasattr(problem[0], 'tocsr') else problem[0]
            if np.iscomplexobj(source_a) or any(np.iscomplexobj(v) for v in problem[1:5]):
                raise ValueError('Real-valued LP inputs required')
        original = _freeze_problem(_validated_problem(problem))
        a, rhs, lower, upper, c, neq = original
        if np.isposinf(lower).any() or np.isneginf(upper).any():
            raise ValueError('Impossible infinite bound value')
        if candidate.row >= neq or any(row >= neq for row in candidate.basis_rows):
            raise ValueError('Target and every support row must be existing equalities')
        defect, rhs_defect = _exact_defect(a, rhs, candidate)
        low = high = -rhs_defect
        for column, coefficient in defect:
            lo, hi = float(lower[column]), float(upper[column])
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise ValueError('Finite bounds required on every exact-defect support column')
            endpoints = (coefficient*Fraction.from_float(lo), coefficient*Fraction.from_float(hi))
            low += min(endpoints)
            high += max(endpoints)
        bound = max(abs(low), abs(high))
        threshold = Fraction.from_float(float(max_defect))
        if bound > threshold:
            raise ValueError('Exact current-box defect bound exceeds max_defect; row retained by rejection')
        self.original = original
        self.original_hash = problem_hash(original)
        self.candidate = candidate
        self.matrix_defect = defect
        self.rhs_defect = rhs_defect
        self.defect_interval = (low, high)
        self.defect_bound = bound
        self.max_defect = threshold
        self.removed_rows = _readonly(np.array([candidate.row], dtype=np.int64))
        self.rows = _readonly(np.array([row for row in range(a.shape[0]) if row != candidate.row], dtype=np.int64))
        self.reduced = _freeze_problem((a[self.rows].tocsr(), rhs[self.rows], lower, upper, c, neq-1))
        self.reduced_hash = problem_hash(self.reduced)
        selector = csr_matrix((np.ones(len(self.rows)), (np.arange(len(self.rows)), self.rows)),
                              shape=(len(self.rows), a.shape[0]))
        for array in (selector.data, selector.indices, selector.indptr):
            array.flags.writeable = False
        self.dual_compression = selector
        self.proof_fingerprint = self._proof_hash()
        self._mapping_fingerprint = self._mapping_hash()
        self._summary = dict(
            kind='finite_box_bounded_one_row_working_relaxation',
            original_hash=self.original_hash, reduced_hash=self.reduced_hash,
            original_rows=a.shape[0], original_equalities=neq,
            reduced_rows=len(self.rows), reduced_equalities=neq-1, removed_rows=1,
            candidate=candidate.as_dict(),
            matrix_defect=[dict(column=column, coefficient=_record(value)) for column, value in defect],
            rhs_defect=_record(rhs_defect),
            defect_interval=[_record(low), _record(high)], defect_bound=_record(bound),
            max_defect=_record(threshold), support_l1=_record(sum(map(abs, candidate.coefficients), Fraction(0))),
            proof_fingerprint=self.proof_fingerprint, exact_coefficient_rhs_identity=not defect and not rhs_defect,
            exact_feasible_set_equivalence_claimed=False, original_lp_unchanged=True,
            relation='a_target=sum(w_i*a_support_i)+r; b_target=sum(w_i*b_support_i)+rho',
            target_residual='sum(w_i*support_residual_i)+r*x-rho',
            requires_original_full_row_certificate=True,
            requires_direct_dual_objective_and_equality_residual_check=True,
            static_defect_bound_is_not_a_solution_certificate=True,
            dual_compression='retained-row selector; old A.T*y need not be preserved',
            dual_lift='zero multiplier at omitted equality; unchanged retained multipliers',
            current_bounds_reverified=True, cpu_lp_calls=0, gpu_calls=0, qr_calls=0)
        self._sealed = True

    def _proof_hash(self):
        payload = dict(candidate=self.candidate.as_dict(),
            defect=[(j, str(value)) for j, value in self.matrix_defect],
            rhs=str(self.rhs_defect), interval=list(map(str, self.defect_interval)),
            bound=str(self.defect_bound), threshold=str(self.max_defect))
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _mapping_hash(self):
        digest = hashlib.sha256()
        for array in (self.rows, self.removed_rows, np.asarray(self.dual_compression.shape),
                      self.dual_compression.data, self.dual_compression.indices, self.dual_compression.indptr):
            digest.update(np.asarray(array).tobytes())
        return digest.hexdigest()

    @property
    def summary(self):
        return deepcopy(self._summary)

    def validate_integrity(self):
        if (problem_hash(self.original) != self.original_hash or problem_hash(self.reduced) != self.reduced_hash
                or self._proof_hash() != self.proof_fingerprint or self._mapping_hash() != self._mapping_fingerprint):
            raise ValueError('Bounded-near LP snapshot, candidate proof or mapping was modified')

    @staticmethod
    def _vector(value, size):
        if np.iscomplexobj(value):
            raise ValueError('Finite real vector with the exact shape required')
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (size,) or not np.isfinite(result).all():
            raise ValueError('Finite real vector with the exact shape required')
        return result

    def compress_dual(self, original_y):
        """Discard the target multiplier; deliberately no near-dependency redistribution."""
        self.validate_integrity()
        return self._vector(original_y, len(self.original[1]))[self.rows].copy()

    def lift_dual(self, reduced_y):
        self.validate_integrity()
        result = np.zeros(len(self.original[1]), dtype=np.float64)
        result[self.rows] = self._vector(reduced_y, len(self.rows))
        return result

    def lift_primal(self, x):
        self.validate_integrity()
        return self._vector(x, len(self.original[4])).copy()
