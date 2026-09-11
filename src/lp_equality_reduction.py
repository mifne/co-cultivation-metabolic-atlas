"""Exact reduction of a conservative subset of homogeneous LP equalities.

Only linearly independent two-entry rows in the leading equality block are
eliminated.  If ``E`` denotes those rows, this module constructs a sparse
null-space map ``x = T z`` with ``E T = 0``.  All retained rows, objectives
and box bounds are transformed algebraically; no optimizer is called.

The reduced LP is useful only as a proposal/correction problem.  Acceptance
must still use the original LP after expanding ``x`` and lifting all row
duals.  The lift allocates each reduced bound multiplier to an original bound
that induced the corresponding reduced endpoint, then solves the eliminated
equality-tree multipliers with a precomputed sparse linear map.  Cycles,
single-entry rows and badly scaled components are retained conservatively.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye


def _normalized_problem(problem: Any):
    if not isinstance(problem, (tuple, list)) or len(problem) != 6:
        raise ValueError("Expected an (a, rhs, lower, upper, c, neq) LP tuple")
    supplied_a, supplied_rhs, supplied_lower, supplied_upper, supplied_c, neq = problem
    a = csr_matrix(supplied_a, dtype=np.float64, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    m, n = a.shape
    if n < 1 or not np.isfinite(a.data).all():
        raise ValueError("The LP matrix must have finite coefficients and columns")
    if isinstance(neq, (bool, np.bool_)) or not isinstance(neq, (int, np.integer)):
        raise ValueError("neq must be an integer")
    neq = int(neq)
    if not 0 <= neq <= m:
        raise ValueError("neq must identify an initial block of equality rows")

    values = []
    for name, supplied, shape, finite in (
        ("rhs", supplied_rhs, (m,), True),
        ("lower", supplied_lower, (n,), False),
        ("upper", supplied_upper, (n,), False),
        ("objective", supplied_c, (n,), True),
    ):
        array = np.asarray(supplied, dtype=np.float64)
        if array.shape != shape:
            raise ValueError(f"{name} has the wrong shape")
        if np.isnan(array).any() or (finite and not np.isfinite(array).all()):
            raise ValueError(f"{name} contains invalid values")
        values.append(array.copy())
    rhs, lower, upper, c = values
    if np.any(lower > upper):
        raise ValueError("Variable lower bounds cannot exceed upper bounds")
    if np.isposinf(lower).any() or np.isneginf(upper).any():
        raise ValueError("Impossible infinite variable bound")
    return a, rhs, lower, upper, c, neq


def _equality_fingerprint(a: csr_matrix, neq: int) -> str:
    equality = a[:neq].tocsr(copy=True)
    equality.sum_duplicates()
    equality.eliminate_zeros()
    equality.sort_indices()
    digest = hashlib.sha256()
    for name, value, dtype in (
        ("shape", equality.shape, "<i8"),
        ("indptr", equality.indptr, "<i8"),
        ("indices", equality.indices, "<i8"),
        ("data", equality.data, "<f8"),
        ("neq", [neq], "<i8"),
    ):
        array = np.asarray(value, dtype=dtype)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _max_abs_sparse(matrix: csr_matrix) -> float:
    return float(np.max(np.abs(matrix.data), initial=0.0))


def _immutable_array(value):
    """Bytes-backed private metadata cannot be made writeable accidentally."""
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class _SegmentedBounds:
    """Static group/tie ordering, shared by every numeric update of one tree.

    Within each group, order by descending |T|, then ascending original
    column. The first candidate attaining an endpoint is therefore exactly
    the historical max-weight/min-index witness. No numeric bounds are cached.
    """

    groups: np.ndarray
    weights: np.ndarray
    order: np.ndarray
    ordered_groups: np.ndarray
    starts: np.ndarray
    positions: np.ndarray
    positive: np.ndarray

    @classmethod
    def build(cls, groups, weights, group_count):
        groups, weights = np.asarray(groups), np.asarray(weights)
        if (groups.ndim != 1 or weights.shape != groups.shape or not len(groups)
                or groups.dtype.kind not in 'iu' or not np.isfinite(weights).all()
                or np.any(weights == 0.) or np.any(groups < 0) or np.any(groups >= group_count)):
            raise ValueError('Invalid forest bound coordinate layout')
        positions = np.arange(len(groups), dtype=np.int64)
        order = np.lexsort((positions, -np.abs(weights), groups))
        ordered_groups = groups[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ordered_groups)) + 1].astype(np.int64)
        if not np.array_equal(ordered_groups[starts], np.arange(group_count)):
            raise ValueError('Every forest component needs at least one original column')
        return cls(*(_immutable_array(value) for value in
            (groups, weights, order, ordered_groups, starts, positions, weights > 0.)))

    def reduce(self, lower, upper, groups, weights, group_count):
        # Legacy forest metadata is mutable. Never combine a cached ordering
        # with edited weights/group indices; a fresh plan is required instead.
        if (group_count != len(self.starts) or groups.shape != self.groups.shape
                or weights.shape != self.weights.shape or groups.dtype.kind not in 'iu'
                or not np.array_equal(groups, self.groups) or not np.array_equal(weights, self.weights)):
            raise ValueError('Forest bound coordinate layout changed; rebuild the reduction plan')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                induced_lower = np.where(self.positive, lower, upper) / self.weights
                induced_upper = np.where(self.positive, upper, lower) / self.weights
        except FloatingPointError as error:
            # Finite input endpoints must not silently overflow to an unbounded
            # reduced box. Legitimate +/-inf bounds divided by finite T remain.
            raise ValueError('Forest bound transformation overflowed or became invalid') from error
        ordered_lower, ordered_upper = induced_lower[self.order], induced_upper[self.order]
        reduced_lower = np.maximum.reduceat(ordered_lower, self.starts)
        reduced_upper = np.minimum.reduceat(ordered_upper, self.starts)
        if np.any(reduced_lower > reduced_upper):
            raise ValueError('Original bounds are infeasible under eliminated equalities')

        def witnesses(ordered, endpoints):
            endpoint = endpoints[self.ordered_groups]
            candidate = np.isfinite(endpoint) & (ordered == endpoint)
            first = np.minimum.reduceat(
                np.where(candidate, self.positions, len(self.order)), self.starts)
            found = first < len(self.order)
            witness = np.full(group_count, -1, dtype=np.int64)
            witness[found] = self.order[first[found]]
            return witness

        return (reduced_lower, reduced_upper, witnesses(ordered_lower, reduced_lower),
                witnesses(ordered_upper, reduced_upper))


@dataclass
class ReducedEqualityLP:
    """One dynamic LP reduced by a fixed :class:`HomogeneousEqualityReduction`."""

    plan: "HomogeneousEqualityReduction"
    problem: tuple
    original_problem: tuple
    kept_matrix: csr_matrix
    lower_witness: np.ndarray
    upper_witness: np.ndarray
    fallback_witness: np.ndarray

    def expand_primal(self, reduced_x):
        return self.plan.expand_primal(reduced_x)

    def compress_primal(self, original_x, *, check: bool = False, atol: float = 1e-9):
        return self.plan.compress_primal(original_x, check=check, atol=atol)

    def allocate_bound_reduced_cost(self, reduced_cost):
        """Allocate reduced costs to deterministic original-bound witnesses.

        A missing finite endpoint uses the component representative.  Such a
        candidate remains algebraically liftable but the unchanged original
        dual certificate rejects its invalid bound sign.
        """

        reduced_cost = np.asarray(reduced_cost, dtype=np.float64)
        if reduced_cost.shape != (self.plan.reduced_variables,) or not np.isfinite(
            reduced_cost
        ).all():
            raise ValueError("reduced_cost has the wrong shape or non-finite values")
        original = np.zeros(self.plan.original_variables, dtype=np.float64)
        for group, value in enumerate(reduced_cost):
            if value > 0.0:
                witness = int(self.lower_witness[group])
            elif value < 0.0:
                witness = int(self.upper_witness[group])
            else:
                continue
            if witness < 0:
                witness = int(self.fallback_witness[group])
            original[witness] = value / self.plan.weights[witness]
        return original

    def lift_dual(self, reduced_y, reduced_cost):
        """Lift retained-row duals and reduced costs to all original rows.

        This performs sparse matrix-vector products only.  It neither solves
        an LP nor certifies the result; callers must expand the primal and run
        the unchanged original-LP certificate afterwards.
        """

        reduced_y = np.asarray(reduced_y, dtype=np.float64)
        if reduced_y.shape != (len(self.plan.kept_rows),) or not np.isfinite(
            reduced_y
        ).all():
            raise ValueError("reduced_y has the wrong shape or non-finite values")
        reduced_cost = np.asarray(reduced_cost, dtype=np.float64)
        if reduced_cost.shape != (self.plan.reduced_variables,) or not np.isfinite(
            reduced_cost
        ).all():
            raise ValueError("reduced_cost has the wrong shape or non-finite values")

        original_a, _rhs, _lower, _upper, original_c, _neq = self.original_problem
        q = original_c - np.asarray(self.kept_matrix.T @ reduced_y).ravel()
        expected = np.asarray(self.plan.transform.T @ q).ravel()
        scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
        if float(np.max(np.abs(expected - reduced_cost), initial=0.0)) > 1e-10 * scale:
            raise ValueError("reduced_cost is inconsistent with the reduced row dual")

        original_cost = self.allocate_bound_reduced_cost(reduced_cost)
        difference = q - original_cost
        removed_y = np.asarray(self.plan.dual_lift_map @ difference).ravel()
        original_y = np.empty(original_a.shape[0], dtype=np.float64)
        original_y[self.plan.kept_rows] = reduced_y
        original_y[self.plan.eliminated_rows] = removed_y

        recovered = original_c - np.asarray(original_a.T @ original_y).ravel()
        if float(np.max(np.abs(recovered - original_cost), initial=0.0)) > 1e-8:
            raise ArithmeticError("Eliminated equality dual lift lost original stationarity")
        return original_y


def prepare_independent_forest_plans(problems, *, max_scale_ratio=1e8):
    """Reuse only identical equality proofs within this batch, never LP values.

    A hash selects candidates; exact canonical equality/RHS comparisons authorize
    reuse. Each lane gets a deep-owned plan, and reduce() still recomputes its
    current bounds, objective, retained rows and witnesses independently.
    """
    import copy
    cache={};plans=[]
    for supplied in problems:
        p=_normalized_problem(supplied);a,rhs,_,_,_,neq=p
        e=a[:neq].tocsr();e.eliminate_zeros();e.sort_indices()
        key=(a.shape,neq,_equality_fingerprint(a,neq),rhs[:neq].tobytes())
        hit=None
        for old_e,old_rhs,old_plan in cache.get(key,[]):
            if (np.array_equal(e.indptr,old_e.indptr) and np.array_equal(e.indices,old_e.indices)
                    and np.array_equal(e.data,old_e.data) and np.array_equal(rhs[:neq],old_rhs)):
                hit=old_plan;break
        if hit is None:
            plan=HomogeneousEqualityReduction.from_problem(p,max_scale_ratio=max_scale_ratio)
            cache.setdefault(key,[]).append((e.copy(),rhs[:neq].copy(),plan))
        else:
            plan=copy.deepcopy(hit)
        plans.append(plan)
    return tuple(plans)


class HomogeneousEqualityReduction:
    """Fixed sparse map for exact two-entry homogeneous equality elimination."""

    def __init__(self, problem, *, max_scale_ratio: float = 1e8):
        if (
            isinstance(max_scale_ratio, (bool, np.bool_))
            or not np.isfinite(max_scale_ratio)
            or float(max_scale_ratio) < 1.0
        ):
            raise ValueError("max_scale_ratio must be finite and at least one")
        a, rhs, lower, upper, c, neq = _normalized_problem(problem)
        self.original_shape = a.shape
        self.original_variables = a.shape[1]
        self.original_equalities = neq
        self.max_scale_ratio = float(max_scale_ratio)
        self.equality_fingerprint = _equality_fingerprint(a, neq)

        parent = np.arange(self.original_variables, dtype=np.int64)

        def find(node):
            node = int(node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = int(parent[node])
            return node

        def union(left, right):
            left, right = find(left), find(right)
            if left == right:
                return False
            # The smaller root makes construction deterministic.
            if left > right:
                left, right = right, left
            parent[right] = left
            return True

        edges: dict[int, tuple[int, int, float, float]] = {}
        selected_rows: list[int] = []
        for row in range(neq):
            start, stop = a.indptr[row], a.indptr[row + 1]
            if stop - start != 2 or rhs[row] != 0.0:
                continue
            columns = a.indices[start:stop]
            coefficients = a.data[start:stop]
            magnitude = np.abs(coefficients)
            if (
                np.min(magnitude) == 0.0
                or np.max(magnitude) / np.min(magnitude) > self.max_scale_ratio
            ):
                continue
            left, right = map(int, columns)
            if not union(left, right):
                # A cycle/parallel relation stays as an explicit equality.
                continue
            edges[row] = (left, right, float(coefficients[0]), float(coefficients[1]))
            selected_rows.append(row)

        adjacency: dict[int, list[tuple[int, int]]] = {}
        for row in selected_rows:
            left, right, _a_left, _a_right = edges[row]
            adjacency.setdefault(left, []).append((right, row))
            adjacency.setdefault(right, []).append((left, row))

        # Build candidate forest components and their null-space weights.
        visited: set[int] = set()
        valid_components: list[tuple[list[int], dict[int, float], list[int]]] = []
        for start in sorted(adjacency):
            if start in visited:
                continue
            stack = [start]
            alpha = {start: 1.0}
            nodes: list[int] = []
            component_rows: set[int] = set()
            valid = True
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                nodes.append(node)
                for neighbor, row in adjacency[node]:
                    component_rows.add(row)
                    if neighbor in alpha:
                        continue
                    left, right, a_left, a_right = edges[row]
                    if node == left:
                        value = -a_left * alpha[node] / a_right
                    else:
                        value = -a_right * alpha[node] / a_left
                    if not np.isfinite(value) or value == 0.0:
                        valid = False
                    alpha[neighbor] = float(value)
                    stack.append(neighbor)
            if not valid or len(nodes) < 2:
                continue
            magnitudes = np.asarray([abs(alpha[node]) for node in nodes])
            if (
                not np.isfinite(magnitudes).all()
                or np.max(magnitudes) / np.min(magnitudes) > self.max_scale_ratio
            ):
                continue
            largest = float(np.max(magnitudes))
            representative = min(node for node in nodes if abs(alpha[node]) == largest)
            scale = alpha[representative]
            alpha = {node: float(alpha[node] / scale) for node in nodes}
            alpha[representative] = 1.0
            valid_components.append(
                (sorted(nodes), alpha, sorted(component_rows))
            )

        valid_components.sort(key=lambda item: item[0][0])
        grouped = {node for nodes, _alpha, _rows in valid_components for node in nodes}
        components = [
            (nodes, weights, rows)
            for nodes, weights, rows in valid_components
        ] + [([node], {node: 1.0}, []) for node in range(self.original_variables) if node not in grouped]
        components.sort(key=lambda item: item[0][0])

        self.component_members = tuple(
            np.asarray(nodes, dtype=np.int64) for nodes, _weights, _rows in components
        )
        self.reduced_variables = len(components)
        self.original_to_reduced = np.empty(self.original_variables, dtype=np.int64)
        self.weights = np.empty(self.original_variables, dtype=np.float64)
        representatives = []
        eliminated = []
        transform_rows, transform_columns, transform_data = [], [], []
        for group, (nodes, weights, rows) in enumerate(components):
            representative = next(node for node in nodes if weights[node] == 1.0)
            representatives.append(representative)
            eliminated.extend(rows)
            for node in nodes:
                self.original_to_reduced[node] = group
                self.weights[node] = weights[node]
                transform_rows.append(node)
                transform_columns.append(group)
                transform_data.append(weights[node])
        self.representatives = np.asarray(representatives, dtype=np.int64)
        self._bound_layout = _SegmentedBounds.build(
            self.original_to_reduced, self.weights, self.reduced_variables)
        self.transform = coo_matrix(
            (transform_data, (transform_rows, transform_columns)),
            shape=(self.original_variables, self.reduced_variables),
            dtype=np.float64,
        ).tocsr()
        # ``T`` is the conventional null-space-map name used by GPU callers.
        self.T = self.transform
        denominator = np.asarray(self.transform.power(2).sum(axis=0)).ravel()
        compression_data = self.weights / denominator[self.original_to_reduced]
        self.compression = coo_matrix(
            (
                compression_data,
                (self.original_to_reduced, np.arange(self.original_variables)),
            ),
            shape=(self.reduced_variables, self.original_variables),
            dtype=np.float64,
        ).tocsr()

        self.eliminated_rows = np.asarray(sorted(set(eliminated)), dtype=np.int64)
        removed = set(map(int, self.eliminated_rows))
        self.kept_rows = np.asarray(
            [row for row in range(a.shape[0]) if row not in removed], dtype=np.int64
        )
        self.reduced_equalities = neq - len(self.eliminated_rows)
        self._eliminated_matrix = a[self.eliminated_rows].tocsr()
        self.dual_lift_map = self._build_dual_lift_map(a, edges, components)

        null_error = self._eliminated_matrix @ self.transform
        if _max_abs_sparse(null_error) > 1e-12:
            raise ArithmeticError("Constructed primal map is not in the equality null space")
        if len(self.eliminated_rows):
            inverse_error = self.dual_lift_map @ self._eliminated_matrix.T - eye(
                len(self.eliminated_rows), format="csr"
            )
            inverse_error.eliminate_zeros()
            if _max_abs_sparse(inverse_error) > 1e-10:
                raise ArithmeticError("Constructed tree dual map is not a left inverse of the eliminated transpose")

    @classmethod
    def from_problem(cls, problem, **kwargs):
        return cls(problem, **kwargs)

    def _build_dual_lift_map(self, a, edges, components):
        row_position = {int(row): i for i, row in enumerate(self.eliminated_rows)}
        expressions: list[dict[int, float] | None] = [None] * len(self.eliminated_rows)
        for nodes, _weights, component_rows in components:
            if not component_rows:
                continue
            node_set = set(nodes)
            root = next(node for node in nodes if self.weights[node] == 1.0)
            adjacency = {node: [] for node in nodes}
            for original_row in component_rows:
                left, right, _a_left, _a_right = edges[original_row]
                if left not in node_set or right not in node_set:
                    raise ArithmeticError("Equality forest component became inconsistent")
                position = row_position[original_row]
                adjacency[left].append((right, position, original_row))
                adjacency[right].append((left, position, original_row))

            parent = {root: None}
            parent_edge: dict[int, tuple[int, int]] = {}
            order = [root]
            for node in order:
                for neighbor, position, original_row in adjacency[node]:
                    if neighbor in parent:
                        continue
                    parent[neighbor] = node
                    parent_edge[neighbor] = (position, original_row)
                    order.append(neighbor)
            if len(order) != len(nodes):
                raise ArithmeticError("Equality forest is disconnected")

            children = {node: [] for node in nodes}
            for node in order[1:]:
                children[parent[node]].append(node)
            for node in reversed(order[1:]):
                position, original_row = parent_edge[node]
                left, right, a_left, a_right = edges[original_row]
                coefficient = a_left if node == left else a_right
                expression = {node: 1.0 / coefficient}
                for child in children[node]:
                    child_position, child_row = parent_edge[child]
                    child_expression = expressions[child_position]
                    if child_expression is None:
                        raise ArithmeticError("Tree dual map order is invalid")
                    c_left, c_right, c_a_left, c_a_right = edges[child_row]
                    parent_coefficient = c_a_left if node == c_left else c_a_right
                    factor = -parent_coefficient / coefficient
                    for variable, value in child_expression.items():
                        expression[variable] = expression.get(variable, 0.0) + factor * value
                expressions[position] = expression

        rows, columns, data = [], [], []
        for row, expression in enumerate(expressions):
            if expression is None:
                raise ArithmeticError("Missing eliminated-row dual expression")
            for column, value in sorted(expression.items()):
                if value != 0.0:
                    rows.append(row)
                    columns.append(column)
                    data.append(value)
        return coo_matrix(
            (data, (rows, columns)),
            shape=(len(self.eliminated_rows), self.original_variables),
            dtype=np.float64,
        ).tocsr()

    def reduce(self, problem) -> ReducedEqualityLP:
        original = _normalized_problem(problem)
        a, rhs, lower, upper, c, neq = original
        if a.shape != self.original_shape or neq != self.original_equalities:
            raise ValueError("LP dimensions/equality block changed from the reduction plan")
        if _equality_fingerprint(a, neq) != self.equality_fingerprint:
            raise ValueError("Equality fingerprint changed from the reduction plan")
        if len(self.eliminated_rows) and np.any(rhs[self.eliminated_rows] != 0.0):
            raise ValueError("An eliminated equality is no longer homogeneous")

        reduced_lower, reduced_upper, lower_witness, upper_witness = self._bound_layout.reduce(
            lower, upper, self.original_to_reduced, self.weights, self.reduced_variables)

        kept_matrix = a[self.kept_rows].tocsr()
        reduced_a = (kept_matrix @ self.transform).tocsr()
        reduced_a.sum_duplicates()
        reduced_a.eliminate_zeros()
        reduced_a.sort_indices()
        reduced_c = np.asarray(self.transform.T @ c).ravel()
        reduced_problem = (
            reduced_a,
            rhs[self.kept_rows].copy(),
            reduced_lower,
            reduced_upper,
            reduced_c,
            self.reduced_equalities,
        )
        return ReducedEqualityLP(
            plan=self,
            problem=reduced_problem,
            original_problem=original,
            kept_matrix=kept_matrix,
            lower_witness=lower_witness,
            upper_witness=upper_witness,
            fallback_witness=self.representatives.copy(),
        )

    def compress_primal(self, original_x, *, check: bool = False, atol: float = 1e-9):
        original_x = np.asarray(original_x, dtype=np.float64)
        if original_x.shape[-1:] != (self.original_variables,) or not np.isfinite(
            original_x
        ).all():
            raise ValueError("original_x has the wrong final dimension or non-finite values")
        if original_x.ndim == 1:
            reduced = np.asarray(self.compression @ original_x).ravel()
        else:
            reduced = np.asarray(original_x @ self.compression.T)
        if check:
            expanded = self.expand_primal(reduced)
            if not np.allclose(expanded, original_x, rtol=0.0, atol=atol):
                raise ValueError("original_x does not satisfy the eliminated equalities")
        return reduced

    def expand_primal(self, reduced_x):
        reduced_x = np.asarray(reduced_x, dtype=np.float64)
        if reduced_x.shape[-1:] != (self.reduced_variables,) or not np.isfinite(
            reduced_x
        ).all():
            raise ValueError("reduced_x has the wrong final dimension or non-finite values")
        if reduced_x.ndim == 1:
            return np.asarray(self.transform @ reduced_x).ravel()
        return np.asarray(reduced_x @ self.transform.T)


# A descriptive alias for integrations that prefer the generic plan name.
EqualityReductionPlan = HomogeneousEqualityReduction


__all__ = [
    "EqualityReductionPlan",
    "HomogeneousEqualityReduction",
    "ReducedEqualityLP",
]
