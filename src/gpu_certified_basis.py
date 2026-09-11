"""Experimental certified basis evaluation, not a surrogate interpolator.

CPU compilation solves an anchor LP and factors its active basis OFFLINE.
Online evaluation uses GPU linear algebra and full primal/numerical KKT
checks; invalid candidates are returned as invalid, never as zero fluxes.
This module alone does not implement a fully GPU-resident dFBA environment.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.sparse import csr_matrix, diags


@dataclass
class NormalizedLP:
    a: object
    rhs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    c: np.ndarray
    neq: int
    col_scale: np.ndarray
    row_scale: np.ndarray


class GpuBasisInputAdapter:
    """Host input preparation without unused GPU copies of the active inverse."""
    def __init__(self,anchor,variable_rows):
        import cupy as cp
        self.cp=cp;self.anchor=anchor;self.host_a=anchor['lp'].a;self.neq=anchor['lp'].neq
        self.variable_rows=np.asarray(sorted(set(variable_rows)),dtype=int)
        if np.any(self.variable_rows<0) or np.any(self.variable_rows>=self.host_a.shape[0]):raise ValueError('Invalid variable rows')

    def prepare_host(self,problems):
        return GpuBasisEvaluator.prepare_host(self,problems)


class CommunityCoordinates:
    """Invertible z_i=X_i*v_i scaling from the actual LP coefficients.

Derive biomass from shared-pool rows, not a possibly pre-dilution snapshot.
No inequality/equality/variable is dropped in this representation.
"""
    def __init__(self, metadata, original_a, neq):
        self.species = list(metadata["growth_terms"])
        self.n_fluxes = len(metadata["reaction_ids"])
        self.column_species = np.array([self.species.index(s) for s in metadata["reaction_species"]])
        shared = sorted(metadata["exchange_terms"])
        self.probes = []
        for species in self.species:
            choices = [(neq+len(self.species)+shared.index(m), i, stoich)
                for m, terms in metadata["exchange_terms"].items()
                for s, i, stoich, _ in terms if s == species and stoich != 0]
            if not choices:
                raise ValueError("No shared exchange coefficient for species")
            self.probes.append(choices[0])
        self.row_species = np.full(neq, -1, dtype=int)
        eq = csr_matrix(original_a)[:neq]
        for row in range(neq):
            cols = eq.indices[eq.indptr[row]:eq.indptr[row+1]]
            cols = cols[cols < self.n_fluxes]
            if len(cols):
                species = np.unique(self.column_species[cols])
                if len(species) != 1:
                    raise ValueError("Equality crosses species; explicit scaling required")
                self.row_species[row] = species[0]

    def normalize(self, a, rhs, lower, upper, c, neq):
        a = csr_matrix(a, dtype=float)
        biomass = np.array([float(a[row, col])/stoich for row, col, stoich in self.probes])
        if not np.isfinite(biomass).all() or np.any(biomass <= 0):
            raise ValueError("Biomass coordinate scaling must be positive and finite")
        col_scale = np.ones(a.shape[1])
        col_scale[:self.n_fluxes] = biomass[self.column_species]
        row_scale = np.ones(a.shape[0])
        valid = self.row_species >= 0
        row_scale[np.flatnonzero(valid)] = biomass[self.row_species[valid]]
        row_scale[neq:neq+len(biomass)] = biomass
        inverse_col_scale = 1/col_scale
        normalized = None
        # For a canonical matrix with finite nonzero coefficients, diagonal
        # sparse products perform exactly these two multiplications per entry.
        # Keep their order (row scale first, reciprocal column scale second),
        # but avoid constructing two diagonal matrices and two SpGEMM outputs.
        # Explicit/underflowed zeros, duplicates, nonfinite values, or unsorted
        # rows retain the original sparse operations and storage semantics.
        if (a.has_canonical_format and np.isfinite(a.data).all()
                and np.all(a.data != 0.) and np.isfinite(inverse_col_scale).all()):
            data_rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
            scaled_data = (row_scale[data_rows]*a.data)*inverse_col_scale[a.indices]
            if np.isfinite(scaled_data).all() and np.all(scaled_data != 0.):
                normalized = csr_matrix(
                    (scaled_data, a.indices.copy(), a.indptr.copy()), shape=a.shape)
        if normalized is None:
            normalized = (diags(row_scale) @ a @ diags(inverse_col_scale)).tocsr()
        return NormalizedLP(normalized, np.asarray(rhs)*row_scale,
            np.asarray(lower)*col_scale, np.asarray(upper)*col_scale,
            np.asarray(c)/col_scale, neq, col_scale, row_scale)


def compile_basis(original_a, rhs, lower, upper, c, neq, coordinates=None, *, factorize=True):
    """Offline CPU HiGHS solve + sparse-LU inverse; explicitly paid setup cost."""
    import highspy as hp
    started = time.perf_counter()
    a = csr_matrix(original_a, dtype=float)
    n, m = a.shape[1], a.shape[0]
    lp = hp.HighsLp(); lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, lower, upper
    lp.row_lower_, lp.row_upper_ = np.r_[rhs[:neq], np.full(m-neq, -np.inf)], rhs
    lp.a_matrix_.format_ = hp.MatrixFormat.kRowwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = a.indptr, a.indices, a.data
    solver = hp.Highs()
    for key, value in dict(output_flag=False, threads=1, solver="simplex").items():
        if solver.setOptionValue(key, value) == hp.HighsStatus.kError:
            raise ValueError(key)
    solver.passModel(lp); solver.run()
    if solver.getModelStatus() != hp.HighsModelStatus.kOptimal:
        raise ValueError("Anchor LP is not optimal")
    basis = solver.getBasis()
    basic = np.array([i for i, s in enumerate(basis.col_status) if s == hp.HighsBasisStatus.kBasic], dtype=int)
    active = np.array([i for i, s in enumerate(basis.row_status) if s != hp.HighsBasisStatus.kBasic], dtype=int)
    if len(basic) != len(active) or not len(basic):
        raise ValueError("Unsupported empty/non-square anchor basis")
    kind = np.zeros(n, dtype=np.int8)
    for i, status in enumerate(basis.col_status):
        if status == hp.HighsBasisStatus.kLower:
            kind[i] = -1
        elif status == hp.HighsBasisStatus.kUpper:
            kind[i] = 1
        elif status == hp.HighsBasisStatus.kBasic:
            kind[i] = 2
        elif status != hp.HighsBasisStatus.kZero:
            raise ValueError("Unsupported nonbasic status")
    normalized = (coordinates.normalize(a, rhs, lower, upper, c, neq) if coordinates else
        NormalizedLP(a, np.asarray(rhs), np.asarray(lower), np.asarray(upper), np.asarray(c), neq,
                     np.ones(n), np.ones(m)))
    row_kind = np.zeros(m, dtype=np.int8)
    for i, status in enumerate(basis.row_status):
        row_kind[i] = (2 if status == hp.HighsBasisStatus.kBasic else
                       -1 if status == hp.HighsBasisStatus.kLower else
                       1 if status == hp.HighsBasisStatus.kUpper else 0)
    result=dict(lp=normalized, basic=basic, active=active, kind=kind,
        row_kind=row_kind,
        cpu_anchor_values=np.asarray(solver.getSolution().col_value),
        cpu_anchor_objective=solver.getInfo().objective_function_value,
        compilation_seconds=time.perf_counter()-started, offline_cpu_lp_calls=1)
    return factor_compiled_basis(result) if factorize else result


def factor_compiled_basis(anchor):
    """Offline factorization separated from LP solve to deduplicate a bank."""
    from scipy.sparse.linalg import splu
    if "inverse" in anchor: return anchor
    started=time.perf_counter()
    basic,active=anchor["basic"],anchor["active"]
    b=anchor["lp"].a[active][:,basic].tocsc()
    factor=splu(b)
    inverse=np.empty(b.shape)
    for start in range(0,len(basic),128):
        stop=min(start+128,len(basic))
        identity=np.zeros((len(basic),stop-start))
        identity[np.arange(start,stop),np.arange(stop-start)]=1.
        inverse[:,start:stop]=factor.solve(identity)
    inverse_sparse=csr_matrix(inverse)
    return dict(anchor,inverse=inverse_sparse,inverse_density=inverse_sparse.nnz/inverse.size,
        compilation_seconds=anchor["compilation_seconds"]+time.perf_counter()-started)


class GpuBasisEvaluator:
    """Device batch solves with row-low-rank updates and no online CPU solver.

Matrices are [environment, variable]. Call prepare_host only when the caller
still builds LPs on CPU; its transfer cost is not part of resident evaluation.
The GPU return mask is authoritative; invalid vectors are NaN, never accepted.
"""
    def __init__(self, anchor, variable_rows, primal_tolerance=1e-5, dual_tolerance=1e-7, gap_tolerance=1e-7):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as gpu_csr
        self.cp = cp
        self.anchor = anchor
        self.primal_tolerance, self.dual_tolerance, self.gap_tolerance = primal_tolerance, dual_tolerance, gap_tolerance
        self.variable_rows = np.asarray(sorted(set(variable_rows)), dtype=int)
        self.host_a = anchor["lp"].a
        if np.any(self.variable_rows < 0) or np.any(self.variable_rows >= self.host_a.shape[0]):
            raise ValueError("Invalid variable rows")
        self.a = gpu_csr(self.host_a)
        self.inverse = gpu_csr(anchor["inverse"])
        # Explicit CSR transposes avoid repeated scatter-style CSC SpMM.
        self.a_transpose = gpu_csr(self.host_a.T.tocsr())
        self.inverse_transpose = gpu_csr(anchor["inverse"].T.tocsr())
        self.basic = cp.asarray(anchor["basic"])
        self.active = cp.asarray(anchor["active"])
        self.kind = cp.asarray(anchor["kind"])
        self.var = cp.asarray(self.variable_rows)
        active_lookup = {row: i for i, row in enumerate(anchor["active"])}
        self.updated_positions = np.array([j for j, row in enumerate(self.variable_rows) if row in active_lookup], dtype=int)
        self.updated_active_rows = np.array([active_lookup[self.variable_rows[j]] for j in self.updated_positions], dtype=int)
        self.update_positions = cp.asarray(self.updated_positions)
        self.update_active = cp.asarray(self.updated_active_rows)
        self.u = cp.asarray(anchor["inverse"][:, self.updated_active_rows].toarray())
        self.neq = anchor["lp"].neq
        self.cpu_lp_calls = 0

    def prepare_host(self, problems):
        """Explicit CPU assembly/transfer adapter, not called by evaluate_device."""
        cp = self.cp
        deltas = []
        for problem in problems:
            if problem.a.shape != self.host_a.shape or problem.neq != self.neq:
                raise ValueError("LP structure changed")
            delta = (problem.a-self.host_a).tocsr()
            fixed = np.ones(delta.shape[0], dtype=bool); fixed[self.variable_rows] = False
            if np.max(np.abs(delta[fixed].data), initial=0) > 2e-12:
                raise ValueError("A changed outside the declared low-rank rows")
            deltas.append(delta[self.variable_rows].toarray())
        return dict(rhs=cp.asarray(np.stack([p.rhs for p in problems])),
            lower=cp.asarray(np.stack([p.lower for p in problems])),
            upper=cp.asarray(np.stack([p.upper for p in problems])),
            c=cp.asarray(np.stack([p.c for p in problems])), delta=cp.asarray(np.stack(deltas)),
            col_scale=cp.asarray(np.stack([p.col_scale for p in problems])),
            row_scale=cp.asarray(np.stack([p.row_scale for p in problems])))

    def evaluate_device(self, *, rhs, lower, upper, c, delta, col_scale, row_scale):
        cp = self.cp
        batch, n = lower.shape
        def max_rows(values):
            return cp.maximum(cp.max(values,axis=1),0.) if values.shape[1] else cp.zeros(batch)
        x = cp.where(self.kind[None] == -1, lower, cp.where(self.kind[None] == 1, upper, 0.))
        finite_input = cp.isfinite(x).all(axis=1) & cp.isfinite(rhs).all(axis=1) & cp.isfinite(c).all(axis=1)
        finite_input &= cp.isfinite(delta).all(axis=(1,2)) & cp.isfinite(col_scale).all(axis=1) & (col_scale>0).all(axis=1)
        finite_input &= cp.isfinite(row_scale).all(axis=1) & (row_scale>0).all(axis=1) & (lower<=upper).all(axis=1)
        # Replace invalid intermediate inputs only to prevent cross-batch NaNs;
        # their original mask remains false and results are always rejected.
        x = cp.where(cp.isfinite(x), x, 0.)
        activity = (self.a @ x.T).T
        activity[:, self.var] += cp.einsum("bkn,bn->bk", delta, x)
        rb = rhs[:, self.active]-activity[:, self.active]
        initial = (self.inverse @ rb.T).T
        changed = delta[:, self.update_positions][:, :, self.basic]
        if len(self.updated_positions):
            small = cp.eye(len(self.updated_positions))[None]+changed @ self.u
            correction = cp.linalg.solve(small, (changed @ initial[:,:,None]))[:,:,0]
            xb = initial-correction @ self.u.T
        else:
            small = None
            xb = initial
        x[:, self.basic] = xb
        # Refinement evaluates the actual basis equations on GPU; do not
        # weaken KKT tolerances when a compiled inverse accumulates roundoff.
        for _ in range(2):
            current=(self.a@x.T).T
            current[:,self.var]+=cp.einsum("bkn,bn->bk",delta,x)
            residual=(rhs-current)[:,self.active]
            correction=(self.inverse@residual.T).T
            if small is not None:
                q=cp.linalg.solve(small,changed@correction[:,:,None])[:,:,0]
                correction-=q@self.u.T
            x[:,self.basic]+=correction
        # Numerical stationarity uses the same low-rank basis, not a guessed
        # objective or a distance-to-dictionary acceptance rule.
        y0 = (self.inverse_transpose @ c[:,self.basic].T).T
        if small is not None:
            q = cp.linalg.solve(small.transpose(0,2,1), y0[:,self.update_active,None])[:,:,0]
            y_active = y0-(self.inverse_transpose @ cp.einsum("bkn,bk->bn", changed,q).T).T
        else:
            y_active = y0
        y = cp.zeros_like(rhs); y[:,self.active] = y_active
        for _ in range(2):
            current=(self.a_transpose@y.T).T+cp.einsum("bkn,bk->bn",delta,y[:,self.var])
            residual=(c-current)[:,self.basic]
            correction=(self.inverse_transpose@residual.T).T
            if small is not None:
                q=cp.linalg.solve(small.transpose(0,2,1),correction[:,self.update_active,None])[:,:,0]
                correction-=(self.inverse_transpose@cp.einsum("bkn,bk->bn",changed,q).T).T
            y[:,self.active]+=correction
        activity = (self.a @ x.T).T
        activity[:,self.var] += cp.einsum("bkn,bn->bk",delta,x)
        reduced = c-(self.a_transpose @ y.T).T-cp.einsum("bkn,bk->bn",delta,y[:,self.var])
        row_error = (activity-rhs)/row_scale
        bound_error = cp.maximum(lower-x,x-upper)/col_scale
        primal = cp.maximum(max_rows(cp.abs(row_error[:,:self.neq])),
            cp.maximum(max_rows(row_error[:,self.neq:]),max_rows(bound_error)))
        original_reduced = reduced*col_scale
        # A dual-simplex warm start requires reduced-cost signs at ALL
        # nonbasic finite bounds, not just the infinite-bound KKT test below.
        # The complementarity gap can be tiny for an infeasible primal basis
        # even when these signs are wrong.
        sign_error=cp.where(self.kind[None]==-1,-original_reduced,
            cp.where(self.kind[None]==1,original_reduced,cp.abs(original_reduced)))
        sign_error=cp.where((self.kind[None]!=2)&(upper>lower),sign_error,0.)
        basis_dual_violation=cp.maximum(max_rows(sign_error),max_rows((y*row_scale)[:,self.neq:]))
        infinite_violation = cp.maximum(cp.where(~cp.isfinite(lower),original_reduced,0.),
                                       cp.where(~cp.isfinite(upper),-original_reduced,0.))
        dual = cp.maximum(max_rows(infinite_violation),max_rows((y*row_scale)[:,self.neq:]))
        chosen_bound = cp.where(reduced>=0,lower,upper)
        bounded = cp.isfinite(chosen_bound)
        # Numerical KKT gap; an infinite-side reduced-cost violation is checked
        # separately above. This is not an exact-rational optimality proof.
        gap = cp.sum(cp.abs(reduced*(x-cp.where(bounded,chosen_bound,x))),axis=1)
        gap += cp.sum(cp.abs(y[:,self.neq:]*(rhs-activity)[:,self.neq:]),axis=1)
        objective = cp.sum(c*x,axis=1)
        relative_gap = gap/cp.maximum(1.,cp.abs(objective))
        accepted = finite_input & cp.isfinite(x).all(axis=1) & cp.isfinite(y).all(axis=1)
        accepted &= cp.isfinite(primal) & cp.isfinite(dual) & cp.isfinite(relative_gap)
        accepted &= (primal<=self.primal_tolerance) & (dual<=self.dual_tolerance) & (relative_gap<=self.gap_tolerance)
        return dict(accepted=accepted, values=cp.where(accepted[:,None],x/col_scale,cp.nan),
            objective=cp.where(accepted,objective,cp.nan), primal_residual=primal,
            dual_violation=dual, relative_kkt_gap=relative_gap,
            basis_dual_violation=basis_dual_violation,
            cpu_lp_calls=0, scope="GPU numerical basis/KKT certificate, not full trajectory qualification")
