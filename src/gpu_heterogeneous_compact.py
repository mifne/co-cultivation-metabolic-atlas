"""Experimental heterogeneous compact-basis screening for B environments × K proposals.

Only the requested bases are reconstructed. Indexed projection kernels read
flat per-candidate weights, without gathering B×K×basis_rows×parameters weights.
The complete, unchanged LP is certified once as a shared-A SpMM batch. This
prototype does not optimize rejected proposals, and is not a measured speedup.
"""
import numpy as np
from collections import OrderedDict

from .gpu_compact_basis import certify


_KERNELS = r'''
extern "C" __global__ void project(
    const double* weights, const long long* offsets, const int* heights,
    const int* widths, const int* candidate, const double* vector,
    double* output, int queries, int max_height, int stride) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)queries * max_height) return;
  int query = i / max_height, row = i % max_height;
  int c = candidate[query], height = heights[c], width = widths[c];
  double total = 0.;
  if (row < height) {
    long long offset = offsets[c];
    for (int column = 0; column < width; ++column)
      total += weights[offset + (long long)column * height + row]
          * vector[(long long)query * stride + column];
  }
  output[i] = total;
}

extern "C" __global__ void low_rank(
    const double* weights, const long long* offsets, const int* heights,
    const int* ranks, const int* candidate, const double* changed,
    double* output, int max_height, int max_rank) {
  int query = blockIdx.x / (max_rank * max_rank);
  int ij = blockIdx.x % (max_rank * max_rank);
  int left = ij / max_rank, right = ij % max_rank;
  int c = candidate[query], height = heights[c], rank = ranks[c];
  double total = 0.;
  if (left < rank && right < rank)
    for (int row = threadIdx.x; row < height; row += blockDim.x)
      total += changed[((long long)query * max_rank + left) * max_height + row]
          * weights[offsets[c] + (long long)right * height + row];
  __shared__ double sums[256];
  sums[threadIdx.x] = total;
  __syncthreads();
  for (int step = blockDim.x / 2; step; step /= 2) {
    if (threadIdx.x < step) sums[threadIdx.x] += sums[threadIdx.x + step];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    output[((long long)query * max_rank + left) * max_rank + right]
        = sums[0] + (left == right ? 1. : 0.);
}

extern "C" __global__ void scatter_basis(
    const int* candidate, const int* heights, const int* basic, const int* active,
    const double* xb, const double* y0, double* x, double* y,
    int queries, int max_height, int columns, int rows) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)queries * max_height) return;
  int query = i / max_height, row = i % max_height, c = candidate[query];
  // Padding must never write basic[padding]=0 or active[padding]=0.
  if (row < heights[c]) {
    x[(long long)query * columns + basic[(long long)c * max_height + row]] = xb[i];
    y[(long long)query * rows + active[(long long)c * max_height + row]] = y0[i];
  }
}

extern "C" __global__ void store_scores(
    const int* candidate, const bool* valid_index, const bool* unique_slot,
    const bool* family, const double* scores, const bool* dual,
    double* all_scores, bool* all_dual, int queries, int proposals, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= queries || !valid_index[i] || !unique_slot[i]) return;
  long long target = (long long)(i / proposals) * count + candidate[i];
  all_scores[target] = family[i] ? scores[i] : 1. / 0.;
  all_dual[target] = family[i] && dual[i];
}
'''


class _HeterogeneousReplay:
    def __init__(self, service):
        self.service = service; self.cp = service.cp
        # CUDA graph parameters alone do not keep Python GPU arrays alive.
        # Retain the projection used at capture even if the bank is reconfigured.
        self.observables = (None if service.certificate_only else
            (service.bank.observable_matrix, service.bank.observable_scales))

    def solve_device(self, inputs, order, conditional=None, _warm_iterations=None):
        return self.service.evaluate(inputs, order)


class HeterogeneousCompactBank:
    """Opt-in prototype; setup duplicates flat projection weights once on GPU.

    ``evaluate(inputs, order)`` accepts a host or GPU integer array of shape
    [environment, proposal]. Invalid candidate IDs fail closed per slot. The
    returned first accepted candidate retains the original LP certificate;
    spread only describes candidate observables and never accepts a solution.
    """
    def __init__(self, bank, *, certificate_only=False):
        import cupy as cp
        from .gpu_capture_math import CaptureMath
        if not isinstance(certificate_only, (bool, np.bool_)):
            raise TypeError('certificate_only must be boolean')
        self.bank = bank; self.cp = cp; self.closed = False
        self.certificate_only = bool(certificate_only)
        self.graph_cache = OrderedDict(); self.graph_cache_size = 2
        self.graph_compilation_seconds = 0.
        self.math = bank.math if bank.math is not None else CaptureMath()
        self.owns_math = bank.math is None
        self.entries = len(bank.evaluators)
        if not self.entries:
            raise ValueError('At least one compact candidate is required')
        self.rows, self.columns = bank.host_a.shape
        self.variable_count = len(bank.variable_rows)
        dictionaries = [evaluator.d for evaluator in bank.evaluators]
        heights = np.array([len(d['basic']) for d in dictionaries], dtype=np.int32)
        if np.any(heights <= 0) or np.any(heights > min(self.rows, self.columns)):
            raise ValueError('Invalid compact basis dimensions')
        self.height = int(heights.max()); self.heights = cp.asarray(heights)
        self.indices = {}; self.counts = {}
        index_names = ('basic', 'active', 'representatives', 'mapping', 'rhs_indices',
            'cost_representatives', 'cost_mapping', 'update_positions', 'update_active', 'delta_support')
        for name in index_names:
            lengths = np.array([d[name].size for d in dictionaries], dtype=np.int32)
            width = int(lengths.max())
            if name in ('basic', 'active', 'cost_mapping'):
                width = self.height
            values = cp.zeros((self.entries, width), dtype=cp.int32)
            for candidate, d in enumerate(dictionaries):
                values[candidate, :len(d[name])] = d[name]
            self.indices[name] = values
            self.counts[name] = cp.asarray(lengths)
        self.kind = cp.stack([d['kind'] for d in dictionaries])
        self.offset = cp.stack([d['offset'].reshape(self.variable_count, self.columns) for d in dictionaries])
        self.supported_rhs = cp.zeros((self.entries, self.height), dtype=bool)
        self.supported_delta = cp.zeros_like(self.supported_rhs)
        for candidate, d in enumerate(dictionaries):
            self.supported_rhs[candidate, d['rhs_indices']] = True
            self.supported_delta[candidate, d['delta_support']] = True
        self.weights = {}
        for name in ('p_rhs', 'p_bound', 'u', 'p_cost', 'p_delta'):
            shapes = [d[name].shape for d in dictionaries]
            if any(len(shape) != 2 or shape[0] != heights[i] for i, shape in enumerate(shapes)):
                raise ValueError('Inconsistent compact projection shape: ' + name)
            widths = np.array([shape[1] for shape in shapes], dtype=np.int32)
            sizes = heights.astype(np.int64) * widths
            offsets = np.r_[0, np.cumsum(sizes[:-1])].astype(np.int64)
            # Column-major candidate blocks make consecutive basis rows
            # contiguous for each parameter, enabling coalesced indexed reads.
            weights = cp.concatenate([d[name].T.ravel() for d in dictionaries]).astype(cp.float64, copy=False)
            self.weights[name] = (weights, cp.asarray(offsets), cp.asarray(widths))
        self.rank = self.indices['update_positions'].shape[1]
        self.project_kernel = cp.RawKernel(_KERNELS, 'project')
        self.low_rank_kernel = cp.RawKernel(_KERNELS, 'low_rank')
        self.scatter_kernel = cp.RawKernel(_KERNELS, 'scatter_basis')
        self.score_kernel = cp.RawKernel(_KERNELS, 'store_scores')
        self.setup_weight_bytes = sum(item[0].nbytes for item in self.weights.values())
        self.last_candidate_scores = None
        self.last_candidate_dual = None
        self.last_candidates = None

    def _project(self, name, candidates, vector):
        cp = self.cp
        vector = cp.ascontiguousarray(vector, dtype=cp.float64)
        queries = len(candidates)
        output = cp.empty((queries, self.height), dtype=cp.float64)
        weights, offsets, widths = self.weights[name]
        self.project_kernel(((output.size+127)//128,), (128,),
            (weights, offsets, self.heights, widths, candidates, vector, output,
             np.int32(queries), np.int32(self.height), np.int32(vector.shape[1])))
        return output

    def _candidates(self, inputs, order):
        """Reconstruct every selected candidate, including all family guards."""
        cp = self.cp
        batch, proposals = order.shape; queries = batch*proposals
        valid_index = (order >= 0) & (order < self.entries)
        candidates = cp.clip(order.ravel(), 0, self.entries-1).astype(cp.int32)
        expanded = {key: cp.repeat(cp.asarray(value, dtype=cp.float64), proposals, axis=0)
            for key, value in inputs.items()}
        get = lambda name: self.indices[name][candidates]
        take = lambda values, positions: cp.take_along_axis(values, positions, axis=1)
        basic, active = get('basic'), get('active')
        height_mask = cp.arange(self.height)[None] < self.heights[candidates, None]
        kind = self.kind[candidates]
        lo, hi, c, rhs = (expanded[key] for key in ('lower', 'upper', 'c', 'rhs'))
        xn = cp.where(kind == -1, lo, cp.where(kind == 1, hi, 0.))
        theta = take(xn, get('representatives'))
        cb = cp.where(height_mask, take(c, basic), 0.)
        theta_c = take(cb, get('cost_representatives'))
        active_rhs = cp.where(height_mask, take(rhs, active), 0.)
        valid = valid_index.ravel() & cp.isfinite(xn).all(axis=1)
        valid &= cp.isfinite(rhs).all(axis=1) & cp.isfinite(c).all(axis=1)
        valid &= cp.isfinite(expanded['delta']).all(axis=(1, 2)) & (lo <= hi).all(axis=1)
        valid &= ~cp.isnan(lo).any(axis=1) & ~cp.isnan(hi).any(axis=1)
        for name in ('col_scale', 'row_scale'):
            valid &= cp.isfinite(expanded[name]).all(axis=1) & (expanded[name] > 0).all(axis=1)
        valid &= cp.max(cp.abs(xn-take(theta, get('mapping'))), axis=1) <= 1e-12
        cb_error = cp.where(height_mask, cb-take(theta_c, get('cost_mapping')), 0.)
        valid &= cp.max(cp.abs(cb_error), axis=1) <= 1e-12
        residual_rhs = cp.where(self.supported_rhs[candidates], 0., active_rhs)
        valid &= cp.max(cp.abs(residual_rhs), axis=1) <= 1e-12
        theta = cp.nan_to_num(theta); theta_c = cp.nan_to_num(theta_c)
        rhs_parameters = take(active_rhs, get('rhs_indices'))
        xb = self._project('p_rhs', candidates, rhs_parameters)-self._project('p_bound', candidates, theta)
        y0 = self._project('p_cost', candidates, theta_c)
        if self.rank:
            rank_mask = cp.arange(self.rank)[None] < self.counts['update_positions'][candidates, None]
            changed_fields = expanded['delta']-self.offset[candidates]
            delta = cp.take_along_axis(changed_fields, get('update_positions')[:, :, None], axis=1)
            delta = cp.where(rank_mask[:, :, None], delta, 0.)
            changed = cp.take_along_axis(delta, basic[:, None, :], axis=2)
            changed = cp.where(height_mask[:, None, :], changed, 0.)
            extra = cp.where(self.supported_delta[candidates, None, :], 0., changed)
            valid &= cp.max(cp.abs(extra), axis=(1, 2)) <= 1e-12
            xb -= self._project('u', candidates, self.math.einsum('bkn,bn->bk', delta, xn))
            small = cp.empty((queries, self.rank, self.rank), dtype=cp.float64)
            weights, offsets, widths = self.weights['u']
            self.low_rank_kernel((queries*self.rank*self.rank,), (256,),
                (weights, offsets, self.heights, widths, candidates, cp.ascontiguousarray(changed), small,
                 np.int32(self.height), np.int32(self.rank)))
            correction = self.math.solve(small, self.math.einsum('bkn,bn->bk', changed, xb)[:, :, None])[:, :, 0]
            xb -= self._project('u', candidates, correction)
            dual_rhs = cp.where(rank_mask, take(y0, get('update_active')), 0.)
            q = self.math.solve(small.transpose(0, 2, 1), dual_rhs[:, :, None])[:, :, 0]
            support = cp.take_along_axis(changed, get('delta_support')[:, None, :], axis=2)
            y0 -= self._project('p_delta', candidates, self.math.einsum('bkn,bk->bn', support, q))
        x = xn.copy(); y = cp.zeros_like(rhs)
        self.scatter_kernel(((queries*self.height+127)//128,), (128,),
            (candidates, self.heights, self.indices['basic'], self.indices['active'], xb, y0, x, y,
             np.int32(queries), np.int32(self.height), np.int32(self.columns), np.int32(self.rows)))
        out = certify(cp, self.bank.a, self.bank.at, self.bank.var, self.bank.neq, expanded, x, y, self.math)
        out['accepted'] &= valid
        out['values'] = cp.where(out['accepted'][:, None], x/expanded['col_scale'], cp.nan)
        if self.certificate_only:
            # The strict original-LP certificate above still evaluates the
            # complete primal, dual and KKT-gap conditions.  Zero-repair
            # speculative routing needs neither rejected-candidate scoring nor
            # observable dispersion, so do not create those grids here.
            return out, expanded, candidates, valid_index.ravel()
        bound_error = cp.maximum(lo-x, x-hi)/expanded['col_scale']
        row_error = (out['activity']-rhs)/expanded['row_scale']
        row_error[:, :self.bank.neq] = cp.abs(row_error[:, :self.bank.neq])
        out['primal_violation_count'] = cp.sum(bound_error > 1e-5, axis=1)+cp.sum(row_error > 1e-5, axis=1)
        out['primal_violation_l1'] = cp.maximum(bound_error, 0.).sum(axis=1)+cp.maximum(row_error, 0.).sum(axis=1)
        rc = out['raw_reduced']*expanded['col_scale']
        sign = cp.where(kind == -1, -rc, cp.where(kind == 1, rc, cp.abs(rc)))
        sign = cp.where((kind != 2) & (hi > lo), sign, 0.)
        out['basis_dual_violation'] = cp.maximum(cp.max(sign, axis=1), 0.)
        if self.rows > self.bank.neq:
            out['basis_dual_violation'] = cp.maximum(out['basis_dual_violation'],
                cp.max((y*expanded['row_scale'])[:, self.bank.neq:], axis=1))
        out['input_family_valid'] = valid
        return out, expanded, candidates, valid_index.ravel()

    def evaluate(self, inputs, order):
        if self.closed:
            raise RuntimeError('Heterogeneous compact prototype is closed')
        cp = self.cp
        order = cp.asarray(order)
        if order.ndim != 2 or order.dtype.kind not in 'iu' or not order.shape[1]:
            raise ValueError('A nonempty B×K integer candidate order is required')
        batch, proposals = order.shape
        expected = dict(rhs=(batch, self.rows), lower=(batch, self.columns), upper=(batch, self.columns),
            c=(batch, self.columns), delta=(batch, self.variable_count, self.columns),
            col_scale=(batch, self.columns), row_scale=(batch, self.rows))
        if set(inputs) != set(expected) or any(inputs[key].shape != shape for key, shape in expected.items()):
            raise ValueError('Heterogeneous compact input shape does not match the bank')
        if not batch:
            raise ValueError('At least one environment is required')
        out, expanded, candidates, valid_index = self._candidates(inputs, order)
        accepted = out['accepted'].reshape(batch, proposals)
        has_solution = accepted.any(axis=1)
        first_slot = cp.argmax(accepted, axis=1)
        if self.certificate_only:
            # For a rejected row the slot is intentionally arbitrary: all
            # returned values are uncertified and values remains NaN.  For an
            # accepted row argmax returns the first True slot, exactly matching
            # the full diagnostic path's acceptance ordering.
            chosen = cp.arange(batch)*proposals+first_slot
            result = {key:out[key][chosen] for key in ('values', 'objective',
                'primal_residual', 'dual_violation', 'relative_kkt_gap')}
            result['accepted'] = has_solution
            result['values'] = cp.where(has_solution[:, None], result['values'], cp.nan)
            result['candidate_index'] = cp.where(has_solution, candidates[chosen], -1)
            result['candidate_evaluations'] = batch*proposals
            # Explicitly clear possibly stale full-diagnostic references if a
            # caller deliberately switches this opt-in mode between calls.
            self.last_candidate_scores = None
            self.last_candidate_dual = None
            self.last_candidates = None
            return result
        score = cp.maximum(out['primal_residual']/1e-5,
            cp.maximum(out['dual_violation']/1e-7, out['relative_kkt_gap']/1e-7))
        if self.bank.candidate_ranking in ('count', 'count_only'):
            l1 = out['primal_violation_l1']
            score = out['primal_violation_count']+l1/(1.+l1)
        dual = out['basis_dual_violation'] <= 1e-8
        eligible = (out['input_family_valid'] & cp.isfinite(score)).reshape(batch, proposals)
        score_rows = score.reshape(batch, proposals)
        if self.bank.candidate_ranking != 'count_only':
            dual_rows = dual.reshape(batch, proposals) & eligible
            eligible &= ~dual_rows.any(axis=1)[:, None] | dual_rows
        best_slot = cp.argmin(cp.where(eligible, score_rows, cp.inf), axis=1)
        chosen_slot = cp.where(has_solution, first_slot, best_slot)
        chosen = cp.arange(batch)*proposals+chosen_slot
        result = {key: value[chosen] for key, value in out.items()}
        result['accepted'] = has_solution
        result['values'] = cp.where(has_solution[:, None], result['values'], cp.nan)
        result.update(candidate_index=cp.where(has_solution, candidates[chosen], -1),
            best_candidate_index=cp.where(has_solution | eligible.any(axis=1), candidates[chosen], -1),
            best_candidate_dual_feasible=dual[chosen] & result['input_family_valid'],
            candidate_evaluations=batch*proposals, cpu_lp_calls=0,
            scope='Experimental heterogeneous compact reconstruction with original full-LP certificate')
        slots = cp.arange(proposals)
        duplicate = ((order[:, :, None] == order[:, None, :]) & (slots[None, None, :] < slots[None, :, None])).any(axis=2)
        unique = cp.ascontiguousarray(~duplicate).ravel()
        self.last_candidate_scores = cp.full((batch, self.entries), cp.inf)
        self.last_candidate_dual = cp.zeros((batch, self.entries), dtype=bool)
        self.score_kernel(((batch*proposals+255)//256,), (256,),
            (candidates, valid_index, unique, out['input_family_valid'], score, dual,
             self.last_candidate_scores, self.last_candidate_dual,
             np.int32(batch*proposals), np.int32(proposals), np.int32(self.entries)))
        result['last_candidate_scores'] = self.last_candidate_scores
        result['last_candidate_dual'] = self.last_candidate_dual
        # Slot-indexed GPU metadata avoids fetching candidate IDs just to build
        # a Python list of environment subsets. This intentionally differs from
        # the legacy bank.last_candidates list interface in this prototype.
        self.last_candidates = dict(indices=order, **{key: value.reshape(batch, proposals)
            for key, value in out.items() if value.ndim == 1})
        result['candidate_metrics'] = self.last_candidates
        if self.bank.observable_matrix is not None:
            physical = out['raw_values']/expanded['col_scale']
            observable = self.math.mm(self.bank.observable_matrix, physical.T).T
            valid = out['input_family_valid'] & cp.isfinite(physical).all(axis=1)
            valid &= cp.isfinite(observable).all(axis=1) & unique
            observable = observable.reshape(batch, proposals, -1)
            valid = valid.reshape(batch, proposals)
            count = cp.zeros(batch, dtype=cp.int64)
            mean = cp.zeros((batch, observable.shape[2])); m2 = cp.zeros_like(mean)
            for slot in range(proposals):
                values = cp.where(valid[:, slot, None], observable[:, slot], mean)
                count += valid[:, slot].astype(cp.int64)
                difference = values-mean
                updated = mean+difference/cp.maximum(count, 1)[:, None]
                m2 += difference*(values-updated); mean = updated
            std = cp.sqrt(cp.maximum(m2, 0.)/cp.maximum(count-1, 1)[:, None])
            spread = cp.max(std/self.bank.observable_scales, axis=1)
            result['candidate_count'] = count
            result['observable_dispersion'] = cp.where(count >= 2, spread, cp.inf)
        return result

    def evaluate_replay(self, inputs, order):
        """One graph for a fixed B,K shape; candidate IDs remain dynamic inputs."""
        from .gpu_replay_call import GpuReplayCall, signature
        if self.closed:
            raise RuntimeError('Heterogeneous compact prototype is closed')
        order = self.cp.asarray(order)
        if order.ndim != 2 or order.dtype.kind not in 'iu' or not order.shape[1]:
            raise ValueError('A nonempty B×K integer candidate order is required')
        arguments = dict(inputs=inputs, order=order)
        key = (signature(arguments), self.certificate_only,
            None if self.certificate_only else self.bank.candidate_ranking,
            None if self.certificate_only else id(self.bank.observable_matrix),
            None if self.certificate_only else id(self.bank.observable_scales))
        if key not in self.graph_cache:
            while len(self.graph_cache) >= self.graph_cache_size:
                self.graph_cache.popitem(last=False)[1].close()
            self.graph_cache[key] = GpuReplayCall(_HeterogeneousReplay(self), arguments)
            self.graph_compilation_seconds += self.graph_cache[key].compilation_seconds
        self.graph_cache.move_to_end(key)
        result = self.graph_cache[key].run(arguments)
        if self.certificate_only:
            self.last_candidate_scores = None
            self.last_candidate_dual = None
            self.last_candidates = None
        else:
            self.last_candidate_scores = result['last_candidate_scores']
            self.last_candidate_dual = result['last_candidate_dual']
            self.last_candidates = result['candidate_metrics']
        return result

    def clear_graph_cache(self):
        while self.graph_cache:
            self.graph_cache.popitem()[1].close()

    def close(self):
        if not self.closed:
            self.clear_graph_cache()
            self.cp.cuda.get_current_stream().synchronize()
            if self.owns_math:
                self.math.close()
            self.weights.clear(); self.indices.clear(); self.counts.clear()
            self.kind = self.offset = self.heights = None
            self.supported_rhs = self.supported_delta = None
            self.closed = True
