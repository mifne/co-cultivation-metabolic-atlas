"""Fixed-layout device gathers for all mutable forest-IPM numeric payloads.

Construction is host-only symbolic work plus initial device validation. Stage
uses CURRENT DeviceLPBatch arrays, not saved host numeric values, and neither
mutates solver buffers nor downloads old numeric vectors. The enclosing
transaction owns input/proof/topology validation, static integrity checks,
generation tracking, commit, factor invalidation and current-LP acceptance.
This binder alone does not validate a new LP or authorize a solve.
"""

import math

import numpy as np
from scipy.sparse import bmat, block_diag, csr_matrix, diags, vstack

from .gpu_batched_ipm import constraint_form, uniform_kkt_pattern
from .gpu_ipm_kkt_payload import _canonical_csr_parts
from .gpu_ipm_numeric_update import _existing_host_state, _payloads, _target
from .gpu_lp_numeric import DeviceLPBatch


_PHASES = ('full', 'forest', 'face', 'secondary', 'core')
_WITNESSES = ('lower_witness', 'upper_witness', 'fallback_witness')


def _same_pattern(a, b):
    return (a.shape == b.shape and np.array_equal(a.indptr, b.indptr)
            and np.array_equal(a.indices, b.indices))


def _same_problem(a, b):
    return (a[-1] == b[-1] and _same_pattern(a[0], b[0])
            and np.array_equal(a[0].data, b[0].data)
            and all(np.array_equal(x, y) for x, y in zip(a[1:5], b[1:5])))


class DeviceIPMPayloadBinder:
    """Compile fixed gathers and expose staged ``(path, target, source)``.

    ``host_states`` has full/forest/face/secondary/core ordered LP tuples.
    ``stage`` takes the same keys with DeviceLPBatch values and witness triples
    of exact int64 [batch, reduced_columns] arrays. Missing secondary forests
    require second_witnesses=None. All input arrays must have already passed
    the enclosing transaction's numeric and fixed-layout proof guards.

    ``static_targets`` contains (path, live_array, owned_device_snapshot).
    Paths beginning '__binder__' are diagnostic labels for binder-owned maps;
    all other paths resolve from the solver. Root must check these snapshots
    before every stage/commit, including the maps used by stage itself.
    """

    def __init__(self, solver, host_states):
        solver.factor._context()
        if (not solver.globalized or solver.near_equality_plans
                or getattr(solver, 'equality_row_scaling', False)):
            raise ValueError('Require globalized IPM without near equalities or row scaling')
        if set(host_states) != set(_PHASES):
            raise ValueError('Complete full/forest/face/secondary/core host states required')
        self.solver, self.cp = solver, solver.cp
        self.batch, self.device = solver.batch, solver.factor.device
        self.static_targets, self._bindings, self._maps = [], {}, {}
        self._binding_specs, self._map_specs = {}, {}
        self._recipes, self._checks, self._shapes = {}, [], {}
        self._static_paths = set()
        self._secondary = bool(solver.secondary_forest_reductions)
        state = _existing_host_state(solver)
        expected = dict(full=state.full_problems, forest=state.forest_problems,
            face=tuple(p.reduced for p in state.face_plans), core=state.problems)
        expected['secondary'] = (tuple(p.problem for p in state.secondary_forest_reductions)
                                 if self._secondary else expected['face'])
        hosts = {}
        for phase in _PHASES:
            ps = tuple(host_states[phase])
            if len(ps) != self.batch or not ps:
                raise ValueError('Incomplete host batch: '+phase)
            shape, neq = ps[0][0].shape, ps[0][-1]
            for i, (p, old) in enumerate(zip(ps, expected[phase])):
                _canonical_csr_parts(p[0], label=f'{phase}[{i}]')
                if (p[0].shape != shape or p[-1] != neq or not _same_problem(p, old)):
                    raise ValueError('Host state differs from current solver: '+phase)
            m, n = shape
            count = sum(p[0].nnz for p in ps)
            if count+5 >= 2**53:
                raise ValueError('Unsupported symbolic token range')
            self._shapes[phase] = (count, m, n, neq)
            hosts[phase] = ps

        baseline_sparse, baseline_arrays = _payloads(solver, state, _direct_kkt_payload=True)
        self._baseline_sparse, self._baseline_arrays = baseline_sparse, baseline_arrays
        tags = {}
        for phase, ps in hosts.items():
            offset, tagged = 0, []
            for p in ps:
                a = p[0]
                tagged.append(csr_matrix((np.arange(offset+1, offset+a.nnz+1, dtype=np.float64),
                    a.indices.copy(), a.indptr.copy()), shape=a.shape))
                offset += a.nnz
            tags[phase] = tagged

        for name, phase in (('assembled', 'core'), ('original_assembled', 'forest'),
                             ('_full_assembled', 'full')):
            a = block_diag(tags[phase], format='csr')
            self._sparse((name, 0), a, phase)
            for i, field in ((1, 'row_lower'), (2, 'rhs'), (3, 'lower'), (4, 'upper'), (5, 'c')):
                self._vector((name, i), phase, field, flatten=True)
            if name != 'assembled':
                self._sparse(('original_at' if name == 'original_assembled' else '_full_at',),
                             a.T.tocsr(), phase)

        forms = tuple(constraint_form(p) for p in hosts['core'])
        n, ne, q = solver.n, solver.ne, solver.q
        fixed, il, iu = forms[0][4:7]
        if (any(not np.array_equal(form[j], forms[0][j]) for form in forms for j in (4, 5, 6))
                or any(form[0].shape != (ne, n) or form[2].shape[0] != solver.ng for form in forms)
                or self._shapes['core'][3]+np.count_nonzero(fixed) != ne
                or self._shapes['core'][1]-self._shapes['core'][3] != q):
            raise ValueError('Core bound masks or dimensions differ across lanes')
        self._fixed = self._map('fixed', np.flatnonzero(fixed))
        self._il, self._iu = self._map('il', il), self._map('iu', iu)
        count = self._shapes['core'][0]
        # Positive symbolic tokens refer to data or [1,-1,delta,-delta,0].
        identity = diags(np.full(n, count+1., dtype=np.float64), format='csr')
        negative_identity = diags(np.full(n, count+2., dtype=np.float64), format='csr')
        eq, ge = [], []
        neq = self._shapes['core'][3]
        for a in tags['core']:
            eq.append(vstack((a[:neq], identity[fixed]), format='csr'))
            ge.append(vstack((a[neq:], negative_identity[il], identity[iu]), format='csr'))
        e, g = block_diag(eq, format='csr'), block_diag(ge, format='csr')
        for path, matrix in ((('e',), e), (('et',), e.T.tocsr()),
                             (('g',), g), (('gt',), g.T.tocsr())):
            self._sparse(path, matrix, 'core')
        self._vector(('b',), 'core', 'b')
        self._vector(('h',), 'core', 'h')
        self._vector(('c',), 'core', 'c')
        if hasattr(solver, '_condensed_h') != hasattr(solver, '_condensed_ht'):
            raise ValueError('Incomplete condensed operator cache')
        if hasattr(solver, '_condensed_h'):
            h = block_diag([a[:q] for a in ge], format='csr')
            self._sparse(('_condensed_h',), h, 'core')
            self._sparse(('_condensed_ht',), h.T.tocsr(), 'core')
        self._had_condensed = hasattr(solver, '_condensed_h')
        matrices = [bmat([[diags(np.full(n, count+3.)), a.T, b[:q].T],
            [a, diags(np.full(ne, count+4.)), None],
            [b[:q], None, diags(np.full(q, count+2.))]], format='csr') for a, b in zip(eq, ge)]
        pattern, tokens, _ = uniform_kkt_pattern(matrices)
        if not _same_pattern(pattern, solver.factor.host_pattern):
            raise ValueError('Symbolic KKT pattern differs from exact full lane union')
        tokens[tokens == 0.] = count+5.
        kkt_map = self._map('kkt', tokens.astype(np.int64)-1)
        for path in (('values',), ('factor', 'values')):
            self._bind(path, baseline_arrays[path], check_values=False)
            self._recipes[path] = ('gather', 'core', kkt_map)
        for name in _WITNESSES:
            path = ('_forest_'+name,)
            self._bind(path, baseline_arrays[path])
            self._recipes[path] = ('witness', 0, _WITNESSES.index(name))

        if self._secondary:
            self._compile_secondary(tags['face'], hosts['face'])

        # Every legacy mutable payload is covered; everything else is static.
        for path, host in baseline_sparse.items():
            target = _target(solver, path)
            for part in ('indptr', 'indices'):
                self._static(path+(part,), getattr(host, part))
            if path+('data',) not in self._recipes:
                self._static(path+('data',), host.data)
        for path, host in baseline_arrays.items():
            if path not in self._recipes:
                self._static(path, host)
        for name in ('batch_index', '_forest_batch'):
            self._static((name,), np.arange(self.batch, dtype=np.int64)[:, None])
        for name in ('indptr', 'indices'):
            if not hasattr(solver.factor, name):
                raise ValueError('Native factor must expose its fixed CSR coordinates')
            self._static(('factor', name), getattr(solver.factor.host_pattern, name))
        if getattr(solver.factor,'execution_layout','uniform')=='block_diagonal':
            pattern=solver.factor.host_pattern
            pointers=np.r_[np.concatenate([pattern.indptr[:-1].astype(np.int64)+i*pattern.nnz
                for i in range(self.batch)]),self.batch*pattern.nnz].astype(np.int32)
            columns=np.concatenate([pattern.indices.astype(np.int64)+i*pattern.shape[0]
                for i in range(self.batch)]).astype(np.int32)
            self._static(('factor','_native_indptr'),pointers)
            self._static(('factor','_native_indices'),columns)
        if getattr(solver.factor, 'transpose_positions', None) is not None:
            pattern = solver.factor.host_pattern
            size = pattern.shape[0]
            rows = np.repeat(np.arange(size), np.diff(pattern.indptr))
            keys = rows*size+pattern.indices
            transpose = np.searchsorted(keys, pattern.indices*size+rows)
            self._static(('factor', 'transpose_positions'), transpose)
        if not bool(self.cp.all(self.cp.stack(self._checks))):
            raise ValueError('Initial device payload does not match verified host state')
        self.numeric_targets = [(path, self._bindings[path]) for path in self._recipes]
        self._pool_phases = tuple(phase for phase in _PHASES if any(
            recipe[:2] == ('gather', phase) for recipe in self._recipes.values()))
        # Stage retains no baseline numeric vectors and performs no host CSR algebra.
        del self._baseline_sparse, self._baseline_arrays, self._checks

    def _array(self, value, shape, dtype, label):
        if (not isinstance(value, self.cp.ndarray) or value.shape != shape
                or value.dtype != np.dtype(dtype) or not value.flags.c_contiguous
                or (self.cp is not np and value.device.id != self.device)):
            raise ValueError('Exact contiguous shape/dtype/device required: '+str(label))

    def _signature(self, value):
        pointer = value.ctypes.data if self.cp is np else value.data.ptr
        return (value.shape, value.dtype, pointer)

    def _bind(self, path, expected, *, check_values=True):
        target = _target(self.solver, path)
        expected = np.asarray(expected)
        dtype = expected.dtype
        if dtype.kind in 'iu':
            if target.dtype not in (np.dtype('int32'), np.dtype('int64')):
                raise ValueError('Integer coordinates required: '+str(path))
            dtype = target.dtype
            if not np.array_equal(expected.astype(dtype), expected):
                raise ValueError('Coordinate cast overflow: '+str(path))
        self._array(target, expected.shape, dtype, path)
        self._bindings[path] = target
        self._binding_specs[path] = self._signature(target)
        if check_values:
            self._checks.append(self.cp.array_equal(target, self.cp.asarray(expected, dtype=dtype)))
        return target

    def _static(self, path, expected):
        if path in self._static_paths:
            return
        target = self._bind(path, expected)
        self.static_targets.append((path, target, target.copy()))
        self._static_paths.add(path)

    def _map(self, key, values):
        values = np.asarray(values, dtype=np.int64)
        target = self.cp.asarray(values)
        self._maps[key] = target
        self._map_specs[key] = (target, self._signature(target))
        self.static_targets.append((('__binder__', 'maps', key), target, target.copy()))
        return target

    def _sparse(self, path, tagged, phase, *, expected=None):
        expected = self._baseline_sparse[path] if expected is None else expected
        if not _same_pattern(tagged, expected):
            raise ValueError('Compiled sparse gather pattern differs: '+str(path))
        target = _target(self.solver, path)
        if target.shape != expected.shape:
            raise ValueError('Sparse target shape differs: '+str(path))
        indices = tagged.data.astype(np.int64)-1
        if (not np.array_equal(indices+1, tagged.data) or np.any(indices < 0)
                or np.any(indices >= self._shapes[phase][0]+5)):
            raise ValueError('Invalid symbolic numeric gather')
        self._bind(path+('data',), expected.data)
        self._recipes[path+('data',)] = ('gather', phase, self._map(str(path), indices))

    def _vector(self, path, phase, field, *, flatten=False):
        self._bind(path, self._baseline_arrays[path])
        self._recipes[path] = ('vector', phase, field, flatten)

    def _compile_secondary(self, tagged_face, face):
        prefix = ('secondary_forest_map',)
        current = self.solver.secondary_forest_map
        current._context()
        if current.device != self.device or current.stream.ptr != self.solver.factor.stream.ptr:
            raise ValueError('Secondary forest device/stream differs')
        host = current._host
        tagged = block_diag(tagged_face, format='csr').T.tocsr()
        expected = block_diag([p[0] for p in face], format='csr').T.tocsr()
        self._sparse(prefix+('_original_at',), tagged, 'face', expected=expected)
        self._bind(prefix+('_objective',), np.stack([p[4] for p in face]))
        self._recipes[prefix+('_objective',)] = ('vector', 'face', 'c', False)
        for i, name in enumerate(_WITNESSES):
            path = prefix+('_'+name,)
            self._bind(path, getattr(host, name))
            self._recipes[path] = ('witness', 1, i)
        for name in ('transform', 'compression', 'dual_lift', 'original_at', 'transform_t'):
            matrix = (host.transform.T.tocsr() if name == 'transform_t'
                      else expected if name == 'original_at' else getattr(host, name))
            for part in ('indptr', 'indices'):
                self._static(prefix+('_'+name, part), getattr(matrix, part))
            if name != 'original_at':
                self._static(prefix+('_'+name, 'data'), matrix.data)
        for name in ('kept_rows', 'eliminated_rows', 'weights'):
            self._static(prefix+('_'+name,), getattr(host, name))
        self._static(prefix+('_batch_rows',), np.arange(self.batch, dtype=np.int64)[:, None])

    def stage(self, batches, first_witnesses, second_witnesses=None):
        """Return owned ready-to-copy sources; no mutation, solve or acceptance.

        The caller must validate static_targets and all incoming numeric /
        proof guards first. Staging cannot certify topology from numeric-only
        arrays. Inputs may be reused or released after these same-stream
        gathers finish; the returned copies retain independent source buffers.
        """
        self.solver.factor._context()
        if self._secondary:
            self.solver.secondary_forest_map._context()
        if (set(batches) != set(_PHASES)
                or self._had_condensed != hasattr(self.solver, '_condensed_h')
                or self._had_condensed != hasattr(self.solver, '_condensed_ht')):
            raise ValueError('Complete unchanged batch/cache layout required')
        for phase in _PHASES:
            batch = batches[phase]
            if not isinstance(batch, DeviceLPBatch):
                raise ValueError('DeviceLPBatch required: '+phase)
            count, m, n, _ = self._shapes[phase]
            for value, shape in zip(batch.arrays(), ((count,), (self.batch, m),
                    (self.batch, n), (self.batch, n), (self.batch, n))):
                self._array(value, shape, np.float64, phase)
        witnesses = (first_witnesses, second_witnesses)
        for which, size in ((0, self._shapes['forest'][2]),
                            (1, self._shapes['secondary'][2])):
            if which == 1 and not self._secondary:
                if witnesses[1] is not None:
                    raise ValueError('Unexpected second forest witnesses')
                continue
            if not isinstance(witnesses[which], tuple) or len(witnesses[which]) != 3:
                raise ValueError('Complete ordered witness triple required')
            for value in witnesses[which]:
                self._array(value, (self.batch, size), np.int64, 'witness')
        for path, target in self._bindings.items():
            current = _target(self.solver, path)
            if current is not target or self._signature(current) != self._binding_specs[path]:
                raise ValueError('Bound target object was replaced: '+str(path))
            shape, dtype, _ = self._binding_specs[path]
            self._array(current, shape, dtype, path)
        for key, (target, signature) in self._map_specs.items():
            if self._maps.get(key) is not target or self._signature(target) != signature:
                raise ValueError('Binder gather map was replaced: '+key)
            self._array(target, signature[0], signature[1], key)
        if any(getattr(self, '_'+name) is not self._maps[name] for name in ('fixed', 'il', 'iu')):
            raise ValueError('Binder vector gather map alias was replaced')
        delta = self.solver.regularization
        if type(delta) not in (float, int) or not math.isfinite(delta) or delta <= 0.:
            raise ValueError('Positive finite scalar regularization required')
        cp = self.cp
        constants = cp.asarray([1., -1., delta, -delta, 0.], dtype=np.float64)
        pools = {phase: cp.concatenate((batches[phase].data, constants)) for phase in self._pool_phases}
        core, neq = batches['core'], self._shapes['core'][3]
        special = {'b': cp.concatenate((core.rhs[:, :neq], core.lower[:, self._fixed]), axis=1),
            'h': cp.concatenate((core.rhs[:, neq:], -core.lower[:, self._il],
                                  core.upper[:, self._iu]), axis=1)}
        result = []
        for path, recipe in self._recipes.items():
            kind = recipe[0]
            if kind == 'gather':
                source = pools[recipe[1]][recipe[2]]
            elif kind == 'witness':
                source = witnesses[recipe[1]][recipe[2]].copy()
            else:
                _, phase, field, flatten = recipe
                batch = batches[phase]
                if field == 'row_lower':
                    _, m, _, ne = self._shapes[phase]
                    source = cp.concatenate((batch.rhs[:, :ne],
                        cp.full((self.batch, m-ne), -cp.inf, dtype=np.float64)), axis=1)
                elif field in special:
                    source = special[field].copy()
                else:
                    source = getattr(batch, field).copy()
                if flatten:
                    source = source.reshape(-1)
            target = self._bindings[path]
            self._array(source, target.shape, target.dtype, path)
            result.append((path, target, source))
        return result
