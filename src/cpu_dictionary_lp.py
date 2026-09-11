"""CPU-only dictionary-initialized HiGHS comparator for repeated LPs.

GPU arrays are copied to immutable host metadata only during construction.
Timed solves use NumPy/SciPy nearest-centroid proposals and persistent HiGHS;
the dictionary never supplies a final flux or bypasses the LP certificate.
This baseline separates a good initial basis from GPU execution benefits.
"""
import time

import numpy as np
from scipy.sparse import csr_matrix

from .cpu_repeated_lp import RepeatedCpuLP, _problem
from .gpu_compiled_community_backend import stage_key


FIELDS = ('rhs', 'lower', 'upper', 'c', 'delta', 'col_scale', 'row_scale')


def _host_copy(value, dtype=None):
    """Setup-only device download; no device object is retained afterward."""
    if hasattr(value, '__cuda_array_interface__'):
        value = value.get()
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def cpu_features(inputs):
    """Same field order and finite float32 encoding as GPU basis routing."""
    joined = np.concatenate([np.asarray(inputs[key]).ravel() for key in FIELDS])
    joined = np.nan_to_num(joined, nan=0., posinf=1e13, neginf=-1e13)
    return (np.sign(joined) * np.log1p(np.abs(joined))).astype(np.float32)


class CpuDictionaryLP:
    """Nearest-centroid initial basis followed by exact persistent CPU LPs.

    ``banks_keyed`` uses the same stage/shape keys as the GPU adapter. Missing
    banks or unsupported coordinate normalization simply omit initialization;
    the requested original LP is still solved by HiGHS. A provided basis is
    only a proposal; the persistent solver prioritizes an existing warm basis.
    """
    name = 'cpu_dictionary_persistent_highs_simplex'

    def __init__(self, coordinates, banks_keyed, workers=4, *, selected_features=True, cpu_service=None,
                 exchange_support_updates=False, coverage_routers=None):
        started = time.perf_counter()
        self.coordinates = coordinates
        if coverage_routers is None:
            coverage_routers = {}
        if not isinstance(coverage_routers, dict) or set(coverage_routers)-set(banks_keyed):
            raise ValueError('CPU coverage routers must match compact bank keys')
        self.banks = {}
        for key, bank in banks_keyed.items():
            a = csr_matrix(bank.host_a, dtype=np.float64, copy=True)
            m, n = a.shape
            variables = _host_copy(bank.variable_rows, np.int64)
            centers = _host_copy(bank.centers)
            indices = _host_copy(bank.feature_indices, np.int64)
            scale = _host_copy(bank.feature_scale)
            if variables.ndim != 1 or np.any(variables < 0) or np.any(variables >= m):
                raise ValueError('Invalid dictionary variable rows')
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= 2*m + 4*n + len(variables)*n):
                raise ValueError('Invalid dictionary feature indices')
            if centers.shape != (len(bank.evaluators), len(indices)) or not len(bank.evaluators):
                raise ValueError('Dictionary centroid dimensions must match candidates and features')
            if scale.shape != (len(indices),) or not np.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError('Dictionary feature scales must be positive and finite')
            if not np.isfinite(centers).all():
                raise ValueError('Nonfinite dictionary centroids')
            coverage_router = coverage_routers.get(key)
            if coverage_router is not None:
                if (getattr(coverage_router, 'xp', None) is not np
                        or getattr(coverage_router, 'candidate_count', None) != len(bank.evaluators)
                        or getattr(coverage_router, 'input_dim', None) != len(indices)
                        or not callable(getattr(coverage_router, 'rank', None))):
                    raise ValueError('CPU coverage router must be a matching NumPy router')
            bases = []
            for evaluator in bank.evaluators:
                kind = _host_copy(evaluator.d['kind'], np.int8)
                active = _host_copy(evaluator.d['active'], np.int64)
                if kind.shape != (n,) or not np.isin(kind, [-1, 0, 1, 2]).all():
                    raise ValueError('Invalid dictionary column basis kind')
                if (active.ndim != 1 or np.any(active < 0) or np.any(active >= m)
                        or len(np.unique(active)) != len(active)
                        or np.count_nonzero(kind == 2) != len(active)):
                    raise ValueError('Invalid dictionary active-row basis')
                # HiGHS: lower=0, basic=1, upper=2, zero=3. Positive
                # coordinate scaling preserves these combinatorial statuses.
                columns = np.where(kind == -1, 0, np.where(kind == 1, 2, np.where(kind == 2, 1, 3))).astype(np.int32)
                rows = np.ones(m, dtype=np.int32)
                rows[active] = np.where(active < bank.neq, 0, 2)
                columns.setflags(write=False); rows.setflags(write=False)
                bases.append(dict(col_status=columns, row_status=rows))
            selector=None
            if selected_features:
                from .selected_lp_features import SelectedLPFeatures
                selector=SelectedLPFeatures(indices,dict(rhs=(m,),lower=(n,),upper=(n,),c=(n,),
                    delta=(len(variables),n),col_scale=(n,),row_scale=(m,)))
            self.banks[key] = dict(a=a, variables=variables, centers=centers,
                indices=indices, scale=scale, bases=bases, selector=selector,
                coverage_router=coverage_router)
        self._owns_cpu = cpu_service is None
        options = dict(exchange_support_updates=True) if exchange_support_updates else {}
        self.cpu = RepeatedCpuLP(workers, n_fluxes=coordinates.n_fluxes, **options) if cpu_service is None else cpu_service
        if cpu_service is not None and exchange_support_updates and not getattr(cpu_service, 'exchange_support_updates', False):
            raise ValueError('The borrowed CPU service must enable exchange support updates')
        self.history = []
        self.setup_seconds = time.perf_counter()-started

    def _propose(self, arrays):
        a, rhs, lower, upper, c, neq = arrays
        key = stage_key(c, a, neq, self.coordinates.n_fluxes)
        bank = self.banks.get(key)
        if bank is None:
            return None, dict(stage=key[0], dictionary_candidate=None, dictionary_reason='missing_bank')
        try:
            problem = self.coordinates.normalize(*arrays)
        except ValueError:
            return None, dict(stage=key[0], dictionary_candidate=None, dictionary_reason='unsupported_coordinates')
        if problem.a.shape != bank['a'].shape:
            return None, dict(stage=key[0], dictionary_candidate=None, dictionary_reason='changed_normalized_shape')
        delta = (problem.a-bank['a']).tocsr()[bank['variables']].toarray()
        inputs = {name: getattr(problem, name) for name in FIELDS if name != 'delta'}
        inputs['delta'] = delta
        selector=bank.get('selector')
        features = selector.cpu_features(inputs) if selector is not None else cpu_features(inputs)[bank['indices']]
        router = bank.get('coverage_router')
        if router is not None:
            order = np.asarray(router.rank(features[None], k=len(bank['bases'])))
            if (order.shape != (1, len(bank['bases'])) or order.dtype.kind not in 'iu'
                    or np.any(order < 0) or np.any(order >= len(bank['bases']))
                    or not np.array_equal(np.sort(order[0]), np.arange(len(bank['bases'])))):
                raise ValueError('CPU coverage router must return a full candidate permutation')
            index = int(order[0, 0])
            return bank['bases'][index], dict(stage=key[0], dictionary_candidate=index,
                dictionary_distance=None, dictionary_reason='learned_coverage')
        distance = np.sum(((features[None]-bank['centers'])/bank['scale'])**2, axis=1)
        index = int(np.argmin(distance))
        return bank['bases'][index], dict(stage=key[0], dictionary_candidate=index,
            dictionary_distance=float(distance[index]), dictionary_reason='nearest_centroid')

    def _has_existing_basis(self, arrays, kwargs, environment_id):
        """Detect a usable persistent basis before paying feature-routing cost."""
        a, rhs, lower, upper, c, neq = arrays
        stage = kwargs.get('_stage')
        if stage is None:
            stage = stage_key(c, a, neq, self.coordinates.n_fluxes)[0]
        entry = self.cpu.models.get((environment_id, stage, a.shape[0], a.shape[1], neq))
        basis = None if entry is None else entry.get('basis')
        if entry is not None and basis is not None and basis.valid and getattr(self.cpu, 'exchange_support_updates', False):
            return self.cpu.can_reuse_structure(a, entry, stage, neq)
        return bool(entry is not None and basis is not None and basis.valid
            and np.array_equal(a.indptr, entry['a'].indptr)
            and np.array_equal(a.indices, entry['a'].indices))

    def prepare_request(self, objective, kwargs, environment_id):
        """Prepare one stable environment's proposal on the calling thread.

        Only one request per environment may be pending. The returned request
        contains ordinary host LP arrays and can be sent to a solver worker;
        COBRA/environment objects are never touched by that worker.
        """
        if self.cpu.closed:
            raise RuntimeError('The CPU dictionary backend is closed')
        row_started=time.perf_counter()
        arrays=_problem(objective,kwargs)
        if self._has_existing_basis(arrays,kwargs,environment_id):
            key=stage_key(arrays[4],arrays[0],arrays[-1],self.coordinates.n_fluxes)
            proposal=None
            metadata=dict(stage=kwargs.get('_stage',key[0]),dictionary_candidate=None,
                dictionary_reason='existing_cpu_basis')
        else:
            proposal,metadata=self._propose(arrays)
        updated=dict(kwargs)
        if proposal is not None:
            updated['_initial_basis']=proposal
        metadata['dictionary_selection_seconds']=time.perf_counter()-row_started
        return (objective,updated),metadata

    def solve_batch(self, requests, *, environment_ids=None):
        if self.cpu.closed:
            raise RuntimeError('The CPU dictionary backend is closed')
        if environment_ids is None:
            environment_ids = list(range(len(requests)))
        if len(environment_ids) != len(requests) or len(set(environment_ids)) != len(environment_ids):
            raise ValueError('One unique stable environment ID is required per request')
        started = time.perf_counter()
        prepared, selected = [], []
        for environment_id, (objective, kwargs) in zip(environment_ids, requests):
            updated, metadata = self.prepare_request(objective, kwargs, environment_id)
            prepared.append(updated)
            selected.append(metadata)
        selection_seconds = time.perf_counter()-started
        results = self.cpu.solve_batch(prepared, environment_ids=environment_ids)
        for result, metadata in zip(results, selected):
            result.diagnostics.update(metadata)
        self.history.append(dict(batch=len(requests), cpu_lp_calls=len(requests),
            dictionary_seconds=selection_seconds,
            cpu_seconds=self.cpu.history[-1]['seconds'],
            seconds=time.perf_counter()-started,
            rows=[dict(result.diagnostics) for result in results]))
        return results

    def reset_trajectory(self):
        self.cpu.models.clear()

    def close(self):
        if self._owns_cpu:
            self.cpu.close()
