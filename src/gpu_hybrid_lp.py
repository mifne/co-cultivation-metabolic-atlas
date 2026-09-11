"""GPU-first certified repeated LPs with explicit, measured CPU escape routes.

Candidate dispersion is an experimental routing feature, not a validated
predictor of correction cost; it never certifies a solution.
All returned rows pass the original LP primal/dual/gap checks. CPU results
are not inserted into the immutable offline dictionary or validation set.
This is a hybrid host dFBA backend, not a fully GPU-resident simulator.
"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time

import numpy as np
from scipy.sparse import csr_matrix

from .gpu_compiled_community_backend import lp_arrays, stage_key
from .gpu_batched_compiled_backend import BatchedCompiledBackend, subset_device_result
from .cpu_repeated_lp import RepeatedCpuLP


def rank_compact_candidates(cp, bank, inputs, coverage_router=None):
    """Return a complete host candidate order; never accept an LP.

    A learned coverage router is an opt-in replacement for nearest-centroid
    ordering only.  Its output must be a full permutation so temporal
    promotion and the existing candidate budget retain their exact semantics.
    """
    from .gpu_neural_basis_proposal import features
    encoded = (bank.proposal_features(inputs) if hasattr(bank, 'proposal_features')
               else features(inputs)[:, bank.feature_indices])
    if coverage_router is None:
        distance = cp.sum(((encoded[:, None]-bank.centers[None])
                           / bank.feature_scale)**2, axis=2)
        # Preserve the pre-router hot path exactly: one device argsort and one
        # host transfer, without opt-in validation or dtype conversion cost.
        ranked = cp.argsort(distance, axis=1)
        return (ranked.get() if hasattr(ranked, 'get') else np.asarray(ranked)), 'nearest_centroid'
    entries = len(bank.evaluators)
    ranked = coverage_router.rank(encoded, k=entries)
    order = np.asarray(ranked.get() if hasattr(ranked, 'get') else ranked)
    if (order.shape != (len(encoded), entries) or order.dtype.kind not in 'iu'
            or np.any(order < 0) or np.any(order >= entries)
            or not np.all(np.sort(order, axis=1) == np.arange(entries)[None])):
        raise ValueError('Candidate router must return one full integer permutation per LP')
    return order, 'learned_coverage'


def observable_projection(metadata, columns):
    """Species growth, species-resolved exchange and PHA reaction rates.

    These are rates per biomass, not pool derivatives: no fictitious biomass
    or physical uncertainty scale is inferred. Unit scales are explicit and
    thresholds remain experimental routing parameters, not accuracy bounds.
    Internal cycle fluxes and exchange absolute-value auxiliary variables
    are deliberately excluded.
    """
    rows, cols, data, labels = [], [], [], []
    for species, (index, coefficient) in metadata['growth_terms'].items():
        rows.append(len(labels)); cols.append(index); data.append(coefficient)
        labels.append('growth:' + species)
    for metabolite, terms in sorted(metadata['exchange_terms'].items()):
        grouped = {}
        for species, index, coefficient, _ in terms:
            grouped.setdefault(species, []).append((index, coefficient))
        for species, terms in sorted(grouped.items()):
            for index, coefficient in terms:
                rows.append(len(labels)); cols.append(index); data.append(coefficient)
            labels.append('exchange:' + species + ':' + metabolite)
    for index, reaction in enumerate(metadata['reaction_ids']):
        if any(token in reaction.upper() for token in ('PHA', 'PHB', 'PHV')):
            rows.append(len(labels)); cols.append(index); data.append(1.)
            labels.append('product:' + metadata['reaction_species'][index] + ':' + reaction)
    return csr_matrix((data, (rows, cols)), shape=(len(labels), columns)), np.ones(len(labels)), labels


def route_rejected(accepted, dispersion, count, threshold):
    """Acceptances take precedence, including degenerate high-variance optima."""
    accepted = np.asarray(accepted, dtype=bool)
    dispersion, count = np.asarray(dispersion), np.asarray(count)
    if dispersion.shape != accepted.shape or count.shape != accepted.shape:
        raise ValueError('Routing arrays must have the same shape')
    if np.isnan(threshold) or threshold < 0:
        raise ValueError('Nonnegative routing threshold required')
    unknown = (count < 2) | ~np.isfinite(dispersion)
    high = (~unknown) & (dispersion > threshold)
    direct_cpu = (~accepted) & (unknown | high)
    return direct_cpu, (~accepted) & ~direct_cpu, unknown


def near_dual_maxmin_rows(stage, accepted, candidate, max_violations=2):
    """Bounded-repair eligibility, never a certificate or pivot-count promise.

    Select a SINGLE candidate that is family-valid and dual/gap compliant but
    violates only a few primal constraints. Dispersion is not used here.
    Unavailable/malformed diagnostics fail closed to the ordinary CPU route.
    """
    accepted=np.asarray(accepted,dtype=bool)
    result=np.zeros_like(accepted)
    if accepted.ndim!=1 or not isinstance(max_violations,int) or max_violations<1:
        raise ValueError('Invalid near-dual repair eligibility inputs')
    if stage!='maxmin':return result
    fields={}
    for name in ('input_family_valid','primal_residual','dual_violation',
                 'relative_kkt_gap','basis_dual_violation','primal_violation_count'):
        value=candidate.get(name)
        if value is None:return result
        value=np.asarray(value)
        if value.shape!=accepted.shape or value.dtype.kind not in 'biuf':return result
        fields[name]=value
    if fields['input_family_valid'].dtype.kind!='b':return result
    finite=np.logical_and.reduce([np.isfinite(value) for value in fields.values()])
    count=fields['primal_violation_count']
    return (~accepted & finite & fields['input_family_valid']
        & (fields['primal_residual']>1e-5) & (fields['dual_violation']<=1e-7)
        & (fields['relative_kkt_gap']<=1e-7) & (fields['basis_dual_violation']<=1e-8)
        & (count>=1) & (count<=max_violations)
        & (count==np.floor(np.where(np.isfinite(count),count,0.))))


def candidate_diagnostic_snapshot(slot_metrics, result, order, entries):
    """Small immutable host snapshot of screening, never an acceptance rule.

    Only B×K scalars and B selection vectors are downloaded. Raw flux/dual
    vectors are not present in heterogeneous slot metadata, so their finite
    flags are explicitly unavailable, not inferred from scalar residuals.
    """
    order = np.asarray(order)
    shape = order.shape
    if order.ndim != 2:
        raise ValueError('Candidate diagnostics require a B×K requested order')
    source = slot_metrics if isinstance(slot_metrics, dict) else {}
    unavailable = {}

    def field(mapping, name, expected, *, kind=None, label=None):
        label = label or name
        value = mapping.get(name)
        if value is None:
            unavailable[label] = 'not_recorded'
            return None
        array = np.asarray(value.get() if hasattr(value, 'get') else value)
        if array.shape != expected or array.dtype.kind not in (kind or 'biuf'):
            unavailable[label] = 'unexpected_shape_or_dtype'
            return None
        return array.copy()

    def serial(array):
        if array is None:
            return None
        if array.dtype.kind == 'f':
            return np.where(np.isfinite(array), array.astype(object), None).tolist()
        return array.tolist()

    fields = {name:field(source, name, shape) for name in (
        'primal_residual', 'dual_violation', 'relative_kkt_gap', 'basis_dual_violation',
        'primal_violation_count', 'primal_violation_l1', 'objective')}
    flags = {name:field(source, name, shape, kind='b') for name in ('input_family_valid', 'accepted')}
    indices = field(source, 'indices', shape, kind='iu')
    selections = {name:field(result, name, (shape[0],), label='selection.'+name) for name in (
        'candidate_index', 'best_candidate_index', 'best_candidate_dual_feasible', 'accepted',
        'candidate_count', 'observable_dispersion')}
    thresholds = dict(primal_residual=1e-5, dual_violation=1e-7, relative_kkt_gap=1e-7)
    passes = {name:(None if fields[name] is None else
        (np.isfinite(fields[name]) & (fields[name] <= tolerance)).tolist())
        for name, tolerance in thresholds.items()}
    certificate_fields = [fields[name] for name in thresholds]
    certificate_finite = None if any(value is None for value in certificate_fields) else (
        np.logical_and.reduce([np.isfinite(value) for value in certificate_fields]).tolist())
    basis_dual = fields['basis_dual_violation']
    return dict(schema_version=1, available=bool(source), batch=shape[0], proposals=shape[1],
        dictionary_entries=int(entries), requested_order=order.tolist(), indices=serial(indices),
        index_in_range=None if indices is None else ((indices >= 0) & (indices < entries)).tolist(),
        **{name:serial(value) for name, value in {**fields, **flags}.items()},
        selection={name:serial(value) for name, value in selections.items()},
        metric_finite={name:None if value is None else np.isfinite(value).tolist() for name, value in fields.items()},
        certificate_metrics_finite=certificate_finite,
        raw_values_finite=None, raw_duals_finite=None, observables_finite=None,
        scalar_gate_pass=passes, scalar_gate_thresholds=thresholds,
        basis_dual_feasible_for_ranking=None if basis_dual is None else (
            np.isfinite(basis_dual) & (basis_dual <= 1e-8)).tolist(),
        unavailable_fields=unavailable,
        scope='Pre-repair/pre-CPU GPU screening only. Scalar passes are necessary but not sufficient; '
              'accepted is the original full certificate. Basis-dual threshold is ranking only; '
              'null means unavailable/nonfinite, never success.')


def candidate_oracle_snapshot(engine, inputs, order, entries, normal_accepted):
    """Diagnostic counterfactual only; restore all normal routing metadata."""
    started = time.perf_counter()
    missing = object()
    names = ('last_candidates', 'last_candidate_scores', 'last_candidate_dual')
    saved = {name:getattr(engine, name, missing) for name in names}
    order = np.asarray(order)
    full_order = np.broadcast_to(np.arange(entries, dtype=np.int32), (len(order), entries)).copy()
    record = dict(status='unavailable', snapshot=None,
        normal_k_accepted=np.asarray(normal_accepted, dtype=bool).tolist(),
        additional_accepted_environment_ids=None, outside_k_rescued_environment_ids=None,
        additional_accepted_count=None, outside_k_rescued_count=None,
        scope='Diagnostic-only all-dictionary counterfactual on the same normalized LP inputs. '
              'No oracle flux, proposal or acceptance is used in the normal trajectory; not a speed benchmark.')
    try:
        # Eager evaluation intentionally avoids adding whole-dictionary
        # graphs to the normal K-candidate replay cache.
        oracle = engine.evaluate(inputs, full_order)
        snapshot = candidate_diagnostic_snapshot(getattr(engine, 'last_candidates', None),
            oracle, full_order, entries)
        record.update(status='completed', snapshot=snapshot)
        selected = snapshot['selection']['accepted']
        if selected is not None and np.asarray(selected).dtype.kind == 'b':
            rescued = ~np.asarray(normal_accepted, dtype=bool) & np.asarray(selected)
            record['additional_accepted_environment_ids'] = np.flatnonzero(rescued).tolist()
            record['additional_accepted_count'] = int(rescued.sum())
        if snapshot['indices'] is not None and snapshot['accepted'] is not None:
            indices, accepted = np.asarray(snapshot['indices']), np.asarray(snapshot['accepted'])
            outside = ~np.any(indices[:, :, None] == order[:, None, :], axis=2)
            rescued = (~np.asarray(normal_accepted, dtype=bool)
                & (outside & accepted & (indices >= 0) & (indices < entries)).any(axis=1))
            record['outside_k_rescued_environment_ids'] = np.flatnonzero(rescued).tolist()
            record['outside_k_rescued_count'] = int(rescued.sum())
    except Exception as error:
        # Failed measurement is explicitly unknown; it does not manufacture
        # an acceptance or displace the already-screened normal K results.
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        for name, value in saved.items():
            if value is missing:
                if hasattr(engine, name): delattr(engine, name)
            else:
                setattr(engine, name, value)
        record['seconds'] = time.perf_counter()-started
    return record


def cpu_fallback_requests(requests, templates, current_choices, previous_choices=None):
    """Attach initialization only; no proposal ever replaces an exact CPU solve.

    A previous choice, when supplied, is a certified dictionary candidate from
    the preceding full-cohort call in the same stable environment order. The
    CPU backend applies that override only to an existing usable native basis;
    cold/rebuilt models retain the current-query initializer.
    """
    if len(current_choices) != len(requests):
        raise ValueError('One current candidate per CPU request is required')
    if previous_choices is not None and len(previous_choices) != len(requests):
        raise ValueError('One previous candidate per CPU request is required')
    output = []
    for index, (objective, kwargs) in enumerate(requests):
        current = int(current_choices[index])
        previous = -1 if previous_choices is None else int(previous_choices[index])
        updated = dict(kwargs)
        if 0 <= current < len(templates):
            updated['_initial_basis'] = templates[current]
        if 0 <= previous < len(templates):
            updated['_basis_override'] = dict(
                basis=templates[previous], source='previous_certified_gpu', age=1)
        output.append((objective, updated))
    return output


class HybridCertifiedBackend:
    """Bank -> original certificate -> bounded GPU repair / persistent CPU.

    Direct CPU rows run concurrently with the remaining GPU repair groups.
    Uncertified rows after the GPU budget use an explicit safety route even
    when candidate dispersion is small. Variance alone cannot ensure accuracy.
    """
    name = 'hybrid_certified_basis'

    def __init__(self, coordinates, banks, *, cpu_workers=4,
                 dispersion_threshold=1., candidate_limit=0, repair_rounds=2,
                 repair_columns=64, repair_pivots=256, max_repair_batch=4,
                 repair_cache_size=2, bucket=True, cohort_candidates=False, cohort_replay=False,
                 cpu_basis_proposals=False, heterogeneous_candidates=False, heterogeneous_replay=False,
                 candidate_diagnostics=False, candidate_oracle=False,
                 repair_policy='dispersion', compact_repair_workspace=False,
                 gpu_stages=('maxmin', 'aggregate', 'exchange'), selected_features=True,
                 cpu_basis_handoff=False, cpu_exchange_support_updates=False,
                 speculative_cpu=False, coverage_routers=None, cpu_coverage_routers=None,
                 temporal_candidate_policy='prepend', certificate_only=False,
                 packed_result_transfer=False, device_routing=False,
                 speculative_dispatch_phase='before-prepare',
                 gpu_first_batch=False, gpu_cpu_fallback='exact'):
        if not isinstance(gpu_first_batch, bool):
            raise ValueError('gpu_first_batch must be boolean')
        if gpu_cpu_fallback not in ('exact', 'reject') or (gpu_cpu_fallback!='exact' and not gpu_first_batch):
            raise ValueError('CPU reject policy requires GPU-first batch mode')
        if gpu_first_batch and (speculative_cpu or not heterogeneous_candidates or repair_rounds
                or candidate_diagnostics or candidate_oracle or cpu_basis_handoff):
            raise ValueError('GPU-first batch requires heterogeneous zero-repair without speculation/handoff/diagnostics')
        if gpu_cpu_fallback == 'reject' and set(gpu_stages) != {'maxmin', 'aggregate', 'exchange'}:
            raise ValueError('CPU reject policy requires all original stages on GPU')
        self.gpu_first_batch, self.gpu_cpu_fallback = gpu_first_batch, gpu_cpu_fallback
        self._gpu_first_previous = {}
        if gpu_first_batch:
            certificate_only = packed_result_transfer = device_routing = True
        if speculative_dispatch_phase not in ('before-prepare','after-submit'):
            raise ValueError('Unknown speculative CPU dispatch phase')
        if speculative_dispatch_phase!='before-prepare' and not speculative_cpu:
            raise ValueError('Delayed CPU dispatch requires speculative mode')
        self.speculative_dispatch_phase=speculative_dispatch_phase
        from .temporal_candidate_order import POLICIES
        if temporal_candidate_policy not in POLICIES:
            raise ValueError('Unknown temporal candidate policy')
        self.temporal_candidate_policy = temporal_candidate_policy
        if not isinstance(certificate_only, bool) or (certificate_only and
                (not (speculative_cpu or gpu_first_batch) or not heterogeneous_candidates)):
            raise ValueError('Certificate-only probes require speculative heterogeneous mode or GPU-first batch')
        self.certificate_only = certificate_only
        for flag in (packed_result_transfer,device_routing):
            if not isinstance(flag,bool) or (flag and not (speculative_cpu or gpu_first_batch)):
                raise ValueError('Device routing/packed transfer require speculative mode or GPU-first batch')
        if device_routing and not heterogeneous_candidates:
            raise ValueError('Device routing requires heterogeneous candidates')
        self.packed_result_transfer=packed_result_transfer
        self.device_routing=device_routing
        if candidate_limit < 0 or repair_rounds < 0 or repair_columns < 1 or repair_pivots < 1:
            raise ValueError('Invalid GPU candidate/repair budget')
        if max_repair_batch < 1:
            raise ValueError('A positive repair batch size is required')
        if cpu_basis_handoff and (not cpu_basis_proposals or repair_rounds != 0
                                  or set(gpu_stages) != {'maxmin'} or candidate_oracle):
            raise ValueError('CPU basis handoff requires maxmin-only, basis proposals, zero repair rounds, and no oracle')
        self.cpu_basis_handoff = bool(cpu_basis_handoff)
        if not isinstance(speculative_cpu,bool):
            raise ValueError('speculative_cpu must be boolean')
        if speculative_cpu and (set(gpu_stages) != {'maxmin'} or repair_rounds != 0
                or cpu_basis_handoff or candidate_diagnostics or candidate_oracle):
            raise ValueError('Speculative CPU requires maxmin-only, zero repair, no handoff or diagnostic oracle')
        self.speculative_cpu = speculative_cpu
        self._speculative_previous = {}
        self._handoff_batch_size = None
        if repair_policy not in ('dispersion','maxmin-dual'):
            raise ValueError('Unknown GPU repair routing policy')
        if repair_policy=='maxmin-dual' and repair_rounds<1:
            raise ValueError('Near-dual repair requires a positive correction budget')
        self.repair_policy=repair_policy
        self.gpu_stages=frozenset(gpu_stages)
        if not self.gpu_stages or not self.gpu_stages <= {'maxmin','aggregate','exchange'}:
            raise ValueError('GPU stages must be a nonempty subset of the original three stages')
        self.compact_repair_workspace=bool(compact_repair_workspace)
        if candidate_oracle and not (candidate_diagnostics and heterogeneous_candidates):
            raise ValueError('Candidate oracle requires heterogeneous candidate diagnostics')
        route_rejected(np.array([], bool), np.array([]), np.array([]), dispersion_threshold)
        self.coordinates, self.banks = coordinates, banks
        if coverage_routers is None:
            coverage_routers = {}
        if not isinstance(coverage_routers, dict):
            raise ValueError('Coverage routers must be keyed by compact stage key')
        unknown = set(coverage_routers)-set(banks)
        if unknown:
            raise ValueError('Coverage router has no matching compact bank')
        self.coverage_routers = dict(coverage_routers)
        if device_routing and ((not self.coverage_routers and not gpu_first_batch) or any(
                not callable(getattr(router,'rank_with_validity',None))
                for router in self.coverage_routers.values())):
            raise ValueError('Device routing requires a coverage router with device validity')
        for key, router in self.coverage_routers.items():
            bank = banks[key]
            if key[0] not in self.gpu_stages:
                raise ValueError('Coverage routing requires its matching original stage to run on GPU')
            if (getattr(router, 'candidate_count', None) != len(bank.evaluators)
                    or getattr(router, 'input_dim', None) != int(bank.feature_indices.size)
                    or not callable(getattr(router, 'rank', None))):
                raise ValueError('Coverage router dimensions disagree with compact bank')
        if cpu_coverage_routers is None:
            cpu_coverage_routers = {}
        if not isinstance(cpu_coverage_routers, dict) or set(cpu_coverage_routers)-set(banks):
            raise ValueError('CPU coverage routers must match compact bank keys')
        if (speculative_cpu or gpu_first_batch) and set(self.coverage_routers) != set(cpu_coverage_routers):
            raise ValueError('Speculative/GPU-first and CPU paths require the same coverage routers')
        self.cpu_coverage_routers = dict(cpu_coverage_routers)
        self.dispersion_threshold, self.candidate_limit = dispersion_threshold, candidate_limit
        self.repair_rounds, self.repair_columns, self.repair_pivots = repair_rounds, repair_columns, repair_pivots
        self.max_repair_batch, self.bucket = max_repair_batch, bucket
        self.cohort_candidates = bool(cohort_candidates)
        self.candidate_diagnostics = bool(candidate_diagnostics)
        self.candidate_oracle = bool(candidate_oracle)
        self.cohort_replay = bool(cohort_replay)
        if self.cohort_replay and not self.cohort_candidates:
            raise ValueError('Whole-cohort replay requires cohort candidate evaluation')
        self.heterogeneous_replay = bool(heterogeneous_replay)
        if heterogeneous_candidates and (self.cohort_candidates or self.cohort_replay):
            raise ValueError('Choose either heterogeneous or whole-cohort candidates')
        if self.heterogeneous_replay and not heterogeneous_candidates:
            raise ValueError('Heterogeneous replay requires heterogeneous candidates')
        self.heterogeneous_banks = {}
        if heterogeneous_candidates:
            from .gpu_heterogeneous_compact import HeterogeneousCompactBank
            self.heterogeneous_banks = {key:HeterogeneousCompactBank(bank, **(
                dict(certificate_only=True) if certificate_only else {})) for key,bank in banks.items()
                if key[0] in self.gpu_stages}
        cpu_options = dict(exchange_support_updates=True) if cpu_exchange_support_updates else {}
        self.cpu = RepeatedCpuLP(cpu_workers, n_fluxes=coordinates.n_fluxes, **cpu_options)
        self.direct_cpu = None
        if speculative_cpu or gpu_first_batch or any(key[0] not in self.gpu_stages for key in banks):
            from .cpu_dictionary_lp import CpuDictionaryLP
            direct_banks = {key:bank for key,bank in banks.items()
                if speculative_cpu or gpu_first_batch or key[0] not in self.gpu_stages}
            direct_routers = {key:router for key,router in self.cpu_coverage_routers.items()
                if key in direct_banks}
            self.direct_cpu = CpuDictionaryLP(coordinates, direct_banks,
                selected_features=selected_features, cpu_service=self.cpu,
                coverage_routers=direct_routers)
        self.dispatch = ThreadPoolExecutor(max_workers=1)
        self.operators = BatchedCompiledBackend(coordinates, banks, pivots=repair_pivots,
            repair_cache_size=repair_cache_size, compact_updates=True,
            require_tie=False, reuse_small_factor=True, skip_unused_dual=True,
            adaptive_refinement_tolerance=1e-12, repair_graph_cache_size=1)
        self.history, self.previous_candidates = [], {}
        self.cpu_basis_proposals = bool(cpu_basis_proposals)
        self.host_bases = {}
        if self.cpu_basis_proposals:
            # Immutable offline basis metadata, NOT a newly solved CPU LP.
            # A rejected GPU candidate is only a simplex initialization.
            for key, bank in banks.items():
                templates = []
                for evaluator in bank.evaluators:
                    kind, active = evaluator.d['kind'].get(), evaluator.d['active'].get()
                    col_status = np.where(kind == 2, 1, np.where(kind == -1, 0, np.where(kind == 1, 2, 3))).astype(np.int32)
                    row_status = np.ones(bank.host_a.shape[0], dtype=np.int32)
                    row_status[active] = np.where(active < bank.neq, 0, 2)
                    templates.append(dict(col_status=col_status, row_status=row_status))
                self.host_bases[key] = templates

    def reset_trajectory(self):
        self.previous_candidates.clear()
        if hasattr(self,'_speculative_previous'):self._speculative_previous.clear()
        if hasattr(self,'_gpu_first_previous'):self._gpu_first_previous.clear()
        self._handoff_batch_size = None
        # Fair repetitions may reuse immutable GPU graphs, but not CPU/GPU
        # trajectory solutions from the previous benchmark repetition.
        self.cpu.models.clear()

    def _solve_cpu_stage(self, requests, stage, started, environment_ids=None):
        """Exact CPU-only stage; skip normalization/upload/GPU screening costs.

        The same offline nearest-centroid initializer and persistent HiGHS
        comparator are used. This is an explicit experiment policy, not an
        accuracy gate or a claim that the requested LP ran on the GPU.
        """
        results=self.direct_cpu.solve_batch(requests,environment_ids=environment_ids)
        elapsed=time.perf_counter()-started
        batch=len(results)
        for row in results:
            row.diagnostics.update(route='cpu_stage_policy')
            row.message='cpu_stage_policy; '+row.message
        self.history.append(dict(stage=stage,batch=batch,bank_accepts=0,
            environment_ids=list(range(batch)) if environment_ids is None else list(environment_ids),
            candidate_engine='disabled_by_stage_policy',candidate_evaluations=0,
            candidate_count=[0]*batch,observable_dispersion=[None]*batch,
            preparation_seconds=0.,bank_seconds=0.,seconds=elapsed,
            cpu_lp_calls=batch,routes=['cpu_stage_policy']*batch,
            repair_policy=self.repair_policy,repair_eligible=[False]*batch,
            cpu_initial_basis_proposals=True,
            groups=[dict(ids=list(range(batch)),
                environment_ids=list(range(batch)) if environment_ids is None else list(environment_ids),route='cpu_fallback',
                rows=[row.diagnostics for row in results])],
            accepted=[row.success for row in results],
            **{field:[row.diagnostics[field] for row in results] for field in
               ('primal_residual','dual_violation','relative_kkt_gap')}))
        return results

    def solve_batch(self, requests, *, environment_ids=None):
        import cupy as cp
        from .gpu_restricted_basis import GpuRestrictedBasis
        started = time.perf_counter()
        ids = list(range(len(requests))) if environment_ids is None else list(environment_ids)
        try:
            unique = len(set(ids))
        except TypeError as error:
            raise ValueError('Stable environment IDs must be hashable and unique') from error
        if not requests or len(ids)!=len(requests) or unique!=len(ids):
            raise ValueError('Nonempty requests and unique stable environment IDs required')
        if not (getattr(self,'speculative_cpu',False) or getattr(self,'gpu_first_batch',False)) and ids!=list(range(len(requests))):
            raise ValueError('Explicit reordered environment IDs require speculative_cpu; legacy uses a fixed full cohort')
        arrays = [lp_arrays(c, **kw) for c, kw in requests]
        keys = [stage_key(a[4], a[0], a[-1], self.coordinates.n_fluxes) for a in arrays]
        if len(set(keys)) != 1 or keys[0][0] == 'exchange_tie':
            raise ValueError('Hybrid backend requires one original three-stage LP batch')
        key, batch = keys[0], len(requests)
        if key[0] not in getattr(self,'gpu_stages',{'maxmin','aggregate','exchange'}):
            return self._solve_cpu_stage(requests,key[0],started,environment_ids=ids)
        if getattr(self,'gpu_first_batch',False):
            from .gpu_first_batch_lp import solve_gpu_first_batch
            return solve_gpu_first_batch(self,requests,key,arrays,ids,started)
        if getattr(self,'speculative_cpu',False):
            from .gpu_speculative_lp import solve_speculative_maxmin
            return solve_speculative_maxmin(self,requests,key,arrays,ids,started)
        if getattr(self, 'cpu_basis_handoff', False):
            # This opt-in API is restricted to a fixed, stably ordered cohort.
            # Subset/reordered callers require explicit environment IDs first.
            if self._handoff_batch_size is not None and self._handoff_batch_size != batch:
                raise ValueError('CPU basis handoff requires a fixed full-cohort batch')
            self._handoff_batch_size = batch
        bank = self.banks[key]
        problems = [self.coordinates.normalize(*a) for a in arrays]
        self.last_problems = problems
        inputs = bank.prepare_host(problems)
        prepared = time.perf_counter()
        order, candidate_router = rank_compact_candidates(
            cp, bank, inputs, getattr(self,'coverage_routers',{}).get(key))
        previous = self.previous_candidates.get(key)
        from .temporal_candidate_order import temporal_candidate_order
        prior = previous if previous is not None and len(previous) == batch else np.full(batch,-1)
        order = temporal_candidate_order(order, prior, self.candidate_limit,
            getattr(self,'temporal_candidate_policy','prepend'))
        engine = self.heterogeneous_banks.get(key)
        if engine is not None:
            evaluate = engine.evaluate_replay if self.heterogeneous_replay else engine.evaluate
            result = evaluate(inputs, cp.asarray(order))
        elif self.cohort_candidates:
            # A candidate already evaluated for the cohort may solve rows
            # outside its nearest-neighbour shortlist at no extra LP cost.
            popularity = np.bincount(order.ravel(), minlength=len(bank.evaluators))
            candidates = np.flatnonzero(popularity)
            candidates = candidates[np.argsort(-popularity[candidates], kind='stable')]
            evaluate = bank.evaluate_cohort_replay if self.cohort_replay else bank.evaluate_cohort
            result = evaluate(inputs, candidates)
        else:
            result = bank.evaluate_device(inputs, order=order)
        accepted = result['accepted'].get()
        selected = result['candidate_index'].get()
        count = result.get('candidate_count', cp.zeros(batch, dtype=cp.int32)).get()
        dispersion = result.get('observable_dispersion', cp.full(batch, cp.inf)).get()
        direct, repair, unknown = route_rejected(accepted, dispersion, count, self.dispersion_threshold)
        repair_eligibility=None
        if getattr(self,'repair_policy','dispersion')=='maxmin-dual':
            repair_eligibility={name:(result[name].get() if name in result else None) for name in
                ('input_family_valid','primal_residual','dual_violation','relative_kkt_gap',
                 'basis_dual_violation','primal_violation_count')}
            repair=near_dual_maxmin_rows(key[0],accepted,repair_eligibility)
            direct=~accepted & ~repair
        record = dict(stage=key[0], batch=batch, bank_accepts=int(accepted.sum()),
            candidate_router=candidate_router,
            candidate_engine='heterogeneous' if engine is not None else ('cohort' if self.cohort_candidates else 'legacy'),
            candidate_evaluations=int(result['candidate_evaluations']),
            candidate_count=count.tolist(), observable_dispersion=[float(v) if np.isfinite(v) else None for v in dispersion],
            preparation_seconds=prepared-started, bank_seconds=time.perf_counter()-prepared,
            groups=[], cpu_lp_calls=0, routes=['gpu_dictionary' if ok else None for ok in accepted])
        record['repair_policy']=getattr(self,'repair_policy','dispersion')
        record['repair_eligible']=repair.tolist()
        if getattr(self, 'candidate_diagnostics', False):
            before = time.perf_counter()
            record['candidate_diagnostics'] = candidate_diagnostic_snapshot(
                None if engine is None else getattr(engine, 'last_candidates', None),
                result, order, len(bank.evaluators))
            record['candidate_diagnostics']['stage_call'] = len(self.history)+1
            # Explicit opt-in diagnostic overhead remains inside end-to-end
            # timing. The default path does not call/download these fields.
            record['candidate_diagnostic_seconds'] = time.perf_counter()-before
        if getattr(self, 'candidate_oracle', False):
            record['candidate_oracle'] = candidate_oracle_snapshot(
                engine, inputs, order, len(bank.evaluators), accepted)
            record['candidate_oracle']['stage_call'] = len(self.history)+1
        values, objective = result['values'].get(), result['objective'].get()
        metrics = {k:result[k].get() for k in ('primal_residual', 'dual_violation', 'relative_kkt_gap')}
        cpu_choices = result['best_candidate_index'].get()
        def cpu_requests(ids):
            if not self.cpu_basis_proposals:
                return [requests[i] for i in ids]
            # No eligible candidate is -1, not Python's last dictionary entry.
            prior = ([previous[i] for i in ids] if getattr(self, 'cpu_basis_handoff', False)
                     and previous is not None and len(previous) == batch else None)
            return cpu_fallback_requests([requests[i] for i in ids], self.host_bases[key],
                                         [cpu_choices[i] for i in ids], prior)
        record['cpu_initial_basis_proposals'] = self.cpu_basis_proposals
        record['cpu_basis_handoff_policy'] = getattr(self, 'cpu_basis_handoff', False)
        # With no GPU correction budget there is no work to overlap: send
        # every rejected row in one CPU wave instead of serial high/low-
        # dispersion batches. Keep a dispatch mask independent of success,
        # so an infeasible CPU row is never retried merely because it failed.
        early_cpu = direct.copy() if self.repair_rounds else ~accepted.copy()
        early_cpu_ids = np.flatnonzero(early_cpu).tolist()
        future = self.dispatch.submit(self.cpu.solve_batch, cpu_requests(early_cpu_ids),
            environment_ids=early_cpu_ids) if early_cpu_ids else None

        def merge_cpu(ids, results, route):
            for i, out in zip(ids, results):
                accepted[i] = out.success
                if out.success:
                    values[i], objective[i] = out.x, out.fun
                for field in metrics: metrics[field][i] = out.diagnostics[field]
                record['routes'][i] = route(i)
            record['cpu_lp_calls'] += len(ids)
            record['groups'].append(dict(ids=ids, route='cpu_fallback', rows=[out.diagnostics for out in results]))

        try:
            choices = result['best_candidate_index'].get()
            if bank.candidate_ranking == 'count' and getattr(self,'repair_policy','dispersion')!='maxmin-dual':
                scores = engine.last_candidate_scores if engine is not None else bank.last_candidate_scores
                choices = cp.where(cp.isfinite(scores).any(axis=1), cp.argmin(scores, axis=1), -1).get()
            for index in np.unique(choices[repair & (choices >= 0)]) if self.repair_rounds else []:
                pending = np.flatnonzero(repair & (choices == index)).tolist()
                for start in range(0, len(pending), self.max_repair_batch):
                    ids = pending[start:start+self.max_repair_batch]
                    before = time.perf_counter(); solver = self.operators.solver(key, int(index))
                    if getattr(solver, 'restricted_solver', None) is None:
                        options=dict(rank_bucket=self.bucket)
                        if getattr(self,'compact_repair_workspace',False):options['pivot_aware_capacity']=True
                        solver.restricted_solver = GpuRestrictedBasis(solver, self.repair_pivots, **options)
                    reducer = solver.restricted_solver
                    query = solver.prepare_host([problems[i] for i in ids]); warm = None
                    setup_seconds = time.perf_counter()-before
                    for round_index in range(self.repair_rounds):
                        compiled_before=reducer.graph_compilation_seconds if hasattr(reducer,'graph_compilation_seconds') else 0.
                        before = time.perf_counter()
                        options=dict(columns=min(self.repair_columns, len(reducer.nonbasic0)),
                            warm_start=warm, reuse_lifted_state=self.bucket and warm is not None)
                        if getattr(self,'compact_repair_workspace',False):options['compute_expansion']=False
                        trial = reducer.run_device(**query, **options)
                        good = trial['accepted'].get(); accepted[ids] = good
                        passed = np.flatnonzero(good)
                        if len(passed):
                            out_ids = np.asarray(ids)[passed]
                            values[out_ids] = trial['values'][cp.asarray(passed)].get()
                            objective[out_ids] = trial['objective'][cp.asarray(passed)].get()
                            for field in metrics: metrics[field][out_ids] = trial[field][cp.asarray(passed)].get()
                            for i in out_ids: record['routes'][i] = 'gpu_restricted_repair'
                        record['groups'].append(dict(ids=ids.copy(), route='gpu_restricted_repair',
                            candidate_index=int(index), round=round_index, accepted=good.tolist(),
                            pivots=trial['restricted_pivots'].get().tolist(),
                            operator_setup_seconds=setup_seconds if round_index==0 else 0.,
                            graph_compilation_seconds=(getattr(reducer,'graph_compilation_seconds',0.)-compiled_before),
                            retained_update_width=int(trial['warm_state']['u'].shape[2]) if 'u' in trial['warm_state'] else None,
                            seconds=time.perf_counter()-before+(setup_seconds if round_index == 0 else 0.)))
                        remaining = np.flatnonzero(~good)
                        if not len(remaining): break
                        warm = subset_device_result(trial, remaining.tolist())['warm_state']
                        ids = [ids[j] for j in remaining]
                        query = {k:v[cp.asarray(remaining)] for k,v in query.items()}
        finally:
            # Join even on GPU exceptions; no solver thread survives this LP.
            if future is not None:
                merge_cpu(early_cpu_ids, future.result(), lambda i:'cpu_not_near_dual_maxmin'
                    if getattr(self,'repair_policy','dispersion')=='maxmin-dual' else
                    ('cpu_unknown_dispersion' if unknown[i] else
                     ('cpu_high_dispersion' if direct[i] else 'cpu_gpu_budget_exhausted')))
        remaining = np.flatnonzero(~accepted & ~early_cpu).tolist()
        if remaining:
            merge_cpu(remaining, self.cpu.solve_batch(cpu_requests(remaining), environment_ids=remaining),
                lambda i:'cpu_gpu_budget_exhausted')
        self.previous_candidates[key] = selected.copy()
        cp.cuda.get_current_stream().synchronize()
        record.update(seconds=time.perf_counter()-started, accepted=accepted.tolist(),
            **{k:v.tolist() for k,v in metrics.items()})
        record['device_used_bytes_at_stage_end'] = cp.cuda.runtime.memGetInfo()[1]-cp.cuda.runtime.memGetInfo()[0]
        self.history.append(record)
        return [SimpleNamespace(success=bool(accepted[i]), x=values[i] if accepted[i] else None,
            fun=float(objective[i]) if accepted[i] else None, message=record['routes'][i],
            diagnostics=dict(success=bool(accepted[i]), max_original_residual=float(metrics['primal_residual'][i]),
                total_seconds=record['seconds'], objective=float(objective[i]),
                cpu_lp_calls=int(record['routes'][i].startswith('cpu_')),
                route=record['routes'][i])) for i in range(batch)]

    def close(self):
        self.dispatch.shutdown(wait=True)
        self.cpu.close()
        for solver in self.operators.repairs.values(): solver.clear_graph_cache()
        for bank in self.heterogeneous_banks.values(): bank.close()
