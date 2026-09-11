"""Microbatched full-LP GPU repair for independent host dFBA environments.

Host grouping/LP assembly/state updates remain explicit. All optimization,
constraint checks, and repaired flux calculations use the GPU. No CPU LP
fallback; a rejected row is returned as a failure, not zero flux.
"""
from types import SimpleNamespace
from collections import OrderedDict
import time
import numpy as np
from .gpu_compiled_community_backend import lp_arrays,stage_key
from .gpu_revised_basis import GpuRevisedBasis
from .gpu_optimal_face import solve_on_primary_face


def subset_device_result(result,positions):
    """Copy selected environments, retaining solver identity for warm states."""
    import cupy as cp
    ix=cp.asarray(positions,dtype=cp.int64)
    def select(value):
        if isinstance(value,cp.ndarray):return value[ix].copy()
        if isinstance(value,dict):return {k:select(v) for k,v in value.items()}
        return value
    return select(result)


def portable_result(result):
    """Keep numerical state without retaining an evicted device operator."""
    return dict(result,warm_state=dict(result['warm_state'],owner=None))


def bind_portable_result(result,solver,identity):
    if solver.operator_identity!=identity:
        raise ValueError('Portable warm operator changed')
    return dict(result,warm_state=dict(result['warm_state'],owner=solver))


class BatchedCompiledBackend:
    name='gpu_microbatched_certified_basis'
    method='GPU_full_LP_repair'

    def __init__(self,coordinates,banks,pivots=32,repair_cache_size=8,aggregate_portfolio=0,dual_edge='dantzig',reserve_banks=None,pivot_refinement=0,compact_updates=False,repair_portfolio=0,portfolio_pivots=32,portfolio_continuation='best',require_tie=True,max_repair_batch=0,verbose=False,reuse_small_factor=False,skip_unused_dual=False,portfolio_ranking='bank',adaptive_refinement_tolerance=0.,repair_graph_cache_size=4,temporal_pivots=0,restricted_columns=(),restricted_pivots=256,restricted_polish_pivots=0,restricted_rounds=1,restricted_stage1_warm=False,restricted_bucket=False):
        self.coordinates,self.banks=coordinates,banks
        if not isinstance(repair_cache_size,int) or repair_cache_size<1:raise ValueError('Positive cache capacity required')
        self.repair_cache_size=repair_cache_size
        if not isinstance(aggregate_portfolio,int) or aggregate_portfolio<0:raise ValueError('Invalid portfolio size')
        self.aggregate_portfolio=aggregate_portfolio
        self.dual_edge=dual_edge
        self.pivot_refinement=pivot_refinement
        self.compact_updates=compact_updates
        self.reuse_small_factor=reuse_small_factor
        self.skip_unused_dual=skip_unused_dual
        self.adaptive_refinement_tolerance=adaptive_refinement_tolerance
        self.repair_graph_cache_size=repair_graph_cache_size
        if not isinstance(temporal_pivots,int) or temporal_pivots<0:raise ValueError('Invalid temporal budget')
        if temporal_pivots and not compact_updates:raise ValueError('Temporal cache requires compact updates')
        self.temporal_pivots=temporal_pivots;self.temporal_cache={}
        if tuple(sorted(set(restricted_columns)))!=tuple(restricted_columns) or any(v<1 for v in restricted_columns):
            raise ValueError('Restricted dimensions must be positive, unique and increasing')
        if restricted_columns and not compact_updates:raise ValueError('Restricted solves require compact updates')
        if restricted_pivots<1 or restricted_polish_pivots<0:raise ValueError('Invalid restricted budget')
        self.restricted_columns=tuple(restricted_columns);self.restricted_pivots=restricted_pivots
        self.restricted_polish_pivots=restricted_polish_pivots
        if restricted_rounds<1 or (restricted_rounds>1 and len(restricted_columns)!=1):
            raise ValueError('Continued column generation needs one fixed tableau width and positive rounds')
        self.restricted_rounds=restricted_rounds
        if restricted_stage1_warm and not restricted_columns:raise ValueError('Stage-1 reduced warm start requires restricted columns')
        self.restricted_stage1_warm=restricted_stage1_warm
        self.restricted_bucket=restricted_bucket
        if portfolio_ranking not in ('bank','primal_count'):raise ValueError('Invalid portfolio ranking')
        self.portfolio_ranking=portfolio_ranking
        self.repair_portfolio=repair_portfolio
        if portfolio_pivots<1 or portfolio_continuation not in ('best','initial'):raise ValueError('Invalid portfolio strategy')
        self.portfolio_pivots=portfolio_pivots;self.portfolio_continuation=portfolio_continuation
        self.require_tie=require_tie
        if not isinstance(max_repair_batch,int) or max_repair_batch<0:raise ValueError('Invalid repair batch limit')
        self.max_repair_batch=max_repair_batch;self.verbose=verbose
        self.reserve_banks=reserve_banks or {}
        self.pivots=pivots;self.repairs=OrderedDict();self.history=[];self.primary_groups=[]
        self.host_operators=OrderedDict();self.host_operator_cache_hits=0;self.host_operator_preparation_seconds=0.
        self.growth_groups=[]

    def chunks(self,ids):
        size=self.max_repair_batch or max(1,len(ids))
        return [ids[i:i+size] for i in range(0,len(ids),size)]

    def solver(self,key,index,reserve=False):
        cache_key=(key,index,reserve)
        if cache_key not in self.repairs:
            while len(self.repairs)>=self.repair_cache_size:
                _,old_solver=self.repairs.popitem(last=False)
                old_solver.clear_graph_cache()
                del old_solver
            anchor=(self.reserve_banks if reserve else self.banks)[key].evaluators[index]
            identity=(key,int(index),anchor.repair_path[1] if hasattr(anchor,'repair_path') else id(anchor.anchor),reserve)
            prepared=None
            if hasattr(anchor,'repair_anchor'):
                from .gpu_certified_basis import GpuBasisInputAdapter
                started=time.perf_counter()
                if cache_key not in self.host_operators:
                    while len(self.host_operators)>=32:self.host_operators.popitem(last=False)
                    host_anchor=anchor.repair_anchor()
                    self.host_operators[cache_key]=GpuRevisedBasis.prepare_operator(host_anchor)
                    self.host_operator_preparation_seconds+=time.perf_counter()-started
                else:self.host_operator_cache_hits+=1
                self.host_operators.move_to_end(cache_key);prepared=self.host_operators[cache_key]
                anchor=GpuBasisInputAdapter(prepared['anchor'],anchor.bank.variable_rows)
            self.repairs[cache_key]=GpuRevisedBasis(anchor.anchor,anchor.variable_rows,
                max_pivots=self.pivots,capture_safe=True,adapter=anchor,dual_edge=self.dual_edge,pivot_refinement=self.pivot_refinement,
                compact_updates=self.compact_updates,prepared_operator=prepared,reuse_small_factor=self.reuse_small_factor,
                skip_unused_dual=self.skip_unused_dual,adaptive_refinement_tolerance=self.adaptive_refinement_tolerance,
                graph_cache_size=self.repair_graph_cache_size)
            self.repairs[cache_key].operator_identity=identity
            self.repairs[cache_key].bank_index=index
        self.repairs.move_to_end(cache_key)
        return self.repairs[cache_key]

    def solve_batch(self,requests):
        """Requests are (objective, keyword arguments), in environment order."""
        import cupy as cp
        started=time.perf_counter()
        arrays=[lp_arrays(c,**kw) for c,kw in requests]
        keys=[stage_key(a[4],a[0],a[-1],self.coordinates.n_fluxes) for a in arrays]
        if len(set(keys))!=1:raise ValueError('A microbatch must contain one LP stage/structure')
        key=keys[0];batch=len(arrays)
        if self.verbose:print(f'GPU stage {key[0]}: batch {batch} start',flush=True)
        record=dict(stage=key[0],batch=batch,cpu_lp_calls=0,groups=[])
        values=cp.full((batch,len(arrays[0][4])),cp.nan)
        objectives=cp.full(batch,cp.nan);accepted=cp.zeros(batch,dtype=bool)
        residual=cp.full(batch,cp.inf);dual=cp.full(batch,cp.inf);gap=cp.full(batch,cp.inf)
        def merge(ids,result):
            ix=cp.asarray(ids)
            accepted[ix]=result['accepted'];values[ix]=result['values'];objectives[ix]=result['objective']
            residual[ix]=result['primal_residual'];dual[ix]=result['dual_violation'];gap[ix]=result['relative_kkt_gap']
        if key[0]=='exchange_tie':
            if sum(len(g['ids']) for g in self.primary_groups)!=batch:raise ValueError('Missing primary batch')
            for group in self.primary_groups:
                ids,solver,primary=group['ids'],group['solver'],group['result']
                problems=[];bounds=[]
                for i in ids:
                    a,rhs,lo,hi,c,neq=arrays[i]
                    if not np.array_equal(a[-1].toarray().ravel(),group['original_c'][i]):
                        raise ValueError('Changed retained primary objective')
                    problems.append(self.coordinates.normalize(a[:-1],rhs[:-1],lo,hi,c,neq))
                    bounds.append(rhs[-1])
                inputs=solver.prepare_host(problems)
                allowance=cp.asarray(bounds)-primary['objective']
                result=solve_on_primary_face(solver,primary,inputs,inputs['c'],allowance,pivot_budget=0)
                zero_pass=bool(result['accepted'].all().get())
                if not zero_pass:
                    result=solve_on_primary_face(solver,primary,inputs,inputs['c'],allowance)
                merge(ids,result)
                record['groups'].append(dict(ids=ids,pivots=result['pivots'].get().tolist(),
                    complementarity_gap=result['complementarity_gap'].get().tolist(),allowance_gap=result['allowance_gap'].get().tolist(),
                    zero_pivot_certificate_passed=zero_pass,route='original_LP_certified_face'))
            self.primary_groups=[]
        else:
            bank=self.banks[key]
            problems=[self.coordinates.normalize(*a) for a in arrays]
            self.last_problems=problems
            record['host_prepare_seconds']=time.perf_counter()-started
            bank_started=time.perf_counter()
            result=bank.evaluate_device(**bank.prepare_host(problems))
            merge(list(range(batch)),result)
            valid=result['accepted'].get()
            bank_valid=valid.copy()
            chosen=result['candidate_index'].get()
            best=result['best_candidate_index'].get()
            first_best=best.copy()
            record['bank_seconds']=time.perf_counter()-bank_started
            record['bank_accepts']=int(valid.sum())
            for field in ('neural_proposal_accepts','exhaustive_fallback_rows'):
                if field in result:record[field]=result[field]
            if key[0]=='exchange':self.primary_groups=[]
            if key[0]=='maxmin':self.growth_groups=[]
            starts=[];best_progress=cp.full(batch,cp.inf);best_start=cp.full(batch,-1,dtype=cp.int32)
            previous_temporal=self.temporal_cache.pop(key,[]) if self.temporal_pivots else []
            next_temporal=[]
            first_start=cp.full(batch,-1,dtype=cp.int32)
            def operator_identity(index):
                candidate=bank.evaluators[index]
                return (key,int(index),candidate.repair_path[1] if hasattr(candidate,'repair_path') else id(candidate.anchor))
            def remember_start(ids,index,trial,budget):
                if not ids:return
                ix=cp.asarray(ids)
                score=cp.maximum(trial['primal_residual']/1e-5,
                    cp.maximum(trial['dual_violation']/1e-7,trial['relative_kkt_gap']/1e-7))
                improve=cp.isfinite(score)&(score<best_progress[ix])
                positions=np.flatnonzero(improve.get()).tolist()
                if not positions:return
                number=len(starts)
                packet=subset_device_result(trial,positions)
                # Retain numerical state, not the large GPU operator owned by
                # an evicted solver. Rebinding requires identical bank/operator
                # provenance and is performed only by this controlled bridge.
                packet['warm_state']['owner']=None
                starts.append(dict(ids=[ids[j] for j in positions],index=int(index),identity=operator_identity(index),result=packet,budget=budget))
                best_progress[ix]=cp.where(improve,score,best_progress[ix]);best_start[ix]=cp.where(improve,number,best_start[ix])
                first_start[ix]=cp.where(improve&(first_start[ix]<0),number,first_start[ix])
                live=set(best_start.get().tolist())
                if self.portfolio_continuation=='initial':live.update(first_start.get().tolist())
                for old in range(number):
                    if old not in live:starts[old]=None
            def retain(group_ids,solver,group_result):
                # Never let a rejected state become a primary face or a
                # stage-1 feasibility certificate during a later retry.
                keep=np.flatnonzero(group_result['accepted'].get()).tolist()
                if not keep:return
                # Graph replay mutates its output buffers even when every
                # row passed. Freeze retained rows before another chunk can
                # replay the same graph with different environments.
                group_ids=[group_ids[j] for j in keep]
                group_result=subset_device_result(group_result,keep)
                if key[0]=='exchange' and self.require_tie:
                    self.primary_groups.append(dict(ids=group_ids,solver=solver,result=group_result,
                        original_c={i:arrays[i][4].copy() for i in group_ids}))
                if key[0]=='maxmin':
                    self.growth_groups.append(dict(ids=group_ids,key=key,index=solver.bank_index,
                        identity=solver.operator_identity,
                        problems=[problems[i] for i in group_ids],result=portable_result(group_result)))
                if self.temporal_pivots:
                    next_temporal.append(dict(ids=group_ids,index=solver.bank_index,
                        identity=solver.operator_identity,result=portable_result(group_result)))
            # Old GPU solutions are only basis proposals. Every temporal
            # query updates the changed matrix/RHS/bounds and recertifies the
            # full current LP before any flux is returned to the simulator.
            for group in previous_temporal:
                carry=[j for j,i in enumerate(group['ids']) if i<batch and bank_valid[i]]
                if carry:
                    next_temporal.append(dict(group,ids=[group['ids'][j] for j in carry],
                        result=subset_device_result(group['result'],carry)))
                positions=[j for j,i in enumerate(group['ids']) if i<batch and not valid[i]]
                for part in self.chunks(positions):
                    ids=[group['ids'][j] for j in part];group_started=time.perf_counter()
                    solver=self.solver(key,group['index'])
                    primary=bind_portable_result(subset_device_result(group['result'],part),solver,group['identity'])
                    trial=solver.run_device(**solver.prepare_host([problems[i] for i in ids]),
                        warm_start=primary['warm_state'],update_warm_matrix=True,
                        pivot_budget=min(self.pivots,self.temporal_pivots))
                    merge(ids,trial);retain(ids,solver,trial)
                    good=trial['accepted'].get();valid[ids]=good
                    record['groups'].append(dict(ids=ids,candidate_index=group['index'],route='temporal_GPU_basis_update',
                        accepted=good.tolist(),pivots=trial['pivots'].get().tolist(),seconds=time.perf_counter()-group_started))
            if self.restricted_columns:
                from .gpu_restricted_basis import GpuRestrictedBasis
                # The aggregate objective changes, but its structural matrix
                # is the same as stage 1. Reuse that certified GPU basis
                # before constructing a fresh feasibility path. Every warm
                # query still checks its RHS/matrix/scales and original KKT.
                if key[0]=='aggregate' and self.restricted_stage1_warm:
                    for group in self.growth_groups:
                        positions=[j for j,i in enumerate(group['ids']) if not valid[i]]
                        for part in self.chunks(positions):
                            ids=[group['ids'][j] for j in part];before=time.perf_counter()
                            solver=self.solver(group['key'],group['index'])
                            primary=group['result']
                            if primary is None:
                                primary=solver.run_device(**solver.prepare_host([group['problems'][j] for j in part]),pivot_budget=0)
                            else:primary=bind_portable_result(subset_device_result(primary,part),solver,group['identity'])
                            if not bool(primary['accepted'].all().get()):raise ValueError('Stage-1 reduced warm basis did not recertify')
                            if getattr(solver,'restricted_solver',None) is None:
                                solver.restricted_solver=GpuRestrictedBasis(solver,self.restricted_pivots,rank_bucket=self.restricted_bucket)
                            warm=primary['warm_state'];query=solver.prepare_host([problems[i] for i in ids])
                            setup_seconds=time.perf_counter()-before
                            for round_index in range(self.restricted_rounds):
                                before=time.perf_counter()
                                trial=solver.restricted_solver.run_device(**query,columns=self.restricted_columns[0],warm_start=warm,
                                    reuse_lifted_state=self.restricted_bucket and round_index>0)
                                merge(ids,trial);good=trial['accepted'].get();valid[ids]=good
                                record['groups'].append(dict(ids=ids.copy(),candidate_index=group['index'],route='restricted_stage1_GPU_columns',
                                    columns=self.restricted_columns[0],round=round_index,accepted=good.tolist(),pivots=trial['restricted_pivots'].get().tolist(),
                                    seconds=time.perf_counter()-before+(setup_seconds if round_index==0 else 0.)))
                                remaining=np.flatnonzero(~good).tolist()
                                if not remaining:break
                                warm=subset_device_result(trial,remaining)['warm_state']
                                ids=[ids[j] for j in remaining];query={k:v[cp.asarray(remaining)] for k,v in query.items()}
                choices=(cp.argmin(bank.last_candidate_scores,axis=1).get()
                    if hasattr(bank,'last_candidate_scores') and bank.candidate_ranking=='count' else best)
                for index in np.unique(choices[~valid]):
                    for initial_ids in self.chunks(np.flatnonzero(~valid&(choices==index)).tolist()):
                        solver=self.solver(key,int(index))
                        if getattr(solver,'restricted_solver',None) is None:
                            solver.restricted_solver=GpuRestrictedBasis(solver,self.restricted_pivots,rank_bucket=self.restricted_bucket)
                        reducer=solver.restricted_solver;ids=initial_ids;selected=None;prior=0;reduced_warm=None
                        query=solver.prepare_host([problems[i] for i in ids])
                        requests=self.restricted_columns*self.restricted_rounds
                        for round_index,requested in enumerate(requests):
                            width=min(requested,len(reducer.nonbasic0))
                            if self.restricted_rounds==1 and width<=prior:continue
                            group_started=time.perf_counter()
                            trial=reducer.run_device(**query,columns=width,selected_columns=selected,warm_start=reduced_warm,
                                reuse_lifted_state=self.restricted_bucket and reduced_warm is not None)
                            merge(ids,trial);retain(ids,solver,trial)
                            good=trial['accepted'].get();valid[ids]=good
                            record['groups'].append(dict(ids=ids.copy(),candidate_index=int(index),route='restricted_GPU_columns',
                                columns=width,accepted=good.tolist(),pivots=trial['restricted_pivots'].get().tolist(),
                                round=round_index,primal_residual=trial['primal_residual'].get().tolist(),
                                dual_violation=trial['dual_violation'].get().tolist(),relative_kkt_gap=trial['relative_kkt_gap'].get().tolist(),
                                lifted_restricted_dual=trial['restricted_lifted_dual_violation'].get().tolist(),
                                seconds=time.perf_counter()-group_started))
                            if self.restricted_polish_pivots and (self.restricted_rounds==1 or round_index==len(requests)-1):
                                eligible=np.flatnonzero(~good&(trial['primal_residual'].get()<=1e-5)).tolist()
                                if eligible:
                                    polish_ids=[ids[j] for j in eligible];before=time.perf_counter()
                                    warm=subset_device_result(trial,eligible)['warm_state']
                                    polish=solver.run_device(**solver.prepare_host([problems[i] for i in polish_ids]),warm_start=warm,
                                        pivot_budget=min(self.pivots,self.restricted_polish_pivots))
                                    merge(polish_ids,polish);retain(polish_ids,solver,polish)
                                    good[eligible]=polish['accepted'].get();valid[polish_ids]=good[eligible]
                                    record['groups'].append(dict(ids=polish_ids,candidate_index=int(index),route='restricted_full_GPU_polish',
                                        columns=width,accepted=good[eligible].tolist(),pivots=polish['pivots'].get().tolist(),seconds=time.perf_counter()-before))
                            remaining=np.flatnonzero(~good)
                            if not len(remaining):break
                            if self.restricted_rounds>1:
                                reduced_warm=subset_device_result(trial,remaining.tolist())['warm_state']
                                ids=[ids[j] for j in remaining];query={k:v[cp.asarray(remaining)] for k,v in query.items()}
                                continue
                            selected=trial['selected_columns'][cp.asarray(remaining)].copy()
                            scores=trial['expansion_scores'][cp.asarray(remaining)]
                            ids=[ids[j] for j in remaining];query={k:v[cp.asarray(remaining)] for k,v in query.items()}
                            # The next directions are priced at this RMP's
                            # lifted solution, not an initial static ranking.
                            next_width=next((min(v,len(reducer.nonbasic0)) for v in self.restricted_columns if min(v,len(reducer.nonbasic0))>width),width)
                            additional=next_width-width
                            if additional:selected=cp.concatenate((selected,reducer.nonbasic0[cp.argsort(-scores,axis=1)[:,:additional]]),axis=1)
                            prior=width
            for index in range(len(bank.evaluators)):
                # Host dispatch is measured; it is not a host optimizer.
                repair_ids=np.flatnonzero(~valid&(best==index)).tolist()
                if key[0]=='maxmin':
                    ids=np.flatnonzero(valid&(chosen==index)).tolist()
                    for part in self.chunks(ids):self.growth_groups.append(dict(ids=part,key=key,index=index,
                        problems=[problems[i] for i in part],result=None))
                direct_ids=np.flatnonzero(valid&(chosen==index)).tolist() if key[0]=='exchange' and self.require_tie else []
                initial_budget=min(self.pivots,8 if key[0]=='maxmin' else 32)
                if key[0]=='aggregate' and (self.aggregate_portfolio or key in self.reserve_banks):initial_budget=min(self.pivots,8)
                jobs=[(part,budget,route) for ids,budget,route in
                    ((repair_ids,initial_budget,'full_variable_repair'),(direct_ids,0,'certified_face_setup')) for part in self.chunks(ids)]
                for ids,budget,route in jobs:
                    if not ids:continue
                    group_started=time.perf_counter()
                    solver=self.solver(key,index)
                    before=solver.graph_compilation_seconds
                    repaired=solver.run_device(**solver.prepare_host([problems[i] for i in ids]),pivot_budget=budget)
                    merge(ids,repaired)
                    record['groups'].append(dict(ids=ids,pivots=repaired['pivots'].get().tolist(),route=route,
                        candidate_index=index,
                        seconds=time.perf_counter()-group_started,
                        graph_compilation_seconds=solver.graph_compilation_seconds-before))
                    good=repaired['accepted'].get()
                    good_positions=np.flatnonzero(good).tolist()
                    bad_positions=np.flatnonzero(~good).tolist()
                    def remember(group_ids,group_result):
                        if group_ids:retain(group_ids,solver,group_result)
                    if not bad_positions:
                        remember(ids,repaired)
                    else:
                        good_ids=[ids[j] for j in good_positions]
                        if good_ids:remember(good_ids,subset_device_result(repaired,good_positions))
                        bad_ids=[ids[j] for j in bad_positions]
                        if self.repair_portfolio:
                            remember_start(bad_ids,index,subset_device_result(repaired,bad_positions),budget)
                        if budget and budget<self.pivots and not self.repair_portfolio and not (key[0]=='aggregate' and (self.aggregate_portfolio or key in self.reserve_banks)):
                            # Only rejected rows consume the larger correction tier.
                            # Raw basis/state may be reused, never its rejected flux.
                            warm=subset_device_result(repaired,bad_positions)['warm_state']
                            before=solver.graph_compilation_seconds
                            retry=solver.run_device(**solver.prepare_host([problems[i] for i in bad_ids]),
                                warm_start=warm,pivot_budget=self.pivots-budget)
                            merge(bad_ids,retry)
                            record['groups'].append(dict(ids=bad_ids,pivots=retry['pivots'].get().tolist(),
                                route='larger_gpu_correction',graph_compilation_seconds=solver.graph_compilation_seconds-before))
                            remember(bad_ids,retry)
            if self.repair_portfolio and starts:
                order=bank.rank().get() if hasattr(bank,'rank') else np.tile(np.arange(len(bank.evaluators)),(batch,1))
                if self.portfolio_ranking=='primal_count' and key[0]=='exchange':
                    if not hasattr(bank,'last_candidate_scores') or bank.candidate_ranking!='count':
                        raise ValueError('Primal portfolio requires compact count scores')
                    order=cp.argsort(bank.last_candidate_scores,axis=1).get()
                order=np.array([[j for j in row if j!=first_best[i]] for i,row in enumerate(order)])
                for tier in range(min(self.repair_portfolio,order.shape[1])):
                    pending=np.flatnonzero(~accepted.get())
                    if not len(pending):break
                    jobs=[(int(index),ids) for index in np.unique(order[pending,tier])
                        for ids in self.chunks(pending[order[pending,tier]==index].tolist())]
                    for index,ids in jobs:
                        group_started=time.perf_counter()
                        solver=self.solver(key,int(index))
                        budget=min(self.pivots,self.portfolio_pivots)
                        trial=solver.run_device(**solver.prepare_host([problems[i] for i in ids]),pivot_budget=budget)
                        merge(ids,trial);retain(ids,solver,trial)
                        record['groups'].append(dict(ids=ids,pivots=trial['pivots'].get().tolist(),route='short_compact_portfolio',candidate_index=int(index),tier=tier+1,
                            seconds=time.perf_counter()-group_started))
                        bad=np.flatnonzero(~trial['accepted'].get()).tolist()
                        remember_start([ids[j] for j in bad],int(index),subset_device_result(trial,bad),budget)
                choices=(cp.where(first_start>=0,first_start,best_start) if self.portfolio_continuation=='initial' else best_start).get()
                pending=~accepted.get()
                for number,start in enumerate(starts):
                    if start is None:continue
                    positions=[j for j,i in enumerate(start['ids']) if pending[i] and choices[i]==number]
                    if not positions:continue
                    ids=[start['ids'][j] for j in positions]
                    if start['identity']!=operator_identity(start['index']):raise ValueError('Portable warm operator changed')
                    group_started=time.perf_counter()
                    solver=self.solver(key,start['index'])
                    warm=subset_device_result(start['result'],positions)['warm_state']
                    warm['owner']=solver
                    retry=solver.run_device(**solver.prepare_host([problems[i] for i in ids]),warm_start=warm,pivot_budget=max(0,self.pivots-start['budget']))
                    merge(ids,retry);retain(ids,solver,retry)
                    record['groups'].append(dict(ids=ids,pivots=retry['pivots'].get().tolist(),route=self.portfolio_continuation+'_portfolio_continuation',
                        candidate_index=start['index'],seconds=time.perf_counter()-group_started,
                        unique_update_columns=retry.get('unique_update_columns',cp.zeros(len(ids),dtype=cp.int32)).get().tolist()))
                starts.clear()
            if key[0]=='aggregate' and self.aggregate_portfolio:
                # Diversify the initial basis BEFORE long retries. A ranking
                # is only a proposal: every attempted solution is certified.
                proposal=getattr(bank,'proposal',None)
                if proposal is not None and hasattr(proposal,'rank'):
                    order=proposal.rank(bank.prepare_host(problems)).get()
                else:order=np.tile(np.arange(len(bank.evaluators)),(batch,1))
                order=np.array([[j for j in row if j!=first_best[i]] for i,row in enumerate(order)])
                for tier in range(min(self.aggregate_portfolio,order.shape[1])):
                    pending=np.flatnonzero(~accepted.get())
                    if not len(pending):break
                    for index in np.unique(order[pending,tier]):
                        ids=pending[order[pending,tier]==index].tolist()
                        solver=self.solver(key,int(index));before=solver.graph_compilation_seconds
                        repaired=solver.run_device(**solver.prepare_host([problems[i] for i in ids]),
                            pivot_budget=min(self.pivots,8))
                        merge(ids,repaired)
                        record['groups'].append(dict(ids=ids,pivots=repaired['pivots'].get().tolist(),
                            route='short_alternative_basis',candidate_index=int(index),tier=tier+1,
                            accepted=repaired['accepted'].get().tolist(),
                            graph_compilation_seconds=solver.graph_compilation_seconds-before))
            if key[0]=='aggregate' and key in self.reserve_banks:
                pending=np.flatnonzero(~accepted.get()).tolist()
                record['reserve_query_rows']=len(pending)
                if pending:
                    reserve=self.reserve_banks[key];before=time.perf_counter()
                    proposed=reserve.evaluate_device(**reserve.prepare_host([problems[i] for i in pending]))
                    merge(pending,proposed)
                    valid_reserve=proposed['accepted'].get();best_reserve=proposed['best_candidate_index'].get()
                    record['reserve_bank_seconds']=time.perf_counter()-before
                    record['reserve_direct_accepts']=int(valid_reserve.sum())
                    for index in np.unique(best_reserve[~valid_reserve]):
                        positions=np.flatnonzero(~valid_reserve&(best_reserve==index))
                        ids=[pending[j] for j in positions]
                        solver=self.solver(key,int(index),reserve=True)
                        before=solver.graph_compilation_seconds
                        repaired=solver.run_device(**solver.prepare_host([problems[i] for i in ids]),
                            pivot_budget=min(self.pivots,32))
                        merge(ids,repaired)
                        record['groups'].append(dict(ids=ids,pivots=repaired['pivots'].get().tolist(),
                            route='reserve_basis_repair',candidate_index=int(index),
                            accepted=repaired['accepted'].get().tolist(),
                            graph_compilation_seconds=solver.graph_compilation_seconds-before))
                        bad_positions=np.flatnonzero(~repaired['accepted'].get()).tolist()
                        if bad_positions and self.pivots>32:
                            # Continue the new basis rather than discarding
                            # its progress and restarting the older bank.
                            bad_ids=[ids[j] for j in bad_positions]
                            warm=subset_device_result(repaired,bad_positions)['warm_state']
                            before=solver.graph_compilation_seconds
                            retry=solver.run_device(**solver.prepare_host([problems[i] for i in bad_ids]),
                                warm_start=warm,pivot_budget=self.pivots-32)
                            merge(bad_ids,retry)
                            record['groups'].append(dict(ids=bad_ids,pivots=retry['pivots'].get().tolist(),
                                route='reserve_basis_continuation',candidate_index=int(index),
                                accepted=retry['accepted'].get().tolist(),
                                graph_compilation_seconds=solver.graph_compilation_seconds-before))
            if result.get('neural_deferred_exhaustive',False):
                pending=np.flatnonzero(~accepted.get()).tolist()
                record['exhaustive_after_neural_repair_rows']=len(pending)
                if pending:
                    fallback=bank.base.evaluate_device(**bank.prepare_host([problems[i] for i in pending]))
                    merge(pending,fallback)
                    ok=fallback['accepted'].get();chosen=fallback['candidate_index'].get()
                    best=fallback['best_candidate_index'].get()
                    for index in range(len(bank.evaluators)):
                        for positions,budget in ((np.flatnonzero(~ok&(best==index)),self.pivots),
                                                (np.flatnonzero(ok&(chosen==index)),0)):
                            if not len(positions):continue
                            ids=[pending[j] for j in positions]
                            solver=self.solver(key,index)
                            repaired=solver.run_device(**solver.prepare_host([problems[i] for i in ids]),pivot_budget=budget)
                            merge(ids,repaired);retain(ids,solver,repaired)
                            record['groups'].append(dict(ids=ids,pivots=repaired['pivots'].get().tolist(),
                                route='exhaustive_after_neural_repair'))
            if key[0]=='aggregate':
                bad=~accepted.get()
                for group in self.growth_groups:
                    positions=[j for j,i in enumerate(group['ids']) if bad[i]]
                    if not positions:continue
                    ids=[group['ids'][j] for j in positions]
                    primary=group['result']
                    if primary is None:
                        solver=self.solver(group['key'],group['index'])
                        primary=solver.run_device(**solver.prepare_host([group['problems'][j] for j in positions]),pivot_budget=0)
                    else:
                        primary=subset_device_result(primary,positions)
                        solver=self.solver(group['key'],group['index'])
                        primary=bind_portable_result(primary,solver,group['identity'])
                    if not bool(primary['accepted'].all().get()):
                        raise ValueError('Stage-1 warm basis did not recertify')
                    before=solver.graph_compilation_seconds
                    # Stage 2 retains the same A/RHS and tightens only tau
                    # around the attained coexistence optimum. Reuse a known
                    # feasible stage-1 basis instead of restarting Phase I.
                    repaired=solver.run_device(**solver.prepare_host([problems[i] for i in ids]),
                        warm_start=primary['warm_state'],pivot_budget=self.pivots)
                    merge(ids,repaired)
                    record['groups'].append(dict(ids=ids,pivots=repaired['pivots'].get().tolist(),
                        route='stage1_feasible_basis_reoptimization',
                        graph_compilation_seconds=solver.graph_compilation_seconds-before))
                self.growth_groups=[]
            if self.temporal_pivots:self.temporal_cache[key]=next_temporal
        host_values=values.get();host_obj=objectives.get();host_ok=accepted.get()
        record.update(accepted=host_ok.tolist(),primal_residual=residual.get().tolist(),
            dual_violation=dual.get().tolist(),relative_kkt_gap=gap.get().tolist(),seconds=time.perf_counter()-started)
        free,total=cp.cuda.runtime.memGetInfo()
        record['device_used_bytes_at_stage_end']=total-free
        self.history.append(record)
        if self.verbose:print(f'GPU stage {key[0]}: {sum(host_ok)}/{batch} accepted; {record["seconds"]:.3f} s; bank {record.get("bank_seconds",0.):.3f} s',flush=True)
        return [SimpleNamespace(success=bool(host_ok[i]),x=host_values[i] if host_ok[i] else None,
            fun=float(host_obj[i]) if host_ok[i] else None,
            diagnostics=dict(success=bool(host_ok[i]),max_original_residual=record['primal_residual'][i],
                objective=float(host_obj[i]),total_seconds=record['seconds'],cpu_lp_calls=0),
            message=str(dict(stage=key[0],accepted=bool(host_ok[i]),
            primal=record['primal_residual'][i],dual=record['dual_violation'][i],gap=record['relative_kkt_gap'][i]))) for i in range(batch)]


class YieldingLPBackend:
    """Per-environment bridge preserving the ordinary backend audit contract."""
    name=BatchedCompiledBackend.name
    method=BatchedCompiledBackend.method

    def __init__(self,parent):self.parent=parent;self.history=[]

    def solve(self,c,**kw):
        result=self.parent.switch((c,kw))
        self.history.append(dict(result.diagnostics))
        return result
