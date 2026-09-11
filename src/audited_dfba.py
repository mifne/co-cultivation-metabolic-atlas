"""Audited cultivation reference, versioned separately from legacy LP teachers.

Conserves the quantities represented by the GEM and dynamic accounting. It
does not repair missing biology, calibrate kinetics, or prove elemental
balance for GEM metabolites with missing/pseudo molecular formulae.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from cobra.core.solution import Solution
from cobra.util.array import create_stoichiometric_matrix
from scipy.optimize import linprog, brentq
from scipy.sparse import csr_matrix, hstack, vstack, coo_matrix

from .resolved_dfba import ResolvedDFBASimulator, oxygen_interval
from .cultivation_numerics import compute_uptake_limits, allocate_oxygen, oxygen_mean_response
from .utils import COEXISTENCE_DEFINED_FEED_MMOL_L_H
from .metabolite_ids import canonical_metabolite_id


class AuditedDFBASimulator(ResolvedDFBASimulator):
    NUMERICS_VERSION = 'audited_cultivation_v2'
    NUMERICAL_REPAIR_REVISION = 'shared_oxygen_endpoint_20260908'
    GROWTH_FLOOR_TOLERANCE_H_INV = 1e-9
    DOSE_ALIASES = {'sn_or16':'mlttr_e', 'sn_ns21':'ptrc_e',
                    'sn_lp':'mnl_e', 'helper_lactate':'lac__L_e'}
    DEFINED_MW = {'glc__D_e':180.16, 'arg__L_e':174.20, 'trp__L_e':204.23,
                  'leu__L_e':131.17, 'glu__L_e':147.13, 'ile__L_e':131.17,
                  'pro__L_e':115.13, 'phe__L_e':165.19, 'gln__L_e':146.14,
                  'pydam_e':168.20}

    def __init__(self, models, initial_biomass, initial_metabolites, *,
                 growth_reactions=None, density_policy='flux_consistent',
                 polymer_oxygen_half_saturation=.01, buffer_mmol_l=50.,
                 flux_selection='parsimonious_exchange', oxygen_scheme='shared_endpoint_v3', **kwargs):
        if kwargs.get('solver_backend', 'highs') != 'highs':
            raise ValueError('audited reference requires highs and its explicit storage inequality')
        if density_policy not in {'none','flux_consistent'}:
            raise ValueError('unknown density_policy')
        if flux_selection not in {'parsimonious_exchange','primary_only'}:raise ValueError('unknown flux_selection')
        self.flux_selection=flux_selection
        if oxygen_scheme not in {'shared_endpoint_v3','quota_mean_v2'}:raise ValueError('unknown oxygen_scheme')
        self.oxygen_scheme=oxygen_scheme
        for key,value in [('polymer_oxygen_half_saturation',polymer_oxygen_half_saturation),
                          ('buffer_mmol_l',buffer_mmol_l)]:
            if not math.isfinite(value) or value<=0:raise ValueError(key+' must be positive and finite')
        for group in [initial_biomass,initial_metabolites]:
            if any(not math.isfinite(v) or v<0 for v in group.values()):
                raise ValueError('initial concentrations must be nonnegative and finite')
        for key in ['initial_rubber','volume','max_uptake_rate','carrying_capacity']:
            if key in kwargs and (not math.isfinite(kwargs[key]) or kwargs[key]<0 or (key in {'volume','carrying_capacity'} and kwargs[key]==0)):
                raise ValueError('invalid '+key)
        self.density_policy=density_policy
        self.polymer_oxygen_half_saturation=float(polymer_oxygen_half_saturation)
        self.configured_buffer_mmol_l=float(buffer_mmol_l)
        # Own the working GEMs: constructing/updating this reference must not
        # mutate a model used by another simulator or an immutable trace.
        working={name:model.copy() for name,model in models.items()}
        for model in working.values():
            if len(model.constraints)!=len(model.metabolites):
                raise ValueError('additional model constraints need an explicit audited LP adapter')
            self._validate_balance_constraints(model)
            if any(not r.lower_bound<=0<=r.upper_bound for r in model.reactions):
                raise ValueError('required nonzero internal flux bounds need an adapter for density/pH scaling')
            seen_pools=set()
            for rxn in model.exchanges:
                if len(rxn.metabolites)!=1:raise ValueError('unsupported nonsingleton exchange: '+rxn.id)
                pool=canonical_metabolite_id(next(iter(rxn.metabolites)).id)
                if pool in seen_pools:raise ValueError('duplicate canonical exchange pool: '+pool)
                seen_pools.add(pool)
                if not rxn.lower_bound<=0<=rxn.upper_bound:
                    raise ValueError('base exchange must allow zero; explicit measured ranges require a supported contract: '+rxn.id)
        kwargs['solver_backend']='highs'
        super().__init__(models=working,initial_biomass=initial_biomass,
                         initial_metabolites=initial_metabolites,**kwargs)
        self.growth_reactions={}
        supplied=growth_reactions or {}
        for name,model in self.models.items():
            candidates=[supplied[name]] if name in supplied else [rid for rid in
                ['R_Growth','Growth','biomass_c0','R_BIOMASS_LLA','BIOMASS_LLA'] if rid in model.reactions]
            if len(candidates)!=1 or candidates[0] not in model.reactions:
                raise ValueError('explicit unique growth reaction required for '+name)
            self.growth_reactions[name]=candidates[0]
        self._lp_templates={name:(create_stoichiometric_matrix(m,array_type='lil',dtype=float).tocsr(),
                                 [r.id for r in m.reactions]) for name,m in self.models.items()}
        self._structure_contract={n:self._structure_token(m) for n,m in self.models.items()}
        self._initial_state=copy.deepcopy(self.state)
        self._initial_buffers=(self.buffer_total,self.buffer_base,self.buffer_acid)
        self._initial_bounds={n:{r.id:r.bounds for r in m.reactions} for n,m in self.models.items()}
        self._invalid=False
        self._oxygen_last_guess=None;self._oxygen_last_slope=1.
        self._oxygen_last_mean=None
        self._cell_scale=1.
        self._integration_biomass={}
        self.cumulative_delivered_mmol_l={}
        self.accounting_audit={'roundoff_added_mmol_l':{}, 'max_lp_residual':0.,
                               'max_storage_excess_g_l':0., 'failed_steps':0,
                               'oxygen_trial_solves':0,'max_oxygen_mean_residual':0.}
        self._initialize_lp_audit()

    def _initialize_lp_audit(self):
        self.accounting_audit.update(lp_requests=0,lp_certified_requests=0,lp_retry_attempts=0,
            rejected_lp_trials=[],max_dual_residual=0.,max_relative_duality_gap=0.,max_oxygen_kinetic_residual=0.)

    @staticmethod
    def _validate_balance_constraints(model):
        expected={met.id:{} for met in model.metabolites}
        for reaction in model.reactions:
            for met,coefficient in reaction.metabolites.items():
                expected[met.id][reaction.forward_variable]=float(coefficient)
                expected[met.id][reaction.reverse_variable]=-float(coefficient)
        if {c.name for c in model.constraints}!=set(expected):
            raise ValueError('noncanonical balance constraints need an explicit audited LP adapter')
        for constraint in model.constraints:
            terms={v:float(c) for v,c in constraint.expression.as_coefficients_dict().items() if c!=0}
            if constraint.lb!=0 or constraint.ub!=0 or terms!=expected[constraint.name]:
                raise ValueError('modified balance constraints need an explicit audited LP adapter')

    @staticmethod
    def _structure_token(model):
        return (tuple((r.id,tuple(sorted((m.id,float(c)) for m,c in r.metabolites.items())),
                       r.gene_reaction_rule,float(r.objective_coefficient)) for r in model.reactions),
                tuple((m.id,m.formula,m.charge,m.compartment) for m in model.metabolites),
                tuple((c.name,c.lb,c.ub,tuple(sorted((getattr(v,'name',None) or str(v),float(co)) for v,co in
                      c.expression.as_coefficients_dict().items() if co!=0))) for c in model.constraints))

    def _safe_lb(self, desired_lb, upper_bound):
        if not math.isfinite(desired_lb) or not math.isfinite(upper_bound):
            raise ValueError('finite uptake bounds required')
        return min(float(desired_lb),float(upper_bound))

    def _identify_exchange_reactions(self):
        mapping=super()._identify_exchange_reactions()
        for name,model in self.models.items():
            clean={};seen=set()
            for met,rid in mapping[name].items():
                if met in self.DOSE_ALIASES or rid in seen:continue
                reaction=model.reactions.get_by_id(rid)
                if len(reaction.metabolites)!=1 or next(iter(reaction.metabolites.values()))!=-1:
                    raise ValueError('unsupported exchange stoichiometry: '+name+'/'+rid)
                clean[met]=rid;seen.add(rid)
            mapping[name]=clean
        return mapping

    def _initialize_medium(self):
        requested_h=self.state.metabolites.get('h_e')
        super()._initialize_medium()
        self.buffer_total=self.configured_buffer_mmol_l
        ph=self.ph_control_target if self.ph_control_target is not None else (
            3-math.log10(requested_h) if requested_h and requested_h>0 else 7.)
        ratio=10**(ph-self.pKa)
        self.buffer_base=self.buffer_total*ratio/(1+ratio)
        self.buffer_acid=self.buffer_total-self.buffer_base
        self.state.metabolites['h_e']=10**(3-ph)

    def reset(self):
        self.state=copy.deepcopy(self._initial_state)
        self.buffer_total,self.buffer_base,self.buffer_acid=self._initial_buffers
        for name,model in self.models.items():
            for rid,bounds in self._initial_bounds[name].items():model.reactions.get_by_id(rid).bounds=bounds
        for key in ['cumulative_co2_emission','cumulative_base_added_mmol_l','cumulative_acid_added_mmol_l',
                    'cumulative_defined_feed_g_l','last_defined_feed_g_l','current_step','controller_steps',
                    'solve_attempts','solve_successes']:
            setattr(self,key,0)
        self.cumulative_continuous_feed={};self.cumulative_delivered_mmol_l={}
        self.oxygen_audit=dict(transferred=0.,polymer_consumed=0.,cellular_consumed=0.,max_balance_error=0.)
        self.accounting_audit={'roundoff_added_mmol_l':{},'max_lp_residual':0.,'max_storage_excess_g_l':0.,'failed_steps':0,
                               'oxygen_trial_solves':0,'max_oxygen_mean_residual':0.}
        self._initialize_lp_audit()
        self.nitrogen_allocation={};self.last_fba_solutions={};self.last_polymer_fluxes={}
        self.last_step_timing={};self.step_timing_totals={k:0. for k in self.step_timing_totals}
        self._oxygen_context=None;self._invalid=False
        self._oxygen_last_guess=None;self._oxygen_last_slope=1.
        self._oxygen_last_mean=None
        return self.state

    def _add_feed(self,met,dose):
        self.state.metabolites[met]=self.state.metabolites.get(met,0.)+dose
        self.cumulative_delivered_mmol_l[met]=self.cumulative_delivered_mmol_l.get(met,0.)+dose

    def _prepare_oxygen(self,kla,rates,kinetic_do):
        c0=self.state.metabolites.get('o2_e',self.oxygen_saturation)
        # Enzyme potential is evaluated without an inventory cap, then given
        # a physical DO-dependent rate. The fixed 25% per-step reservation is
        # deliberately absent. This affinity is provisional, not measured.
        saved=(self.state.rubber_concentration,dict(self.state.metabolites))
        original_fraction=self.polymer_oxygen_fraction
        self.state.metabolites['o2_e']=1e12;self.polymer_oxygen_fraction=1.
        try:
            super(ResolvedDFBASimulator,self).degrade_rubber(rates)
            potential=dict(self.last_polymer_fluxes)
            requested=sum([self.last_polymer_fluxes['lcp_c5_mmol_l_step']/6,
                self.last_polymer_fluxes['roxb_c5_mmol_l_step']/6,
                self.last_polymer_fluxes['roxa_direct_c5_mmol_l_step']/3,
                self.last_polymer_fluxes['roxa_oligo_c30_mmol_l_step']])
        finally:
            self.state.rubber_concentration=saved[0];self.state.metabolites=dict(saved[1])
            self.polymer_oxygen_fraction=original_fraction
        affinity=kinetic_do/(self.polymer_oxygen_half_saturation+kinetic_do)
        polymer_rate=requested/self.dt*affinity
        cellular={}
        for name in self.models:
            rid=self.exchange_reactions[name].get('o2_e')
            if rid:
                original=self.original_bounds[name].get(rid,(0,0))[0]
                limit=min(max(0.,-original),self.max_uptake_rate*kinetic_do/(.01+kinetic_do))
                cellular[name]=limit*self._integration_biomass[name]*self._cell_scale
        if self.oxygen_scheme=='quota_mean_v2':
            budget=allocate_oxygen(c0,kla,self.dt,self.oxygen_saturation,polymer_rate,cellular)
        else:
            # Species see the same DO. Only their actual jointly solved OUR
            # consumes oxygen; unused hypothetical uptake gets no reservation.
            free,response,mean,total=oxygen_interval(c0,kla,self.dt,self.oxygen_saturation)
            budget=dict(free=free,response=response,mean=mean,budget=total,scale=1.,
                polymer_amount=polymer_rate*self.dt,
                cellular_amounts={name:rate*self.dt for name,rate in cellular.items()})
        self._oxygen_context=dict(budget,c0=c0,polymer=0.)
        # Scale every extracellular reaction extent by the same factor;
        # allocating oxygen to bulk reactions first would starve the oligo
        # reaction according to implementation order.
        factor=affinity*budget['scale']
        actual={k:potential[k]*factor for k in ['lcp_c5_mmol_l_step','roxb_c5_mmol_l_step',
                'roxa_direct_c5_mmol_l_step','roxa_oligo_c30_mmol_l_step']}
        lcp=actual['lcp_c5_mmol_l_step'];roxb=actual['roxb_c5_mmol_l_step']
        direct=actual['roxa_direct_c5_mmol_l_step'];oligo=actual['roxa_oligo_c30_mmol_l_step']
        removed=(lcp+roxb+direct)*self.RUBBER_C5_MOLAR_MASS_G_PER_MOL/1000.
        self.state.rubber_concentration=saved[0]-removed
        self.state.metabolites['C30_oligo_e']=saved[1].get('C30_oligo_e',0.)+(lcp+roxb)/6-oligo
        self.state.metabolites['odtd_e']=saved[1].get('odtd_e',0.)+direct/3+2*oligo
        polymer=(lcp+roxb)/6+direct/3+oligo
        carbon_before=saved[0]*1000/self.RUBBER_C5_MOLAR_MASS_G_PER_MOL+6*saved[1].get('C30_oligo_e',0.)+3*saved[1].get('odtd_e',0.)
        carbon_after=self.state.rubber_concentration*1000/self.RUBBER_C5_MOLAR_MASS_G_PER_MOL+6*self.state.metabolites['C30_oligo_e']+3*self.state.metabolites['odtd_e']
        self.last_polymer_fluxes=dict(actual,oxygen_mmol_l_step=polymer,rubber_degraded_g_l_step=removed,
            carbon_c5_equivalent_error_mmol_l=carbon_after-carbon_before,bulk_substrate_scale=potential['bulk_substrate_scale']*factor,
            polymer_oxygen_fraction=None,kinetic_oxygen_factor=affinity,shared_oxygen_scale=budget['scale'])
        self._oxygen_context['polymer']=polymer
        self.state.metabolites['o2_e']=c0
        concentration=self.state.metabolites.get('C30_oligo_e',0.)
        if -1e-12<=concentration<0.:
            addition=-concentration
            self.state.metabolites['C30_oligo_e']=0.
            self.last_polymer_fluxes['carbon_c5_equivalent_error_mmol_l']+=6*addition
            self._oxygen_context['c30_roundoff_added']=addition

    def _solve_oxygen_coupled(self,kla,rates):
        initial_rubber=self.state.rubber_concentration
        initial_medium=dict(self.state.metabolites)
        c0=initial_medium.get('o2_e',self.oxygen_saturation)
        free,response,free_mean,budget=oxygen_interval(c0,kla,self.dt,self.oxygen_saturation)
        mean_response=oxygen_mean_response(kla,self.dt)
        endpoint=self.oxygen_scheme=='shared_endpoint_v3'
        upper_do=free if endpoint else free_mean
        latest={};evaluations=[]
        def trial(guess):
            self.state.rubber_concentration=initial_rubber
            self.state.metabolites=dict(initial_medium)
            self._prepare_oxygen(kla,rates,guess)
            self._uptake_limits=compute_uptake_limits(self.state.metabolites,self._integration_biomass,self.exchange_reactions,self.original_bounds,self.dt,self.max_uptake_rate)
            for name in self.models:self.set_uptake_constraints(name,self.state.metabolites,kla)
            solutions={n:self.solve_fba(n) for n in self.models}
            self.accounting_audit['oxygen_trial_solves']+=1
            if any(s is None for s in solutions.values()):raise RuntimeError('LP failed; numerical failure is not a biological death rate')
            cellular=sum(-s.fluxes.get(self.exchange_reactions[n].get('o2_e'),0.)*self._integration_biomass[n]*self._cell_scale for n,s in solutions.items())
            our=cellular+self._oxygen_context['polymer']/self.dt
            actual_mean=free_mean-our*mean_response
            target=free-our*response if endpoint else actual_mean
            latest.update(solutions=solutions,guess=guess,residual=guess-target,actual_mean=actual_mean)
            evaluations.append((guess,guess-target))
            return guess-target
        if upper_do<=1e-12:trial(0.)
        else:
            root=None
            # Checked warm secant predictions save LPs after the fast oxygen
            # transient. They are accepted only by the same physical residual
            # gate; failures fall back to a bracketed solve.
            if self._oxygen_last_guess is not None:
                guess=min(upper_do,max(0.,self._oxygen_last_guess));slope=self._oxygen_last_slope
                for _ in range(3):
                    value=trial(guess)
                    if abs(value)<=1e-8:root=guess;break
                    candidate=guess-value/slope
                    if not 0<=candidate<=upper_do or abs(candidate-guess)<1e-12:break
                    if len(evaluations)>=2:
                        a,fa=evaluations[-2];b,fb=evaluations[-1]
                        if abs(b-a)>1e-12 and (fb-fa)/(b-a)>0:slope=(fb-fa)/(b-a)
                    guess=candidate
            if root is None:
                lower=trial(0.);upper=trial(upper_do)
                if lower>1e-8 or upper < -1e-8:raise RuntimeError('oxygen coupling is not bracketed; inspect oxygen-producing fluxes')
                if abs(lower)<1e-9:root=0.
                elif abs(upper)<1e-9:root=upper_do
                else:root=brentq(trial,0.,upper_do,xtol=1e-11,rtol=1e-9,maxiter=64)
            # Restore the exact accepted trial, including generated polymer
            # pools, constraints and diagnostics; exploratory trials do not
            # advance time or commit their state.
            if latest['guess']!=root:trial(root)
        residual=abs(latest['residual'])
        self.accounting_audit['max_oxygen_kinetic_residual']=max(self.accounting_audit['max_oxygen_kinetic_residual'],residual)
        if not endpoint:self.accounting_audit['max_oxygen_mean_residual']=max(self.accounting_audit['max_oxygen_mean_residual'],residual)
        if residual>1e-6:raise RuntimeError('oxygen consumption/concentration did not converge')
        self._oxygen_last_guess=latest['guess']
        self._oxygen_last_mean=latest['actual_mean']
        nearby=sorted(evaluations,key=lambda item:abs(item[0]-latest['guess']))
        for guess,value in nearby:
            delta=guess-latest['guess']
            if abs(delta)>1e-9:
                slope=(value-latest['residual'])/delta
                if math.isfinite(slope) and .1<=slope<=1e5:self._oxygen_last_slope=slope;break
        return latest['solutions']

    def set_uptake_constraints(self,name,medium,dynamic_kla=50.):
        model=self.models[name]
        # Bound changes are deterministic and never silently swallowed.
        for rxn in model.exchanges:rxn.lower_bound=0.
        for met,rid in self.exchange_reactions[name].items():
            reaction=model.reactions.get_by_id(rid)
            if met in {'pha_c','phv_c','rubber_e'}:continue
            if met in {'h2o_e','h_e'}:
                lower=max(self.original_bounds[name].get(rid,(0,0))[0],-1000. if met=='h2o_e' else -.1)
            elif met=='o2_e':
                amount=self._oxygen_context['cellular_amounts'].get(name,0.)
                denom=self._integration_biomass[name]*self.dt*self._cell_scale
                lower=-amount/denom if denom>0 else 0.
            else:lower=-self._uptake_limits[name].get(met,0.)
            reaction.lower_bound=min(reaction.upper_bound,lower)
        self._enforce_polymer_boundary_separation()

    def _certified_linprog(self,name,stage,c,*,A_eq,b_eq,bounds,A_ub=None,b_ub=None,primary=None,growth_floor=None):
        """Retry the identical LP, never relax its original-scale certificate."""
        self.accounting_audit['lp_requests']+=1
        numeric_bounds=np.array([[-np.inf if lo is None else lo,np.inf if hi is None else hi]
                                 for lo,hi in bounds],dtype=float)
        digest=None
        override=getattr(self,'_lp_algorithm_override',None)
        methods=([override]*3 if override is not None else
                 [('highs',True,1e-9),('highs-ds',False,1e-10),('highs-ipm',False,1e-10)])
        for attempt,(method,presolve,tol) in enumerate(methods):
            self.solve_attempts+=1
            if attempt:self.accounting_audit['lp_retry_attempts']+=1
            result=linprog(c,A_eq=A_eq,b_eq=b_eq,A_ub=A_ub,b_ub=b_ub,bounds=bounds,method=method,
                options={'presolve':presolve,'primal_feasibility_tolerance':tol,
                         'dual_feasibility_tolerance':tol,'ipm_optimality_tolerance':1e-12})
            metrics=dict(status=int(result.status),method=method,presolve=presolve,tolerance=tol)
            finite_solution=result.success and all(np.all(np.isfinite(v)) for v in
                [result.x,result.eqlin.marginals,result.lower.marginals,result.upper.marginals,result.ineqlin.marginals])
            if result.success and not finite_solution:metrics['nonfinite_primal_or_dual']=True
            if finite_solution:
                eq=np.abs(A_eq@result.x-b_eq)
                inequalities=A_ub@result.x-b_ub if A_ub is not None else np.array([0.])
                metrics.update(equality=float(np.max(eq,initial=0.)),equality_row=int(np.argmax(eq)) if len(eq) else None,
                    lower_bound=float(np.max(numeric_bounds[:,0]-result.x,initial=0.)),
                    upper_bound=float(np.max(result.x-numeric_bounds[:,1],initial=0.)),
                    inequality=float(np.max(inequalities,initial=0.)),objective=float(np.asarray(c)@result.x))
                stationarity=np.asarray(c)-A_eq.T@result.eqlin.marginals-result.lower.marginals-result.upper.marginals
                dual=float(np.asarray(b_eq)@result.eqlin.marginals)
                infinite_bound_marginal=0.
                for side,marginal in [(0,result.lower.marginals),(1,result.upper.marginals)]:
                    finite=np.isfinite(numeric_bounds[:,side]);dual+=float(numeric_bounds[finite,side]@marginal[finite])
                    infinite_bound_marginal=max(infinite_bound_marginal,float(np.max(np.abs(marginal[~finite]),initial=0.)))
                sign=max(float(np.max(-result.lower.marginals,initial=0.)),float(np.max(result.upper.marginals,initial=0.)))
                if A_ub is not None:
                    stationarity-=A_ub.T@result.ineqlin.marginals;dual+=float(np.asarray(b_ub)@result.ineqlin.marginals)
                    sign=max(sign,float(np.max(result.ineqlin.marginals,initial=0.)))
                metrics['dual_residual']=max(sign,infinite_bound_marginal,float(np.max(np.abs(stationarity),initial=0.)))
                metrics['relative_duality_gap']=abs(metrics['objective']-dual)/(1+abs(metrics['objective']))
                residual=max(metrics[k] for k in ['equality','lower_bound','upper_bound','inequality'])
                metrics['primary_loss']=float(primary[1]-primary[0]@result.x) if primary is not None else 0.
                metrics['growth_floor_shortfall_h_inv']=float(growth_floor[1]-result.x[growth_floor[0]]) if growth_floor is not None else 0.
                certificate_finite=all(math.isfinite(v) for v in metrics.values() if isinstance(v,float))
                if (certificate_finite and residual<=1e-7 and metrics['dual_residual']<=1e-7 and metrics['relative_duality_gap']<=1e-7
                        and metrics['primary_loss']<=1.1e-9
                        and metrics['growth_floor_shortfall_h_inv']<=self.GROWTH_FLOOR_TOLERANCE_H_INV+1e-10):
                    self.solve_successes+=1;self.accounting_audit['lp_certified_requests']+=1
                    for target,value in [('max_lp_residual',residual),('max_dual_residual',metrics['dual_residual']),
                                         ('max_relative_duality_gap',metrics['relative_duality_gap'])]:
                        self.accounting_audit[target]=max(self.accounting_audit[target],value)
                    if override is not None and stage!='exchange_selection':
                        self._recomputed_primary_values.append(float(result.fun))
                    return result
            if digest is None:
                h=hashlib.sha256()
                for array in [np.asarray(c),np.asarray(b_eq),numeric_bounds]:h.update(np.ascontiguousarray(array,dtype=float).tobytes())
                for matrix in [A_eq,A_ub]:
                    if matrix is not None:
                        matrix=matrix.tocsr()
                        for array in [matrix.data,matrix.indices,matrix.indptr]:h.update(array.tobytes())
                if b_ub is not None:h.update(np.ascontiguousarray(b_ub,dtype=float).tobytes())
                digest=h.hexdigest()
            metrics={k:(v if not isinstance(v,float) or math.isfinite(v) else None) for k,v in metrics.items()}
            self.accounting_audit['rejected_lp_trials'].append(dict(metrics,species=name,stage=stage,
                time=self.state.time,attempt=attempt,lp_sha256=digest))
        return None

    def _solve_lp(self,name,objective,*,growth_lower=None,storage=False,select_fluxes=True):
        # A primary value within the general LP certificate may still exceed
        # the true optimum by more than the tighter lexicographic precision.
        # Recompute the primary only after secondary selection has failed.
        kwargs=dict(growth_lower=growth_lower,storage=storage,select_fluxes=select_fluxes)
        before_requests=self.accounting_audit['lp_requests']
        before_certified=self.accounting_audit['lp_certified_requests']
        solution=self._solve_lp_once(name,objective,**kwargs)
        if solution is not None:return solution
        if self.accounting_audit['rejected_lp_trials'][-1]['stage']!='exchange_selection':return None
        history=[]
        for method in ['highs-ds','highs-ipm']:
            self._lp_algorithm_override=(method,False,1e-10)
            self._recomputed_primary_values=[]
            try:
                solution=self._solve_lp_once(name,objective,**kwargs)
                values=list(self._recomputed_primary_values)
            finally:
                self._lp_algorithm_override=None
                self._recomputed_primary_values=[]
            history.append(dict(method=method,primary_minimization_values=values,recovered=solution is not None))
            if solution is not None:
                abandoned=(self.accounting_audit['lp_requests']-before_requests)-(self.accounting_audit['lp_certified_requests']-before_certified)
                self.accounting_audit['superseded_uncertified_requests']=self.accounting_audit.get('superseded_uncertified_requests',0)+abandoned
                self.accounting_audit.setdefault('primary_recomputations',[]).append(dict(species=name,time=self.state.time,history=history,superseded_requests=abandoned))
                return solution
        return None

    def _solve_lp_once(self,name,objective,*,growth_lower=None,storage=False,select_fluxes=True):
        model=self.models[name];matrix,ids=self._lp_templates[name]
        if ids!=[r.id for r in model.reactions]:raise RuntimeError('model structure changed after reference initialization')
        idx={rid:i for i,rid in enumerate(ids)}
        bounds=np.array([r.bounds for r in model.reactions],dtype=float)
        growth=idx[self.growth_reactions[name]]
        if growth_lower is not None:
            # A certified numerical optimum can still contain a tiny positive
            # growth value at true zero growth. Preserve the requested floor
            # to an explicitly declared absolute precision in h^-1,
            # rather than making the next LP spuriously infeasible.
            bounds[growth,0]=max(bounds[growth,0],0.,growth_lower-self.GROWTH_FLOOR_TOLERANCE_H_INV)
            self.nitrogen_allocation.setdefault(name,{}).update(growth_floor_requested=growth_lower,growth_floor_applied=float(bounds[growth,0]))
        c=np.zeros(len(ids))
        for rid,weight in objective.items():c[idx[rid]]=weight
        storage_row=np.zeros(len(ids))
        state=self.state.species[name]
        for met,mass in [('pha_c',self.PHB_REPEAT_G_PER_MMOL),('phv_c',self.PHV_REPEAT_G_PER_MMOL)]:
            rid=self.exchange_reactions[name].get(met)
            if not rid:continue
            allowed=storage and (met=='pha_c' or not self.phv_requires_rubber_intermediate or any(self.state.metabolites.get(k,0.)>1e-12 for k in ['C30_oligo_e','odtd_e']))
            bounds[idx[rid],1]=max(0.,self.original_bounds[name].get(rid,(0,1000))[1]) if allowed else 0.
            storage_row[idx[rid]]=mass
        aub=None;bub=None
        if storage and np.any(storage_row):
            ratio=self.max_pha_fraction_g_gdcw/(1-self.max_pha_fraction_g_gdcw)
            current=state.phb_accumulated*self.PHB_REPEAT_G_PER_MMOL+state.phv_accumulated*self.PHV_REPEAT_G_PER_MMOL
            scale=self._integration_biomass[name]*self.dt*self._cell_scale
            storage_row[growth]-=ratio
            aub=csr_matrix(storage_row.reshape(1,-1))
            bub=np.array([max(0.,ratio*state.biomass-current)/scale if scale>0 else 0.])
        result=self._certified_linprog(name,'storage' if storage else 'growth',-c,A_eq=matrix,
            b_eq=np.zeros(matrix.shape[0]),A_ub=aub,b_ub=bub,bounds=bounds,
            growth_floor=(growth,growth_lower) if growth_lower is not None else None)
        if result is None:return None
        if self.flux_selection=='parsimonious_exchange' and select_fluxes:
            # A declared lexicographic convention, not measured fluxes: retain
            # primary growth/storage within absolute 1e-9, then minimize total
            # physical exchange magnitude. Intracellular cycles may remain.
            exchange_indices=sorted({idx[rid] for met,rid in self.exchange_reactions[name].items()
                if met not in {'h_e','h2o_e','pha_c','phv_c','rubber_e'}})
            size=len(ids);count=len(exchange_indices)
            rows=[];cols=[];values=[]
            for j,column in enumerate(exchange_indices):
                rows.extend([2*j,2*j,2*j+1,2*j+1]);cols.extend([column,size+j,column,size+j]);values.extend([1.,-1.,-1.,-1.])
            absolute=coo_matrix((values,(rows,cols)),shape=(2*count,size+count)).tocsr()
            primary=hstack([csr_matrix(-c.reshape(1,-1)),csr_matrix((1,count))],format='csr')
            blocks=[absolute,primary];rhs=[np.zeros(2*count),np.array([-float(c@result.x)+1e-9])]
            if aub is not None:blocks.append(hstack([aub,csr_matrix((aub.shape[0],count))],format='csr'));rhs.append(bub)
            selected=self._certified_linprog(name,'exchange_selection',np.r_[np.zeros(size),np.ones(count)],
                A_eq=hstack([matrix,csr_matrix((matrix.shape[0],count))],format='csr'),b_eq=np.zeros(matrix.shape[0]),
                A_ub=vstack(blocks,format='csr'),b_ub=np.concatenate(rhs),
                bounds=list(map(tuple,bounds))+[(0.,None)]*count,primary=(np.r_[c,np.zeros(count)],float(c@result.x)),
                growth_floor=(growth,growth_lower) if growth_lower is not None else None)
            if selected is None:return None
            if float(c@result.x)-float(c@selected.x[:size])>1.1e-9:raise RuntimeError('exchange selection changed primary objective')
            result.x=selected.x[:size]
        residual=max(float(np.max(np.abs(matrix@result.x),initial=0)),
                     float(np.max(bounds[:,0]-result.x,initial=0)),float(np.max(result.x-bounds[:,1],initial=0)),
                     float(np.max(aub@result.x-bub,initial=0)) if aub is not None else 0.)
        self.accounting_audit['max_lp_residual']=max(self.accounting_audit['max_lp_residual'],residual)
        if residual>1e-7:raise RuntimeError('LP residual exceeds reference tolerance')
        if growth_lower is not None:
            self.nitrogen_allocation[name].update(lp_growth_realized_h_inv=float(result.x[growth]),
                lp_growth_shortfall_h_inv=max(0.,float(growth_lower-result.x[growth])))
        return Solution(objective_value=float(c@result.x),status='optimal',fluxes=pd.Series(result.x,index=ids))

    def solve_fba(self,name):
        growth=self.growth_reactions[name]
        if 'NS21' not in name:return self._solve_lp(name,{growth:1.})
        potential=self._solve_lp(name,{growth:1.},select_fluxes=False)
        if potential is None:return None
        nh4=self.state.metabolites.get('nh4_e',0.)
        fraction=nh4/(self.nitrogen_half_saturation+nh4)
        mu=max(0.,float(potential.fluxes[growth]))
        self.nitrogen_allocation[name]=dict(nh4=nh4,growth_fraction=fraction,growth_potential=mu)
        objective={self.exchange_reactions[name][met]:mass for met,mass in
                   [('pha_c',self.PHB_REPEAT_G_PER_MMOL),('phv_c',self.PHV_REPEAT_G_PER_MMOL)] if met in self.exchange_reactions[name]}
        if not objective:return potential
        return self._solve_lp(name,objective,growth_lower=fraction*mu,storage=True)

    def _integrate(self,solutions):
        changes=defaultdict(float);protons=0.;cell_o2=0.
        for name,solution in solutions.items():
            state=self.state.species[name];x=self._integration_biomass[name]
            state.metabolite_uptake.clear();state.metabolite_secretion.clear()
            fluxes=solution.fluxes*self._cell_scale
            mu=float(fluxes[self.growth_reactions[name]])
            state.biomass=x+mu*x*self.dt;state.growth_rate=mu;state.last_inhibition_factor=self._cell_scale
            for met,rid in self.exchange_reactions[name].items():
                flux=float(fluxes.get(rid,0.));delta=flux*x*self.dt
                if flux<0:state.metabolite_uptake[met]=-flux
                elif flux>0:state.metabolite_secretion[met]=flux
                if met=='pha_c':state.phb_accumulated+=delta
                elif met=='phv_c':state.phv_accumulated+=delta
                elif met=='h_e':protons+=delta
                elif met=='o2_e':cell_o2-=delta
                elif met not in {'h2o_e','rubber_e'}:changes[met]+=delta
            state.pha_accumulated=state.phb_accumulated+state.phv_accumulated
            pha=state.phb_accumulated*self.PHB_REPEAT_G_PER_MMOL+state.phv_accumulated*self.PHV_REPEAT_G_PER_MMOL
            ceiling=state.biomass*self.max_pha_fraction_g_gdcw/(1-self.max_pha_fraction_g_gdcw)
            excess=max(0.,pha-ceiling);self.accounting_audit['max_storage_excess_g_l']=max(self.accounting_audit['max_storage_excess_g_l'],excess)
            if excess>1e-7:raise RuntimeError('storage capacity violated after constrained LP')
        for met,delta in changes.items():
            raw=self.state.metabolites.get(met,0.)+delta
            if raw < -1e-8:raise RuntimeError(f'negative shared inventory {met}: {raw}')
            if raw<0:
                ledger=self.accounting_audit['roundoff_added_mmol_l'];ledger[met]=ledger.get(met,0.)-raw
            self.state.metabolites[met]=max(0.,raw)
        self.cumulative_co2_emission+=changes.get('co2_e',0.)
        ctx=self._oxygen_context;withdrawal=ctx['polymer']+cell_o2
        end=ctx['free']-withdrawal/self.dt*ctx['response']
        if end < -1e-8:raise RuntimeError('oxygen inventory violated')
        end=max(0.,end);transfer=end-ctx['c0']+withdrawal
        self.state.metabolites['o2_e']=end
        self.oxygen_audit['transferred']+=transfer;self.oxygen_audit['polymer_consumed']+=ctx['polymer'];self.oxygen_audit['cellular_consumed']+=cell_o2
        self.oxygen_audit['max_balance_error']=max(self.oxygen_audit['max_balance_error'],abs(end-ctx['c0']-transfer+withdrawal))
        self.buffer_base-=protons;self.buffer_acid+=protons
        if self.ph_control_target is not None:self._apply_ph_stat()
        else:
            if self.buffer_base<=0 or self.buffer_acid<=0:raise RuntimeError('buffer model exhausted; acid/base equilibrium model required')
            ph=self.pKa+math.log10(self.buffer_base/self.buffer_acid)
            self.state.metabolites['h_e']=10**(3-ph)

        addition=self._oxygen_context.get('c30_roundoff_added',0.)
        if addition:
            ledger=self.accounting_audit['roundoff_added_mmol_l']
            ledger['C30_oligo_e']=ledger.get('C30_oligo_e',0.)+addition

    def step(self,rubber_degradation_rates,nutrient_supplementation,dynamic_kla=50.,*,feed_rates_mmol_l_h=None):
        if self._invalid:raise RuntimeError('failed reference state cannot be reused; reset required')
        if set(self.models)!=set(self._structure_contract) or any(self._structure_token(m)!=self._structure_contract[n] for n,m in self.models.items()):
            raise RuntimeError('GEM structure/objective changed after initialization; construct a new reference')
        if any(not r.lower_bound<=0<=r.upper_bound for m in self.models.values() for r in m.reactions):
            raise ValueError('required nonzero internal flux bounds need an adapter for density/pH scaling')
        control_dt=float(self.dt);rates=dict(feed_rates_mmol_l_h or {});nutrients=dict(nutrient_supplementation or {})
        if not math.isfinite(control_dt) or control_dt<=0 or not math.isfinite(dynamic_kla) or dynamic_kla<0:raise ValueError('invalid time/kLa')
        if any(not math.isfinite(v) or v<0 for v in list(rates.values())+list(nutrients.values())+list((rubber_degradation_rates or {}).values())):raise ValueError('finite nonnegative inputs required')
        if set(nutrients)-set(self.DOSE_ALIASES)-{'coexistence_feed_rate'}:raise ValueError('unsupported nutrient dose; declare molecular feed rates')
        if set(rates)&{'h_e','h2o_e','o2_e','pha_c','phb_c','phv_c','rubber_e','rubber_bulk_e','M_rubber_bulk_e'}:
            raise ValueError('use physical pH/water/gas/solid interfaces; intracellular storage cannot be fed')
        known=set(self.state.metabolites)|{m for d in self.exchange_reactions.values() for m in d}
        if set(rates)-known:raise ValueError('unknown feed metabolite')
        if set(rubber_degradation_rates or {})-set(self.DEFAULT_POLYMER_RATES)-{'OR16','NS21'}:raise ValueError('unknown polymer rate')
        self.last_defined_feed_g_l=0.;first=True;end=self.state.time+control_dt
        try:
            while self.state.time<end-1e-12:
                begin=time.perf_counter();self.dt=min(self.max_internal_dt,end-self.state.time)
                for met,rate in rates.items():
                    dose=rate*self.dt;self._add_feed(met,dose)
                    self.cumulative_continuous_feed[met]=self.cumulative_continuous_feed.get(met,0.)+dose
                if first:
                    for alias,met in self.DOSE_ALIASES.items():self._add_feed(met,nutrients.get(alias,0.))
                for met,rate in COEXISTENCE_DEFINED_FEED_MMOL_L_H.items():
                    dose=nutrients.get('coexistence_feed_rate',0.)*rate*self.dt;self._add_feed(met,dose)
                    mass=dose*self.DEFINED_MW[met]/1000.;self.cumulative_defined_feed_g_l+=mass;self.last_defined_feed_g_l+=mass
                self._integration_biomass={n:float(s.biomass) for n,s in self.state.species.items()}
                total=sum(self._integration_biomass.values())
                self._cell_scale=max(0.,1-total/self.carrying_capacity) if self.density_policy=='flux_consistent' else 1.
                # pH inhibition remains an explicit common phenomenological
                # factor; unlike legacy it scales every cellular flux.
                ph=3-math.log10(max(self.state.metabolites.get('h_e',1e-4),1e-15))
                self._cell_scale*=math.exp(-2*max(0.,abs(ph-7)-1.))
                solving=time.perf_counter();solutions=self._solve_oxygen_coupled(dynamic_kla,rubber_degradation_rates)
                solved=time.perf_counter();self.last_fba_solutions=solutions;self._integrate(solutions)
                self.state.time+=self.dt;self.current_step+=1;self._log_telemetry()
                self.last_step_timing=dict(pre_solve_seconds=solving-begin,solve_seconds=solved-solving,post_solve_seconds=time.perf_counter()-solved,total_seconds=time.perf_counter()-begin)
                for key,value in self.last_step_timing.items():self.step_timing_totals[key]+=value
                first=False
            self.state.time=end;self.controller_steps+=1
        except Exception:
            self._invalid=True;self.accounting_audit['failed_steps']+=1
            raise
        finally:self.dt=control_dt;self._oxygen_context=None
        return self.state

    def get_solver_diagnostics(self):
        diagnostics=super().get_solver_diagnostics()
        diagnostics['audited_cultivation']=dict(version=self.NUMERICS_VERSION,valid=not self._invalid,
            numerical_repair_revision=self.NUMERICAL_REPAIR_REVISION,
            accounting=copy.deepcopy(self.accounting_audit),density_policy=self.density_policy,
            flux_selection=self.flux_selection,primary_objective_tolerance=1e-9,
            growth_allocation_tolerance_h_inv=self.GROWTH_FLOOR_TOLERANCE_H_INV,
            oxygen_scheme=self.oxygen_scheme,
            oxygen_allocation=('shared DO with actual joint OUR; no unused uptake reservation' if self.oxygen_scheme=='shared_endpoint_v3' else 'proportional kinetic-demand quotas with analytic total inventory'),
            oxygen_kinetics=('implicit endpoint kinetics with analytic constant-OUR transfer; refinement required' if self.oxygen_scheme=='shared_endpoint_v3' else 'implicit interval-mean DO; refinement required'),
            mean_residual_applicable=self.oxygen_scheme=='quota_mean_v2',
            polymer_oxygen_half_saturation=self.polymer_oxygen_half_saturation,
            buffer_mmol_l=self.configured_buffer_mmol_l,growth_reactions=self.growth_reactions.copy(),
            cumulative_delivered_mmol_l=self.cumulative_delivered_mmol_l.copy(),
            biological_parameters_calibrated=False,stable_coexistence_validated=False,
            limitations=['NH4-only allocation ignores alternative nitrogen sufficiency',
                         'maintenance, death, PHA remobilization and strain-specific inhibition uncalibrated',
                         'Pf oxygen phenotype and B12 receiver dependence not validated',
                         'exchange parsimony is a solution convention, not measured or unique metabolism',
                         'GEM molecular-formula coverage incomplete; no complete elemental closure claim'])
        if self.oxygen_scheme=='shared_endpoint_v3':
            diagnostics['audited_cultivation']['accounting']['max_oxygen_mean_residual']=None
        return diagnostics
