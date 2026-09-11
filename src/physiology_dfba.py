"""Explicit, uncalibrated physiology layered on the frozen audited reference.

Maintenance is prioritized, then growth/PHA allocation. Deficit-dependent death
is a declared population-average hypothesis, never a solver-error fallback.
Dead matter remains in an inert compartment; no unmeasured lysis is invented.
"""
from __future__ import annotations

import copy
import math
import re
from collections import defaultdict
from contextlib import contextmanager

from .audited_dfba import AuditedDFBASimulator


def is_atp_hydrolysis(reaction):
    """Require one ATP hydrolysis, not a kinase or arbitrary internal drain."""
    identities={'cpd00001':'water','cpd00002':'atp','cpd00008':'adp','cpd00009':'pi','cpd00067':'h'}
    actual={}
    for met,coefficient in reaction.metabolites.items():
        if not met.formula or met.charge is None or not math.isfinite(met.charge):return False
        match=re.search(r'cpd\d{5}',met.id)
        key=identities.get(match.group()) if match else {'atp':'atp','adp':'adp','pi':'pi','h':'h','h2o':'water'}.get(met.id.split('_')[0])
        if key is None or key in actual:return False
        actual[key]=coefficient
    return actual=={'atp':-1.,'water':-1.,'adp':1.,'pi':1.,'h':1.} and not reaction.check_mass_balance()


class PhysiologyDFBASimulator(AuditedDFBASimulator):
    NUMERICS_VERSION = 'physiology_cultivation_v1'
    NUMERICAL_REPAIR_REVISION = 'physiology_validation_20260908'

    def __init__(self, models, initial_biomass, initial_metabolites, *,
                 maintenance=None, basal_death_rates=None, starvation_death_rates=None,
                 nitrogen_policy='capacity_ratio', remobilize_pha=False, **kwargs):
        self.maintenance = copy.deepcopy(maintenance or {})
        self.basal_death_rates = dict(basal_death_rates or {})
        self.starvation_death_rates = dict(starvation_death_rates or {})
        if nitrogen_policy not in {'capacity_ratio', 'nh4_legacy'}:
            raise ValueError('unknown nitrogen policy')
        if not isinstance(remobilize_pha, bool):
            raise ValueError('remobilize_pha must be boolean')
        self.nitrogen_policy = nitrogen_policy
        self.remobilize_pha = remobilize_pha
        for mapping in (self.maintenance, self.basal_death_rates, self.starvation_death_rates):
            if set(mapping)-set(models):
                raise ValueError('unknown physiology species')
        for mapping in (self.basal_death_rates, self.starvation_death_rates):
            if any(not math.isfinite(v) or v < 0 for v in mapping.values()):
                raise ValueError('death rates must be nonnegative finite h^-1')
        if any(v > 0 and n not in self.maintenance for n,v in self.starvation_death_rates.items()):
            raise ValueError('starvation death needs an explicit maintenance requirement')
        working = {n:m.copy() for n,m in models.items()}
        from .b12_evidence import inspect_and_curate
        working,self.b12_evidence=inspect_and_curate(working)
        self.maintenance_original_bounds = {}
        for name, spec in self.maintenance.items():
            if set(spec) != {'reaction', 'rate_mmol_g_h'}:
                raise ValueError('maintenance needs reaction and rate_mmol_g_h')
            rate = spec['rate_mmol_g_h']
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError('maintenance rate must be positive finite')
            reaction = working[name].reactions.get_by_id(spec['reaction'])
            if not is_atp_hydrolysis(reaction):
                raise ValueError('maintenance needs a balanced unit ATP hydrolysis reaction')
            if reaction.boundary or reaction.lower_bound < 0 or reaction.upper_bound < rate:
                raise ValueError('maintenance must be a forward internal reaction with sufficient capacity')
            if reaction.lower_bound > rate:
                raise ValueError('configured maintenance cannot silently lower a required GEM bound')
            self.maintenance_original_bounds[name] = tuple(reaction.bounds)
            # The base integrator requires zero-containing bounds between LPs.
            # This declared requirement is enforced in physical units in every
            # accepted LP below, rather than lost to the common flux multiplier.
            reaction.lower_bound = 0.
        super().__init__(working, initial_biomass, initial_metabolites, **kwargs)
        self.pool_elements, self.formula_gaps = self._element_map()
        self.nitrogen_relief_pools = {}
        for name in self.models:
            self.nitrogen_relief_pools[name] = [p for p,rid in self.exchange_reactions[name].items()
                if p=='nh4_e' and self.pool_elements.get(p,{}).get('N',0)==1
                and self.pool_elements[p].get('C',0)==0
                and self.original_bounds[name][rid][0]<0]
            if nitrogen_policy == 'capacity_ratio' and 'NS21' in name and not self.nitrogen_relief_pools[name]:
                raise ValueError('capacity ratio needs an identified ammonium uptake route')
        self._initialize_physiology()
        self._physiology_contract_at_init=self.physiology_contract()

    def step(self,*args,**kwargs):
        if self.physiology_contract()!=self._physiology_contract_at_init:
            raise RuntimeError('physiology assumptions changed; construct a new reference')
        return super().step(*args,**kwargs)

    def _element_map(self):
        candidates = defaultdict(list)
        for name, model in self.models.items():
            for pool,rid in self.exchange_reactions[name].items():
                met = next(iter(model.reactions.get_by_id(rid).metabolites))
                # Unknown / pseudo elements are not interpreted as zero C or N.
                try:
                    elements = met.elements if met.formula else None
                except (ValueError, TypeError):
                    elements = None
                if elements and elements.get('C',0)==0 and re.search(r'toluene|benzene',met.name,re.I):
                    elements=None  # Known organic names with a truncated formula (e.g. TNT=NO2).
                if elements and not (set(elements)-{'C','H','O','N','P','S','Co','Fe','Mg','Ca','K','Na','Cl','Zn','Mn','Cu','Ni','Mo','Se','F','Br','I'}):
                    candidates[pool].append({k:float(elements.get(k,0)) for k in ('C','N')})
                else:
                    candidates[pool].append(None)
        result = {}; gaps = {}
        for pool, entries in candidates.items():
            known = [e for e in entries if e is not None]
            if known and all(e==known[0] for e in known):
                result[pool] = known[0]
                if len(known)!=len(entries):gaps[pool]='some GEM formulas missing; consistent known formula used'
            else:gaps[pool]='missing or conflicting formula; excluded from known-element totals'
        for pool in self.state.metabolites:
            if pool not in result:gaps.setdefault(pool,'no supported exchange formula')
        return result,gaps

    def _initialize_physiology(self):
        self.physiology_trials = {}
        self.dead_matter = {n:dict(biomass_g_l=0.,phb_mmol_l=0.,phv_mmol_l=0.) for n in self.models}
        self.element_exchanges = {n:{direction:defaultdict(float) for direction in ('uptake','secretion')} for n in self.models}
        self.physiology_totals = {n:dict(maintenance_requested_mmol_l=0.,maintenance_used_mmol_l=0.,
            maintenance_deficit_mmol_l=0.,phb_remobilized_mmol_l=0.,phv_remobilized_mmol_l=0.) for n in self.models}
        self._physiology_initial_medium = dict(self.state.metabolites)
        self.physical_events=[]
        self.polymer_pool_changes=defaultdict(float)
        self.physical_pool_changes=defaultdict(float)

    def record_volume_addition(self, old_volume, new_volume, additions):
        dilution=old_volume/new_volume
        for pools in self.dead_matter.values():
            for key in pools:pools[key]*=dilution
        for pool,current in self.state.metabolites.items():
            previous=(current*new_volume-additions.get(pool,0.))/old_volume
            self.physical_pool_changes[pool]+=current-previous
        self.physical_events.append(dict(time_h=self.state.time,kind='addition',
            old_volume_l=old_volume,new_volume_l=new_volume,added_mmol=dict(additions)))

    def record_sample(self, sample_l):
        self.physical_events.append(dict(time_h=self.state.time,kind='sample',sample_l=sample_l,
            removed_mmol={p:v*sample_l for p,v in self.state.metabolites.items()},
            dead_matter_removed={n:{p.removesuffix('_l'):v*sample_l for p,v in pools.items()} for n,pools in self.dead_matter.items()}))

    def physiology_contract(self):
        return copy.deepcopy(dict(version=self.NUMERICS_VERSION,maintenance=self.maintenance,
            maintenance_original_bounds=self.maintenance_original_bounds,
            basal_death_rates=self.basal_death_rates,starvation_death_rates=self.starvation_death_rates,
            nitrogen_policy=self.nitrogen_policy,nitrogen_relief_pools=self.nitrogen_relief_pools,
            remobilize_pha=self.remobilize_pha,pool_elements=self.pool_elements,
            b12_evidence=self.b12_evidence,
            death_model='post-growth exponential survival; inert dead biomass and storage; no lysis'))

    def reset(self):
        result=super().reset()
        self._initialize_physiology()
        return result

    @contextmanager
    def _temporary_bounds(self, name, changes):
        reactions=self.models[name].reactions
        saved={rid:reactions.get_by_id(rid).bounds for rid in changes}
        try:
            for rid,bounds in changes.items():reactions.get_by_id(rid).bounds=bounds
            yield
        finally:
            for rid,bounds in saved.items():reactions.get_by_id(rid).bounds=bounds

    def _storage_import_bounds(self, name):
        state=self.state.species[name]
        denominator=self._integration_biomass[name]*self.dt*self._cell_scale
        changes={}
        for pool,amount in [('pha_c',state.phb_accumulated),('phv_c',state.phv_accumulated)]:
            rid=self.exchange_reactions[name].get(pool)
            if rid:
                # Only existing biochemical degradation routes can use this
                # finite intracellular source. No depolymerase is added here.
                limit=max(0.,amount-1e-10)/denominator if self.remobilize_pha and denominator>0 else 0.
                changes[rid]=(-limit,self.models[name].reactions.get_by_id(rid).upper_bound)
        return changes

    def solve_fba(self, name):
        growth=self.growth_reactions[name]
        requested=self.maintenance.get(name,{}).get('rate_mmol_g_h',0.)
        maintenance_bounds={}
        with self._temporary_bounds(name,self._storage_import_bounds(name)):
            if requested and self._cell_scale>0:
                rid=self.maintenance[name]['reaction']
                upper=self.maintenance_original_bounds[name][1]
                with self._temporary_bounds(name,{rid:(0.,min(upper,requested/self._cell_scale))}):
                    potential=self._solve_lp(name,{rid:1.},select_fluxes=False)
                if potential is None:return None
                attainable=max(0.,float(potential.fluxes[rid]))
                # Preserve the certified attainable requirement to the same
                # declared absolute precision as growth allocation.
                maintenance_bounds[rid]=(max(0.,attainable-1e-9),min(upper,requested/self._cell_scale))
            elif requested:
                maintenance_bounds[self.maintenance[name]['reaction']]=(0.,0.)
            with self._temporary_bounds(name,maintenance_bounds):
                solution=self._solve_allocation(name,growth)
            if solution is not None:
                used=max(0.,float(solution.fluxes[self.maintenance[name]['reaction']])*self._cell_scale) if requested else 0.
                self.physiology_trials[name]=dict(requested=requested,used=used,
                    deficit_fraction=max(0.,1-used/requested) if requested else 0.)
            return solution

    def _solve_allocation(self,name,growth):
        if 'NS21' not in name:return self._solve_lp(name,{growth:1.})
        if self.nitrogen_policy=='nh4_legacy':return super().solve_fba(name)
        actual=self._solve_lp(name,{growth:1.},select_fluxes=False)
        if actual is None:return None
        changes={}
        for pool in self.nitrogen_relief_pools[name]:
            rid=self.exchange_reactions[name][pool]
            reaction=self.models[name].reactions.get_by_id(rid)
            changes[rid]=(-min(self.max_uptake_rate,-self.original_bounds[name][rid][0]),reaction.upper_bound)
        # Hypothetical reference only: relax ammonium uptake. A formula-only
        # classifier can mistake incompletely annotated organic N pools for
        # inorganic N, or add alternative electron acceptors. All carbon,
        # oxygen, maintenance and other constraints stay fixed. Its fluxes are
        # never integrated or recorded as real consumption.
        with self._temporary_bounds(name,changes):
            relieved=self._solve_lp(name,{growth:1.},select_fluxes=False)
        if relieved is None:return None
        mu=max(0.,float(actual.fluxes[growth]));capacity=max(0.,float(relieved.fluxes[growth]))
        fraction=min(1.,mu/capacity) if capacity>1e-9 else 0.
        self.nitrogen_allocation[name]=dict(policy=self.nitrogen_policy,growth_potential=mu,
            nitrogen_relieved_growth=capacity,growth_fraction=fraction,
            relief_pools=self.nitrogen_relief_pools[name],reference_fluxes_integrated=False)
        objective={self.exchange_reactions[name][p]:mass for p,mass in
            [('pha_c',self.PHB_REPEAT_G_PER_MMOL),('phv_c',self.PHV_REPEAT_G_PER_MMOL)] if p in self.exchange_reactions[name]}
        return self._solve_lp(name,objective,growth_lower=fraction*mu,storage=True) if objective else actual

    def _integrate(self,solutions):
        super()._integrate(solutions)
        for pool,amount in self._polymer_pool_trial.items():self.polymer_pool_changes[pool]+=amount
        for name, solution in solutions.items():
            x=self._integration_biomass[name];state=self.state.species[name]
            trial=self.physiology_trials[name];totals=self.physiology_totals[name]
            totals['maintenance_requested_mmol_l']+=trial['requested']*x*self.dt
            totals['maintenance_used_mmol_l']+=trial['used']*x*self.dt
            totals['maintenance_deficit_mmol_l']+=max(0.,trial['requested']-trial['used'])*x*self.dt
            for pool,rid in self.exchange_reactions[name].items():
                amount=float(solution.fluxes[rid])*self._cell_scale*x*self.dt
                if pool in ('pha_c','phv_c'):
                    totals[('phb' if pool=='pha_c' else 'phv')+'_remobilized_mmol_l']+=max(0.,-amount)
                    continue
                if pool in ('h_e','h2o_e','rubber_e'):continue
                direction='uptake' if amount<0 else 'secretion'
                self.element_exchanges[name][direction][pool]+=abs(amount)
            rate=self.basal_death_rates.get(name,0.)+self.starvation_death_rates.get(name,0.)*trial['deficit_fraction']
            loss=-math.expm1(-rate*self.dt)
            for attribute,key in [('biomass','biomass_g_l'),('phb_accumulated','phb_mmol_l'),('phv_accumulated','phv_mmol_l')]:
                amount=getattr(state,attribute)
                self.dead_matter[name][key]+=amount*loss
                setattr(state,attribute,amount*(1-loss))
            state.pha_accumulated=state.phb_accumulated+state.phv_accumulated
            state.growth_rate=(state.biomass/x-1)/self.dt if x>0 else 0.

    def _prepare_oxygen(self,*args,**kwargs):
        before=dict(self.state.metabolites)
        super()._prepare_oxygen(*args,**kwargs)
        self._polymer_pool_trial={p:self.state.metabolites.get(p,0.)-before.get(p,0.)
            for p in set(before)|set(self.state.metabolites) if p not in {'o2_e','h_e','h2o_e'}}

    def element_accounting(self):
        def inventory(medium):
            return {e:sum(v*self.pool_elements.get(p,{}).get(e,0.) for p,v in medium.items()) for e in ('C','N')}
        exchanges={n:{d:inventory(pools) for d,pools in directions.items()} for n,directions in self.element_exchanges.items()}
        initial=inventory(self._physiology_initial_medium);current=inventory(self.state.metabolites)
        delivered=inventory(self.cumulative_delivered_mmol_l)
        polymer=inventory(self.polymer_pool_changes);physical=inventory(self.physical_pool_changes)
        # C30 correction is already in the accepted extracellular reaction
        # delta; the other small negative-pool corrections happen in integrate.
        rounding=inventory({p:v for p,v in self.accounting_audit['roundoff_added_mmol_l'].items() if p!='C30_oligo_e'})
        residual={e:current[e]-initial[e]-delivered[e]-polymer[e]-physical[e]-rounding[e]
            -sum(d['secretion'][e]-d['uptake'][e] for d in exchanges.values()) for e in ('C','N')}
        return dict(units='mmol element/L; molecular exchange ledger in mmol/L',
            initial_known=initial,current_known=current,delivered_known=delivered,species_known=exchanges,
            extracellular_polymer_changes_known=polymer,physical_changes_known=physical,
            roundoff_known=rounding,known_medium_closure_residual=residual,
            molecular_exchanges=copy.deepcopy(self.element_exchanges),formula_gaps=self.formula_gaps,
            physical_events=copy.deepcopy(self.physical_events),
            concentration_ledger_requires_volume_events=bool(self.physical_events),
            unclassified_current={p:v for p,v in self.state.metabolites.items() if p not in self.pool_elements and v!=0},
            unclassified_delivered={p:v for p,v in self.cumulative_delivered_mmol_l.items() if p not in self.pool_elements and v!=0},
            complete_element_balance_claim=False,pha_carbon_origin_identified=False)

    def get_solver_diagnostics(self):
        result=super().get_solver_diagnostics()
        result['physiology']=dict(contract=self.physiology_contract(),last_trials=copy.deepcopy(self.physiology_trials),
            cumulative=copy.deepcopy(self.physiology_totals),dead_matter=copy.deepcopy(self.dead_matter),
            elements=self.element_accounting(),parameters_calibrated=False)
        result['audited_cultivation']['limitations']=[
            'nitrogen capacity ratio is an allocation hypothesis, not measured regulation',
            'maintenance and death parameters uncalibrated; dead matter is inert with no lysis',
            'PHA reuse requires an existing open biochemical degradation route',
            'Pf oxygen phenotype and B12 receiver dependence not validated',
            'incomplete GEM formulas; no complete elemental balance or PHA carbon-origin claim']
        return result
