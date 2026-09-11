"""Reproducible in-silico coverage, not experimentally calibrated ranges."""
import numpy as np


def trajectory_design(seed,index,steps):
    if type(seed) is not int or type(index) is not int or index<0 or type(steps) is not int or not 1<=steps<=120:
        raise ValueError('Nonnegative index and 1..120 steps required')
    rng=np.random.default_rng(seed)
    biomass=rng.uniform(.75,1.25,3)
    nh4=float(2.**rng.uniform(-1.,1.))
    kind=index%4
    if kind==0:actions=rng.uniform(.05,.95,(120,5))
    elif kind==1:actions=np.repeat(rng.uniform(.05,.95,(12,5)),10,axis=0)
    elif kind==2:
        actions=np.empty((120,5));actions[0]=rng.uniform(.05,.95,5)
        for t in range(1,120):actions[t]=np.clip(actions[t-1]+rng.normal(0,.06,5),.05,.95)
    else:
        low=rng.uniform(.05,.25,5);high=rng.uniform(.75,.95,5)
        actions=np.where(((np.arange(120)//15)%2)[:,None],high,low)
    return dict(schema='coverage_v1',index=index,seed=seed,
        action_pattern=('uniform','block','smooth','pulse')[kind],
        biomass_multipliers=biomass.tolist(),nh4_multiplier=nh4,
        calibration='hypothetical in-silico ranges; not measured physiology'),actions[:steps].astype(np.float32)


def apply_design(env,design):
    species=list(env.simulator.state.species.values())
    if len(species)!=3:raise ValueError('Coverage design requires the declared three-species GEM')
    for state,multiplier in zip(species,design['biomass_multipliers']):state.biomass*=multiplier
    env.simulator.state.metabolites['nh4_e']*=design['nh4_multiplier']
