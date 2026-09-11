from types import SimpleNamespace

import numpy as np

from scripts.benchmark_gpu_lexicographic import ppo_telemetry


def test_raw_reward_repeat_amount_and_endpoint_flags_are_not_mass_or_branch_claims():
    state=SimpleNamespace(species={'a':SimpleNamespace(pha_accumulated=1.,biomass=.009),
                                  'b':SimpleNamespace(pha_accumulated=2.,biomass=.019)},
                          metabolites={'nh4_e':.1,'o2_e':0.,'C30_oligo_e':1e-12})
    r=ppo_telemetry(state,np.array([.2,.3]),-3.,np.array([.5]*5),False,True,{'ph':7.,'do':0.})
    assert r['pha_repeat_mmol_l']==3. and r['reward_raw']==-3.
    assert 'pha' not in r
    assert r['truncated'] and not r['terminated']
    flags=r['state_thresholds_after_step']
    assert not flags['all_biomass_below_001'] and flags['any_biomass_below_002']
    assert not flags['nh4_below_01'] and not flags['rubber_intermediate_present']
    assert flags['oxygen_at_or_below_zero']
