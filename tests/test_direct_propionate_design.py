import pytest
from scripts.analysis.compare_direct_propionate_20260909 import study_cases,rates
@pytest.mark.parametrize('dt',[.25,.025,.0125])
def test_carbon_and_nitrogen_budgets(dt):
    for c in study_cases('screen'):
        totals={k:0. for k in ['lac__L_e','ppa_e','nh4_e']}
        for i in range(round(24/dt)):
            r=rates(i*dt,c['profile'])
            assert r['lac__L_e']+r['ppa_e']<=.5+1e-12
            for k,v in r.items():totals[k]+=v*dt
        assert 3*(totals['lac__L_e']+totals['ppa_e'])==pytest.approx(18.)
        assert totals['ppa_e']==pytest.approx(6*c['profile']['propionate_fraction'])
        assert totals['nh4_e']==pytest.approx(.4)
def test_each_arm_has_identical_timing_opportunities():
    for phase in ['screen','refine']:
        cases=study_cases(phase)
        assert len({c['id'] for c in cases})==36
        for k in [2.,10.]:
            sets=[{(c['profile']['lactate'],c['profile']['nitrogen']) for c in cases if c['kla']==k and c['arm']==arm} for arm in ['three','two_propionate','two_mixed']]
            assert sets[0]==sets[1]==sets[2] and len(sets[0])==6
