"""Resource-integral invariance for the predeclared paired search."""
import pytest
from scripts.analysis.compare_equal_budget_20260908 import PROFILES,KLAS,rates,case

@pytest.mark.parametrize('dt',[.25,.025,.0125])
def test_each_profile_delivers_same_budget_at_each_clock(dt):
    for profile in PROFILES:
        delivered={'lac__L_e':0.,'nh4_e':0.}
        for i in range(round(24/dt)):
            r=rates(i*dt,profile)
            assert 0<=r['lac__L_e']<=.5 and 0<=r['nh4_e']<=.1
            for k in delivered:delivered[k]+=r[k]*dt
        assert delivered['lac__L_e']==pytest.approx(6.,abs=1e-9)
        assert delivered['nh4_e']==pytest.approx(.4,abs=1e-9)

def test_symmetric_candidate_space():
    for k in KLAS:
        a=[case(k,p,'two','screen') for p in PROFILES]
        b=[case(k,p,'three','screen') for p in PROFILES]
        assert len(a)==len(b)==6
        assert len({c['id'] for c in a+b})==12
        for x,y in zip(a,b):
            assert {k:v for k,v in x.items() if k not in ['id','arm']}=={k:v for k,v in y.items() if k not in ['id','arm']}
