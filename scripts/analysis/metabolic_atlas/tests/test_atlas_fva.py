import json,tempfile
from pathlib import Path
import fba_service as service

def run(reactions,rid,medium):
    mids={m for r in reactions for m in r['stoich']}
    d={'medium':medium,'species':[{'short':'test','metabolites':{m:{'compartment':'c'} for m in mids},'reactions':reactions,'objective':{'Growth':1},'objective_direction':'max'}]}
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'data.json';path.write_text(json.dumps(d));old=service.DATA;service.DATA=path
        try:return service.check('test',rid,1,fva=True)
        finally:service.DATA=old

def r(id,stoich,exchange=False,bounds=(0,1000)):
    return dict(id=id,stoich=stoich,exchange=exchange,bounds=bounds)
# A forced reverse conversion at the original growth optimum.
base=[r('EX_a',{'a_e':-1},True,(-10,1000)),r('reverse',{'b_c':-1,'a_e':1},bounds=(-1000,1000)),r('Growth',{'b_c':-1})]
forced=run(base,'reverse',{'a_e':1});assert forced['status']=='optimal',forced
assert forced['direction_stability']=='reverse' and forced['maximum']<0,forced
# Parallel pathway permits a formerly nonzero reaction to be zero in another optimum.
optional=run(base+[r('parallel',{'a_e':-1,'b_c':1})],'reverse',{'a_e':1})
assert optional['minimum']<0 and optional['maximum']>=-1e-6,optional
assert forced['mass_balance_residual']<1e-7
assert service.fva_direction(-1,1)=='both'
assert service.fva_direction(0,0)=='zero'
assert service.fva_direction(0,1)=='forward_or_zero'
print('PASS FVA forced reverse, alternative optimum, original objective retention and flux balance')
