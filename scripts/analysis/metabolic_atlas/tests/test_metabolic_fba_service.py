"""Supply certificate controls: all co-substrates required, missing supply and cycles."""
import json,tempfile
from pathlib import Path
import fba_service as service

def run(medium,reactions):
 mids={m for r in reactions for m in r['stoich']}
 payload={'medium':medium,'species':[{'short':'test','metabolites':{m:{'compartment':'c'} for m in mids},'reactions':reactions}]}
 with tempfile.TemporaryDirectory() as folder:
  path=Path(folder)/'data.json';path.write_text(json.dumps(payload));old=service.DATA;service.DATA=path
  try:return service.check('test','R',1)
  finally:service.DATA=old

def r(id,stoich,exchange=False,bounds=(0,1000)):
 return dict(id=id,stoich=stoich,exchange=exchange,bounds=bounds)
reactions=[r('EX_a',{'a_e':-1},True,(-10,1000)),r('EX_b',{'b_e':-1},True,(-10,1000)),r('EX_p',{'p_e':-1},True,(0,1000)),r('R',{'a_e':-1,'b_e':-1,'p_e':1})]
yes=run({'a_e':1,'b_e':1},reactions);assert yes['status']=='feasible',yes
no=run({'a_e':1},reactions);assert no['status']=='infeasible',no
cycle=run({},[r('R',{'a_c':-1,'b_c':1}),r('back',{'b_c':-1,'a_c':1})]);assert cycle['status']!='feasible',cycle
print('PASS simultaneous supply, missing co-substrate, cycle-only rejection')
