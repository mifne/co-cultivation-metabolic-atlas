"""Predeclared controller invariance and internal-step refinement comparisons."""
from pathlib import Path
import subprocess,sys,json
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/resolved_dynamics_20260908';OUT.mkdir(exist_ok=False)
cases=[dict(id=f'three_h{h}_control1',arm='three',internal=h,control=1.) for h in [.05,.025,.0125]]
cases += [dict(id='three_h0.025_control0.2',arm='three',internal=.025,control=.2),dict(id='two_h0.025_control1',arm='two_equal_total',internal=.025,control=1.)]
(OUT/'design.json').write_text(json.dumps(dict(cases=cases,hours=12,feed_rate=.25,nh4=.05,kla=50,nitrogen_half_saturation=.1,checks='Controller partition invariance; internal-step convergence assessed separately; no three-species positive selection'),indent=2))
def run(c):
    print('START',c['id'],flush=True)
    with (OUT/(c['id']+'.log')).open('w') as log:
        p=subprocess.run([sys.executable,str(ROOT/'scripts/simulate_resolved_dfba.py'),'--output',str(OUT/c['id']),'--arm',c['arm'],'--internal-dt',str(c['internal']),'--control-dt',str(c['control'])],stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    path=OUT/c['id']/'result.json';r=json.loads(path.read_text()) if path.exists() else {}
    outcome=dict(case=c,returncode=p.returncode,pha=r.get('final',{}).get('pha_g_l'))
    print('DONE',json.dumps(outcome),flush=True);return outcome
progress=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    for future in as_completed([pool.submit(run,c) for c in cases]):
        progress.append(future.result());(OUT/'progress.json').write_text(json.dumps(progress,indent=2))
