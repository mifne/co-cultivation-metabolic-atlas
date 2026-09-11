"""Independent-process refinement; preserves production collection code."""
from pathlib import Path
import sys,json,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'results/pfreud_refine_20260908';out.mkdir(exist_ok=False)
cases=[dict(id=f'fine_{dt}_{arm}',arm=arm,dt=dt,profile='continuous') for dt in [.0125,.00625] for arm in ['two_fixed','two_equal_total','three']]
(out/'design.json').write_text(json.dumps(cases,indent=2))
def run(c):
    dest=out/c['id'];dest.mkdir();print('START',c['id'],flush=True)
    with (dest/'worker.log').open('w') as f:
        p=subprocess.run([sys.executable,str(ROOT/'scripts/analysis/pfreud_causal_20260908.py'),'--case',json.dumps(c),'--output',str(dest)],stdout=f,stderr=subprocess.STDOUT,timeout=2400)
    r=json.loads((dest/'result.json').read_text()) if (dest/'result.json').exists() else {}
    print('DONE',c['id'],p.returncode,r.get('pha'),flush=True)
    return dict(case=c,returncode=p.returncode,pha=r.get('pha'))
with ThreadPoolExecutor(max_workers=2) as pool:
    progress=[]
    for f in as_completed([pool.submit(run,c) for c in cases]):
        progress.append(f.result());(out/'progress.json').write_text(json.dumps(progress,indent=2))
