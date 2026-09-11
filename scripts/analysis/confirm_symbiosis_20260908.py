"""Paired time-step refinement, kept separate from the screening outputs."""
from pathlib import Path
import sys,json,subprocess,hashlib
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
worker=ROOT/'scripts/analysis/reassess_symbiosis_20260908.py'
condition=sys.argv[1] if len(sys.argv)>1 else 'base'
out=ROOT/'results/symbiosis_reassessment_20260908'/('refinement_'+condition)
out.mkdir(exist_ok=False)
design=dict(condition=condition,internal_dt=.0125,sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [worker,ROOT/'src/resolved_dfba.py',ROOT/'src/dfba_simulator.py']})
(out/'design.json').write_text(json.dumps(design,indent=2))
def run(arm):
    dest=out/arm;dest.mkdir()
    case=dict(id=condition+'_'+arm+'_fine',condition=condition,arm=arm,internal_dt=.0125)
    with (dest/'worker.log').open('w') as log:
        p=subprocess.run([sys.executable,str(worker),'--case',json.dumps(case),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    print(arm,p.returncode,flush=True)
    if p.returncode:raise RuntimeError(arm)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,['two_equal_total','three']))
