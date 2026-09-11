"""Third-step paired refinement triggered by the remaining seven-percent shift."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/audited_symbiosis_20260908'
parent=OUT/'third_refinement';parent.mkdir(exist_ok=False)
cases=[dict(id='oxygen_low_'+arm+'_finer',condition='oxygen_low',arm=arm,internal_dt=.00625)
       for arm in ['two_equal_total','three']]
sources=json.loads((OUT/'design.json').read_text())['sources'].copy()
for name in ['scripts/analysis/audited_precision_adapter_20260908.py',
             'scripts/analysis/audited_roundoff_adapter_20260908.py',str(Path(__file__).relative_to(ROOT))]:
    sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
(parent/'design.json').write_text(json.dumps(dict(cases=cases,sources=sources,
    trigger='Both low-kLa arms changed PHA by approximately 7% after halving .025 to .0125 h; check another halving without selecting for a positive outcome.'),indent=2))
def run(case):
    dest=parent/case['id']
    with (parent/(case['id']+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/analysis/audited_precision_adapter_20260908.py'),
            '--case',json.dumps(case),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,timeout=10800)
    return dict(case=case,returncode=result.returncode,relative_directory=str(dest.relative_to(OUT)))
results=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    for future in as_completed([pool.submit(run,case) for case in cases]):
        result=future.result();results.append(result)
        temporary=parent/'progress.tmp';temporary.write_text(json.dumps(results,indent=2));temporary.replace(parent/'progress.json')
        print(json.dumps(result),flush=True)
after={file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in sources}
assert after==sources,'Execution sources changed during refinement'
(parent/'execution_source_verification.json').write_text(json.dumps(after,indent=2))
assert all(r['returncode']==0 for r in results),results
