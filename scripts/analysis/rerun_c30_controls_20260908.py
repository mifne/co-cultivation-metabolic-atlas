"""Run explicit boundary-patched controls without overwriting failed attempts."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/audited_symbiosis_20260908'
parent=OUT/'numerical_patch_runs';parent.mkdir(exist_ok=False)
cases=[dict(id='base_ns21_alone',condition='base',arm='ns21_alone'),
       dict(id='base_ns21_pf',condition='base',arm='ns21_pf'),
       dict(id='oxygen_low_ns21_alone',condition='oxygen_low',arm='ns21_alone')]
def run(case):
    dest=parent/case['id']
    with (parent/(case['id']+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/analysis/audited_roundoff_adapter_20260908.py'),
            '--case',json.dumps(case),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,timeout=7200)
    return dict(case=case,returncode=result.returncode,relative_directory=str(dest.relative_to(OUT)))
results=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    for future in as_completed([pool.submit(run,case) for case in cases]):
        result=future.result();results.append(result)
        (OUT/'numerical_patch_reruns.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(result),flush=True)
