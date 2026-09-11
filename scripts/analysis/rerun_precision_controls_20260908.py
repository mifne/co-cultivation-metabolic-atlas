"""Retain failed attempts and retry only controls requiring primary recomputation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/audited_symbiosis_20260908'
parent=OUT/'precision_patch_runs';parent.mkdir(exist_ok=False)
cases=[dict(id='base_ns21_alone',condition='base',arm='ns21_alone'),
       dict(id='oxygen_low_ns21_alone',condition='oxygen_low',arm='ns21_alone')]
def run(case):
    dest=parent/case['id']
    with (parent/(case['id']+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/analysis/audited_precision_adapter_20260908.py'),
            '--case',json.dumps(case),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,timeout=7200)
    return dict(case=case,returncode=result.returncode,relative_directory=str(dest.relative_to(OUT)))
results=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    for future in as_completed([pool.submit(run,case) for case in cases]):
        result=future.result();results.append(result)
        temporary=OUT/'precision_patch_reruns.json.tmp';temporary.write_text(json.dumps(results,indent=2))
        temporary.replace(OUT/'precision_patch_reruns.json')
        print(json.dumps(result),flush=True)
