"""Input-only reverse mapping of saved maxmin community constraints; no LP solve."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_basis_bank_rollout import environment
from scripts.benchmark_ipm_cpu_inputs import load_inputs
from src.fba_surrogate import model_fingerprint


def main():
    trace=ROOT/'results/pf_lp_trace_dev32x41_20260905'
    manifest=json.loads((trace/'manifest.json').read_text())
    env,layout=environment(manifest['configuration']['seed'])
    actual={k:model_fingerprint(v) for k,v in env.simulator.models.items()}
    expected=manifest.get('model_fingerprints')
    if expected is None or actual!=expected:
        raise ValueError(f'Model identity not verified: actual={actual}, expected={expected}')
    growth=len(layout._growth_terms)
    names=sorted(layout._exchange_terms)
    rows=[]
    for step in range(2,9):
        problems,provenance=load_inputs(trace,'maxmin',step,4)
        for lane,(a,b,lo,hi,c,neq) in enumerate(problems):
            assert a.shape[1]==layout.n_fluxes+1 and a.shape[0]-neq==growth+len(names)
            row=4738;name=names[row-neq-growth]
            terms=layout._exchange_terms[name]
            indices=sorted(set(t[1] for t in terms))
            if indices!=a.indices[a.indptr[row]:a.indptr[row+1]].tolist():
                raise ValueError('Saved row columns do not match live layout')
            rows.append(dict(step=step,lane=lane,row=row,metabolite=name,terms=terms,
                rhs=float(b[row]),coefficients=a.data[a.indptr[row]:a.indptr[row+1]].tolist(),
                problem_sha256=provenance['problem_sha256'][lane]))
    out=dict(scope=__doc__,cpu_lp_calls=0,model_fingerprints=actual,rows=rows)
    path=ROOT/'results/pf_hot_supply_row_mapping_20260907.json'
    with path.open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
    env.close()


if __name__=='__main__':main()
