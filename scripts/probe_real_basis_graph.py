"""Real-GEM graph setup diagnostic; no online CPU LP calls."""
import argparse,sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.compiled_basis_artifact import load_anchor
from src.gpu_revised_basis import GpuRevisedBasis
p=argparse.ArgumentParser();p.add_argument('--pivots',type=int,default=32)
p.add_argument('--batch',type=int,default=1);args=p.parse_args()
anchor=load_anchor('results/pf_basis_train20286311_4_compiled/anchor_0004.npz')
solver=GpuRevisedBasis(anchor,range(anchor['lp'].neq,anchor['lp'].neq+3),max_pivots=args.pivots,capture_safe=True)
inputs=solver.prepare_host([anchor['lp']]*args.batch)
started=time.perf_counter()
result=solver.run_device(**inputs)
print('setup',time.perf_counter()-started,'accepted',result['accepted'].get().tolist(),flush=True)
times=[]
for _ in range(5):
    started=time.perf_counter();result=solver.run_device(**inputs)
    solver.cp.cuda.get_current_stream().synchronize();times.append(time.perf_counter()-started)
print(json.dumps(dict(times=times,accepted=result['accepted'].get().tolist(),
    primal=result['primal_residual'].get().tolist(),gap=result['relative_kkt_gap'].get().tolist())),flush=True)
