"""Local feature-encoding ablation on synthetic arrays with real LP shapes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.selected_lp_features import SelectedLPFeatures,FIELDS
from src.cpu_dictionary_lp import cpu_features
from src.gpu_neural_basis_proposal import features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--iterations',type=int,default=30)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if min(args.batch,args.repeats,args.iterations)<1:raise ValueError('Positive sizes required')
    import cupy as cp
    manifest=json.loads((args.bank/'manifest.json').read_text())
    rng=np.random.default_rng(20291501)
    report=dict(scope='Local encoding microbenchmark, synthetic inputs at stored real LP shapes; NOT dFBA speedup',
        configuration={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        bank_manifest_sha256=hashlib.sha256((args.bank/'manifest.json').read_bytes()).hexdigest(),
        stages=[],source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('src/selected_lp_features.py','src/cpu_dictionary_lp.py','src/gpu_neural_basis_proposal.py',
             'scripts/benchmark_selected_lp_features.py')})
    for stage in manifest['stages']:
        folder=args.bank/stage['stage']
        for filename,digest in (('root.npz',stage['root_sha256']),('router.npz',stage['router_sha256'])):
            if hashlib.sha256((folder/filename).read_bytes()).hexdigest()!=digest:raise ValueError('Artifact hash mismatch')
        with np.load(folder/'root.npz',allow_pickle=False) as root:
            m,n=map(int,root['a_shape']);v=len(root['variable_rows'])
        with np.load(folder/'router.npz',allow_pickle=False) as router:indices=router['indices'].copy()
        shapes=dict(rhs=(m,),lower=(n,),upper=(n,),c=(n,),delta=(v,n),col_scale=(n,),row_scale=(m,))
        selector=SelectedLPFeatures(indices,shapes)
        host={key:rng.normal(size=(args.batch,*shapes[key])) for key in FIELDS}
        # Encoding-edge stress, not physically meaningful LP inputs.
        host['lower'].flat[:4]=[np.nan,-np.inf,np.inf,-0.]
        device={key:cp.asarray(value) for key,value in host.items()}
        one={key:value[0] for key,value in host.items()}
        np.testing.assert_array_equal(cpu_features(one)[indices],selector.cpu_features(one))
        cp.testing.assert_array_equal(features(device)[:,cp.asarray(indices)],selector.gpu_features(device))
        device_indices=cp.asarray(indices)
        calls=dict(cpu_full=lambda:cpu_features(one)[indices],cpu_selected=lambda:selector.cpu_features(one),
                   gpu_full=lambda:features(device)[:,device_indices],gpu_selected=lambda:selector.gpu_features(device))
        for call in calls.values():call()
        cp.cuda.get_current_stream().synchronize()
        samples={key:[] for key in calls}
        for repeat in range(args.repeats):
            names=list(calls) if repeat%2==0 else list(reversed(calls))
            for name in names:
                if name.startswith('gpu_'):cp.cuda.get_current_stream().synchronize()
                before=time.perf_counter()
                for _ in range(args.iterations):value=calls[name]()
                if name.startswith('gpu_'):cp.cuda.get_current_stream().synchronize()
                samples[name].append((time.perf_counter()-before)/args.iterations)
        medians={key:float(np.median(value)) for key,value in samples.items()}
        row=dict(stage=stage['stage'],batch=args.batch,full_width=selector.feature_width,
            selected_width=selector.selected_width,exact_cpu_equality=True,exact_gpu_equality=True,
            seconds_per_call=samples,median_seconds_per_call=medians,
            cpu_full_over_selected=medians['cpu_full']/medians['cpu_selected'],
            gpu_full_over_selected=medians['gpu_full']/medians['gpu_selected'])
        report['stages'].append(row)
        print(json.dumps({key:value for key,value in row.items() if key!='seconds_per_call'}),flush=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
