"""Training-only coverage audit; not an end-to-end performance benchmark."""
import argparse,json,shutil,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.compact_training_data import load_compact_training_source,checked_npz,sha256_file,COMPACT_TRAINING_STAGES
from src.compact_bank_coverage import measure_compact_bank_coverage,summarize_routing_coverage,save_compact_coverage_npz
from src.gpu_compact_basis import CompactBank


def load_training_bank(source,directory,*,entries=None):
    folder=Path(directory)/source.stage;row=source.stage_manifest
    arrays=[checked_npz(folder,e['filename'],e['sha256']) for e in row['entries']] if entries is None else entries
    router=checked_npz(folder,'router.npz',row['router_sha256'])
    return CompactBank(source.root,source.variable_rows,arrays,router['centers'],
        router['indices'],router['scale'],capture=False,selected_features=True)


def grouped_summary(result,source):
    rows=[]
    oracle=result.coverage.any(axis=1)
    for t,seed in enumerate(source.seeds):
        for label,lo,hi in [('all',0,source.steps),('early',0,source.steps//3),
            ('middle',source.steps//3,2*source.steps//3),('late',2*source.steps//3,source.steps)]:
            ids=np.arange(t*source.steps+lo,t*source.steps+hi)
            row=dict(seed=seed,period=label,rows=len(ids),oracle_covered=int(oracle[ids].sum()))
            for k in result.routing_topk:
                row['nearest_k'+str(k)]=int(result.coverage[ids[:,None],result.routing_order[ids,:k]].any(axis=1).sum())
            rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=ROOT/'results/pf_neural_basis_actual4x120_20260904')
    p.add_argument('--bank',type=Path,default=ROOT/'results/pf_compact120_v3_20260904')
    p.add_argument('--stage',choices=COMPACT_TRAINING_STAGES,default='maxmin')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--forbidden-seeds',type=int,nargs='*',default=[])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    from scripts.benchmark_basis_bank_rollout import environment
    from src.fba_surrogate import model_fingerprint
    sample,_=environment(20287101)
    fingerprints={name:model_fingerprint(model) for name,model in sample.simulator.models.items()}
    source=load_compact_training_source(args.data,args.bank,stage=args.stage,forbidden_seeds=args.forbidden_seeds,
        expected_model_fingerprints=fingerprints)
    if len(source.seeds)!=4 or source.steps!=120 or source.size!=480:
        raise ValueError('This audit requires the declared 4 x 120 training source')
    sources={}
    (args.output/'sources').mkdir()
    for file in [Path(__file__),ROOT/'src/compact_training_data.py',ROOT/'src/compact_bank_coverage.py',
                 ROOT/'src/gpu_compact_basis.py',ROOT/'src/selected_lp_features.py']:
        shutil.copyfile(file,args.output/'sources'/file.name);sources[file.name]=sha256_file(file)
    bank=load_training_bank(source,args.bank)
    print(f'Validated {source.size} cached {source.stage} training LPs, {len(bank.evaluators)} candidates; current GEM hashes match.',flush=True)
    def batches():
        for i,batch in enumerate(source.batches(args.batch_size)):
            print(f'coverage batch {i+1}/{(source.size+args.batch_size-1)//args.batch_size}',flush=True)
            yield batch
    result=measure_compact_bank_coverage(bank,batches(),routing_topk=(1,2,4))
    metadata=dict(**source.provenance,sources=sources,bank_path=str(args.bank.resolve()),
        source_path=str(args.data.resolve()),scope='Training coverage, not independent validation or speedup')
    save_compact_coverage_npz(args.output/'coverage.npz',result,metadata)
    np.savez_compressed(args.output/'features.npz',features=source.selected_features(),
        trajectory_ids=np.repeat(source.seeds,source.steps),steps=np.tile(np.arange(1,source.steps+1),len(source.seeds)))
    summary=summarize_routing_coverage(result)
    report=dict(status='completed',summary=summary,by_trajectory_period=grouped_summary(result,source),
        uncovered_rows=np.flatnonzero(~result.coverage.any(axis=1)).tolist(),timing=result.timing,
        total_seconds=time.perf_counter()-started,coverage_sha256=sha256_file(args.output/'coverage.npz'),
        features_sha256=sha256_file(args.output/'features.npz'),provenance=metadata)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
