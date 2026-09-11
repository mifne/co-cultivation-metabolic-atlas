"""Read-only numerical/provenance audit, writing a NEW external addendum.

The completed bank/router remain byte-identical. Hashes absent from the legacy
builder are pinned at audit time, not misrepresented as historical guarantees.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.compact_training_data import checked_npz,checked_child,sha256_file
from src.compact_basis_rebuild import basis_status_signature
from src.coverage_router import CoverageRouter,coverage_metrics
from scripts.train_coverage_router import validate_labels


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--router',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    manifest=json.loads((args.bank/'manifest.json').read_text())
    base=json.loads((args.base/'manifest.json').read_text())
    report=json.loads((args.bank/'rebuild_report.json').read_text())
    collector=json.loads((args.bank/'collector_report.json').read_text())
    router_report=json.loads((args.router/'report.json').read_text())
    for value in (manifest,base,report,collector,router_report):
        if value['status']!='completed':raise ValueError('Incomplete artifact')
    coverage=checked_npz(args.bank,'coverage.npz',report['coverage_sha256'])
    features=checked_npz(args.bank,'features.npz',report['features_sha256'])
    validate_labels(coverage,features,manifest,sha256_file(args.bank/'manifest.json'))
    pool=checked_npz(args.bank,'pool_coverage.npz')
    statuses=checked_npz(args.bank,'new_basis_statuses.npz')
    selected=np.asarray(report['selection']['selected'])
    if not np.array_equal(selected,pool['selected']) or not np.array_equal(coverage['coverage'],pool['coverage'][:,selected]):
        raise ValueError('Selected coverage/order mismatch')
    if pool['coverage'].dtype!=np.bool_ or int(pool['coverage'].any(axis=1).sum())!=report['pool_union_covered_rows']:
        raise ValueError('Pool coverage count mismatch')
    stages={s['stage']:s for s in manifest['stages']};old_stages={s['stage']:s for s in base['stages']}
    inherited=0;pool_ids=[]
    for name,old in old_stages.items():
        stage=stages[name]
        if stage['entries'][:len(old['entries'])]!=old['entries']:raise ValueError('Mandatory metadata changed')
        for filename,key in [('root.npz','root_sha256'),('router.npz','router_sha256')]:
            checked_child(args.bank/name,filename,stage[key])
            if name!='maxmin' or filename=='root.npz':
                if stage[key]!=old[key]:raise ValueError('Inherited root/router changed')
        for entry in old['entries']:
            checked_child(args.base/name,entry['filename'],entry['sha256'])
            checked_child(args.bank/name,entry['filename'],entry['sha256']);inherited+=1
            if name=='maxmin':pool_ids.append(entry['filename']+':'+entry['sha256'])
    for record in report['projected_candidates']:
        checked_child(args.bank/'pool',record['pool_filename'],record['sha256'])
        if not record['self_row_gpu_certified'] or not record['all_training_coverage_measured']:raise ValueError('Missing projection gates')
        pool_ids.append(record['pool_filename']+':'+record['sha256'])
    for i,(columns,rows) in enumerate(zip(statuses['col_statuses'],statuses['row_statuses'])):
        if basis_status_signature(columns,rows)!=str(statuses['signatures'][i]):raise ValueError('Status signature mismatch')
    for row in collector['rows']:
        certificate=row['original_certificate']
        if not certificate['certificate_passed']:raise ValueError('Uncertified CPU anchor')
        for metric,threshold in [('primal_residual',1e-5),('dual_violation',1e-7),('relative_kkt_gap',1e-7)]:
            value=certificate[metric]
            if value is None or not np.isfinite(value) or not 0<=value<=threshold:raise ValueError('CPU gate mismatch')
    for entry in stages['maxmin']['entries']:
        checked_child(args.bank/'maxmin',entry['filename'],entry['sha256'])
    router=CoverageRouter.load(args.router/'router.npz',expected_sha256=router_report['artifact']['sha256'],
        expected_provenance=router_report['provenance'])
    calculated=coverage_metrics(coverage['coverage'],router.rank(features['features']),top_k=(1,2,4))
    if calculated!=router_report['final_training_metrics']:raise ValueError('Final router metric mismatch')
    folds=[]
    for internal in router_report['internal_comparison']:
        for fold in range(len(internal['folds'])):
            filename=f'internal_h{internal["hidden_dim"]}_fold{fold}'
            folds.append(dict(report_filename=filename+'.json',report_sha256=sha256_file(args.router/(filename+'.json')),
                artifact_filename=filename+'.npz',artifact_sha256=sha256_file(args.router/(filename+'.npz'))))
    result=dict(status='passed',scope='Audit-time addendum; existing bank/router bytes were not changed',
        bank_manifest_sha256=sha256_file(args.bank/'manifest.json'),rebuild_report_sha256=sha256_file(args.bank/'rebuild_report.json'),
        collector_report_sha256=sha256_file(args.bank/'collector_report.json'),
        new_basis_statuses_sha256=sha256_file(args.bank/'new_basis_statuses.npz'),
        pool_coverage_sha256=sha256_file(args.bank/'pool_coverage.npz'),pool_candidate_ids=pool_ids,
        router_report_sha256=sha256_file(args.router/'report.json'),internal_fold_evidence=folds,
        inherited_candidate_files_verified=inherited,selected_candidates=len(selected),
        original_cpu_certified_rows=len(collector['rows']),unique_new_basis_statuses=len(statuses['signatures']),
        reconstructed_seconds=report['total_seconds'],reconstructed_cpu_lp_calls=collector['offline_cpu_lp_calls'],
        legacy_field_errata=dict(
            manifest_total_seconds=manifest.get('total_seconds'),manifest_cpu_lp_calls=manifest.get('cpu_lp_calls'),
            interpretation='In this first snapshot, top-level total_seconds/cpu_lp_calls were inherited from the OLD bank. Use reconstruction_seconds/reconstruction_cpu_lp_calls and rebuild_report. Future builder code corrects the names. Existing hashes remain unchanged.'),
        historical_hash_limit='Intermediate hashes here were first pinned during this audit, not before construction.')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ['status','inherited_candidate_files_verified','selected_candidates',
        'original_cpu_certified_rows','unique_new_basis_statuses','reconstructed_seconds']},indent=2))


if __name__=='__main__':main()
