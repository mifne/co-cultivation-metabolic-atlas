"""Matched development ablation; no confidence interval from two seed groups."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _finite_scalar(value):
    try:
        return np.ndim(value)==0 and bool(np.isfinite(float(value)))
    except (TypeError,ValueError,OverflowError):
        return False


def _check_endpoints(endpoints,count):
    if not isinstance(endpoints,list) or len(endpoints)!=count:
        raise ValueError('One endpoint per environment is required')
    for row in endpoints:
        if not isinstance(row,dict) or not all(_finite_scalar(row.get(key)) for key in ('pha','phv_fraction')):
            raise ValueError('Nonfinite or missing endpoint scalar')
        for key in ('biomass','metabolites'):
            if not isinstance(row.get(key),dict) or (key=='biomass' and not row[key]):
                raise ValueError('Missing endpoint state dictionary')
            if not all(_finite_scalar(value) for value in row[key].values()):
                raise ValueError('Nonfinite endpoint state')


def compare(serial, pipeline):
    for report in (serial,pipeline):
        cfg=report['configuration']
        if report.get('status')!='completed' or not report.get('runs'):
            raise ValueError('Both runs must be completed')
        if cfg['tie_policy']!='original3' or cfg['cpu_backend']!='dictionary':
            raise ValueError('Original three-stage dictionary CPU comparison required')
        if any(row.get('failure') or not row.get('all_endpoint_gates_passed')
               or row['gpu_completed_steps']!=[cfg['steps']]*cfg['environments']
               or row['cpu_completed_steps']!=[cfg['steps']]*cfg['environments']
               for row in report['runs']):
            raise ValueError('Every trajectory must pass accuracy and finish')
        for row in report['runs']:
            if len(row.get('seeds',[]))!=cfg['environments']:
                raise ValueError('One seed per environment is required')
            errors=row.get('errors')
            if not isinstance(errors,list) or len(errors)!=cfg['environments']:
                raise ValueError('One error record per environment is required')
            for error in errors:
                if not isinstance(error,dict) or any(not _finite_scalar(error.get(key))
                        or not 0<=float(error[key])<=.01
                        for key in ('pha_relative','biomass_g_l','phv_fraction')):
                    raise ValueError('Original endpoint error gate failed')
            for side in ('cpu','gpu'):
                _check_endpoints(row.get(side+'_rows'),cfg['environments'])
    if serial['configuration'].get('pipeline_cpu_stages') or not pipeline['configuration'].get('pipeline_cpu_stages'):
        raise ValueError('Expected serial and pipelined scheduler pair')
    ignored={'output','pipeline_cpu_stages'}
    configs=[{k:v for k,v in report['configuration'].items() if k not in ignored} for report in (serial,pipeline)]
    if configs[0]!=configs[1]:
        raise ValueError('Ablation configurations differ beyond scheduler')
    common=set(serial['source_hashes'])
    if set(pipeline['source_hashes'])-{'scripts/pipelined_microbatch.py'}!=common-{'scripts/pipelined_microbatch.py'}:
        raise ValueError('Source inventories changed between ablations')
    if any(serial['source_hashes'][key]!=pipeline['source_hashes'][key] for key in common):
        raise ValueError('Common source files changed between ablations')
    if serial['model_fingerprints']!=pipeline['model_fingerprints']:
        raise ValueError('Models changed between ablations')
    if len(serial['runs'])!=len(pipeline['runs']):
        raise ValueError('Repeat counts differ')
    rows=[]
    for old,new in zip(serial['runs'],pipeline['runs']):
        if old['repeat']!=new['repeat'] or old['seeds']!=new['seeds'] or old['execution_order']!=new['execution_order']:
            raise ValueError('Matched seed/order required')
        times={name:float(value) for name,value in dict(cpu_serial=old['cpu_seconds'],
            cpu_pipeline=new['cpu_seconds'],hybrid_serial=old['gpu_seconds'],
            hybrid_pipeline=new['gpu_seconds']).items()}
        if any(not np.isfinite(value) or value<=0 for value in times.values()):
            raise ValueError('Positive finite wall times required')
        endpoint_deltas={}
        for side in ('cpu','gpu'):
            if len(old[side+'_rows'])!=serial['configuration']['environments'] or len(new[side+'_rows'])!=len(old[side+'_rows']):
                raise ValueError('One endpoint per environment is required')
            deltas=[]
            for a,b in zip(old[side+'_rows'],new[side+'_rows']):
                if any(set(a[key])!=set(b[key]) for key in ('biomass','metabolites')):
                    raise ValueError('Endpoint species or metabolite keys changed')
                deltas.append(dict(pha_relative=abs(a['pha']-b['pha'])/max(abs(a['pha']),1e-9),
                    biomass_g_l=max(abs(a['biomass'][key]-b['biomass'][key]) for key in a['biomass']),
                    phv_fraction=abs(a['phv_fraction']-b['phv_fraction'])))
            endpoint_deltas[side]={key:max(row[key] for row in deltas) for key in deltas[0]}
        if any(value>.01 or not np.isfinite(value) for item in endpoint_deltas.values() for value in item.values()):
            raise ValueError('Scheduler ablation changes endpoints beyond the existing gate')
        rows.append(dict(repeat=old['repeat'],seeds=old['seeds'],seconds=times,
            cpu_scheduler_ratio=times['cpu_serial']/times['cpu_pipeline'],
            hybrid_scheduler_ratio=times['hybrid_serial']/times['hybrid_pipeline'],
            pipeline_cpu_over_hybrid=times['cpu_pipeline']/times['hybrid_pipeline'],
            cpu_lp_calls_serial=old['online_cpu_lp_calls'],
            cpu_lp_calls_pipeline=new['online_cpu_lp_calls'],
            scheduler_endpoint_errors=endpoint_deltas))
    return dict(scope='Development ablation with matched seeds/actions, common source, and the same four LP workers; '
        'online wall time includes host work, GPU work and CPU fallbacks. No CI or general superiority claim.',
        environments=serial['configuration']['environments'],steps=serial['configuration']['steps'],
        simulated_hours_per_environment=serial['simulated_hours_per_environment'],pairs=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial',type=Path,required=True)
    parser.add_argument('--pipeline',type=Path,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args()
    targets=[args.output_prefix.with_suffix(suffix) for suffix in ('.json','.svg','.png')]
    if any(path.exists() for path in targets):raise FileExistsError(args.output_prefix)
    reports=[json.loads(path.read_text()) for path in (args.serial,args.pipeline)]
    result=compare(*reports)
    result['sources']=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (args.serial,args.pipeline)]
    targets[0].parent.mkdir(parents=True,exist_ok=True)
    targets[0].write_text(json.dumps(result,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
    x=np.arange(len(result['pairs']))
    shared_limit=1.3*max(value for row in result['pairs'] for value in row['seconds'].values())
    for ax,side,label in zip(axes,('cpu','hybrid'),('CPU','GPU/CPU hybrid')):
        for offset,scheduler,color in ((-.18,'serial','#999999'),(.18,'pipeline','#0072B2')):
            values=[row['seconds'][side+'_'+scheduler] for row in result['pairs']]
            ax.bar(x+offset,values,width=.32,color=color,label='Barrier' if scheduler=='serial' else 'Pipelined')
        ax.set(xticks=x,xticklabels=[str(row['repeat']+1) for row in result['pairs']],
            xlabel='Matched seed group',ylabel='Online wall time (s)',ylim=(0,shared_limit))
        ax.set_title(label,fontsize=10)
        ax.legend(frameon=False,fontsize=8)
    for path in targets[1:]:fig.savefig(path,dpi=300)
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
