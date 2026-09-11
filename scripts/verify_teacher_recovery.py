"""Replay certified prefix to reproduce failed CPU teacher inputs, then finish.

Diagnostic only. Current LP hashes must match every restored prefix problem.
No failed candidate is reused; only previously certified solutions are read.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.offline_scipy_teacher import OfflineScipyTeacher
from src.cpu_repeated_lp import _problem
from src.lp_trace import load_trace_lp,problem_hash
from scripts import capture_dfba_lp_trace as capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();m=json.loads((args.prefix/'manifest.json').read_text());cfg=m['configuration']
    if m['status']!='failed' or not cfg['scipy_teacher']:raise ValueError('Expected failed SciPy reference')
    n=cfg['environments'];entries=m['entries']
    if len(entries)%n:raise ValueError('Incomplete saved batch')
    recovery=[]
    class Replay(OfflineScipyTeacher):
        def __init__(self,**kwargs):super().__init__(**kwargs);self.batch=0
        def solve_batch(self,requests):
            batch=self.batch;self.batch+=1;first=batch*n
            if first>=len(entries):
                results=super().solve_batch(requests)
                report=dict(step=batch//3+1,stage=('maxmin','aggregate','exchange')[batch%3],
                    diagnostics=[r.diagnostics for r in results])
                recovery.append(report)
                if first==len(entries):print(json.dumps(dict(first_unrecorded_batch=report)),flush=True)
                return results
            self.models.clear();results=[]
            for lane,request in enumerate(requests):
                e=entries[first+lane];problem,x,y=load_trace_lp(args.prefix,e)
                current=_problem(*request)
                if (e['environment_id']!=lane or e['step']!=batch//3+1 or
                    e['stage']!=('maxmin','aggregate','exchange')[batch%3] or
                    problem_hash(current)!=problem_hash(problem)):
                    raise ValueError('Reconstructed trajectory differs from certified prefix')
                a,_,_,_,c,neq=problem
                solution=SimpleNamespace(col_value=x,row_dual=y)
                self.models[lane,e['stage'],*a.shape,neq]={'solver':SimpleNamespace(getSolution=lambda s=solution:s)}
                results.append(SimpleNamespace(success=True,x=x,fun=float(c@x),message='certified prefix replay',
                    diagnostics=dict(e['reference_diagnostics'],certified_prefix_replay=True)))
            return results
    argv=['capture','--output',str(args.output),'--steps',str(cfg['steps']),
        '--environments',str(n),'--workers','1','--seed',str(cfg['seed']),
        '--role','development_diagnostic_not_training','--frozen-inputs','--design-profile',cfg['design_profile'],
        '--design-offset',str(cfg['design_offset']),'--scipy-teacher']
    with patch.object(sys,'argv',argv),patch('src.offline_scipy_teacher.OfflineScipyTeacher',Replay):
        capture.main()
    new=json.loads((args.output/'manifest.json').read_text())
    if new['model_fingerprints']!=m['model_fingerprints']:raise ValueError('GEM differs')
    report=dict(status='completed',scope=__doc__,source=str(args.prefix),
        replayed_certified_lp=len(entries),newly_solved_lp=sum(len(r['diagnostics']) for r in recovery),
        tail_batches=recovery)
    with (args.output/'recovery_report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='tail_batches'}),flush=True)


if __name__=='__main__':main()
