"""Read-only progress snapshot; rates are observed, not promised completion times."""
from pathlib import Path
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/audited_symbiosis_20260908'
design=json.loads((OUT/'design.json').read_text())
finished={}
if (OUT/'progress.json').exists():
    try:finished={r['case']['id']:r for r in json.loads((OUT/'progress.json').read_text())}
    except json.JSONDecodeError:pass
states=[];rates=[]
for c in design['cases']:
    p=OUT/c['id'];result=p/'result.json';progress=p/'progress.json'
    weight=(.025/c.get('internal_dt',.025))*({'three':7,'inert_pf':7,'two_equal_total':5,'two_fixed':5,'ns21_pf':5,'pf_alone':2,'ns21_alone':3}[c['arm']]/7)
    if c['id'] in finished and finished[c['id']]['returncode']!=0:
        hours=0.;seconds=0.;state='failed'
    elif result.exists():
        d=json.loads(result.read_text());hours=d['final']['time'];seconds=d['seconds'];state='complete'
    elif progress.exists():
        try:d=json.loads(progress.read_text())
        except json.JSONDecodeError:continue
        hours=d['hours'];seconds=d['seconds'];state='running'
    else:hours=0.;seconds=0.;state='starting' if p.exists() else 'queued'
    if hours>0:rates.append(seconds/(hours*weight))
    states.append(dict(case=c['id'],state=state,hours=hours,seconds=seconds,weighted_remaining=(12-hours)*weight))
now=datetime.now(ZoneInfo('Asia/Tokyo'))
report=dict(as_of=now.isoformat(),complete=sum(s['state']=='complete' for s in states),failed=sum(s['state']=='failed' for s in states),total=len(states),states=states)
if rates and not report['failed']:
    rates.sort();rate=rates[len(rates)//2]
    remaining=max(sum(s['weighted_remaining'] for s in states)/design['runtime']['workers'],
                  max(s['weighted_remaining'] for s in states))*rate
    report.update(estimated_remaining_minutes=round(remaining/60,1),rough_eta=(now+timedelta(seconds=remaining)).isoformat(),
        caveat='Median observed normalized rate; convergence effort and remaining-case mix can change this substantially.')
print(json.dumps(report,indent=2))
