"""Record the isolated defect and the analytic shared-oxygen control."""
from pathlib import Path
import hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tests.test_shared_oxygen_coupling import fixed_population_equilibrium
rows=[]
for dt in [.1,.025,.00625]:
    for scheme in ['quota_mean_v2','shared_endpoint_v3']:
        value,sim=fixed_population_equilibrium(dt,scheme)
        rows.append(dict(dt_h=dt,scheme=scheme,steady_do_mmol_l=float(value),
            lp_requests=sim.accounting_audit['lp_requests'],lp_certified=sim.accounting_audit['lp_certified_requests']))
b=20.+.01-.25;expected=2*.25*.01/(b+math.sqrt(b*b+4*.25*.01))
result=dict(rows=rows,continuous_expected_do_mmol_l=expected,
    assumptions='Frozen populations 1 g/L each and nutrient bath; A oxygen cap 20*D/(.01+D), growth ceiling .5; B same oxygen exchange capacity but its only consuming route is shut; kLa 1/h, saturation .25 mM.',
    equilibrium_equation='20*D/(.01+D) = .25-D',
    old_quota_fixed_point='When the A quota binds and B uses none, D = .25*(1-exp(-dt))/(2-exp(-dt)); maximum demand is not actual uptake.',
    scope='Numerical reference fixture, not measured microbial kinetics',
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/audited_dfba.py','tests/test_shared_oxygen_coupling.py']})
out=ROOT/'results/cultivation_model_audit_20260908/oxygen_quota_followup'
(out/'isolated_defect.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
