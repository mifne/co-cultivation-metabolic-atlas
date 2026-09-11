"""Short real-GEM compatibility check for the audited reference and guards."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.analysis import screen_propionibacterium_helper as helper
from src.audited_dfba import AuditedDFBASimulator
models=helper._models(True)
biomass,medium=helper.get_initial_params(models)
biomass={helper.OR16_NAME:.5,helper.NS21_NAME:.1,helper.HELPER_NAME:.03}
medium.update(yeast_extract_e=0.,glc__D_e=0.,lac__L_e=0.,ppa_e=0.,ac_e=0.,nh4_e=.05)
sim=AuditedDFBASimulator(models,biomass,medium,initial_rubber=10.,dt=.025,
    max_internal_dt=.025,solver_backend='highs',ph_control_target=7.)
for _ in range(2):
    sim.step({}, {},dynamic_kla=10.,feed_rates_mmol_l_h={'lac__L_e':.25})
    assert abs(sim.last_polymer_fluxes['carbon_c5_equivalent_error_mmol_l'])<1e-8
diag=sim.get_solver_diagnostics()
assert diag['audited_cultivation']['valid']
accounting=diag['audited_cultivation']['accounting']
assert accounting['lp_requests']==accounting['lp_certified_requests']
assert abs(sim.state.time-.05)<1e-12
result=dict(diagnostics=diag,source_sha256=hashlib.sha256((ROOT/'src/audited_dfba.py').read_bytes()).hexdigest())
(ROOT/'results/cultivation_model_audit_20260908/actual_model_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(real_model_smoke='passed',time=sim.state.time,lp_solves=diag['solve_successes'])))
