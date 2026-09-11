"""Versioned CPU reference run with independently specified control/integration steps."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analysis import screen_propionibacterium_helper as s
from src.resolved_dfba import ResolvedDFBASimulator

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    models=s._models(args.arm=='three');biomass,medium=s.get_initial_params(models)
    biomass[s.OR16_NAME]=.525 if args.arm=='two_equal_total' else .5
    biomass[s.NS21_NAME]=.105 if args.arm=='two_equal_total' else .1
    if args.arm=='three':biomass[s.HELPER_NAME]=.03
    medium.update(yeast_extract_e=0.,glc__D_e=0.,lac__L_e=0.,ppa_e=0.,ac_e=0.,nh4_e=args.nh4)
    config=vars(args).copy()
    config.update(numerics_version=ResolvedDFBASimulator.NUMERICS_VERSION,
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'src/resolved_dfba.py',ROOT/'src/dfba_simulator.py']},
        evidence='Qualitative growth/storage tradeoff informed by supplied RDE2 report; no NS21 parameter fit',
        initial_biomass=biomass.copy(),initial_medium=medium.copy())
    (out/'design.json').write_text(json.dumps(config,indent=2))
    sim=ResolvedDFBASimulator(models=models,initial_biomass=biomass,initial_metabolites=medium,
        initial_rubber=10.,dt=args.control_dt,max_internal_dt=args.internal_dt,
        nitrogen_half_saturation=args.nitrogen_half_saturation,
        solver_backend='highs',ph_control_target=7.)
    scenario=s.Scenario('resolved',args.arm=='three','lac__L_e',args.feed_rate,.03,args.kla)
    def row():
        data=s._row(sim,scenario,sim.cumulative_continuous_feed.get('lac__L_e',0.))
        data['o2_mmol_l']=sim.state.metabolites['o2_e'];return data
    rows=[row()];start=time.perf_counter()
    while sim.state.time < args.hours-1e-10:
        sim.dt=min(args.control_dt,args.hours-sim.state.time)
        sim.step({}, {},dynamic_kla=args.kla,feed_rates_mmol_l_h={'lac__L_e':args.feed_rate})
        rows.append(row())
        (out/'progress.json').write_text(json.dumps(dict(time_h=sim.state.time,seconds=time.perf_counter()-start)))
    result=dict(final=rows[-1],seconds=time.perf_counter()-start,diagnostics=sim.get_solver_diagnostics())
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (out/'trajectory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(arm=args.arm,control_dt=args.control_dt,internal_dt=args.internal_dt,pha=rows[-1]['pha_g_l'],seconds=result['seconds'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--arm',choices=['two_fixed','two_equal_total','three'],default='three')
    p.add_argument('--hours',type=float,default=12.);p.add_argument('--control-dt',type=float,default=1.)
    p.add_argument('--internal-dt',type=float,default=.025);p.add_argument('--nh4',type=float,default=.05)
    p.add_argument('--nitrogen-half-saturation',type=float,default=.1);p.add_argument('--kla',type=float,default=50.)
    p.add_argument('--feed-rate',type=float,default=.25)
    a=p.parse_args()
    if a.hours<=0:p.error('hours must be positive')
    run(a)
