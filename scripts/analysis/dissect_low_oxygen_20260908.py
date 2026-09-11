"""Post-screen exploratory controls: fixed inoculum, alternative producer, inert Pf.

The inert arm is a mathematical counterfactual, not a viable mutant strain.
It holds the Pf pool/allocation bookkeeping while removing all Pf metabolism.
"""
from pathlib import Path
import sys,json,subprocess,hashlib
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/symbiosis_reassessment_20260908/low_oxygen_controls'
if len(sys.argv)>1:
    name=sys.argv[1]
    from scripts.analysis import reassess_symbiosis_20260908 as runner
    from scripts.analysis import screen_propionibacterium_helper as s
    original=s._models
    if name=='inert_pf':
        def models(helper):
            result=original(helper)
            for reaction in result[s.HELPER_NAME].reactions:reaction.bounds=(0.,0.)
            return result
        s._models=models
    runner.worker(dict(id=name,condition='oxygen_low',arm='three' if name=='inert_pf' else name),OUT/name)
else:
    OUT.mkdir(exist_ok=False)
    (OUT/'design.json').write_text(json.dumps(dict(trigger='Exploratory low-kLa hit: PHA +8.71%, rubber removal -4.15%; controls selected after seeing this result.',controls=['two_fixed','ns21_alone','inert_pf'],inert_pf='All Pf reaction bounds zero in memory; biomass pool retained. Not biological knockout evidence.',sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/analysis/reassess_symbiosis_20260908.py',ROOT/'src/resolved_dfba.py',ROOT/'src/dfba_simulator.py']}),indent=2))
    def run(name):
        dest=OUT/name;dest.mkdir()
        with (dest/'worker.log').open('w') as log:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),name],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        print(name,p.returncode,flush=True)
        if p.returncode:raise RuntimeError(name)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,['two_fixed','ns21_alone','inert_pf']))
