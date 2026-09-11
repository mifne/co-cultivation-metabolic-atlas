"""Replay a complete previously failing GEM control with the integrated core."""
from pathlib import Path
import hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src
candidate=ROOT/'results/cultivation_model_audit_20260908/audited_dfba_candidate.py'
spec=importlib.util.spec_from_file_location('src.audited_dfba',candidate)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
spec.loader.exec_module(module);src.audited_dfba=module
from scripts.analysis.reassess_audited_symbiosis_20260908 import worker
reference=ROOT/'results/audited_symbiosis_20260908/precision_patch_runs/base_ns21_alone'
dest=ROOT/'results/cultivation_model_audit_20260908/candidate_actual_ns21';dest.mkdir(exist_ok=False)
case=json.loads((reference/'result.json').read_text())['settings']['case']
worker(case,dest)
matches={name:(dest/name).read_bytes()==(reference/name).read_bytes()
         for name in ['trajectory.csv','internal_trajectory.csv']}
a=json.loads((dest/'result.json').read_text());b=json.loads((reference/'result.json').read_text())
matches.update(final=a['final']==b['final'],exchanges=a['exchanges']==b['exchanges'])
result=dict(candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    exact_dynamics_matches=matches,primary_recomputations=a['diagnostics']['audited_cultivation']['accounting'].get('primary_recomputations',[]))
(dest.parent/'candidate_actual_equivalence.json').write_text(json.dumps(result,indent=2))
assert all(matches.values()),result
print('Integrated production candidate exactly reproduced the complete 12 h corrected control.',flush=True)
