"""Install the verified conditional fixes only after all frozen runs finish."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/audited_symbiosis_20260908'
AUDIT=ROOT/'results/cultivation_model_audit_20260908'
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
deadline=time.monotonic()+10800
required=[OUT/'execution_source_verification.json',OUT/'third_refinement/execution_source_verification.json',
          AUDIT/'candidate_actual_equivalence.json']
while not all(path.exists() for path in required):
    if time.monotonic()>deadline:raise TimeoutError('Frozen execution or integrated-core validation incomplete')
    time.sleep(20)
assert all(r['returncode']==0 for r in read(OUT/'precision_patch_reruns.json'))
assert all(r['returncode']==0 for r in read(OUT/'third_refinement/progress.json'))
for design in [read(OUT/'design.json'),read(OUT/'third_refinement/design.json')]:
    assert all(sha(ROOT/file)==expected for file,expected in design['sources'].items()),'Source changed during frozen execution'
equivalence=read(AUDIT/'candidate_actual_equivalence.json')
assert all(equivalence['exact_dynamics_matches'].values())
assert sha(AUDIT/'audited_dfba_candidate.py')==equivalence['candidate_sha256']
assert not (AUDIT/'production_repair_installation.json').exists(),'Do not install twice'
changes={}
for filename in ['audited_dfba','rl_environment']:
    target=ROOT/'src'/f'{filename}.py';candidate=AUDIT/f'{filename}_candidate.py'
    changes[str(target.relative_to(ROOT))]=dict(before=sha(target),after=sha(candidate))
    target.write_bytes(candidate.read_bytes())
(AUDIT/'production_repair_installation.json').write_text(json.dumps(dict(
    installed_at_utc=datetime.now(timezone.utc).isoformat(),changes=changes,
    equivalence='candidate_actual_equivalence.json; identical complete 12h corrected NS21 dynamics',
    scope='C30 rounding and failed-secondary primary recomputation; policy contract includes repair revision'),indent=2))
print('Frozen runs complete; verified repairs installed into production modules.',flush=True)
test_files=['test_audited_dfba','test_cultivation_numerics','test_resolved_dfba','test_one_l_jar',
    'test_graph_learning_curve','test_trace_script_lifecycle','test_rl_audit_contract','test_main_cultivation_profiles',
    'test_legacy_evaluation_callers','test_model_selection','test_ppo_agent_stability','test_ppo_observation_schema',
    'test_ppo_reward_scale','test_rl_feed_action_scaling','test_polymer_pathway_curation','test_ns21_pha_pathway',
    'test_propionibacterium_helper','test_audited_c30_roundoff','test_audited_primary_recomputation','test_audited_production_repairs']
command=[sys.executable,'-m','pytest']+['tests/'+file+'.py' for file in test_files]+['-q']
for filename in ['verification_final.log','verification_command.txt','actual_model_verification.json']:
    path=AUDIT/filename
    if path.exists():(AUDIT/('before_production_repairs_'+filename)).write_bytes(path.read_bytes())
with (AUDIT/'verification_final.log').open('w') as log:
    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
(AUDIT/'verification_command.txt').write_text('Working directory: '+str(ROOT)+'\n'+' '.join(command)+'\nExit code: '+str(result.returncode)+'\n'+(AUDIT/'verification_final.log').read_text())
assert result.returncode==0,'Final tests failed; inspect verification_final.log'
for script in ['verify_audited_actual_models_20260908.py','report_audited_symbiosis_20260908.py','report_third_refinement_20260908.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts/analysis'/script)],cwd=ROOT,check=True)
manifest=read(AUDIT/'source_change_manifest.json')
for record in manifest:
    record['after_sha256']=sha(ROOT/record['file']).upper()
    record['unchanged']=record['before_sha256'].upper()==record['after_sha256']
(AUDIT/'source_change_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Final verification and all comparison artifacts completed.',flush=True)
