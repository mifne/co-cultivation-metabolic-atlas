"""Verify the frozen shared-oxygen study and produce its final artifacts."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/audited_symbiosis_shared_oxygen_20260908'
AUDIT=ROOT/'results/cultivation_model_audit_20260908';FOLLOW=AUDIT/'oxygen_quota_followup'
def read(path):return json.loads(path.read_text())
deadline=time.monotonic()+10800
while not (OUT/'execution_source_verification.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Shared-oxygen study incomplete; inspect retained logs')
    time.sleep(20)
progress=read(OUT/'progress.json');assert len(progress)==23 and all(r['returncode']==0 for r in progress),progress
design=read(OUT/'design.json')
assert all(hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha for file,sha in design['sources'].items())
names=['test_audited_dfba','test_cultivation_numerics','test_resolved_dfba','test_one_l_jar',
    'test_graph_learning_curve','test_trace_script_lifecycle','test_rl_audit_contract','test_main_cultivation_profiles',
    'test_legacy_evaluation_callers','test_model_selection','test_ppo_agent_stability','test_ppo_observation_schema',
    'test_ppo_reward_scale','test_rl_feed_action_scaling','test_polymer_pathway_curation','test_ns21_pha_pathway',
    'test_propionibacterium_helper','test_audited_c30_roundoff','test_audited_primary_recomputation',
    'test_audited_production_repairs','test_shared_oxygen_coupling']
command=[sys.executable,'-m','pytest']+['tests/'+name+'.py' for name in names]+['-q']
with (FOLLOW/'verification_final.log').open('w') as log:
    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
(AUDIT/'verification_final.log').write_bytes((FOLLOW/'verification_final.log').read_bytes())
(AUDIT/'verification_command.txt').write_text('Working directory: '+str(ROOT)+'\n'+' '.join(command)+'\nExit code: '+str(result.returncode)+'\n'+(FOLLOW/'verification_final.log').read_text())
assert result.returncode==0,'Final shared-oxygen tests failed'
for script in ['verify_audited_actual_models_20260908.py','report_shared_oxygen_20260908.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts/analysis'/script)],cwd=ROOT,check=True)
manifest=read(AUDIT/'source_change_manifest.json')
for record in manifest:
    record['after_sha256']=hashlib.sha256((ROOT/record['file']).read_bytes()).hexdigest().upper()
    record['unchanged']=record['before_sha256'].upper()==record['after_sha256']
(AUDIT/'source_change_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Shared-oxygen study, final tests, source checks and reports completed.',flush=True)
