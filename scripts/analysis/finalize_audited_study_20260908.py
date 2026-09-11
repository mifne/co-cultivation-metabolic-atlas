"""Finish the already-running study's artifact pipeline; no scheduled notifications."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/audited_symbiosis_20260908'
def records(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return []
deadline=time.monotonic()+10800
while len(records(OUT/'progress.json'))<21 or len(records(OUT/'numerical_patch_reruns.json'))<3:
    if time.monotonic()>deadline:raise TimeoutError('Study incomplete; retain all attempts and inspect logs')
    time.sleep(20)
design=records(OUT/'design.json')
after={file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in design['sources']}
assert after==design['sources'],'Frozen execution sources changed before final verification'
(OUT/'execution_source_verification.json').write_text(json.dumps(dict(
    verified_at_utc=datetime.now(timezone.utc).isoformat(),after_run_sha256=after),indent=2))
subprocess.run([sys.executable,str(ROOT/'scripts/analysis/report_audited_symbiosis_20260908.py')],check=True)
print('All selected cases audited; REPORT_JA.md and comparison figures generated.',flush=True)
