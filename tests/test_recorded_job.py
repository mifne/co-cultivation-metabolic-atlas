import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name != "posix", reason="WSL/Linux launcher")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_detached_worker_records_exit_and_preserves_existing_files(tmp_path, exit_code):
    record, log = tmp_path/"record.json", tmp_path/"output.log"
    cmd = [sys.executable, str(ROOT/"scripts/launch_recorded_job.py"),
           "--record", str(record), "--log", str(log), "--cwd", str(ROOT), "--",
           sys.executable, "-c", f"print('test worker'); raise SystemExit({exit_code})"]
    launch = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert launch.returncode == 0, launch.stderr
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        result = json.loads(record.read_text())
        if "finished_at_utc" in result:
            break
        time.sleep(.02)
    assert result["returncode"] == exit_code
    assert result["status"] == ("completed" if exit_code == 0 else "failed")
    assert result["worker_pid"] != result["supervisor_pid"]
    assert "test worker" in log.read_text()
    before = record.read_bytes()
    repeat = subprocess.run(cmd, capture_output=True, timeout=10)
    assert repeat.returncode != 0
    assert record.read_bytes() == before
