"""One-shot Linux job launcher with durable lifecycle records (not a scheduler).

Run in WSL. The detached supervisor and worker do not inherit the terminal's
stdin/session. A stale 'running' record must still be checked against the
recorded PID and /proc start time. SIGKILL, WSL shutdown or power loss cannot
be caught; in that case absence of an exit record means interrupted/unknown.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (FileNotFoundError, IndexError, ProcessLookupError):
        return None


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def supervise(record_path, command):
    record = json.loads(record_path.read_text())
    record.update(status="starting", supervisor_pid=os.getpid(),
                  supervisor_start_ticks=process_start(os.getpid()))
    save(record_path, record)
    child = None
    requested_signal = None

    def terminate(signum, frame):
        nonlocal requested_signal
        requested_signal = signum
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        if requested_signal is not None:
            raise InterruptedError("Signal before worker startup")
        child = subprocess.Popen(command, cwd=record["cwd"], stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        record.update(status="running", started_at_utc=stamp(), worker_pid=child.pid,
                      worker_start_ticks=process_start(child.pid))
        save(record_path, record)
        code = child.wait()
        record.update(status="completed" if code == 0 else "interrupted" if code < 0 else "failed",
                      returncode=code, finished_at_utc=stamp(), requested_signal=requested_signal)
    except BaseException as error:
        record.update(status="supervisor_error", error=repr(error), finished_at_utc=stamp())
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    save(record_path, record)
    return 0 if record["status"] == "completed" else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--log", type=Path)
    p.add_argument("--cwd", type=Path, default=ROOT)
    p.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("Run this launcher inside WSL/Linux")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("A command is required")
    record_path = args.record.resolve()
    if args.supervise:
        return supervise(record_path, command)
    if args.log is None:
        raise ValueError("--log is required")
    log_path = args.log.resolve()
    if record_path == log_path:
        raise ValueError("Record and log must be separate files")
    if record_path.exists() or log_path.exists():
        raise FileExistsError("Never overwrite an existing job record/log")
    if not args.cwd.is_dir():
        raise NotADirectoryError(args.cwd)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(status="launch_requested", created_at_utc=stamp(), command=command,
                  cwd=str(args.cwd.resolve()), log=str(log_path),
                  environment={key: os.environ.get(key) for key in
                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    with record_path.open("x") as target:
        json.dump(record, target, indent=2)
    with log_path.open("xb") as output:
        supervisor = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            "--supervise", "--record", str(record_path), "--", *command],
            cwd=record["cwd"], stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    print(json.dumps(dict(supervisor_pid=supervisor.pid, record=str(record_path), log=str(log_path))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
