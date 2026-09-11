"""Run a full regression, then require its independent audit to succeed."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--separate-stage-cache", action="store_true")
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    run = subprocess.run([sys.executable, str(ROOT/"scripts/run_gpu_device_regression.py"),
        "--reference", str(args.reference), "--output-dir", str(args.output_dir)] +
        (["--separate-stage-cache"] if args.separate_stage_cache else []), cwd=ROOT)
    if run.returncode != 0:
        return run.returncode
    audit = subprocess.run([sys.executable, str(ROOT/"scripts/audit_gpu_device_replay.py"),
        "--reference", str(args.reference), "--replay", str(args.output_dir/"replay.json"),
        "--output", str(args.output_dir/"audit.json")], cwd=ROOT)
    return audit.returncode


if __name__ == "__main__":
    raise SystemExit(main())
