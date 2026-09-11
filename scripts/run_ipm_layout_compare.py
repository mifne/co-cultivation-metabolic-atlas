"""Serialized, fixed-order development comparison; never overlap GPU and CPU."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    jobs=[]
    for trial in (1,2,3):
        for layout in ('uniform','block_diagonal','cpu16'):
            output=ROOT/f'results/pf_ipm_layout_{layout}_32_2to8_20260907_r{trial}.json'
            if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
            command=[sys.executable,str(ROOT/'scripts/benchmark_ipm_sequence.py'),
                '--batch','32','--steps','2','3','4','5','6','7','8','--output',str(output)]
            env=os.environ.copy()
            env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUPY_ACCELERATORS='')
            if layout=='cpu16':
                env['PYTHONPATH']=str(ROOT/'tmp/highspy-1.15.1-comparison')
                command.extend(['--mode','cpu','--workers','16'])
            else:
                command.extend(['--mode','gpu','--reuse-numeric-workspace','--device-numeric-updates',
                    '--restart-mu','0.00001','--factor-layout',layout])
            jobs.append((trial,layout,command,env))
    for trial,layout,command,env in jobs:
        print(f'Comparison trial {trial}: {layout}',flush=True)
        subprocess.run(command,env=env,cwd=ROOT,check=True)


if __name__=='__main__':main()
