"""Serialized, rotated-order comparison of within-batch proof reuse and CPU16."""
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    jobs=[]
    for trial,order in enumerate((('baseline','proof_reuse','cpu16'),
                                 ('proof_reuse','cpu16','baseline'),
                                 ('cpu16','baseline','proof_reuse')),1):
        for group in order:
            output=ROOT/f'results/pf_ipm_proof_{group}_32_2to8_20260907_r{trial}.json'
            if output.exists():raise FileExistsError(output)
            env=os.environ.copy()
            env.update(CUPY_ACCELERATORS='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            command=[sys.executable,str(ROOT/'scripts/benchmark_ipm_sequence.py'),
                '--batch','32','--steps','2','3','4','5','6','7','8','--output',str(output)]
            if group=='cpu16':
                env['PYTHONPATH']=str(ROOT/'tmp/highspy-1.15.1-comparison')
                command+=['--mode','cpu','--workers','16']
            else:
                command+=['--mode','gpu','--reuse-numeric-workspace','--device-numeric-updates',
                    '--restart-mu','0.00001','--factor-layout','block_diagonal']
                if group=='proof_reuse':command+=['--reuse-equality-proofs']
            jobs.append((trial,group,command,env))
    for trial,group,command,env in jobs:
        print(f'Proof comparison {trial}: {group}',flush=True)
        subprocess.run(command,cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
