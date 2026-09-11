"""Intrusive cuDSS phase diagnostic; NOT a speed-comparison benchmark.

Each native factor/solve gets CUDA events and an explicit completion fence.
Host-return time, device event interval and wall interval are separate. Events
include launch gaps and are NOT summed per-kernel busy time.
"""
import time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cupy as cp
from src.gpu_sparse_factor import UniformCudssFactor
from src.gpu_forest_ipm import ForestGpuBatchedIPM
from scripts import benchmark_ipm_sequence


def main():
    native=UniformCudssFactor._execute;solve=ForestGpuBatchedIPM.solve
    active=None
    def measured_native(self,phase):
        if active is None or phase not in (4,1008):return native(self,phase)
        first=cp.cuda.Event();last=cp.cuda.Event();first.record()
        start=time.perf_counter();result=native(self,phase);returned=time.perf_counter()
        last.record();last.synchronize();end=time.perf_counter()
        active.setdefault(str(phase),[]).append(dict(host_return_seconds=returned-start,
            completion_wall_seconds=end-start,event_seconds=cp.cuda.get_elapsed_time(first,last)*.001))
        return result
    def measured_solve(self,**kwargs):
        nonlocal active
        active={}
        try:
            result=solve(self,**kwargs)
            result['intrusive_native_profile']=active
            summary={phase:{name:sum(row[name] for row in rows) for name in rows[0]}|
                dict(count=len(rows)) for phase,rows in active.items()}
            print('NATIVE_PROFILE '+str(summary),flush=True)
            result['intrusive_native_profile_summary']=summary
            result['native_profile_source']=(ROOT/'scripts/profile_ipm_native_phases.py').read_text()
            return result
        finally:active=None
    UniformCudssFactor._execute=measured_native
    ForestGpuBatchedIPM.solve=measured_solve
    try:benchmark_ipm_sequence.main()
    finally:
        UniformCudssFactor._execute=native
        ForestGpuBatchedIPM.solve=solve


if __name__=='__main__':main()
