"""Small diagnostic of replayable GPU-only solver operations."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cupy as cp
from tests.test_gpu_revised_basis import problem
from src.gpu_certified_basis import compile_basis
from src.gpu_revised_basis import GpuRevisedBasis
from src.gpu_conditional_capture import ConditionalCapture

p=problem()
s=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=24,capture_safe=True)
q=s.prepare_host([p])
s.solve_device(**q)
cp.cuda.get_current_stream().synchronize()
stream=cp.cuda.Stream(non_blocking=True)
pool=cp.cuda.MemoryPool()
conditional=ConditionalCapture()
with cp.cuda.using_allocator(pool.malloc),stream:
    warm=s.solve_device(**q);stream.synchronize();del warm
    stream.begin_capture()
    r=s.solve_device(**q,conditional=conditional)
    g=stream.end_capture()
    g.launch(stream)
stream.synchronize()
print('capture',r['accepted'].get(),flush=True)
for name,fn in [('eager',lambda:s.solve_device(**q)),('graph',lambda:g.launch(stream))]:
    started=time.perf_counter()
    for _ in range(20):fn()
    cp.cuda.Device().synchronize()
    print(name,(time.perf_counter()-started)/20,flush=True)
