from types import SimpleNamespace
import threading
import numpy as np
from greenlet import getcurrent
from scripts.microbatch_comparison_support import ParallelCpuLP,drive_microbatch


def test_parallel_cpu_matches_independent_lp():
    service=ParallelCpuLP(2)
    try:
        results=service.solve_batch([(np.array([-1.]),dict(bounds=[(0.,bound)],method='highs-ds'))
                                     for bound in (1.,3.)])
    finally:service.close()
    assert all(result.success for result in results)
    np.testing.assert_allclose([result.fun for result in results],[-1.,-3.])
    assert service.history[0]['cpu_lp_calls']==2


def test_host_environment_stays_on_caller_thread():
    parent=getcurrent();thread=threading.get_ident()
    class Environment:
        def __init__(self):self.value=0.;self.model=SimpleNamespace(stats=SimpleNamespace(status='optimal;'))
        def step(self,bound):
            assert threading.get_ident()==thread
            result=parent.switch((np.array([-1.]),dict(bounds=[(0.,bound)],method='highs-ds')))
            self.value+=result.x[0]
    envs=[Environment(),Environment()];service=ParallelCpuLP(2)
    try:
        result=drive_microbatch([(env,env.model) for env in envs],[[1.,2.],[3.,4.]],service,lambda env:env.value)
    finally:service.close()
    np.testing.assert_allclose(result,[3.,7.])
    assert len(service.history)==2
