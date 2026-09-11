import numpy as np
import pytest


@pytest.mark.parametrize('backend',['augmented','dual_schur'])
def test_cuda_separate_steps_preserve_original_lp_gates(backend):
    from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
    from tests.test_gpu_dual_schur import inputs,options
    opts=options();opts.update(newton_backend=backend,step_policy='primal_dual')
    with GpuCondensedBatchedIPM(inputs(),**opts) as s:
        r=s.solve(iterations=100)
        assert r['accepted'].all()
        steps=np.asarray(r['primal_dual_step_diagnostics'])
        assert steps.ndim==3 and np.isfinite(steps).all()
        assert np.all((steps>=0.)&(steps<=1.))
        history=np.asarray(r['globalization_diagnostics'])
        assert np.all(history[:,:,4]<=history[:,:,3])


def test_separate_steps_reject_unverified_compound_configuration():
    from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
    from tests.test_gpu_dual_schur import inputs,options
    opts=options();opts.update(step_policy='primal_dual',centrality_corrections=1)
    with pytest.raises(ValueError,match='Separate primal/dual'):GpuCondensedBatchedIPM(inputs(),**opts)
