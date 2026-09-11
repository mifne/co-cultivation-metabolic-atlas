import json
import numpy as np
import pytest
from src.gpu_certified_basis import compile_basis
from src.gpu_basis_bank import GpuBasisBank
from src.gpu_neural_basis_proposal import NeuralRoutedBasisBank,NeuralBasisProposal,bank_identity,features
from tests.test_gpu_revised_basis import problem,cpu


def make_bank():
    problems=[problem(),problem(rhs=(.1,3.))]
    return GpuBasisBank([compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq) for p in problems],[0])


def test_wrong_neural_proposal_cannot_bypass_certificate(monkeypatch):
    cp=pytest.importorskip('cupy')
    import highspy
    bank=make_bank()
    class Wrong:
        def predict(self,inputs):return cp.zeros(len(inputs['lower']),dtype=cp.int32)
    routed=NeuralRoutedBasisBank(bank,Wrong())
    queries=[problem(),problem(rhs=(.15,3.)),problem(cost=(2.,1.))]
    expected=[cpu(q).fun for q in queries[:2]]
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CPU LP')))
    result=routed.evaluate_device(**routed.prepare_host(queries))
    assert result['accepted'].get().tolist()==[True,True,False]
    np.testing.assert_allclose(result['objective'].get()[:2],expected,atol=1e-8)
    assert result['exhaustive_fallback_rows']==2
    assert np.isnan(result['values'].get()[2]).all() and result['cpu_lp_calls']==0
    deferred=NeuralRoutedBasisBank(bank,Wrong(),defer_exhaustive=True)
    deferred_result=deferred.evaluate_device(**deferred.prepare_host(queries))
    assert deferred_result['neural_deferred_exhaustive']
    assert deferred_result['accepted'].get().tolist()==[True,False,False]
    assert np.isnan(deferred_result['values'].get()[1:]).all()


def test_artifact_identity_and_nonfinite_input(tmp_path):
    cp=pytest.importorskip('cupy');bank=make_bank();data=bank.prepare_host([problem()])
    path=tmp_path/'network.npz';metadata=dict(bank_identity=bank_identity(bank),feature_width=features(data).shape[1])
    weights=dict(indices=np.array([0]),mean=np.zeros(1,dtype=np.float32),scale=np.ones(1,dtype=np.float32),
        w1=np.zeros((1,2),dtype=np.float32),b1=np.zeros(2,dtype=np.float32),
        w2=np.zeros((2,2),dtype=np.float32),b2=np.array([1.,0.],dtype=np.float32))
    np.savez(path,**weights,metadata=json.dumps(metadata))
    network=NeuralBasisProposal(path,bank);routed=NeuralRoutedBasisBank(bank,network)
    data['rhs'][0,0]=cp.nan
    assert not routed.evaluate_device(**data)['accepted'].get()[0]
    metadata['bank_identity']='incorrect'
    np.savez(path,**weights,metadata=json.dumps(metadata))
    with pytest.raises(ValueError,match='identity'):NeuralBasisProposal(path,bank)
