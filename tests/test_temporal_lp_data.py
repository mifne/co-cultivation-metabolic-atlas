import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.temporal_lp_data import fit_codec, trajectory_arrays, TemporalLPCodec


def toy(seed):
    return [( (csr_matrix([[1.,float(seed+t)]]), np.array([2.+t]),
        np.zeros(2), np.ones(2)*10, np.array([-1.,0.]),0),
        np.array([2.+t,0.]),np.array([-1.])) for t in range(3)]


def test_fit_never_uses_development_targets_or_matrix_changes(tmp_path):
    data = [toy(i) for i in (1,2,3)]
    first = fit_codec(data,[0,1],rank=2,max_features=4)
    data[2] = toy(999)
    for p,x,y in data[2]:
        x[:] = 10000; y[:] = 30000
        p[0][0,0] = 15
    second = fit_codec(data,[0,1],rank=2,max_features=4)
    for k in first.arrays:
        np.testing.assert_array_equal(first.arrays[k],second.arrays[k])
    assert 0 not in first.arrays['matrix_keys']
    first.save(tmp_path/'codec.npz')
    restored = TemporalLPCodec.load(tmp_path/'codec.npz')
    assert restored.feature_dim == first.feature_dim
    with pytest.raises(FileExistsError):first.save(tmp_path/'codec.npz')


def test_previous_and_delta_are_causal_and_first_step_has_no_reference():
    data = [toy(i) for i in (1,2,3)]
    codec = fit_codec(data,[0,1],rank=2,max_features=4)
    x,prev,labels,target = trajectory_arrays(data,codec)
    assert np.all(prev[:,0] == 0) and np.all(x[:,0,-1] == 0)
    np.testing.assert_array_equal(prev[:,1:],labels[:,:-1])
    assert x.shape[-1] == codec.feature_dim
    assert codec.decode(labels).shape == target.shape


def test_invalid_codec_or_training_split_fails():
    data = [toy(i) for i in (1,2,3)]
    with pytest.raises(ValueError):fit_codec(data,[0,0])
    codec = fit_codec(data,[0,1],rank=2,max_features=4)
    with pytest.raises(ValueError):
        TemporalLPCodec(dict(codec.arrays,target_scale=np.zeros(3)),codec.metadata)
