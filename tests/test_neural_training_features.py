import numpy as np
from scripts.train_neural_basis_proposals import training_feature_indices


def test_later_state_features_included_without_validation_leakage():
    # Column 1 changes only in held-out data; columns 0 and 2 are duplicates
    # in training. Column 3 captures a late training-state change.
    x=np.array([[0,0,0,0],[1,0,1,0],[2,0,2,1],[3,9,8,2]],dtype=float)
    selected=training_feature_indices(x,np.array([0,1,2]))
    assert 1 not in selected and 3 in selected
    assert len(set(selected)&{0,2})==1


def test_constant_training_features_have_safe_single_input():
    assert training_feature_indices(np.zeros((4,3)),[0,1]).tolist()==[0]


def test_joint_training_preserves_splits_and_feature_identity():
    from scripts.merge_basis_router_training import align_datasets
    a=dict(indices=np.array([4,2]),x=np.array([[40,20],[41,21]]),y=np.array([0,1]),train=np.array([0]),validation=np.array([1]))
    b=dict(indices=np.array([2,7,4]),x=np.array([[22,70,42],[23,71,43]]),y=np.array([1,0]),train=np.array([1]),validation=np.array([0]))
    indices,x,y,train,test=align_datasets([a,b])
    assert indices.tolist()==[2,4]
    assert x.tolist()==[[20,40],[21,41],[22,42],[23,43]]
    assert train.tolist()==[0,3] and test.tolist()==[1,2]
