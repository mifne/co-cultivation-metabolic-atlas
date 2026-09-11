import numpy as np
from scipy.sparse import csr_matrix
from scripts.audit_structural_zero_flux import forced_zero_columns


def test_sign_propagation_and_reversible_escape():
    a=csr_matrix([[1.,1.,0.],[0.,1.,-1.]])
    zero,stats=forced_zero_columns(a,np.zeros(3),np.ones(3)*10)
    assert zero.all() and stats['remaining_equations']==0
    zero,_=forced_zero_columns(a,np.array([-10.,0.,0.]),np.ones(3)*10)
    assert not zero.any()
