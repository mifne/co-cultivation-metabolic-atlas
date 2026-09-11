"""Cluster-shared GPU factors as preconditioners, never shared LP solutions.

The mean CURRENT scaled K_delta of active members supplies one factor per
cluster. Every member retains its own RHS, scaling, original K0 operator,
full nonlinear forcing test and final unchanged LP certificate. Multi-RHS
triangular solves batch the independent directions. No CPU solver is used.
"""
import numpy as np
from .gpu_sparse_factor import UniformCudssFactor


class GpuSharedNewtonFactor:
    def __init__(self,solver,group_size):
        if type(group_size) is not int or group_size<2 or solver.batch%group_size:
            raise ValueError('Shared factor group size must divide the environment batch')
        self.solver=solver;self.cp=solver.cp;self.group_size=group_size
        self.groups=solver.batch//group_size
        self.factor=UniformCudssFactor(solver.factor.host_pattern,batch_size=self.groups,nrhs=group_size,
            matrix_type=solver.matrix_type,refinement_steps=solver.factor_refinements)
        self.shape=(solver.batch,solver.factor.nnz)

    def factor_newton(self,values,active=None):
        s=self.solver;cp=self.cp;s.factor._context();self.factor._context()
        if (not isinstance(values,cp.ndarray) or values.shape!=self.shape or values.dtype!=cp.float64
                or values.device.id!=s.factor.device):raise ValueError('Current scaled per-environment KKT values required')
        if active is None:active=cp.ones(s.batch,dtype=cp.bool_)
        if (not isinstance(active,cp.ndarray) or active.dtype!=cp.bool_ or active.shape!=(s.batch,)
                or active.device.id!=s.factor.device):raise ValueError('Current per-environment active mask required')
        grouped=values.reshape(self.groups,self.group_size,-1)
        mask=active.reshape(self.groups,self.group_size)
        count=cp.sum(mask,axis=1)
        mean=cp.sum(cp.where(mask[:,:,None],grouped,0.),axis=1)/cp.maximum(1,count)[:,None]
        mean=cp.where((count>0)[:,None],mean,grouped[:,0,:])
        self.factor.factor(cp.ascontiguousarray(mean))

    def solve(self,rhs):
        s=self.solver;cp=self.cp;s.factor._context();self.factor._context()
        n=s.factor.n
        if (not isinstance(rhs,cp.ndarray) or rhs.shape!=(s.batch,n,1) or rhs.dtype!=cp.float64
                or rhs.device.id!=s.factor.device):raise ValueError('Exact FP64 independent RHS lanes required')
        packed=cp.ascontiguousarray(rhs[:,:,0].reshape(self.groups,self.group_size,n).transpose(0,2,1))
        result=self.factor._solve_columns_device_checked(packed)
        return cp.ascontiguousarray(result.transpose(0,2,1).reshape(s.batch,n,1))

    def close(self):self.factor.close()
