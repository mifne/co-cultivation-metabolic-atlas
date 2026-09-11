"""Experimental multiple-halfspace projection inside an affine GPU subspace.

This is a proposal, never an acceptance decision. Current G/h select violated
rows; no saved CPU solution or future input is used. A small nonnegative dual
QP replaces repeated factorizations, without weakening the original LP gate.
"""
import numpy as np
from .gpu_affine_feasibility import GpuAffineFeasibility


SOURCE=r'''
extern "C" __global__ void directions(const int* rp,const int* ci,const double* a,
 const double* p,const int* rows,const double* norms,double* w,int n,int ng,int k) {
 int lane=blockIdx.x,slot=blockIdx.y,row=lane*ng+rows[lane*k+slot];
 double scale=sqrt(fmax(norms[row],1e-30));
 for(int j=threadIdx.x;j<n;j+=blockDim.x){double v=0.;
  if(norms[row]>1e-14)for(int t=rp[row];t<rp[row+1];t++)
    v+=a[t]*p[(long long)(ci[t]-lane*n)*n+j];
  w[((long long)lane*k+slot)*n+j]=v/scale;
 }
}
extern "C" __global__ void dual_qp(const double* gram,const double* r,
 double* lambda,int k,int sweeps) {
 int lane=blockIdx.x;if(threadIdx.x)return;
 const double* q=gram+(long long)lane*k*k;
 double* l=lambda+lane*k;
 for(int j=0;j<k;j++)l[j]=0.;
 for(int sweep=0;sweep<sweeps;sweep++)for(int j=0;j<k;j++){
  double v=r[lane*k+j];for(int t=0;t<k;t++)v-=q[j*k+t]*l[t];
  l[j]=fmax(0.,l[j]+v/fmax(q[j*k+j],1e-12));
 }
}
'''


class GpuBlockAffineFeasibility(GpuAffineFeasibility):
    def __init__(self,solver,*,block_size=32,dual_sweeps=40,**kwargs):
        if type(block_size) is not int or not 1<=block_size<=min(256,solver.ng):
            raise ValueError('Block size must be an integer in [1, min(256, ng)]')
        if type(dual_sweeps) is not int or not 1<=dual_sweeps<=1000:
            raise ValueError('Dual sweeps must be an integer in [1, 1000]')
        super().__init__(solver,**kwargs)
        self.block_size=block_size;self.dual_sweeps=dual_sweeps
        self.direction_kernel=self.cp.RawKernel(SOURCE,'directions')
        self.dual_kernel=self.cp.RawKernel(SOURCE,'dual_qp')

    def _iterate(self,x,d,iterations,chunk,tolerance):
        s=self.solver;cp=self.cp;k=self.block_size
        w=cp.empty((s.batch,k,s.n),dtype=cp.float64)
        multipliers=cp.empty((s.batch,k),dtype=cp.float64)
        root=cp.sqrt(cp.maximum(d,1e-30))
        completed=0;history=[]
        while completed<iterations:
            for _ in range(min(chunk,iterations-completed)):
                residual=(s.g@x.ravel()).reshape(s.batch,s.ng)-s.h
                scores=cp.where(d>1e-14,residual/root,-cp.inf)
                rows=cp.ascontiguousarray(cp.argsort(scores,axis=1)[:,-k:],dtype=cp.int32)
                r=cp.ascontiguousarray(cp.take_along_axis(scores,rows,axis=1))
                # Rows with zero affine direction cannot be corrected. Keep
                # their violation in the final original gate, not in this QP.
                r=cp.where(cp.isfinite(r),r,0.)
                self.direction_kernel((s.batch,k),(256,),
                    (s.g.indptr,s.g.indices,s.g.data,self.p,rows,d,w,
                     np.int32(s.n),np.int32(s.ng),np.int32(k)))
                gram=cp.ascontiguousarray(w@w.transpose(0,2,1))
                self.dual_kernel((s.batch,),(1,),
                    (gram,r,multipliers,np.int32(k),np.int32(self.dual_sweeps)))
                x-=cp.einsum('bk,bkn->bn',multipliers,w)
                completed+=1
            maximum=float(cp.max(s.g@x.ravel()-s.h.ravel()))
            history.append([completed,maximum])
            if maximum<=tolerance or not np.isfinite(maximum):break
        return x,completed,history
