"""GPU Motzkin proposals in a numerically computed affine nullspace.

No equality is removed from the original LP. Truncated SVD is used only to
propose a point. Full original-unit primal/dual certification is mandatory.
The one-coordinate objective is fixed to its box optimum in this auxiliary
feasibility search; failure does not prove original LP infeasibility.
"""
import time
import numpy as np
from .gpu_batched_ipm import constraint_form

SOURCE=r'''
extern "C" __global__ void norms(const int* rp,const int* ci,const double* a,
 const double* p,double* d,int n,int ng,int batch) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=batch*ng)return;
 int lane=row/ng;double v=0.;
 for(int k=rp[row];k<rp[row+1];k++) for(int l=rp[row];l<rp[row+1];l++)
   v+=a[k]*a[l]*p[(long long)(ci[k]-lane*n)*n+(ci[l]-lane*n)];
 d[row]=v;
}
extern "C" __global__ void choose(const int* rp,const int* ci,const double* a,
 const double* h,const double* x,const double* d,int* chosen,double* alpha,
 double* violation,int n,int ng,double tol) {
 __shared__ double scores[256],steps[256],worst[256];__shared__ int rows[256];
 int lane=blockIdx.x,tid=threadIdx.x;double best=0.,step=0.,mx=0.;int pick=-1;
 for(int j=tid;j<ng;j+=256) {
  int row=lane*ng+j;double r=-h[row];
  for(int k=rp[row];k<rp[row+1];k++)r+=a[k]*x[ci[k]];
  mx=fmax(mx,isfinite(r)?r:__longlong_as_double(0x7ff0000000000000LL));
  double score=(r>tol && d[row]>1e-14)?r/sqrt(d[row]):0.;
  if(score>best){best=score;pick=row;step=r/d[row];}
 }
 scores[tid]=best;steps[tid]=step;rows[tid]=pick;worst[tid]=mx;__syncthreads();
 for(int k=128;k;k>>=1){if(tid<k){
  if(scores[tid+k]>scores[tid]){scores[tid]=scores[tid+k];steps[tid]=steps[tid+k];rows[tid]=rows[tid+k];}
  worst[tid]=fmax(worst[tid],worst[tid+k]);}__syncthreads();}
 if(!tid){chosen[lane]=rows[0];alpha[lane]=steps[0];violation[lane]=worst[0];}
}
extern "C" __global__ void project(const int* rp,const int* ci,const double* a,
 const double* p,double* x,const int* chosen,const double* alpha,int n,int ng) {
 int lane=blockIdx.x,row=chosen[lane];if(row<0)return;
 for(int j=threadIdx.x;j<n;j+=256){double w=0.;
  for(int k=rp[row];k<rp[row+1];k++) w+=a[k]*p[(long long)(ci[k]-lane*n)*n+j];
  x[lane*n+j]-=alpha[lane]*w;
 }
}
'''


class GpuAffineFeasibility:
    def __init__(self,solver,*,rcond=1e-12):
        started=time.perf_counter();self.solver=solver;cp=self.cp=solver.cp
        solver.factor._context()
        if not np.isfinite(rcond) or not 0.<rcond<1.:raise ValueError('SVD proposal threshold required')
        forms=[constraint_form(p) for p in solver.problems]
        e=forms[0][0]
        for other in forms:
            if (other[0].shape!=e.shape or not np.array_equal(other[0].indptr,e.indptr)
                or not np.array_equal(other[0].indices,e.indices) or not np.array_equal(other[0].data,e.data)):
                raise ValueError('Shared exact equality operator required for this prototype')
        cost=solver.problems[0][4];positions=np.flatnonzero(cost)
        if len(positions)!=1 or cost[positions[0]]>=0.:
            raise ValueError('Prototype requires one negative objective coordinate with an upper bound')
        self.objective=int(positions[0])
        upper=np.flatnonzero(forms[0][-1]==self.objective)
        if len(upper)!=1:raise ValueError('Finite optimizing upper bound required')
        self.upper_row=solver.q+len(solver.il)+int(upper[0])
        self.base_cost=solver.c.copy()
        augmented=cp.concatenate((cp.asarray(e.toarray()),cp.eye(1,solver.n,k=self.objective)),axis=0)
        u,s,v=cp.linalg.svd(augmented,full_matrices=False)
        keep=s>rcond*s[0];self.rank=int(cp.count_nonzero(keep))
        self.inverse=(u[:,keep]/s[keep])@v[keep,:]
        self.p=cp.eye(solver.n)-v[keep,:].T@v[keep,:]
        self.p=cp.ascontiguousarray((self.p+self.p.T)*.5)
        self.expected=(solver.g.indptr.copy(),solver.g.indices.copy(),self.p.copy())
        self.expected_e=solver.e.data.copy()
        self.expected_inverse=self.inverse.copy()
        self.g_shape=solver.g.shape
        self.dimensions=(solver.batch,solver.n,solver.ng,solver.ne)
        self.norm_kernel=cp.RawKernel(SOURCE,'norms');self.choose_kernel=cp.RawKernel(SOURCE,'choose')
        self.project_kernel=cp.RawKernel(SOURCE,'project')
        cp.cuda.get_current_stream().synchronize();self.setup_seconds=time.perf_counter()-started

    def propose(self,initial=None,*,iterations=2000,chunk=100,tolerance=1e-7):
        s=self.solver;cp=self.cp;s.factor._context();started=time.perf_counter()
        if (type(iterations) is not int or iterations<1 or type(chunk) is not int or chunk<1
                or not np.isfinite(tolerance) or tolerance<=0.):raise ValueError('Positive proposal budget required')
        if self.dimensions!=(s.batch,s.n,s.ng,s.ne) or s.g.shape!=self.g_shape:
            raise ValueError('Projection coordinates changed')
        arrays=[(self.p,(s.n,s.n),cp.float64),(self.inverse,(s.ne+1,s.n),cp.float64),
            (s.h,(s.batch,s.ng),cp.float64),(s.b,(s.batch,s.ne),cp.float64),
            (s.g.indptr,(s.batch*s.ng+1,),cp.int32),(s.g.indices,self.expected[1].shape,cp.int32),
            (s.g.data,self.expected[1].shape,cp.float64)]
        if any(not isinstance(v,cp.ndarray) or v.shape!=shape or v.dtype!=dtype
            or v.device.id!=s.factor.device or not v.flags.c_contiguous for v,shape,dtype in arrays):
            raise ValueError('Projection arrays require exact contiguous device coordinates')
        if not bool(cp.all(cp.stack([cp.array_equal(s.g.indptr,self.expected[0]),
            cp.array_equal(s.g.indices,self.expected[1]),cp.array_equal(self.p,self.expected[2]),
            cp.array_equal(s.e.data,self.expected_e),cp.array_equal(self.inverse,self.expected_inverse),
            cp.array_equal(s.c,self.base_cost),cp.all(cp.isfinite(s.g.data)),cp.all(cp.isfinite(s.h)),
            cp.all(cp.isfinite(s.b))]))):
            raise ValueError('Projection static layout/cost or current numeric values invalid')
        b=cp.concatenate((s.b,s.h[:,self.upper_row,None]),axis=1)
        particular=b@self.inverse
        if initial is None:x=particular.copy()
        else:
            if (not isinstance(initial,cp.ndarray) or initial.dtype!=cp.float64
                    or initial.shape!=(s.batch,s.n) or initial.device.id!=s.factor.device
                    or not bool(cp.all(cp.isfinite(initial)))):
                raise ValueError('Finite current-device primal proposal required')
            x=cp.ascontiguousarray(particular+(initial-particular)@self.p)
        d=cp.empty((s.batch,s.ng),dtype=cp.float64)
        ni,ngi,bi=map(np.int32,(s.n,s.ng,s.batch))
        self.norm_kernel(((s.batch*s.ng+255)//256,),(256,),
            (s.g.indptr,s.g.indices,s.g.data,self.p,d,ni,ngi,bi))
        x,completed,history=self._iterate(x,d,iterations,chunk,tolerance)
        cp.cuda.get_current_stream().synchronize()
        return x,dict(iterations=completed,violation_history=history,seconds=time.perf_counter()-started,
            numerical_svd_rank=self.rank,setup_seconds=self.setup_seconds,cpu_lp_calls=0,
            original_certificate_required=True,auxiliary_failure_does_not_prove_infeasibility=True)

    def _iterate(self,x,d,iterations,chunk,tolerance):
        s=self.solver;cp=self.cp
        chosen=cp.empty(s.batch,dtype=cp.int32);alpha=cp.empty(s.batch,dtype=cp.float64)
        violation=cp.empty(s.batch,dtype=cp.float64)
        ni,ngi=map(np.int32,(s.n,s.ng))
        completed=0;history=[]
        while completed<iterations:
            count=min(chunk,iterations-completed)
            for _ in range(count):
                self.choose_kernel((s.batch,),(256,),(s.g.indptr,s.g.indices,s.g.data,
                    s.h,x,d,chosen,alpha,violation,ni,ngi,np.float64(tolerance)))
                self.project_kernel((s.batch,),(256,),(s.g.indptr,s.g.indices,s.g.data,
                    self.p,x,chosen,alpha,ni,ngi))
            completed+=count
            # Assess the returned iterate, not the point before the last update.
            maximum=float(cp.max(s.g@x.ravel()-s.h.ravel()));history.append([completed,maximum])
            if maximum<=tolerance or not np.isfinite(maximum):break
        return x,completed,history
