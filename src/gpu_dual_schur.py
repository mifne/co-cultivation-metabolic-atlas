"""GPU normal-equation preconditioner for a diagonal-primal Newton system.

For K_delta = [[D,H.T],[H,-R]], factor SPD M=R+H inv(D) H.T,
solve M u=H inv(D) p-q, and recover dx=inv(D)(p-H.T u).
This is elimination of a LINEAR SYSTEM, not reduction of LP constraints.
Full original Newton forcing and original LP certification remain mandatory.
Host work compiles sparsity recipes once; current numerical work stays on GPU.
"""
import time
import numpy as np
from scipy.sparse import csr_matrix,block_diag,vstack
from .gpu_batched_ipm import constraint_form,uniform_kkt_pattern
from .gpu_sparse_factor import UniformCudssFactor


_SOURCE=r'''
extern "C" __global__ void schur_values(const int* offsets,const int* left,
 const int* right,const int* column,const int* diagonal,const double* h,
 const double* d,const double* r,double* out,int entries,int nnz,int m) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=entries)return;
 int lane=j/nnz,row=diagonal[j%nnz];double value=row<0?0.:r[lane*m+row];
 for(int k=offsets[j];k<offsets[j+1];k++)value+=h[left[k]]*h[right[k]]/d[column[k]];
 out[j]=value;
}
'''


class GpuDualSchurNewton:
    def __init__(self,solver):
        started=time.perf_counter();self.solver=solver;cp=self.cp=solver.cp
        solver.factor._context()
        if not solver._condense_bounds or solver.ne<1:
            raise ValueError('Dual Schur requires bound condensation and nonempty equality block')
        self.m=solver.ne+solver.q
        forms=[constraint_form(p) for p in solver.problems]
        host_h=[vstack((f[0],f[2][:solver.q]),format='csr') for f in forms]
        patterns=[]
        for a in host_h:
            structural=a.copy();structural.data[:]=1.
            patterns.append((structural@structural.T).tocsr())
        pattern,_,diagonal=uniform_kkt_pattern(patterns)
        m=self.m;nnz=pattern.nnz
        pattern_rows=np.repeat(np.arange(m,dtype=np.int64),np.diff(pattern.indptr))
        keys=pattern_rows*m+pattern.indices
        diagonal_rows=np.where(pattern_rows==pattern.indices,pattern_rows,-1).astype(np.int32)
        targets=[];left=[];right=[];columns=[];ho=0
        edest=[];esource=[];gdest=[];gsource=[];eo=go=0
        for lane,(a,f) in enumerate(zip(host_h,forms)):
            en=f[0].nnz;gn=f[2][:solver.q].nnz
            edest.extend(range(ho,ho+en));esource.extend(range(eo,eo+en))
            gdest.extend(range(ho+en,ho+en+gn));gsource.extend(range(go,go+gn))
            tagged=csr_matrix((np.arange(a.nnz,dtype=np.int64)+ho,a.indices,a.indptr),shape=a.shape).tocsc()
            for column in range(a.shape[1]):
                first,last=tagged.indptr[column:column+2]
                rows=tagged.indices[first:last];ids=tagged.data[first:last]
                count=len(rows)
                if not count:continue
                rr=np.repeat(rows,count);cc=np.tile(rows,count)
                dest=np.searchsorted(keys,rr.astype(np.int64)*m+cc)
                targets.extend((lane*nnz+dest).tolist())
                left.extend(np.repeat(ids,count).tolist());right.extend(np.tile(ids,count).tolist())
                columns.extend([lane*solver.n+column]*(count*count))
            ho+=a.nnz;eo+=f[0].nnz;go+=f[2].nnz
        if max(len(targets),ho,solver.batch*nnz,solver.batch*solver.n)>=2**31:
            raise ValueError('Dual Schur recipe exceeds supported int32 capacity')
        order=np.argsort(targets,kind='stable')
        offsets=np.r_[0,np.cumsum(np.bincount(targets,minlength=solver.batch*nnz))]
        self.offsets=cp.asarray(offsets,dtype=cp.int32)
        self.left=cp.asarray(np.asarray(left)[order],dtype=cp.int32)
        self.right=cp.asarray(np.asarray(right)[order],dtype=cp.int32)
        self.column=cp.asarray(np.asarray(columns)[order],dtype=cp.int32)
        self.diagonal_rows=cp.asarray(diagonal_rows)
        self.diagonal=cp.asarray(diagonal)
        self.rows=cp.asarray(pattern_rows,dtype=cp.int32);self.columns=cp.asarray(pattern.indices,dtype=cp.int32)
        self.edest=cp.asarray(edest,dtype=cp.int32);self.esource=cp.asarray(esource,dtype=cp.int32)
        self.gdest=cp.asarray(gdest,dtype=cp.int32);self.gsource=cp.asarray(gsource,dtype=cp.int32)
        from cupyx.scipy.sparse import csr_matrix as device_csr
        h=block_diag(host_h,format='csr')
        self.h=device_csr(h);self.ht=self.h.T.tocsr()
        tags=csr_matrix((np.arange(h.nnz,dtype=np.float64),h.indices,h.indptr),shape=h.shape)
        self.transpose_order=cp.asarray(tags.T.tocsr().data,dtype=cp.int32)
        self._static_names=('offsets','left','right','column','diagonal_rows','diagonal',
            'rows','columns','edest','esource','gdest','gsource','transpose_order')
        self._static=[(getattr(self,n),getattr(self,n).copy()) for n in self._static_names]
        for a in (solver.e,solver.g,self.h,self.ht):
            self._static.extend((v,v.copy()) for v in (a.indptr,a.indices))
        self._metadata=tuple((v,v.shape,v.strides,v.dtype.str,v.device.id,v.data.ptr) for v,_ in self._static)
        self.values=cp.empty((solver.batch,nnz),dtype=cp.float64)
        self._dimensions=(solver.batch,solver.n,solver.ne,solver.ng,solver.q,
            solver.e.shape,solver.g.shape,self.h.shape,self.ht.shape)
        self._numeric_metadata=tuple((v,v.shape,v.strides,v.dtype.str,v.device.id,v.data.ptr)
            for v in (solver.e.data,solver.g.data,self.h.data,self.ht.data,self.values))
        self.kernel=cp.RawKernel(_SOURCE,'schur_values')
        self.factor=UniformCudssFactor(pattern,batch_size=solver.batch,matrix_type='spd',
            refinement_steps=solver.factor_refinements,execution_layout=solver.factor_layout)
        self.d=self.scaling=None
        cp.cuda.get_current_stream().synchronize()
        self.setup_seconds=time.perf_counter()-started

    def refresh_current_operator(self):
        s=self.solver;cp=self.cp;s.factor._context();self.factor._context()
        if self._dimensions!=(s.batch,s.n,s.ne,s.ng,s.q,s.e.shape,s.g.shape,self.h.shape,self.ht.shape):
            raise ValueError('Schur coordinate dimensions changed')
        for v,expected in zip((s.e.data,s.g.data,self.h.data,self.ht.data,self.values),self._numeric_metadata):
            if v is not expected[0] or (v.shape,v.strides,v.dtype.str,v.device.id,v.data.ptr)!=expected[1:]:
                raise ValueError('Schur numeric buffer identity or metadata changed')
        live=[getattr(self,n) for n in self._static_names]
        for a in (s.e,s.g,self.h,self.ht):live.extend((a.indptr,a.indices))
        for v,expected in zip(live,self._metadata):
            if v is not expected[0] or (v.shape,v.strides,v.dtype.str,v.device.id,v.data.ptr)!=expected[1:]:
                raise ValueError('Schur static coordinate identity changed')
        if not bool(cp.all(cp.stack([cp.array_equal(v,expected) for v,expected in self._static]))):
            raise ValueError('Schur static indices were modified')
        self.h.data[self.edest]=s.e.data[self.esource]
        self.h.data[self.gdest]=s.g.data[self.gsource]
        self.ht.data[:]=self.h.data[self.transpose_order]

    def factor_newton(self,ratio):
        s=self.solver;cp=self.cp
        self.refresh_current_operator()
        if (not isinstance(ratio,cp.ndarray) or ratio.shape!=(s.batch,s.ng) or ratio.dtype!=cp.float64
                or ratio.device.id!=s.factor.device or not bool(cp.all(cp.isfinite(ratio)&(ratio>0.)))):
            raise ValueError('Finite positive current device ratio required')
        d=cp.full((s.batch,s.n),s.regularization,dtype=cp.float64)
        d[:,s.il]+=1./ratio[:,s.q:s.q+len(s.il)]
        d[:,s.iu]+=1./ratio[:,s.q+len(s.il):]
        r=cp.concatenate((cp.full((s.batch,s.ne),s.regularization),ratio[:,:s.q]),axis=1)
        self.factor_diagonal(d,r)
        # Full-coordinate Krylov routes preconditioning through solve(), not
        # the unused augmented native factor. Its scaling argument is neutral.
        return cp.ones((s.batch,s.condensed_size),dtype=cp.float64)

    def factor_diagonal(self,d,r):
        """Factor current H with supplied positive diagonal metrics (proposal)."""
        s=self.solver;cp=self.cp;self.factor._context()
        for value,shape in ((d,(s.batch,s.n)),(r,(s.batch,self.m))):
            if (not isinstance(value,cp.ndarray) or value.shape!=shape or value.dtype!=cp.float64
                    or value.device.id!=s.factor.device or not value.flags.c_contiguous):
                raise ValueError('Exact contiguous FP64 Schur diagonals required')
        if not bool(cp.all(cp.isfinite(d)&(d>0.))&cp.all(cp.isfinite(r)&(r>0.))&cp.all(cp.isfinite(self.h.data))):
            raise ValueError('Nonfinite Schur operator or invalid positive diagonal')
        count=self.values.size
        self.kernel(((count+255)//256,),(256,),(self.offsets,self.left,self.right,self.column,
            self.diagonal_rows,self.h.data,d,r,self.values,np.int32(count),np.int32(self.factor.nnz),np.int32(self.m)))
        diagonal=self.values[:,self.diagonal]
        if not bool(cp.all(cp.isfinite(diagonal)&(diagonal>0.))):
            raise ValueError('Nonpositive/nonfinite Schur diagonal')
        scaling=1./cp.sqrt(diagonal)
        # Form the symmetric scale product first: (v*di)*dj can round
        # differently from (v*dj)*di in the opposite triangle.
        scaled=self.values*(scaling[:,self.rows]*scaling[:,self.columns])
        self.factor.factor(scaled)
        self.d=d.copy();self.scaling=scaling

    def solve(self,packed):
        s=self.solver;cp=self.cp;self.factor._context()
        if self.d is None:raise ValueError('No current Schur factor')
        if (not isinstance(packed,cp.ndarray) or packed.shape!=(s.batch,s.n+self.m)
                or packed.dtype!=cp.float64 or packed.device.id!=s.factor.device):
            raise ValueError('FP64 packed RHS in exact current Schur coordinates required')
        p=packed[:,:s.n];q=packed[:,s.n:]
        rhs=(self.h@(p/self.d).ravel()).reshape(s.batch,self.m)-q
        rhs=cp.ascontiguousarray((self.scaling*rhs)[:,:,None])
        result=(self.factor._solve_device_checked(rhs) if s.device_checked_solves else self.factor.solve(rhs))
        dual=self.scaling*result[:,:,0]
        primal=(p-(self.ht@dual.ravel()).reshape(s.batch,s.n))/self.d
        return cp.concatenate((primal,dual),axis=1)

    def close(self):self.factor.close()
