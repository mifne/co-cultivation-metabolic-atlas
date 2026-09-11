"""Two fused CUDA guards around a native batched solve; no host numeric read.

Each independent lane is sanitized as a whole before cuDSS, and an invalid
input/output lane is returned entirely NaN. This is not an LP acceptance gate.
"""
SOURCE=r'''
extern "C" __global__ void pack_guard(const double* src,double* dst,
    bool* valid,long long* bad,int n,int nrhs,long long sb,long long sn,long long sr) {
    __shared__ int ok[256];
    int lane=blockIdx.x,tid=threadIdx.x,good=1;
    long long size=(long long)n*nrhs;
    for(long long t=tid;t<size;t+=256) {
        long long row=t%n,col=t/n;
        if(!isfinite(src[lane*sb+row*sn+col*sr])) good=0;
    }
    ok[tid]=good; __syncthreads();
    for(int d=128;d;d>>=1) {if(tid<d) ok[tid]&=ok[tid+d]; __syncthreads();}
    if(tid==0) {valid[lane]=ok[0]; if(!ok[0]) bad[lane]++;}
    for(long long t=tid;t<size;t+=256) {
        long long row=t%n,col=t/n;
        dst[lane*size+t]=ok[0]?src[lane*sb+row*sn+col*sr]:0.;
    }
}
extern "C" __global__ void unpack_guard(const double* src,double* dst,
    const bool* input_valid,long long* bad,int n,int nrhs) {
    __shared__ int ok[256];
    int lane=blockIdx.x,tid=threadIdx.x,good=1;
    long long size=(long long)n*nrhs;
    for(long long t=tid;t<size;t+=256) if(!isfinite(src[lane*size+t])) good=0;
    ok[tid]=good; __syncthreads();
    for(int d=128;d;d>>=1) {if(tid<d) ok[tid]&=ok[tid+d]; __syncthreads();}
    if(tid==0 && !ok[0]) bad[lane]++;
    for(long long t=tid;t<size;t+=256) {
        long long row=t%n,col=t/n;
        dst[lane*size+row*nrhs+col]=(ok[0] && input_valid[lane])?src[lane*size+t]:nan("");
    }
}
'''


def _meta(value):
    return (value.shape,value.strides,value.dtype.str,value.device.id,value.data.ptr)


class FusedSolveGuards:
    def __init__(self,factor):
        factor._context()
        cp=factor.cp
        for name in ('rhs','solution','_internal_invalid_rhs_count','_internal_nonfinite_output_count'):
            value=getattr(factor,name)
            numeric=name in ('rhs','solution')
            shape=(factor.batch,factor.nrhs,factor.n) if numeric else (factor.batch,)
            dtype=cp.float64 if numeric else cp.int64
            if (not isinstance(value,cp.ndarray) or value.shape!=shape or value.dtype!=dtype
                    or value.device.id!=factor.device or not value.flags.c_contiguous):
                raise ValueError('Invalid native fused guard buffer')
        self.factor=factor
        self.dimensions=(factor.batch,factor.n,factor.nrhs)
        self.input_valid=factor.cp.empty(factor.batch,dtype=factor.cp.bool_)
        self.static={name:(getattr(factor,name),_meta(getattr(factor,name))) for name in
            ('rhs','solution','_internal_invalid_rhs_count','_internal_nonfinite_output_count')}
        self.valid_snapshot=(self.input_valid,_meta(self.input_valid))
        self.pack_kernel=factor.cp.RawKernel(SOURCE,'pack_guard')
        self.unpack_kernel=factor.cp.RawKernel(SOURCE,'unpack_guard')
        self.pending=False

    def _check(self):
        f=self.factor; f._context()
        if self.dimensions!=(f.batch,f.n,f.nrhs):raise ValueError('Fused guard dimensions changed')
        for name,(original,meta) in self.static.items():
            current=getattr(f,name)
            if current is not original or _meta(current)!=meta:
                raise ValueError('Fused guard native buffer changed')
        if self.input_valid is not self.valid_snapshot[0] or _meta(self.input_valid)!=self.valid_snapshot[1]:
            raise ValueError('Fused guard validity buffer changed')

    def pack(self,rhs):
        import numpy as np
        self._check();f=self.factor;cp=f.cp
        if self.pending:raise RuntimeError('Previous fused solve guard has no output')
        if (not isinstance(rhs,cp.ndarray) or rhs.shape!=(f.batch,f.n,f.nrhs)
                or rhs.dtype!=cp.float64 or rhs.device.id!=f.device or any(s%8 for s in rhs.strides)):
            raise ValueError('Exact-shape FP64 RHS on the bound device required')
        strides=tuple(np.int64(s//8) for s in rhs.strides)
        self.pack_kernel((f.batch,),(256,),(rhs,f.rhs,self.input_valid,
            f._internal_invalid_rhs_count,np.int32(f.n),np.int32(f.nrhs),*strides))
        self.pending=True

    def finish(self):
        import numpy as np
        self._check();f=self.factor;cp=f.cp
        if not self.pending:raise RuntimeError('Fused output guard requires a current input guard')
        result=cp.empty((f.batch,f.n,f.nrhs),dtype=cp.float64)
        self.unpack_kernel((f.batch,),(256,),(f.solution,result,self.input_valid,
            f._internal_nonfinite_output_count,np.int32(f.n),np.int32(f.nrhs)))
        self.pending=False
        return result
