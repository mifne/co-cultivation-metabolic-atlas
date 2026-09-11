"""Experimental device-controlled Harris simplex pivots, no CPU LP calls.

The pricing/ratio kernel is one block; the tableau update is bandwidth-parallel.
Host reads are needed only between batches/refactorizations, not each pivot.
Not selected automatically by the production or reference backend.
"""
import numpy as np

KERNEL_SOURCE=r'''
#define INFINITY (__longlong_as_double(0x7ff0000000000000LL))
extern "C" __global__ void choose(
 const double* T, const double* x, const long long* basis,
 const double* rc, const double* upper, const unsigned char* basic,
 double* col, double* row, long long* s, double* d,
 int m, int n, double tolerance) {
 // s: status, iterations, stalls, entering, row, leaving, pivot_flag, max_pivots
 // d: direction, delta, pivot, reduced_entering, min_pivot
 if(s[0]) return;
 if(s[1]>=s[7]) {if(threadIdx.x==0) s[0]=3; return;}
 __shared__ double v[256]; __shared__ long long ix[256];
 int t=threadIdx.x;
 double best=-INFINITY; long long bi=-1;
 for(int j=t;j<n;j+=256) {
   double value=-INFINITY;
   if(!basic[j] && upper[j]>tolerance) {
     if(x[j]<=tolerance) value=-rc[j];
     else if(x[j]>=upper[j]-tolerance) value=rc[j];
   }
   if(s[2]>100) value=value>tolerance ? (double)(n-j) : -INFINITY;
   if(value>best || (value==best && (bi<0 || j<bi))) {best=value; bi=j;}
 }
 v[t]=best; ix[t]=bi; __syncthreads();
 for(int k=128;k;k/=2) {if(t<k && (v[t+k]>v[t] ||
       (v[t+k]==v[t] && ix[t+k]>=0 && (ix[t]<0 || ix[t+k]<ix[t])))) {
     v[t]=v[t+k]; ix[t]=ix[t+k];} __syncthreads();}
 if(t==0) {
   if(!(v[0]>tolerance)) s[0]=1;
   else {s[3]=ix[0]; d[0]=rc[ix[0]]<0 ? 1. : -1.; d[3]=rc[ix[0]];}
 }
 __syncthreads(); if(s[0]) return;
 int j=(int)s[3]; double direction=d[0];
 double relaxed=INFINITY;
 for(int i=t;i<m;i+=256) {
   double a=T[(long long)i*n+j]; col[i]=a;
   double move=a*direction, dist=move>0 ? fmax(x[basis[i]],0.) : fmax(upper[basis[i]]-x[basis[i]],0.);
   if(fabs(move)>1e-9) relaxed=fmin(relaxed,(dist+1e-8)/fabs(move));
 }
 v[t]=relaxed; __syncthreads();
 for(int k=128;k;k/=2) {if(t<k) v[t]=fmin(v[t],v[t+k]); __syncthreads();}
 relaxed=v[0]; __syncthreads();
 best=-1.; bi=-1;
 for(int i=t;i<m;i+=256) {
   double move=col[i]*direction;
   double dist=move>0 ? fmax(x[basis[i]],0.) : fmax(upper[basis[i]]-x[basis[i]],0.);
   double ratio=fabs(move)>1e-9 ? dist/fabs(move) : INFINITY;
   double value=ratio<=relaxed ? fabs(col[i]) : -1.;
   if(value>best || (value==best && (bi<0 || i<bi))) {best=value; bi=i;}
 }
 v[t]=best; ix[t]=bi; __syncthreads();
 for(int k=128;k;k/=2) {if(t<k && (v[t+k]>v[t] ||
       (v[t+k]==v[t] && ix[t+k]>=0 && (ix[t]<0 || ix[t+k]<ix[t])))) {
     v[t]=v[t+k]; ix[t]=ix[t+k];} __syncthreads();}
 if(t==0) {
   int i=(int)ix[0];
   double move=col[i]*direction;
   double dist=move>0 ? fmax(x[basis[i]],0.) : fmax(upper[basis[i]]-x[basis[i]],0.);
   double basic_step=fabs(move)>1e-9 ? dist/fabs(move) : INFINITY;
   double own_step=direction>0 ? upper[j]-x[j] : x[j];
   d[1]=fmin(basic_step,own_step);
   if(!isfinite(d[1])) s[0]=2;
   else {
     s[4]=i; s[5]=basis[i]; s[6]=basic_step<=own_step;
     d[2]=col[i]; s[2]=d[1]<1e-10 ? s[2]+1 : 0;
     if(s[6]) d[4]=fmin(d[4],fabs(d[2]));
     s[1]+=1;
   }
 }
 __syncthreads(); if(s[0] || !s[6]) return;
 for(int k=t;k<n;k+=256) row[k]=T[s[4]*(long long)n+k]/d[2];
}

extern "C" __global__ void update(
 double* T, double* x, long long* basis, double* rc,
 const double* upper, unsigned char* basic, const double* col,
 const double* row, const long long* s, const double* d, int m, int n) {
 if(s[0]) return;
 long long k=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 int enter=(int)s[3], leave_row=(int)s[4], leave=(int)s[5];
 if(k<m) {
   if(s[6] && k==leave_row) x[leave]=col[k]*d[0]>0 ? 0. : upper[leave];
   else x[basis[k]]-=col[k]*d[0]*d[1];
 }
 if(k==0) {
   x[enter]=s[6] ? x[enter]+d[0]*d[1] : (d[0]>0 ? upper[enter] : 0.);
   if(s[6]) {basis[leave_row]=enter; basic[leave]=0; basic[enter]=1;}
 }
 if(s[6]) {
   if(k<n) rc[k]-=d[3]*row[k];
   if(k<(long long)m*n) {
     int i=k/n,j=k%n;
     T[k]=i==leave_row ? row[j] : T[k]-col[i]*row[j];
   }
 }
}
'''


class PivotBatch:
    def __init__(self,tableau,x,basis,reduced,limits,tolerance=1e-8,batch_size=100):
        import cupy as cp
        self.cp=cp
        self.tableau=cp.ascontiguousarray(tableau.copy())
        self.x=x.copy(); self.basis=basis.copy(); self.reduced=reduced.copy(); self.limits=limits.copy()
        self.m,self.n=tableau.shape
        self.basic=cp.zeros(self.n,dtype=cp.uint8); self.basic[basis]=1
        self.col=cp.empty(self.m); self.row=cp.empty(self.n)
        if not isinstance(batch_size,int) or batch_size<1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size=batch_size
        self.state=cp.zeros(8,dtype=cp.int64); self.state[7]=batch_size
        self.values=cp.zeros(5,dtype=cp.float64); self.values[4]=1.
        self.choose=cp.RawKernel(KERNEL_SOURCE,"choose",options=("--fmad=false",))
        self.update=cp.RawKernel(KERNEL_SOURCE,"update",options=("--fmad=false",))
        self.choose.compile(); self.update.compile()
        self.args_choose=(self.tableau,self.x,self.basis,self.reduced,self.limits,self.basic,
            self.col,self.row,self.state,self.values,np.int32(self.m),np.int32(self.n),np.float64(tolerance))
        self.args_update=(self.tableau,self.x,self.basis,self.reduced,self.limits,self.basic,
            self.col,self.row,self.state,self.values,np.int32(self.m),np.int32(self.n))
        self.grid=((self.m*self.n+255)//256,)
        cp.cuda.get_current_stream().synchronize()
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            for _ in range(batch_size):
                self.choose((1,),(256,),self.args_choose)
                self.update(self.grid,(256,),self.args_update)
            self.graph=stream.end_capture()

    def run(self):
        self.graph.launch(stream=self.cp.cuda.get_current_stream())
        return self.state.get(),self.values.get()

    def refresh(self,tableau,x,basis,reduced,limits,stalls=0,max_pivots=None):
        cp=self.cp
        max_pivots=self.batch_size if max_pivots is None else max_pivots
        if not isinstance(max_pivots,int) or not 0 <= max_pivots <= self.batch_size:
            raise ValueError("max_pivots must be an integer within the captured batch")
        self.tableau[:]=tableau; self.x[:]=x; self.basis[:]=basis
        self.reduced[:]=reduced; self.limits[:]=limits
        self.basic[:]=0; self.basic[basis]=1
        self.state[:]=cp.asarray([0,0,stalls,0,0,0,0,max_pivots],dtype=cp.int64)
