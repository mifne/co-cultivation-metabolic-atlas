"""CUDA conditional IF bodies inside a CuPy capture, without host decisions.

The parent owns all conditional nodes. A separate stream captures each body;
the parent stream then depends on that conditional node. All device state
used after a skipped body must have been initialized outside that body.
"""
from contextlib import contextmanager
import numpy as np


class ConditionalCapture:
    def __init__(self,while_mode=False):
        import cupy as cp
        from cuda.bindings import runtime as rt
        self.cp,self.rt=cp,rt
        self.while_mode=while_mode
        self.zero_counter=cp.zeros(1,dtype=cp.int32)
        self.body_stream=cp.cuda.Stream(non_blocking=True)
        self.set_condition=cp.RawKernel(r'''
        #include <cuda_runtime.h>
        extern "C" __global__ void set_condition(unsigned long long handle,
                const bool* stopped,const bool* failed,int n,const int* counter,int limit){
            if(threadIdx.x==0){unsigned value=0;
                for(int i=0;i<n;i++)if(!stopped[i]&&!failed[i]&&counter[0]<limit)value=1;
                cudaGraphSetConditional((cudaGraphConditionalHandle)handle,value);}
        }''','set_condition',options=('--include-path=/usr/local/cuda/include','--device-c'))
        self.set_condition.compile()

    @staticmethod
    def checked(result):
        if int(result[0]):raise RuntimeError(f'CUDA conditional API: {result[0]}')
        return result[1:]

    @contextmanager
    def iteration(self,stopped,failed,counter=None,limit=2147483647):
        cp,rt=self.cp,self.rt
        parent=cp.cuda.get_current_stream()
        if counter is None:
            if self.while_mode:raise ValueError('Device WHILE requires an iteration counter')
            counter=self.zero_counter
        info=self.checked(rt.cudaStreamGetCaptureInfo(parent.ptr))
        if int(info[0])!=1:raise RuntimeError('Conditional body requires active capture')
        graph=info[2]
        # CUDA requires a distinct handle for each conditional node.
        handle=self.checked(rt.cudaGraphConditionalHandleCreate(graph,1,
            int(rt.cudaGraphConditionalHandleFlags.cudaGraphCondAssignDefault)))[0]
        condition_args=(np.uint64(int(handle)),stopped,failed,np.int32(stopped.size),counter,np.int32(limit))
        self.set_condition((1,),(1,),condition_args)
        info=self.checked(rt.cudaStreamGetCaptureInfo(parent.ptr))
        dependencies=info[3]
        params=rt.cudaGraphNodeParams()
        params.type=rt.cudaGraphNodeType.cudaGraphNodeTypeConditional
        params.conditional.handle=handle
        params.conditional.type=(rt.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile if self.while_mode else
            rt.cudaGraphConditionalNodeType.cudaGraphCondTypeIf)
        params.conditional.size=1
        node=self.checked(rt.cudaGraphAddNode(graph,dependencies,len(dependencies),params))[0]
        self.checked(rt.cudaStreamUpdateCaptureDependencies(parent.ptr,[node],1,
            int(rt.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies)))
        with self.body_stream:
            self.checked(rt.cudaStreamBeginCaptureToGraph(self.body_stream.ptr,
                params.conditional.phGraph_out[0],None,None,0,rt.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed))
            try:
                yield
                if self.while_mode:self.set_condition((1,),(1,),condition_args)
            finally:self.checked(rt.cudaStreamEndCapture(self.body_stream.ptr))
