"""Private-pool, fixed-shape CUDA replay for an experimental device solver.

Compilation is charged to the caller. Persistent inputs are copied on-device
before launch. The allocation pool and solver stay alive with the graph.
Not thread-safe: callers must serialize calls to one replay instance.
"""
import time


def signature(value):
    if hasattr(value,'__cuda_array_interface__'):return ('device',value.shape,value.dtype.str)
    if isinstance(value,dict):return tuple((k,signature(v)) for k,v in sorted(value.items()))
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return ('identity',id(value))


class GpuReplayCall:
    def __init__(self,solver,inputs):
        self.raw_graph=None;self.graph_exec=None;self.last_stream=None
        from .gpu_conditional_capture import ConditionalCapture
        from cuda.bindings import runtime as rt
        self.cp=cp=solver.cp;self.solver=solver
        self.pool=cp.cuda.MemoryPool();self.stream=cp.cuda.Stream(non_blocking=True)
        self.conditional=ConditionalCapture(while_mode=True)
        self.key=signature(inputs)
        started=time.perf_counter()
        cp.cuda.get_current_stream().synchronize()
        def clone(value):
            if isinstance(value,cp.ndarray):return value.copy()
            if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
            return value
        with cp.cuda.using_allocator(self.pool.malloc),self.stream:
            self.inputs=clone(inputs)
            warm=solver.solve_device(**self.inputs,_warm_iterations=1)
            self.stream.synchronize();del warm
            self.stream.begin_capture()
            self.result=solver.solve_device(**self.inputs,conditional=self.conditional)
            raw=rt.cudaStreamEndCapture(self.stream.ptr)
            if int(raw[0]):raise RuntimeError(f'End capture: {raw[0]}')
            self.raw_graph=raw[1]
            params=rt.cudaGraphInstantiateParams()
            built=rt.cudaGraphInstantiateWithParams(self.raw_graph,params)
            if int(built[0]):
                node_type=rt.cudaGraphNodeGetType(params.errNode_out)
                raise RuntimeError(f'Instantiate {built[0]}: {params}; node_type={node_type}')
        self.graph_exec=built[1]
        self.compilation_seconds=time.perf_counter()-started

    def run(self,inputs):
        if signature(inputs)!=self.key:raise ValueError('Replay shape/static argument mismatch')
        cp=self.cp
        def copy(target,source):
            if isinstance(source,cp.ndarray):target[:]=source
            elif isinstance(source,dict):
                # A warm state may reference a previous graph's input/output.
                # Preserve it before overwriting current RHS/matrix buffers.
                keys=sorted(source,key=lambda key:(key!='warm_start',key))
                for key in keys:copy(target[key],source[key])
        copy(self.inputs,inputs)
        from cuda.bindings import runtime as rt
        self.last_stream=cp.cuda.get_current_stream()
        launched=rt.cudaGraphLaunch(self.graph_exec,self.last_stream.ptr)
        if int(launched[0]):raise RuntimeError(f'Graph launch: {launched[0]}')
        return self.result

    def close(self):
        from cuda.bindings import runtime as rt
        if self.last_stream is not None:self.last_stream.synchronize()
        if self.graph_exec is not None:
            status=rt.cudaGraphExecDestroy(self.graph_exec)
            if int(status[0]):raise RuntimeError(f'Graph executable destruction: {status[0]}')
            self.graph_exec=None
        if self.raw_graph is not None:
            status=rt.cudaGraphDestroy(self.raw_graph)
            if int(status[0]):raise RuntimeError(f'Graph destruction: {status[0]}')
            self.raw_graph=None
        # Closing is terminal. External result arrays remain valid through
        # their own allocations, but must not retain all freed temporaries in
        # a retired private pool. Never release blocks of a live graph.
        self.inputs=None;self.result=None;self.solver=None
        if hasattr(self,'pool'):self.pool.free_all_blocks()

    def __del__(self):
        try:self.close()
        except Exception:pass  # CUDA may already be unloaded at interpreter exit.
