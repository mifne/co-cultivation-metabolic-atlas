"""Opt-in learned warm-start plus original-LP GPU correction service.

No CPU LP fallback. Not a fully device-resident dFBA environment: input
normalization/features, batching and diagnostic decisions remain on the host.
Only independently certified solves enter the previous-solution cache.
"""
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .cpu_repeated_lp import _problem
from .temporal_lp_data import TemporalLPCodec


class TemporalGpuLPBackend:
    name = 'experimental_temporal_gpu_pdhg'
    method = 'GRU_or_MLP_warm_start_full_space_GPU_PDHG'

    def __init__(self, artifact, *, mode='gru', iterations=1000, check_interval=100,
                 primal_weight=1., equality_reduction=False, resident_execution=None,
                 graph_chunk=64):
        if mode not in ('cold','previous','mean','mlp','gru'):
            raise ValueError('Unknown warm-start mode')
        if iterations < 0 or check_interval < 1:
            raise ValueError('Invalid correction budget')
        import torch
        import cupy as cp
        from .temporal_lp_model import model_from_checkpoint, TemporalLPSession
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA inference is required')
        self.torch, self.cp = torch, cp
        self.artifact = Path(artifact)
        self.codec = TemporalLPCodec.load(self.artifact/'codec.npz')
        self.mode, self.iterations, self.check_interval = mode, int(iterations), int(check_interval)
        if not np.isfinite(primal_weight) or primal_weight <= 0:
            raise ValueError('Positive finite primal weight required')
        self.primal_weight = float(primal_weight)
        if not isinstance(equality_reduction, bool):
            raise ValueError('equality_reduction must be a boolean')
        self.equality_reduction = equality_reduction
        self.reduction_plan = None
        if resident_execution not in (None,'graph','loop'):
            raise ValueError('resident_execution must be None, graph or loop')
        if resident_execution is not None and not equality_reduction:
            raise ValueError('Resident execution requires equality_reduction')
        if isinstance(graph_chunk,bool) or not isinstance(graph_chunk,int) or graph_chunk < 1:
            raise ValueError('graph_chunk must be a positive integer')
        self.resident_execution,self.graph_chunk = resident_execution,graph_chunk
        self._resident_corrector = None
        self.stage = self.codec.metadata['stage']
        self.session = None
        if mode in ('mlp','gru'):
            payload = torch.load(self.artifact/f'{mode}.pt',map_location='cpu',weights_only=True)
            if payload['metadata']['codec_sha256'] != hashlib.sha256((self.artifact/'codec.npz').read_bytes()).hexdigest():
                raise ValueError('Checkpoint/codec mismatch')
            model = model_from_checkpoint(payload,device='cuda')
            if model.config.kind != mode or model.config.feature_dim != self.codec.feature_dim or model.config.latent_dim != self.codec.rank:
                raise ValueError('Model architecture/codec mismatch')
            model.eval(); self.session = TemporalLPSession(model)
        a = self.codec.arrays
        self.components = torch.as_tensor(a['components'],device='cuda',dtype=torch.float64)
        self.target_scale = torch.as_tensor(a['target_scale'],device='cuda',dtype=torch.float64)
        self.target_mean = torch.as_tensor(a['target_mean'],device='cuda',dtype=torch.float64)
        self.latent_scale = torch.as_tensor(a.get('latent_scale',np.ones(self.codec.rank)),device='cuda',dtype=torch.float64)
        self.last_features, self.last_certified = {}, {}
        self.history = []
        torch.cuda.synchronize(); cp.cuda.runtime.deviceSynchronize()

    def reset(self, environment_ids=None):
        ids = None if environment_ids is None else list(environment_ids)
        if ids is not None and len(set(ids)) != len(ids):
            raise ValueError('Reset requires unique environment IDs')
        # Materialize a generator once, and let the session validate before
        # mutating caches. All histories must reset the same environments.
        if self.session is not None:
            self.session.reset(ids)
        if ids is None:
            self.last_features.clear(); self.last_certified.clear()
        else:
            for env in ids:
                self.last_features.pop(env,None); self.last_certified.pop(env,None)

    def solve_batch(self, requests, *, environment_ids=None):
        from .gpu_pdhg_corrector import GpuPdhgCorrector
        started = time.perf_counter()
        torch, cp = self.torch, self.cp
        ids = list(range(len(requests))) if environment_ids is None else list(environment_ids)
        if not requests or len(ids) != len(requests) or len(set(ids)) != len(ids):
            raise ValueError('Nonempty requests and unique environment IDs required')
        problems = [_problem(c,kw) for c,kw in requests]
        if any(kw.get('_stage',self.stage) != self.stage for _,kw in requests):
            raise ValueError('Use a separate model and history for each LP stage')
        base = np.stack([self.codec.base_features(p) for p in problems])
        n,m,_ = self.codec.metadata['dimensions']
        delta = np.stack([b-self.last_features[env] if env in self.last_features else np.zeros_like(b)
                          for env,b in zip(ids,base)])
        available = np.array([env in self.last_certified for env in ids],dtype=np.float32)
        features = np.concatenate((base,delta,available[:,None]),axis=1)
        preparation = time.perf_counter()-started
        before = time.perf_counter()
        reuse = False
        rebuild_reason = 'nonresident_execution'
        # Validate fixed equality fingerprints and dynamic transformed bounds
        # before advancing a GRU hidden state for this physical step.
        if self.resident_execution is not None:
            from .gpu_resident_reduced_pdhg import GpuResidentReducedPdhg
            corrector = self._resident_corrector
            if corrector is not None:
                reuse = corrector.try_update(problems,environment_ids=ids)
                rebuild_reason = corrector.last_update_timing['reason']
            else:
                rebuild_reason = 'initial_setup'
            if not reuse:
                corrector = GpuResidentReducedPdhg(problems,environment_ids=ids,
                    plan=self.reduction_plan,primal_weight=self.primal_weight,
                    chunk_size=self.graph_chunk,use_graph=self.resident_execution=='graph')
                self._resident_corrector = corrector
                self.reduction_plan = corrector.plan
        elif self.equality_reduction:
            from .gpu_reduced_pdhg import GpuReducedPdhgCorrector
            corrector = GpuReducedPdhgCorrector(problems,plan=self.reduction_plan,
                                               primal_weight=self.primal_weight)
            self.reduction_plan = corrector.plan
        else:
            corrector = GpuPdhgCorrector(problems,primal_weight=self.primal_weight)
        setup = (dict(total_seconds=time.perf_counter()-before,resident=True,reused=reuse,
                      rebuild_reason=rebuild_reason,
                      initial_setup=dict(corrector.setup_timing) if not reuse else None,
                      update=dict(corrector.last_update_timing) if reuse else None)
                 if self.resident_execution is not None else dict(corrector.setup_timing))
        before = time.perf_counter()
        pending_hidden = None
        with torch.inference_mode():
            if self.mode == 'cold':
                candidate = torch.zeros((len(ids),n+m),device='cuda',dtype=torch.float64)
            elif self.mode == 'mean':
                candidate = self.target_mean.expand(len(ids),-1).contiguous()
            else:
                previous_full = torch.stack([self.last_certified[env] if env in self.last_certified else
                    torch.zeros(n+m,device='cuda',dtype=torch.float64) for env in ids])
                if self.mode == 'previous':
                    candidate = previous_full
                else:
                    previous_latent = (((previous_full-self.target_mean)/self.target_scale) @ self.components.T)/self.latent_scale
                    previous_latent *= torch.as_tensor(available[:,None],device='cuda')
                    latent, pending_hidden = self.session.propose(
                        ids,torch.as_tensor(features,device='cuda'),previous_latent.float())
                    candidate = ((latent.double()*self.latent_scale) @ self.components)*self.target_scale+self.target_mean
            initial_x = cp.from_dlpack(candidate[:,:n].contiguous())
            initial_y = cp.from_dlpack(candidate[:,n:].contiguous())
        cp.cuda.runtime.deviceSynchronize(); torch.cuda.synchronize()
        inference_seconds = time.perf_counter()-before
        result = corrector.solve(initial_x=initial_x,initial_y=initial_y,
                                 iterations=self.iterations,check_interval=self.check_interval)
        before = time.perf_counter()
        # Device-to-device exchange only for accepted cache values; CPU labels
        # are never passed to this service. Clear stale values after a failure.
        with torch.inference_mode():
            full = torch.cat((torch.from_dlpack(result['x']),torch.from_dlpack(result['y'])),dim=1)
            for i,env in enumerate(ids):
                if result['accepted'][i]:
                    self.last_certified[env] = full[i].clone()
                else:
                    self.last_certified.pop(env,None)
                self.last_features[env] = base[i].copy()
        # The existing cooperative solver consumes host arrays. This explicit
        # boundary is included in timing; no claim of full dFBA residency.
        host_x = cp.asnumpy(result['x'])
        rows = []
        for i,env in enumerate(ids):
            ok = bool(result['accepted'][i])
            diagnostic = dict(result['metrics'][i],environment_id=env,stage=self.stage,
                success=ok,cpu_lp_calls=0,mode=self.mode)
            rows.append(SimpleNamespace(success=ok,x=host_x[i].copy() if ok else None,
                fun=float(problems[i][4]@host_x[i]) if ok else None,
                message='Original-LP certified GPU solution' if ok else 'GPU correction budget exhausted; no CPU fallback',
                diagnostics=diagnostic))
        torch.cuda.synchronize(); cp.cuda.runtime.deviceSynchronize()
        # Completed rejected proposals still consume this observed step, but
        # a correction exception must not advance the recurrent state twice
        # when the caller retries the same physical input.
        if self.session is not None:
            self.session.commit(ids,pending_hidden)
        result_seconds = time.perf_counter()-before
        self.history.append(dict(stage=self.stage,mode=self.mode,batch=len(ids),environment_ids=ids,
            primal_weight=self.primal_weight,
            equality_reduction=self.equality_reduction,
            resident_execution=self.resident_execution,graph_chunk=self.graph_chunk,
            accepted=sum(bool(v) for v in result['accepted']),cpu_lp_calls=0,
            host_preparation_seconds=preparation,inference_decode_seconds=inference_seconds,
            result_cache_and_host_copy_seconds=result_seconds,total_seconds=time.perf_counter()-started,
            correction=result['timing'], setup=setup,
            iterations_run=result['iterations_run'],checkpoints=result['checkpoints'],
            rows=[row.diagnostics for row in rows],
            scope='LP warm-start/correction replay; host inputs and output bridge remain'))
        return rows
