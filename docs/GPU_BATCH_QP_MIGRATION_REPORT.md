# Cooperative dFBA GPU batch-QP migration

## Implemented online architecture

The online FBA path is now:

1. Build the live three-GEM context (biomass, shared-medium supply, reaction bounds and objectives).
2. Rank 33,728 offline exact cooperative-FBA anchors with a CUDA neural decision head.
3. Keep the best 128 anchors resident on CUDA.
4. Apply a batched convex-hull QP projection. Because every anchor satisfies the block-diagonal GEM equality constraints, any projected convex combination retains intracellular steady-state mass balance. The QP additionally enforces live reaction bounds and biomass-weighted shared-medium inequalities.
5. If the 128-anchor hull is insufficient, retry only that row with 2,048 CUDA anchors, six-way per-species block composition and up to 1,200 primal-dual iterations.
6. Return the flux vector to the dFBA state update. The online path does not invoke SciPy/HiGHS.

PPO workers do not load separate copies of the approximately 1 GiB dictionary. A spawned manager process owns the CUDA context and combines worker requests into microbatches. Independent GPUs remain suitable for independent seed/agent/environment shards; their VRAM is not treated as shared memory.

This follows the general principle of embedding an optimization layer in a neural pipeline (OptNet and differentiable convex optimization layers) while retaining mechanistic feasibility through exact metabolic anchors. It also follows neural-mechanistic hybrid GEM work in using a learned map to accelerate the repeated mechanistic calculation rather than replacing biochemical constraints with an unconstrained black box.

Primary references:

- Amos and Kolter, OptNet (2017): https://proceedings.mlr.press/v70/amos17a.html
- Agrawal et al., Differentiable Convex Optimization Layers (2019): https://papers.nips.cc/paper_files/paper/2019/hash/9ce3c52fc54362e22053399d3181c638-Abstract.html
- Faure et al., Artificial metabolic networks (2023): https://www.nature.com/articles/s41467-023-40380-0
- Multiobjective growth-media design with GEMs and Bayesian optimization: https://spj.science.org/doi/full/10.34133/csbj.0072

## Measured results on RTX 4060 Laptop GPU

| Test | Result |
|---|---:|
| Five-seed, 24 h, 120-step CPU HiGHS comparison | mean speed-up 2.67× (95% CI 2.61–2.74) |
| GPU feasibility in that comparison | 600/600 steps |
| Online CPU LP calls | 0 |
| Mean / maximum terminal PHA relative error | 1.24% / 1.71% |
| Maximum terminal biomass absolute error | 0.00116 g/L |
| PPO-like parallel smoke test | 4 environments × 128 steps = 512 transitions |
| Valid GPU solutions after adaptive retry | 512/512 |
| Retry use / recovery | 5 / 5 requests |
| Smoke wall time | 52.39 s |
| Maximum observed service batch | 4 |

The isolated projection/ranking kernel scales from 161.8 environments/s at batch 1 to 1,264.5 environments/s at batch 64, with peak PyTorch allocation increasing from about 900 MiB to 2,718 MiB. This is kernel throughput, not full Python environment throughput.

## Qualification status and interpretation

The operational goal has been met for the tested parallel smoke workload: every online FBA request was handled on the NVIDIA GPU and no CPU LP fallback occurred. The host CPU still performs Python environment orchestration, state integration and inter-process communication; “GPU-only” refers specifically to the repeated FBA candidate search and feasibility projection.

The current artifact is not yet qualified under the deliberately strict maximum PHA relative-error threshold of 1% across all five seeds. It achieved 1.71% maximum error. Therefore:

- use the GPU-only path for high-throughput PPO exploration;
- evaluate selected policies and final paper results with exact CPU HiGHS;
- do not describe the current surrogate as numerically identical to the CPU LP;
- add independent active-learning trajectories around states that trigger high PHA error before changing the scientific qualification manifest.

Increasing the online pool from 128 to 256 was rejected: it improved one seed but produced a large trajectory divergence in another seed. Expanding the offline dictionary from 33,728 to 34,688 without recalibrating candidate selection was also rejected as the default. These negative controls are retained in `results/` and are not used by the training configuration.

## Reproducible artifacts

- GPU QP implementation: `src/gpu_batch_qp.py`
- Shared microbatch service: `src/cooperative_gpu_service.py`
- Neural CUDA ranking: `src/cooperative_neural_surrogate.py`
- Cooperative solver integration: `src/community_solver.py`
- PPO integration: `scripts/train_until_effect_wcfs1.py`
- Parallel smoke runner: `scripts/smoke_gpu_qp_training_env.py`
- Five-seed validation: `models/cooperative_surrogate/cooperative_neural_reranker_33728_uniform_late_trained_pha256.validation.json`
- 100% recovery smoke report: `results/gpu_qp_parallel_training_smoke_retry2048.json`
- Batch scaling report: `results/gpu_qp_batch_scaling_rtx4060.json`

