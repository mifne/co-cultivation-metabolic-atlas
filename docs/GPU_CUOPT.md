# NVIDIA cuOpt FBA backend

The simulator now has an optional NVIDIA cuOpt backend. GLPK remains the
default so existing results stay reproducible on CPU-only machines.

## Why cuOpt is a plausible target

Each FBA call is a continuous LP of the form `S v = 0`, with reaction bounds
and one linear objective. NVIDIA cuOpt exposes LP through a Python API and
supports CSR constraint matrices, GPU PDLP/barrier methods, variable-bound
updates, and crossover to a basic solution. The latter matters for FBA because
the default PDLP solution is not necessarily a vertex solution.

The adapter builds each species' stoichiometric matrix once and updates bounds
and objective coefficients on every dFBA step. It does not rebuild the LP from
scratch for every step.

The repeated-solve path now also:

- sends only changed reaction bounds/objective coefficients to the persistent
  cuOpt problem;
- disables per-solve console logging;
- validates `max(abs(S @ v))` and reaction-bound violations independently of
  cuOpt's reported status;
- removes numerically dependent stoichiometric rows with pivoted QR, while
  validating returned fluxes against the complete original matrix;
- audits the first GPU objective against an identical signed-flux HiGHS LP;
- retries a failed Barrier solve with rank-reduced PDLP;
- uses crossover after PDLP where possible, and retries without crossover only
  when crossover itself fails on a degenerate zero-objective state; and
- sends transient time limits to the exact CPU LP for that step instead of
  treating a failed LP as biological decay.

## Measured bottleneck on RTX 4060 (2026-08-25)

The initial 0% valid-GPU result was caused by multiple interacting issues, not
by lack of CUDA execution:

1. The first adapter revision did not force the COBRA maximize sense on the
   first cuOpt solve.
2. Fixed reactions were changed from `[0, 0]` to `[-1e-6, 0]`, altering the
   feasible polytope and creating ghost fluxes.
3. The GEM stoichiometric matrices contain dependent conservation rows. These
   are tolerated by simplex but destabilize Barrier/crossover and slow PDLP.
4. Disabling crossover globally avoided a cuDSS error but left inaccurate
   interior PDLP solutions.
5. A single time limit permanently disabled a species even when the next state
   could converge.

Pivoted QR removes 39 dependent rows for OR16, 31 for NS21, and 52 for LP.
The full original `S @ v` is still used for acceptance, so row reduction cannot
silently relax mass balance.

With exact fixed bounds, 12 CPU steps and three GEMs per step measured:

| Phase | Mean seconds/step | Share |
|---|---:|---:|
| Dynamic bounds/objective preparation | 0.0527 | 9.5% |
| Three FBA solves | 0.4973 | 89.9% |
| Environment/state update | 0.0027 | 0.5% |
| Total | 0.5527 | 100% |

After the fixes, the three-step GPU validation completed all 9/9 LPs on cuOpt.
Across six steps, 17/18 GPU attempts were accepted; one OR16 solve hit its time
limit and used the exact CPU fallback for that step. No species was permanently
disabled.

| Path | step/s | Valid GPU LP rate | End-to-end valid result |
|---|---:|---:|---|
| Exact-bound GLPK | 1.808 | n/a | 100% |
| Rank-reduced PDLP, 3 steps | 0.195 | 100% | 100% GPU |
| Rank-reduced PDLP, 6 steps | 0.191 | 94.4% | 100% with one exact CPU fallback |

On a same-action three-step comparison, the CPU and GPU final state vectors
were identical (`max_abs_diff = 0.0`), GPU validity was 9/9, and no fallback was
used. This checks the state actually observed by the RL agent, in addition to
per-LP residual and objective checks.

Correctness is therefore repaired, but single-environment GPU acceleration is
not achieved on RTX 4060: PDLP iterations and crossover make it about 9.4x
slower than exact-bound GLPK. A corrected BatchSolve experiment also saturated
near 0.33 environment-step/s for 4--8 identical environments, with more time
limits under contention. The current cuOpt batch API is therefore neither fast
enough nor reliable enough to integrate into the training path on this card.
Use CPU environment workers and CUDA for PPO until a target GPU passes both the
validity and end-to-end throughput gates.

Relevant NVIDIA behavior is documented in the
[LP/QP feature guide](https://docs.nvidia.com/cuopt/user-guide/latest/lp-qp-features.html),
[solver settings](https://docs.nvidia.com/cuopt/user-guide/26.02.00/lp-qp-milp-settings.html),
and [convex optimization features](https://docs.nvidia.com/cuopt/user-guide/latest/convex-features.html).
NVIDIA states that PDLP defaults to lower accuracy than Barrier, crossover has
a significant accuracy/runtime effect, and LP batch mode is deprecated in the
latest documentation.

## Joint three-GEM mode

`--fba-mode joint` constructs one block-diagonal sparse LP from the OR16, NS21,
and LP GEMs (6,137 reaction variables, 4,117 mass-balance rows, and 25,932
nonzeros for the current final consortium). This is equivalent to the current
three independent FBA objectives while keeping the shared-medium coupling in
the dFBA state update. It is therefore a safe performance experiment, not yet
a new OptCom-style cross-feeding objective.

The CPU joint path uses SciPy/HiGHS. On the current RTX 4060/cuOpt 26.2
combination, barrier+crossover can report a cuDSS CSR error for this highly
disconnected block matrix. The adapter retries once with PDLP without
crossover; `concurrent` is remapped to that safe PDLP path. Explicit `cuopt`
mode fails loudly only if both GPU attempts fail, while `auto` then falls back
once to the CPU joint solver.

On the current RTX 4060, the fixed-size consortium LP completed with cuOpt
status `optimal`, but end-to-end time was about 2.8--3.3 s/step including
initialization, versus about 0.33 s/step for the CPU joint HiGHS path. PDLP
residuals must also be checked against the project's scientific tolerances;
GPU success does not by itself establish a speed or accuracy win.

## Installation

Install the cuOpt wheel matching the CUDA runtime and driver on the target
machine. NVIDIA's current installation selector is authoritative; for CUDA 13
the package family is `cuopt-cu13` and for CUDA 12 it is `cuopt-cu12`:

```bash
python -m pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu13
```

NVIDIA のリポジトリには CUDA/Python の組み合わせごとに複数の版があるため、実運用では
`cuopt-cu13` の互換性を確認してから版を固定してください（例: `cuopt-cu13==26.2.*`）。

Do not add this package to the base requirements: it is an optional NVIDIA
wheel and is not available in the default development environment.

## Running

```bash
# Explicit GPU probe; invalid species solutions circuit-break to GLPK
DFBA_SOLVER=cuopt python main.py train --sbml-dir models/sbml/final_consortium \
  --solver-backend cuopt --fba-mode separate --cuopt-method pdlp

# One larger CPU LP for the three GEMs (useful for validating the formulation)
python main.py train --sbml-dir models/sbml/final_consortium \
  --solver-backend glpk --fba-mode joint

# Probe cuOpt and safely fall back to GLPK
DFBA_SOLVER=auto python main.py train --solver-backend auto

# Assign workers round-robin to physical GPUs 0, 1, and 2
DFBA_SOLVER=cuopt DFBA_GPU_IDS=0,1,2 python main.py train \
  --solver-backend cuopt --fba-mode separate \
  --gpu-ids 0,1,2 --n-envs 3 --gpu-slots-per-device 1 --device cpu
```

For one consumer GPU, multiple environment workers can share the card through
separate CUDA/cuOpt contexts:

```bash
DFBA_SOLVER=cuopt DFBA_GPU_IDS=0 python main.py train \
  --solver-backend cuopt --fba-mode separate \
  --gpu-ids 0 --gpu-slots-per-device 4 --n-envs 4 --device cpu
```

This is cooperative process-level concurrency, not MIG or a hard VRAM
partition. Benchmark `n-envs`/slot counts because oversubscription eventually
reduces throughput.

### RTX 4060 scaling snapshot

The end-to-end PPO benchmark used 64 total timesteps, separate FBA, CPU PPO,
and one RTX 4060 for cuOpt. The CPU column used GLPK under the same worker
count:

| Environments | CPU GLPK (step/s) | GPU cuOpt (step/s) | GPU/CPU |
|---:|---:|---:|---:|
| 4 | 7.46 | 5.83 | 0.78x |
| 8 | 10.46 | 6.91 | 0.66x |
| 16 | 6.92 | 4.01 | 0.58x |

The GPU throughput peaked near eight workers but did not exceed the CPU on
this model/hardware. These older raw-throughput figures predate residual and
objective auditing and must not be interpreted as valid solve throughput.
Sixteen workers oversubscribed the GPU contexts. A GPU win requires larger,
numerically suitable LPs and a validated batched path; increasing environment
count alone is not sufficient.

For these GEMs, use rank-reduced `pdlp` with crossover. `barrier` remains
available as a diagnostic request, but the simulator retries it with PDLP after
a cuDSS failure. Every returned solution is checked against the original
stoichiometric matrix and reaction bounds.

## Benchmark protocol

Run the same seed and short horizon with `--solver-backend glpk`, `cuopt`, and
`auto`. Record wall time, `time/fps`, objective values, and final state metrics.
The GPU result should only replace GLPK after objective/flux differences are
within the project's scientific tolerance and the end-to-end wall time (not
only cuOpt's internal `SolveTime`) improves.

## GPU-resident kernel check

`scripts/benchmark_gpu_resident.py` keeps the stoichiometric CSR matrices and
state vectors on CUDA during the timed sparse-matrix section. It also reports an
Amdahl projection for RTX PRO 4000 SFF using published bandwidth/FP32 ceilings.
This is a hardware-scaling estimate, not a substitute for an actual cuOpt LP
solve on the target card.

## Parallel PPO benchmark

`scripts/benchmark_ppo_devices.py` compares the full parallel dFBA/PPO path on
CPU versus CUDA. The environment workers still use CPU COBRApy/GLPK; `--device
cuda` moves the PPO MLP/value network to the GPU. Use `--n-envs` to test the
parallelism that matters for this project.

For an actual cuOpt run, use `--solver-backend cuopt`; each worker receives one
physical GPU in round-robin order through `CUDA_VISIBLE_DEVICES`. Inside that
worker, cuOpt sees its assigned card as device 0. This avoids multiple workers
contending for the same GPU when `DFBA_GPU_IDS` contains one ID per card.
