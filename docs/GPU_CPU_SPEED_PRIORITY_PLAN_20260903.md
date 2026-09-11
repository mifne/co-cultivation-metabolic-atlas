# Priority: beat CPU at matched accuracy, not complete five slow trials first

User-authorized change of order, 2026-09-03 21:50 JST.
The supervised five-seed run was stopped by SIGTERM to the verified supervisor
PID/start-time. Lifecycle record confirms requested signal 15 and worker exit
-15 at 21:50:47 JST. Seed 20286201 passed; seed 20286202 is partial; the
remaining seeds were not run. Keep all evidence. Do not count it as five passes.

## Frozen reference

- Original CPU 3-stage HiGHS model, unchanged GEMs and bounds.
- GPU device-pivot 120-step regression passed; independent first seed passed.
- First independent seed CPU 115.005 s, GPU 2447.095 s. The current GPU solver
  is not faster. Moving only host setup cannot remove the many GPU pivots.

## Implementation sequence

1. Compile candidate optimal bases offline. Apply exact coordinate scaling
   z_i = X_i v_i so that most constraint coefficients become fixed. This is
   an algebraic variable change, not a changed biological model.
2. GPU evaluation: reuse basis linear algebra, correct the few changing rows
   with a low-rank update, check full original primal residuals and numerical
   dual/KKT optimality. Reject unknown structure or invalid candidates.
3. Batch independent environments/queries on GPU. Report basis-building cost,
   GPU resident memory, transfer-inclusive timings and acceptance rate. A
   saved-state kernel speedup alone is not full dFBA speedup.
4. Connect validated fast path with explicit GPU-only correction/rejection;
   no silent CPU LP substitution or accepting an inaccurate surrogate.
5. Short dynamic regression and CPU comparison, then competitive parallel
   CPU/batch GPU throughput. Only then run the larger final qualification.

No claim of fully resident simulation until bounds, objectives, state updates
and checks stay on device during the online loop. Initial offline CPU
compilation must be disclosed and amortization reported.

Primary references: [Brunner & Chia 2020](https://arxiv.org/abs/2003.03638),
[Fan et al. 2023](https://proceedings.mlr.press/v202/fan23d.html).
Their algorithms/results motivate the design; their speedups are not reused
as measurements of this implementation.

## Implemented experiments and current evidence

- `src/gpu_certified_basis.py`: offline HiGHS anchor solve, exact biomass
  coordinate change, sparse inverse, GPU Woodbury updates, two rounds of
  primal/dual iterative refinement, full original residual and numerical KKT
  checks. Original LP tolerances and biological objectives are not relaxed.
- `src/gpu_basis_bank.py`: multiple offline bases, candidate selection and
  acceptance masks on GPU. Invalid/missing values are NaN, never zero flux.
- `src/gpu_revised_basis.py`: experimental GPU bounded revised-simplex repair,
  including original-cost dual repair when dual feasible and a bound-distance
  feasibility phase otherwise. Final certificate always uses original cost.
  Fixed pivot budgets, no online CPU optimizer, fail closed when insufficient.
- `src/gpu_compiled_community_backend.py`: opt-in integration adapter; explicit
  optional GPU repair. CPU LP assembly, normalization, setup on cache misses,
  result transfer and simulator updates REMAIN HOST WORK. Do not call this
  whole-environment GPU residency. No default production solver was changed.

### Component speed, not whole-dFBA speed

`results/pf_gpu_certified_basis_refined_comparison_20260903.json`
contains the original three stages for 32 RHS perturbations per stage around
saved step 16 (nonzero shared-pool RHS multiplied by 0.999--1.001; zero stays
zero). Every one of these 96 GPU candidates was accepted. Three-stage totals
below sum separate stage measurements, not a coupled dynamic trajectory.

| Comparator / scope | Time, s |
|---|---:|
| GPU resident, sum of five-repeat medians | 0.372279 |
| GPU host-adapter + transfers, sum of one measurement per stage | 0.424453 |
| CPU NumPy/SciPy, SAME batched basis/certificate algorithm, sum of medians | 1.459129 |
| CPU SciPy HiGHS, sequential cold solve per query, one measurement | 26.615284 |

GPU vs the same CPU candidate algorithm: 3.92x for the resident component;
this is not a benchmark against tuned parallel/warm-start HiGHS. The 71.49x
cold-HiGHS/component ratio must NOT be advertised as training/dFBA speedup.
Offline anchor compilation cost 20.489451 s; GEM/state/LP construction excluded.
The saved step 17, 18, 113--116 queries were all rejected by the single-basis
evaluator; this local acceptance rate does not establish broad coverage.

### Short integration before any long qualification

`results/pf_gpu_basis_bank_short_20260903.json`: offline train seed 20286311,
4 steps, 16 CPU LPs including fourth-policy anchors, 16 distinct bases.
Offline training 5.971424 s; factorization/GPU setup 110.839284 s; GPU pool
1338.425 MiB. Test seed 20286312 was not supplied to the bank.
The test stopped at the first step's aggregate LP (bank miss). It completed
ZERO dynamic steps, so its 0.187980 s runtime is a rejection time, NOT a
speedup against the CPU's 8-step 8.100660 s run. CPU LP calls online: zero.

The next short probe enables GPU repair after a bank miss. This is a
development seed once inspected, not an untouched final test. Do not enlarge
the five-seed qualification until a complete short rollout is accurate and
faster. Current obstacles are candidate coverage, repair iterations and
remaining host simulation work, not insufficient validation duration.

## Subsequent short probes and restored tiered-architecture context

24-pivot repair recovered the first test step's aggregate LP (3 pivots) and
primary exchange LP (8 pivots), but the separate fourth LP remained invalid.
64 pivots did not fix it and amplified numerical error; neither run completed
a dynamic step. Retained files: `pf_gpu_basis_bank_repair_short_20260903.json`
and `pf_gpu_basis_bank_repair64_short_20260903.json`. No full speedup ratio.

An additional opt-in route now reuses the primary exchange primal/dual/basis
state, fixes nonzero reduced-cost variables, and solves the secondary objective.
The resulting candidate must ALSO pass a reconstructed numerical KKT test for
the original allowed-loss fourth LP; exact-face optimization alone is not
sufficient. CPU reference/GEMs unchanged. Source: `src/gpu_optimal_face.py`.
The reduced-cost-fixing principle is documented in
[Gurobi's hierarchical LP reference](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/multiobjective.html);
no Gurobi dependency or its performance numbers are used here.
The cached-bank probe `pf_gpu_basis_face_short_20260903.json` completed step 1,
then rejected step 2 aggregate LP. Its 5.427419 s cannot be compared as an
8-step speedup to CPU 8.842965 s. Online CPU LP calls: 0. All probes finished.
Source snapshots are retained beside the last two probe JSONs.

The user's reminder of the earlier tiered design is correct. The prior
`cooperative_gpu_service.py` path tries 128 candidates and retries failed rows
with 2,048, with per-species block-composed candidates in the projector.
`gpu_qp_parallel_training_smoke_retry2048.json` recorded 5 retries, all 5
recovered, and 512/512 feasible requests. These are historical-model results.
The newer basis-bank experiments do NOT yet reconnect that neural/convex-hull
front end. Preserve it as the inexpensive tier; assess a nested larger-hull
tier and a full-original-variable GPU correction tier for rejected requests.
Large dictionary candidate counts alone cannot prove LP objective fidelity or
the same long-term exchange trajectory. Per-tier gates and escalation rates
must distinguish physical feasibility, objective error and rollout error.

## Implementation continuation: repaired warm states and sparse orientation

The 24-pivot dual-priority probe completed one step and rejected step 2;
all available aggregate anchors lacked a dual-feasible warm start there.
No end-to-end speedup has been established by this probe.

Changes now under short testing:

1. Compute nonbasic reduced-cost sign feasibility separately from the
   primal residual / complementarity score. Prefer a dual-feasible repair
   start when one exists; do not change acceptance tolerances.
2. Cache explicit CSR transposes of the inverse and original matrix to
   avoid repeatedly using scatter-style transposed sparse multiplication.
   Apply the same orientation improvement to the CPU algebra comparator.
3. Reuse a previously certified GPU repair basis across time. Apply exact
   low-rank changes on its CURRENT columns and recertify the new LP; do not
   reuse old fluxes or silently assume unchanged medium/biomass.
4. Preserve original cache hashes. A caller may provide the hash-matched
   old compiler source; reuse is allowed only if ASTs of ALL module-level
   definitions/imports other than the online evaluator are unchanged.

Rejected alternatives / diagnostics:

- Eliminating explicitly fixed columns removed only 13/7373 variables on
  a representative LP; not an effective size reduction here.
- Direct CuPy stream capture rejected cuBLAS calls in installed CuPy
  14.2.0. No environment package was replaced and no unsafe bypass applied.
  See `scripts/probe_basis_graph.py` and CuPy stream-capture documentation.

Long five-condition validation remains deferred until a complete short
dynamic comparison passes accuracy and speed. Production CPU/GEM defaults
and the historical QP frontend have not been replaced.

### Device control and microbatch implementation (continuation)

- Added capture-safe FP64 operations through a private documented cuBLAS C
  API handle, pivoted GPU small solves, and GPU CSR products. A persistent
  cuBLAS workspace is required: automatic allocation nodes are not allowed
  inside CUDA conditional graph bodies. No installed package was patched.
- Added GPU conditional control. An initially unrolled IF graph consumed
  excessive VRAM; the 128-pivot integration completed one step then rejected
  step 2 after 257.47 s including failed work. It finished BEFORE an attempted
  stop signal. Its result is retained, and no speedup is claimed.
- Replaced the unrolled capture with one device WHILE body, one iteration
  counter and reusable arrays. This is GPU-side stopping, not polling each
  pivot from Python. Bound/primal/dual/gap acceptance remains unchanged.
- Replaced the dense full-matrix column gather with a CSC-column GPU kernel;
  the LP only needs the entering and leaving columns. Sharing an existing
  bank evaluator also avoids duplicated inverse buffers.
- Added `gpu_batched_compiled_backend.py` and a greenlet-based benchmark that
  runs actual independent environments to their next LP barrier and batches
  that stage. Host assembly, dispatch, transfers, four GPU LPs, and state
  updates are included in online wall time; bank/environment setup is separate.
  The baseline is explicitly sequential single-thread HiGHS environments,
  not a tuned parallel CPU implementation. One-step throughput is not a
  long-trajectory accuracy qualification.
- The first 8-environment wiring probe found a missing per-environment
  backend history bridge after stage 1. This was fixed with YieldingLPBackend;
  its 2.126 s failure runtime is NOT a speedup versus 7.441 s CPU.

References for these engineering mechanisms:

- [NVIDIA CUDA conditional graph nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/)
- [CUDA Python graph/runtime API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html)
- [Brunner and Chia: basis reuse in community dFBA](https://arxiv.org/abs/2003.03638)

These references motivate the architecture, not measured speed/accuracy of
this repository. All performance conclusions require the local run results.

### 2026-09-04: matched microbatch comparison

`pf_gpu_microbatch2_step1_fixed_20260903.json` completed two independent
environments, one step each, without online CPU LP calls. Cold CPU/GPU times
were 1.9050/3.0515 s (GPU slower). The one prepared-cache repeat took
2.4153/0.9472 s (2.550x); all endpoint gates passed. This is a limited actual
dFBA result, not a statistical or long-horizon claim. Earlier CPU thread
limits applied to OpenMP/BLAS, not explicit HiGHS options.

Next checks, in order:

1. Eight environments, three repetitions including cold setup. Explicitly
   set HiGHS threads=1 and parallel=False; compare four concurrent CPU LP
   workers. Only SciPy LP arrays enter threads, never COBRA/environment objects.
2. Use independently deep-copied/reset pristine environments for faster
   untimed setup (clone probe matched a full CPU step exactly); preserve
   model fingerprints and offer fresh construction as an option.
3. Extend to multiple steps; investigate stage-1 feasible basis reuse in
   stage 2 if cold anchors cannot repair a changed LP. Do not relax gates.
4. Only after complete short trajectories pass speed and accuracy, widen
   held-out conditions and reconnect the inexpensive surrogate/QP tier.

LP assembly, dispatch, transfers, graph compilation during an online call,
GPU correction and state integration remain in the timer. Offline bank
construction/loading and environment cloning are separately disclosed.

Eight-environment results and resulting revisions:

- Four concurrent CPU LP workers finished one step in 3.430 s. The initial
  32-pivot GPU run rejected one environment at the exchange LP (residual
  1.93248e-5 > 1e-5); no speedup is reported for that failed run.
- Raising the GPU budget to 64 completed all eight environments and all
  endpoint gates, but prepared GPU times 4.575/4.516 s were slower than
  four-worker CPU 3.478/3.382 s. Sequential-CPU gains alone do not establish
  an advantage over an appropriate parallel CPU baseline.
- Implemented small initial budgets (maxmin 8, later LPs 32) with larger
  GPU correction only for rejected rows; recertify the full original LP.
  Added a zero-pivot fourth-stage certificate before additional work and
  increased bounded graph cache capacity to avoid four signatures thrashing
  a two-entry cache. The prepared GPU repeat decreased to 3.828 s versus
  CPU 3.505 s, still not a win.
- Added device-counter-sized elimination of the EXACT identity-padded
  unused update block. This is not truncation or lower precision: all
  actual basis updates remain, and full original numerical gates are
  unchanged. Added tests for identity padding with nonzero padded RHS and
  continuation from a rejected short-budget state. 25 targeted tests pass.
- Next: measure two complete steps across eight environments; diagnose
  cross-stage warm starts if a changed LP fails. Do not launch long training
  or present failures as speedups.

### Continuation outcome: cross-stage warm start and independent coverage

- Eight environments/two steps still reject step-2 aggregate LPs. A stage-1
  feasible warm basis removes primal infeasibility (~1e-12 residual), but six
  rows remain nonoptimal after 64 pivots. Do not count partial work as a win.
- Sixteen environments/one step completes with all endpoint gates passing,
  but prepared GPU 7.552 s is slower than four-worker CPU 6.773 s (0.897x).
- Added reproducible offline aggregate-bank extension: independent training
  seeds 20286501–20286504, two steps each, eight new bases. Training/assembly
  18.63 s plus factorization 52.15 s (70.78 s total), 24 offline CPU LP calls.
  Loader verifies completion, model fingerprints, compiler AST/hash, artifact
  hashes and training/evaluation seed separation. Nothing is fit to test LPs.
- Expanded aggregate bank (4→12) reduces step-2 aggregate rejections from
  6/8 to 2/8. No environment completes the second step because the barrier
  stops when any LP is rejected. Full-run speedup remains unestablished.
- 26 targeted tests pass, including cross-objective warm-state recertification,
  rejected-state continuation, active identity padding, and parallel CPU LPs.

Next development priorities: diagnose the two remaining stage-2 rejections;
expand independent training coverage together with inexpensive basis routing
(not an ever-growing exhaustive bank scan); then validate the later exchange
LPs and complete short rollouts. Keep the old neural/QP front-end connection
and fully device-resident environment as separate uncompleted integration
tasks. Do not restart long validation/training yet. All current probes ended.

### AMN-inspired continuation, 2026-09-04

See `AMN_GUIDED_ACCELERATION_PLAN_20260904.md` for the paper/code review,
adaptation limits, complete measurements and next priorities. Implemented
learned starting-basis proposals followed by full original-LP certification,
GPU repair and exhaustive fallback only when needed. Actual independent
training: 32 seeds, two steps, 192 CPU LPs; offline total 121.81 s.

With eight environments/one step, two prepared neural repeats passed all
gates and exceeded four-worker CPU by 1.127/1.122x (GPU 3.158/3.231 s).
A constant-basis repair-first control also won by 1.071x: not all of the gain
is due to learning. Neural initialization reduced aggregate pivots 55→15
on the matched one-step inputs. Initial GPU setup still makes the cold run
slower. Two-step validation continues to reject two stage-2 LPs; no claim of
long-horizon or whole-training speedup. 29 targeted tests passed, all jobs
ended, production defaults unchanged.
