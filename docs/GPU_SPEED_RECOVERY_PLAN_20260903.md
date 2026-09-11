# GPU speed recovery after interrupted validation

## Observed state, 2026-09-03 19:32 JST

The five-seed run did **not** complete. Its last durable checkpoint is seed
20286101, GPU step 85/120 (2589.53 s); CPU 120/120 completed (113.58 s).
PID 205684 is absent, the original terminal session no longer exists, and
there is no final error/exit record. The cause of termination is unknown.
No OOM event was visible in the inspected kernel-log tail; this does not
exclude all external causes. Preserve the original files and stale manifest
as evidence; never interpret `running_frozen_test` alone as a live process.

The prior 120-step development regression remains a pass, not five-seed
qualification. Original CPU/GPU reference sources remain unchanged.

## Bounded implementation sequence

1. Record the interrupted state and preserve hashes/checkpoints. DONE.
2. Improve the isolated device-pivot prototype, without changing the frozen
   reference: inherit GPU primal repair, reuse one LU factorization for the
   multiple solves in each refactorization, and retain original residual
   gates. Device pivot batches already remove per-pivot host reads.
3. Test toy LPs, iteration limits, infeasibility and GPU repair. Compare
   reference/device implementations sequentially on identical saved full
   GEM LPs, including the fourth exchange-selection stage. Count CPU LP
   calls and audit the original unreduced constraints and primary objective.
4. Only after saved-state checks pass, run the known 120-step regression
   with a new output path. No CPU-reference change or tolerance relaxation.
   Add a process-independent launcher with durable PID/start/exit records
   for long jobs; do not silently call partial trajectory restoration exact.
5. Requalify all five action sequences from their initial states after any
   implementation change. The partially inspected first seed is no longer
   untouched if its values guide tuning; distinguish it from held-out seeds.
6. Next algorithmic work: separate objective-stage basis caches, verified
   basis reuse and multi-environment batching, measured against a competitive
   parallel CPU reference. Do not claim speedup from utilization alone.

## Scope of the next result

A faster saved-state LP is not faster 24-hour dFBA, PPO learning or a fully
GPU-resident environment. Host presolve/QR/postsolve remain in this phase.
CUDA Graphs lower launch overhead but do not reduce the required LP pivots.
LU factorization reuse reduces repeated linear algebra, not the LP problem.

References: [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html),
[CuPy LU factorization](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.linalg.lu_factor.html),
[basis reuse in dFBA](https://arxiv.org/abs/2003.03638).

## First measured result

The first device-pivot/refactor prototype passed all 9 saved-state cases
(steps 16, 17 and 115, three primary stages and a fourth LP inside each
exchange stage). Each backend made 12 actual GPU LP calls, no CPU LP calls.

|Metric|Reference GPU|Device-pivot + shared LU GPU|
|---|---:|---:|
|Sum of the 9 recorded case times|191.103 s|123.911 s|
|Maximum recomputed original-LP violation|2.174e-10|4.463e-9|
|Recorded objective/residual checks|Pass|Pass|

The ratio of summed times is 1.542 (35.16% less time) for this saved-state
workload, not CPU speedup, full dFBA speedup or an independent accuracy
qualification. There is one run per case, fixed backend order, warmed CUDA
libraries, no confidence interval. Two of nine cases became slower.
Maximum difference in the sampled post-tie net exchange rates between GPU
implementations was 4.261e-7 mM/h. This is not a full-trajectory guarantee.

Raw data: `results/pf_gpu_device_saved_comparison_20260903.json`.
Independent re-evaluation: `results/pf_gpu_device_saved_comparison_audit_20260903.json`.

83 combined regression/unit tests passed in 6.60 s; log
`results/pf_gpu_device_combined_tests_20260903.log`. This includes the 30 new
device/audit/cache/launcher tests; do not add overlapping intermediate totals.

A second isolated experiment separates `exchange_primary` and `exchange_tie`
basis caches. The original backend uses `parsimony` for both, so the tie LP
overwrites the preceding primary cache. Whether separating them improves
performance must be measured; a prior basis can also be worse after a large
nutrient/state change. Do not select this second experiment on intuition.

The separate-cache experiment completed and passed the saved-state numerical
checks, but took 171.191 s, versus 123.911 s without separation. At step 115
its exchange stage took 68.752 s versus 20.649 s. This variant is **not selected**
for the next full regression. It remains available only as an explicit option.
Audit: `results/pf_gpu_device_variants_audit_20260903.json` (27 recorded cases,
three variants, 12 actual LP calls per variant; same-state checks only).

One-shot long-job launcher: `scripts/launch_recorded_job.py` records separate
supervisor/worker PIDs, `/proc` start times, command, thread settings and exit
code. It is not a recurring automation and cannot survive WSL shutdown or
power loss; missing exit records still require a liveness check.

## Current long regression

COMPLETED: 2026-09-03 20:24:57 JST, exit code 0, full numerical audit passed.
Elapsed replay 2231.646 s. PHA endpoint error 0.001799586%, biomass error
2.31835e-5 g/L, 3HV fraction error 1.55727e-6, 480 GPU LPs and no CPU LP.
The next five-seed test is preregistered in `GPU_DEVICE_QUALIFICATION_20260903.md`.
The following launch details are historical, not an active-process claim.

Launched 2026-09-03 around 19:48 JST, supervisor PID 210469. No other GPU jobs.
Runtime record: `results/pf_gpu_device_lu_regression120_20260903.job.json`.
Log: `results/pf_gpu_device_lu_regression120_20260903.log`.
Output: `results/pf_gpu_device_lu_regression120_20260903/`.
Seed 20286001, unchanged original CPU reference, full 120 steps, device
pivots + shared LU, **without** separate stage caches. The wrapper runs the
independent full-trajectory audit after replay and propagates its exit code.
No five-seed pass, full GPU residency or CPU speedup has been established.

Freeze numerical and audit source files listed in this run's metadata until
it finishes. Do not launch another GPU test/profile concurrently. A failed
audit must be investigated before independent qualification or PPO training.
