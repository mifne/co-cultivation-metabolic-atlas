# Objective fidelity and qualification recovery

## CURRENT STATUS (supersedes historical launch commands below)

- PRIORITY CHANGE 21:50 JST: user requested CPU-beating implementation before
  completing the five slow tests. The verified supervisor received SIGTERM;
  lifecycle record confirms worker exit -15. Seed 20286201 passed; seed
  20286202 is partial; three seeds unstarted. The old running manifest is stale.
  Evidence is preserved. Current implementation work is tracked in
  `GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md`; the ACTIVE entry below is historical.

- UPDATE 20:49 JST: device-pivot/LU 120-step regression COMPLETED with a passing
  independent audit and exit code 0 at 20:24:57 JST. 480 GPU LPs, CPU LP 0;
  PHA endpoint error 0.001799586%, biomass 2.31835e-5 g/L, 3HV 1.55727e-6.
  New five-seed protocol: `GPU_DEVICE_QUALIFICATION_20260903.md`, seeds
  20286201--20286205, unchanged CPU loop and fixed GPU implementation.
- ACTIVE at 20:52 JST: supervised `scripts/qualify_gpu_device.py`, supervisor
  PID 223440 / worker 223441. One-shot lifecycle record and log use the stem
  `results/pf_gpu_device_qualification_20260903`. Five new random feed-action
  sequences with the SAME initial model/medium, sequential CPU then GPU.
  Freeze this run's source hashes; no concurrent GPU profiling/benchmarks.

- UPDATE 2026-09-03 19:32 JST: session 87946/PID 205684 no longer exists.
  The last checkpoint is seed 20286101 GPU 85/120, not a completed run.
  No final exception/exit record is present; termination cause unknown.
  The old manifest's `running_frozen_test` is stale. Preserve its evidence.
  Next bounded implementation and validation steps are tracked in
  `GPU_SPEED_RECOVERY_PLAN_20260903.md`. Original frozen sources stay unchanged.
- NEXT JOB launched around 19:48 JST: one-shot supervised device-pivot/LU
  120-step regression, supervisor PID 210469. Status and exit code in
  `results/pf_gpu_device_lu_regression120_20260903.job.json`. The separate
  cache experiment is NOT selected. 83 tests passed; 9 saved cases took
  123.911 s vs reference GPU 191.103 s, not a CPU or end-to-end speedup.

- PASSED: `results/pf_gpu_simplex_repaired_development120.json`, all 120 steps,
  endpoint PHA relative error 5.09242e-8, biomass error 5.97380e-10 g/L,
  3HV fraction error 2.75153e-9. This is development, not independent validation.
- STOPPED FAILED: `results/pf_gpu_simplex_qualification_repaired_20260903/qualification.json`.
  Seed 20286001 failed accuracy; seed 20286002 was interrupted; last three unstarted.
  Parent PID 189540 was terminated after exact-target verification; child PID
  194387 also exited. Unified session 44158 can be closed. Preserve all logs.
  Do not consume the original five-seed set again as an untouched final test.
- Final seed 20286001 completed but FAILED joint accuracy: PHA 3.11865%,
  biomass 0.0247287 g/L, 3HV fraction 0.000931930. All 120 steps accepted.
  Do not label the incomplete five-seed test a pass.
- Offline diagnosis of the now-tested first seed: CPU replay with state/LP
  capture in `results/pf_seed20286001_cpu_diagnostic`; first biomass departure
  is at step 17, major growth/nutrient branch divergence at step 115.
  Full LP residuals remain small; test optimal-face/trajectory sensitivity.
  This is post-test diagnosis, not untouched validation or new training data.
- Completed GPU replay: `capture_cpu_qualification_diagnostic.py --backend tableau`
  `--stop-after 17`, directory `results/pf_seed20286001_gpu_diagnostic17`,
  session 38575 / PID 196373 finished. It overlapped seed 2 briefly, so timings are not
  comparable. Its PHA/biomass match the original GPU run exactly for all 17 steps.
- Root cause identified at step 16: nonunique exchange fluxes (valine/leucine/
  formate/H2/water) despite almost unique growth/storage fluxes. GPU step 17 LP
  re-solved on CPU matches its GPU optimum. This is dynamic state divergence,
  not just a GPU objective/residual error. DFBAlab literature supports this issue.
- COMPLETED GPU TRIAL: `results/pf_seed20286001_gpu_exchange_tie17/`, session 49660
  finished. New isolated `src/gpu_exchange_tie_break.py` adds a fourth GPU LP
  on the existing exchange-optimal face, minimizing exchange L1 excluding
  water/proton/hydroxide. CPU reference unchanged. Experimental, not qualified.
  Unit tests 3 passed; original selected+audit tests 45 passed.
  All 17 steps completed, 68 actual GPU LPs. PHA max absolute difference
  3.66456e-11 g/L; biomass max difference 9.80698e-6 g/L. Valine secretion
  divergence at step 16 removed, but other pool differences start at step 7
  (H2/ethanol/leucine); maximum H2 pool difference ~0.00114 mM. Not a full pass.
- PASSED: 120-step regression of the already-tested seed 20286001,
  `results/pf_seed20286001_gpu_exchange_tie120/`, session 15770 completed.
  Independent audit `pf_seed20286001_gpu_exchange_tie120_audit.json` PASSED.
  Endpoint PHA relative 2.1027123e-5 (0.00210271%), biomass 3.0032573e-5 g/L,
  3HV fraction 1.0679670e-5. 480 actual GPU LPs / 480 attempts, no CPU LP,
  maximum original residual 4.6345840e-8. Replay elapsed 3412.87 s.
  This is a development/regression pass, NOT the five-seed qualification.
- COMPLETED: saved-state Nsight profile, session 51009, two GPU LPs successful,
  107.915 s with profiling. Prefix `results/pf_gpu_exchange_profile_20260903`.
  IMPORTANT: Nsight 2024.6.2 reports driver 13.1 unsupported; it used CUPTI
  12.8. No GPU kernel/memory activity tables, incomplete event warnings.
  Only partial CUDA API counts/times are available. Do NOT claim GPU active
  fraction or attribute memcpy API duration entirely to transfer bandwidth.
  Analysis: `results/pf_gpu_exchange_profile_analysis_20260903.json`.
- CURRENT JOB: five-seed frozen test via `scripts/qualify_gpu_exchange_tie.py`,
  session 87946 / Python PID 205684. Output dir
  `results/pf_gpu_exchange_tie_qualification_20260903/`; log same stem `.log`.
  Seed 20286101 CPU completed; GPU underway. No other GPU jobs.
- A question requesting approval for a common CPU/GPU solution-selection rule
  is pending. Do not change the original CPU reference while awaiting an answer.
  Proposed common rule would be separately versioned and compared to the old
  reference, not presented as achieving unchanged-reference accuracy.
- CPU replay was exact (all saved PHA/biomass differences zero). Step 17 FVA
  critical growth widths ~3.7e-8; step 115 CPU common growth is sulfate-limited
  at zero. Investigate exchange pools/optimality before changing algorithms.
- After diagnosis and a new passing development/regression, use NEW final
  seeds 20286101--20286105 (none consumed yet). Requires an explicit runner
  revision and preregistration, not silently reusing the old frozen outputs.
- LAUNCHED: `scripts/qualify_gpu_exchange_tie.py`, fixed new seeds,
  sequential, original CPU three-stage reference, tolerance 1e-8, 600 s per GPU
  LP, 120 steps, 480 actual GPU LPs per seed. Fail-fast retains failed result
  and lists unstarted seeds; it cannot claim five-seed success from fewer runs.
  `audit_gpu_exchange_tie_replay.py` recomputes all endpoint gates and audits
  nested fourth-LP results. 20 focused synthetic tests passed (including
  rejecting nonfinite CPU iteration counters and fractional GPU attempt counts).
  New runner CPU-only regression completed (session 99951): original CPU
  recorded trajectories match exactly (maximum difference 0), 360 CPU stages,
  0 GPU stages. `results/pf_exchange_new_runner_cpu_regression.json` records
  runner hash ea1d8f753ea9d8a323e4a8e093af3eb7cf6927c66efc22be29944b1465e54e47.
  Freeze all source files listed in qualification.json while the test runs.
  No profile or other GPU process may overlap qualification. It proceeds CPU
  reference then GPU for EACH seed; GPU idles during each CPU-reference phase.
- Selected: bounded GPU simplex with FP64 refinement and GPU primal repair;
  host presolve/QR/postsolve remain. No CPU LP fallback. Not fully resident.
- Both earlier `pf_gpu_simplex_development120` and
  `pf_gpu_simplex_refined_development120` FAILED; keep their evidence.
- Qualification launched with `scripts/qualify_gpu_simplex.py`
  `--development results/pf_gpu_simplex_repaired_development120.json`
  `--output-dir results/pf_gpu_simplex_qualification_repaired_20260903 --workers 1`.
- Then plot its `qualification.json` with `scripts/plot_gpu_simplex_qualification.py`,
  inspect images, and update the report with actual five-seed outcomes.
- A question asking authorization for app follow-up monitoring is unanswered.
  No automation has been created. User authorized unlimited GPU occupation.
- User asked why GPU utilization is low. Read-only live check at local
  18:02:35--48: 12 samples averaged 75.75% (6--100%), whole-device VRAM
  1473 MiB; only GPU regression PID 198176, no CPU reference job. CPU process
  averages ~one logical core. This is NOT a whole-run resource profile.
  `results/pf_gpu_usage_spotcheck_20260903_180235.json` records the samples.
  CPU/GPU per-pivot synchronization is visible in source, but the split between
  host gaps and GPU math requires profiling; do not infer it from utilization.
- COMPLETED BEFORE the five new seeds:
  run one isolated saved-state profile via `scripts/profile_saved_exchange_lp.py`
  under installed Nsight Systems 2024.6.2 (`nsys`), capture-range cudaProfilerApi,
  trace cuda,nvtx, sample none, cpuctxsw none, kill none, capture-range-end stop.
  Input: `pf_seed20286001_cpu_diagnostic/step_016_stage_3.npz` and its metadata.
  This profiled exchange stage + extra tie LP, not a full trajectory/throughput.
  No concurrent GPU jobs. Partial trace only, see driver compatibility warning.
- Do not claim GPU speedup or full qualification from the short/replay tests.

## Scope and diagnosis

Current OR16 + NS21 PHB/PHV + P. freudenreichii, unchanged medium and
three-stage cooperative LP. Previous legacy-model maximum PHA error 0.861%
does not qualify this model. Latest current-model PHA error 0.513% was a
single development seed with biomass error 0.028451 g/L: never a joint pass.

Working hypotheses, to test separately:
1. Surrogate objective differs from the CPU's lexicographic objective.
2. Finite convex hull excludes some live CPU fluxes.
3. Incomplete QP convergence forces tiny feasible correction steps.
4. LP degeneracy may produce different growth/exchange vectors at the same
   objective; feasibility and objective optimality alone need not match dynamics.

## Sequence

1. COMPLETE: audit CPU optimal faces at identical saved states; distinguish
   numerical optimization error from genuine non-uniqueness.
2. COMPLETE: install cuOpt 26.8 in an isolated workspace environment and
   implement an explicitly GPU-only LP adapter (no Concurrent, DualSimplex,
   crossover, or silent CPU fallback). Try the SAME three LP stages first.
3. COMPLETE: original-matrix checks, regression tests and full development
   passed with bounded GPU simplex plus primal feasibility repair. cuOpt
   numerical/convergence failures are retained as negative results.
4. IN PROGRESS: after development passed, froze configuration and test the
   five untouched seeds 20286001--20286005. PHA maximum <=1%, biomass maximum
   <=0.01 g/L, 3HV fraction maximum <=0.01, all steps accepted, no CPU LP.
5. IN PROGRESS: report evidence and limitations; if non-uniqueness is material,
   report it explicitly before changing
   a tie-break rule; do not silently substitute a new CPU reference or tune to
   held-out endpoint values. Compare speed separately from scientific accuracy.

## Primary references

- Applegate et al. (2021), Practical Large-Scale Linear Programming using
  Primal-Dual Hybrid Gradient: https://research.google/pubs/practical-large-scale-linear-programming-using-primal-dual-hybrid-gradient/
- NVIDIA cuOpt methods, tolerances and CPU/GPU execution distinction:
  https://docs.nvidia.com/cuopt/user-guide/latest/lp-qp-features.html
- The earlier AMN / OptNet / DAgger mappings and limitations remain documented
  in GPU_LITERATURE_REFINEMENT_REPORT_20260903.md. Exact GPU LP is an alternative
  mechanistic backend, not a claim to reproduce those neural architectures.

## Evidence update

- At saved states 0/60/90/119, HiGHS dual simplex and IPM returned nearly
  identical critical fluxes. FVA on the parsimonious optimal face (objective
  allowance 1e-8*max(1,|optimum|)) found very narrow growth/PHB/PHV ranges.
  Internal degeneracy can remain, but this audit does not support it as the
  explanation for the observed large critical-flux errors.
- See results/pf_optimal_face_full_audit.json. These are offline diagnostics,
  not training data; no final seeds were consumed.
- Added an optional LP backend to the existing three-stage builder, preserving
  the matrix, objectives and 0.99 retention rules exactly. Failures in any GPU
  stage stop the step; they cannot silently return an earlier stage or call CPU.
- Added a bounded-cache cuOpt adapter with original-matrix GPU residual checks.
  Installed dependencies in .venv-cuopt-26.8, separate from the existing env.

## Numerical alternatives tested

- cuOpt Barrier (normal/augmented systems), FP64 PDLP, three presolve modes,
  exact interval-redundant row removal, cached independent equality rows,
  and objective rescaling. None has completed the current sequential smoke.
- Training-reaction support (2,213 / 6,733 reactions) is an explicitly
  restricted LP, not a full-GEM optimality claim. It also failed its smoke;
  this diagnostic cannot qualify the production model.
- Small quadratic regularization changes the numerical objective and is
  recorded as an approximate QP; it did not solve the convergence problem.
- HiGHS algebraic presolve + cuOpt FP64 PDLP + primal-only HiGHS postsolve
  reduced the initial LP to ~2,400 variables. First dFBA step completed all
  three GPU stages. Next step failed in exchange parsimony at 60 s.
  CPU simplex/IPM/crossover iterations are disabled and checked. Matrix
  assembly and presolve/postsolve remain HOST operations, not GPU residency.
  Never pass a basis to Highs.postsolve: that enables CPU simplex cleanup.
- Bounded-variable two-phase GPU simplex is being tested as an alternative.
  Initial naive pivoting was unstable; Harris-style ratio selection, basis
  refactorization and original-matrix checks have been added. Remains
  experimental; no production default has been changed.
- GPU simplex now PASSES the 3-step current-GEM smoke (all 9 stages, CPU LP 0).
  Cold-start implementation: 228.52 s; incremental QR/basis reuse: 81.45 s.
  Both have critical endpoint differences ~1e-12 or smaller. This is only a
  short smoke and much slower than CPU; not a speedup or 24 h qualification.
- Configuration frozen for development120: tableau + host algebraic presolve,
  tolerance 1e-8, stage time limit 120 s. Output pf_gpu_simplex_development120.
  No final seeds have been used. Do not change GEMs or thresholds to pass.
- During development120, a separate GPU dual-simplex repair probe was tried.
  Most starts were not dual-feasible; one repair used the 20 s budget and
  declined. It was removed from the selected implementation. The SHA256 of
  gpu_bounded_simplex.py returned exactly to
  732f6b409fb235e82a7ffada6fe2657a7ac9b370a1fb50b25698db9c6ea1d73a,
  identical to incremental_smoke3 and the running development120.
- A separate cuOpt probe restored original bounds removed by presolve;
  the hybrid PDLP/Barrier smoke still failed numerically at the first step.
  Do not promote it. Development120's settings/process were not changed.

## Frozen final-test launch

User explicitly confirmed there is no GPU occupation time limit. After the
development120 JSON reports a full joint pass, run (within the project):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-cuopt-26.8/bin/python scripts/qualify_gpu_simplex.py --development results/pf_gpu_simplex_development120.json --output-dir results/pf_gpu_simplex_qualification_20260903 --workers 2
```

The runner verifies implementation and model fingerprints, refuses a failed
or short development run, and creates a new output directory (no overwrite).
The final numerical tolerance is unchanged at 1e-8. The per-LP wall budget is
fixed in advance to 240 s to account for two processes sharing the device;
this is not an accuracy-threshold change. There is no whole-run time limit.
Do not use contended verification times as a fair CPU/GPU speed comparison.
Generate the figure only after five full valid trajectories have finished:

```sh
.venv-cuopt-26.8/bin/python scripts/plot_gpu_simplex_qualification.py --qualification results/pf_gpu_simplex_qualification_20260903/qualification.json
```

Do not retune to these final seeds. If any fails, retain that failure and
distinguish subsequent regression checks from untouched validation.

Primary implementation references:
- HiGHS primal-only postsolve versus basis-triggered cleanup:
  https://github.com/ERGO-Code/HiGHS/blob/master/highs/lp_data/Highs.cpp
- Classical simplex and numerical caveats (not claimed as a new algorithm):
  https://github.com/scipy/scipy/blob/v1.10.1/scipy/optimize/_linprog_simplex.py

## Development120 failure and numerical refinement

The first frozen development run failed at step 113/120: the reduced original
LP residual was 3.08724e-5, above the unchanged 1e-5 feasibility gate. Its PHA
error was tiny, but that does not qualify an incomplete/rejected trajectory.
The failed JSON/log remain `pf_gpu_simplex_development120.*`; final seeds have
NOT been consumed. The preceding frozen launch command is conditional and
must not be used with this failed report.

Next revision adds GPU iterative refinement after basis solves and requires
a freshly refactored basis/reduced costs before declaring either phase
optimal. These correct roundoff without changing the GEM, LP or thresholds.
An exact failed-LP snapshot (including the incoming basis) is now saved for
fast replay if another failure occurs. Start with saved late-state LP probes
and unit tests, then a new full development trajectory. Only a passing new
development report may authorize untouched five-seed verification.

Refinement tests: `pf_gpu_simplex_refinement_tests_v2.log`, 20 passed. Saved
GPU-visited states 90/119 were CPU-relabelled offline and all six GPU LPs
passed (`pf_gpu_simplex_refinement_late_v2.json`). This is not a replay of
the failed trajectory: its context provenance is explicitly recorded.

New development: `pf_gpu_simplex_refined_development120.json`, tolerance 1e-8,
feasibility 1e-5, 600 s per LP, 200,000 maximum iterations. User authorized
unlimited occupation; only operational iteration/time caps were increased.
No mathematical or accuracy criterion changed. The final runner derives
its operational time cap from the passing development configuration.
Plots will include both endpoint gates and all 120 trajectory differences;
additional trajectory summaries do not replace the predeclared endpoint gates.

## Current selected route and final scheduling

The selected source remains the refined `gpu_bounded_simplex.py`; the running
development process is unchanged. A separate experimental device-pivot fork
(`gpu_device_bounded_simplex.py`, `gpu_simplex_pivot_batch.py`) uses 100-pivot
CUDA Graph batches. Its 12 small tests and six saved-state LP probes passed,
but it has NOT passed trajectory qualification and is not selected. Do not
silently replace the selected backend with that experiment.

The two simultaneous GPU jobs showed substantial contention. Before any
final seed is consumed, choose **one worker** for the final accuracy tests
(this is execution scheduling, not changing any tested condition or gate).
Use the following updated command after the refined development passes:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-cuopt-26.8/bin/python scripts/qualify_gpu_simplex.py --development results/pf_gpu_simplex_refined_development120.json --output-dir results/pf_gpu_simplex_qualification_refined_20260903 --workers 1
```

The plotting command should point at that new directory's qualification.json.
Progress can be read compactly with `scripts/summarize_gpu_solver_log.py LOG`;
its summaries explicitly do not qualify partial trajectories. An async
question about authorizing app follow-up monitoring was sent; no automation
has been created without the user's reply. Computation continues in this turn.

## Second failure: primal bounds, and selected repair revision

`pf_gpu_simplex_refined_development120` failed at step 112/120. Its saved LP
replays exactly: equality residual 7.50e-12, but a reaction lower-bound
violation 7.94408e-5. Thus iterative refinement of equations alone did not
restore primal feasibility. Tightening the pivot cutoff to 1e-12 did not
help and was reverted. A cold basis also failed on the same saved LP.

Rebuilding Phase I from the failed candidate basis, on GPU, corrected the
warm case in two extra pivots (~0.72 s; residual 2.24e-11), and the cold case
in 16 extra pivots (~0.74 s; residual 2.19e-9). No CPU solution, bound or
objective change is involved. The selected backend now permits up to three
such repairs within the original per-logical-LP 600 s budget. Each attempt
has a 200,000 iteration limit. Unrepaired failure remains fail-closed.
Reports distinguish logical LP stages from GPU attempts and repair counts.
Related tests: `pf_gpu_primal_repair_tests_final.log`, 36 passed (includes
unselected device-pivot prototype tests; not a whole-repository suite).

CURRENT RUN: `pf_gpu_simplex_repaired_development120.json` / `.log`.
The preceding two development JSONs are failures and must not launch final
seeds. After this NEW complete trajectory passes all unchanged gates:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-cuopt-26.8/bin/python scripts/qualify_gpu_simplex.py --development results/pf_gpu_simplex_repaired_development120.json --output-dir results/pf_gpu_simplex_qualification_repaired_20260903 --workers 1
```

No final seed has yet been consumed. Numerical implementation files must
remain fixed during this development and independent qualification.

Separate cuOpt control: Stable3 PDLP + host presolve, internal tolerance 1e-5,
same 1e-5 original-feasibility gate and unchanged endpoint gates, still failed
at the first step's third stage (30 s, dual convergence). Saved as
`pf_pdlp_practical_tolerance_smoke3.*`. The 42.09 s partial GPU run was
contended with development; not a speed comparison. It was not adopted.
