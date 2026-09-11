# Device-pivot GPU qualification — preregistration and progress

## Development regression completed

2026-09-03 20:24:57 JST: supervised job exited with code 0 and its independent
audit passed. Seed 20286001, original CPU HiGHS dual-simplex reference,
unchanged GEMs, 120 steps / 24 simulated hours.

|Endpoint criterion|Observed error|Required maximum|
|---|---:|---:|
|PHA, relative|0.001799586%|1%|
|Maximum species biomass, absolute|0.0000231835 g/L|0.01 g/L|
|3HV mole fraction, absolute|0.00000155727|0.01|

480 actual GPU LPs, 480 attempts, zero online CPU LP calls. Maximum audited
original-LP residual 2.41234e-7, below 1e-5. Replay time 2231.646 s (37.194 min).
The previous GPU four-LP regression took 3412.870 s (56.881 min), a recorded
ratio of 1.529. This is a historical single-run timing comparison: the older
run overlapped CPU diagnostics for part of its duration. It is not a
controlled CPU/GPU speedup measurement or a confidence interval.

Files: `results/pf_gpu_device_lu_regression120_20260903/{replay,audit}.json`
and `results/pf_gpu_device_lu_regression120_20260903.job.json`.

## Frozen independent test, registered before launch

- Seeds: **20286201, 20286202, 20286203, 20286204, 20286205**.
- Five newly generated random feed-action sequences, **same initial state,
  medium, kinetic parameters and GEMs**, not five biological conditions.
- Actions: NumPy default_rng(seed), uniform [0.05,0.95), (120,5), float32.
  Per-seed action-array SHA256 is stored in the result.
- dt = 0.2 h; 120 steps; initial NH4 = 0.05 mM; common growth cap = 0.005 h^-1.
- CPU uses the exact original reference runner (delegation, not a rewrite).
- GPU uses device pivot batches, shared LU/refinement, the unchanged fourth
  exchange-selection LP, and original basis-cache namespaces.
- **Separate-stage-cache experiment is NOT selected.**
- GPU tolerance 1e-8, original residual gate 1e-5, 600 s per logical GPU LP.
- All three endpoint gates above, all 120 steps, 480 GPU LPs and zero CPU LP
  calls required. No parameter tuning, threshold relaxation or CPU fallback.
- CPU reference then GPU for each seed, one environment at a time; no other
  GPU workload. Fail-fast on the first numerical/accuracy failure; report
  unstarted seeds explicitly. Only all five passing is qualification success.
- Preserve all prior failed/interrupted runs; the interrupted 20286101 run
  is not relabeled as completed or included in this new independent set.
- Freeze source/GEM hashes; source changes invalidate the run.

Runner: `scripts/qualify_gpu_device.py`.
19 targeted runner/auditor tests passed, including reuse of the original CPU
loop, incomplete trajectories, missing fourth stage, incorrect GPU backend,
CPU fallback, missing source hash, and endpoint error exceeding 1%.
These tests overlap earlier audit tests; do not sum overlapping totals.

Launched at **2026-09-03 20:52:08 JST**. Supervisor PID 223440, worker PID
223441. At startup the first CPU reference is progressing normally.
One-shot lifecycle record: `results/pf_gpu_device_qualification_20260903.job.json`.
Live log: `results/pf_gpu_device_qualification_20260903.log`.
Manifest and per-seed files: `results/pf_gpu_device_qualification_20260903/`.
These PIDs are a launch record; check process start ticks and result files
before reporting a later live status. Do not assume a stale running manifest
means an active process.

The completed development-regression figure is
`results/pf_gpu_device_regression_errors_20260903.png` (editable SVG alongside).
Its three panels compare PHA, species biomass and 3HV-fraction errors against
the unchanged CPU reference. Figure was rendered and visually inspected;
the source hashes and development-only scope are recorded in its JSON file.

## Scope and remaining work

This validates numerical agreement with a particular CPU reference over a
finite test set, not experimental biology, broad parameter generalization or
PPO training. Host presolve/QR/postsolve and environment state updates remain;
full device residency and speedup over a competitive CPU solver are not yet
achieved. Multi-environment batching and verified basis reuse require their
own subsequent accuracy and throughput benchmarks.
