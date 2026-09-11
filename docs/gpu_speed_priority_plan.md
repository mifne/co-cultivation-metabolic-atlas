# GPU Speed-First Plan / Precision-Maintaining GPU Acceleration — Integrated Document

> This document integrates two related records of GPU-accelerated dFBA/LP solver development: the user-authorized speed-first priority plan (`GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md`, 2026-09-03, with continuations appended through 2026-09-04) and the precision-maintaining GPU acceleration design note (`GPU_NEXT_SPEED_ARCHITECTURE_20260903.md`, 2026-09-03). The latest status is: an AMN-inspired learned-basis proposal with full original-LP certification passes an eight-environment, one-step gate with prepared caches and slightly beats a four-worker CPU baseline (1.127x/1.122x), but cold start remains slower, two-step validation still rejects two stage-2 LPs, and no whole-training or long-horizon speedup has been established. The user-authorized speed-first priority (beat CPU at matched accuracy before completing the five slow trials) is in effect; production defaults and the frozen regression model are unchanged.

## 最新の結論 (Latest conclusions, as of 2026-09-04)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-04, AMN-inspired continuation section)

- Learned starting-basis proposals followed by full original-LP certification, GPU repair and exhaustive fallback only when needed are implemented.
- Actual independent training: 32 seeds, two steps, 192 CPU LPs; offline total 121.81 s.
- Eight environments / one step: two prepared neural repeats passed all gates and exceeded four-worker CPU by 1.127/1.122x (GPU 3.158/3.231 s).
- A constant-basis repair-first control also won by 1.071x: not all of the gain is due to learning.
- Neural initialization reduced aggregate pivots 55→15 on the matched one-step inputs.
- Initial GPU setup still makes the cold run slower.
- Two-step validation continues to reject two stage-2 LPs; no claim of long-horizon or whole-training speedup.
- 29 targeted tests passed; all jobs ended; production defaults unchanged.
- Overall status stated in the design note (出典: GPU_NEXT_SPEED_ARCHITECTURE_20260903.md, 2026-09-03): 設計・実装中 (design/implementation in progress); full-dFBA CPU-beating speed, GPU-contained residency, and scalability across unknown conditions are 未達・未検証 (not achieved / not verified).

## 1. 開発ログの診断 (Dev-log diagnostics / starting point)

(出典: GPU_NEXT_SPEED_ARCHITECTURE_20260903.md, 2026-09-03)

- The current exact-form GPU simplex met the 3-output criteria on the development 24 h trajectory but took 3212.30 s, slower than a CPU reference of 129.17 s at a different time point.
- 90.83% of all iterations were Phase I (searching for an initial feasible solution).
- Host algebraic preprocessing was 19.34 s total; porting only this would leave the main search load.
- These are development-log diagnostics, not a fair performance test or an independent five-condition conclusion.

## 2. 凍結参照 (Frozen reference)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03)

- Original CPU 3-stage HiGHS model, unchanged GEMs and bounds.
- GPU device-pivot 120-step regression passed; independent first seed passed. ⚠️ 食い違いあり — see Discrepancy Note 1.
- First independent seed CPU 115.005 s, GPU 2447.095 s. The current GPU solver is not faster. Moving only host setup cannot remove the many GPU pivots.

## 3. 独立1条件目の不合格と交換フラックスの非一意性 (First independent condition failure and exchange-flux non-uniqueness)

(出典: GPU_NEXT_SPEED_ARCHITECTURE_20260903.md, 2026-09-03)

- Update: the first independent condition was rejected (不合格): PHA 3.11865%, biomass 0.0247287 g/L.
- Exchange flux non-uniqueness was found; therefore, before the speedup work, the solution-selection rule in the state update must be validated.
- Even if the optimal basis, objective value and feasibility match, an arbitrary optimal solution does not necessarily reproduce the same long-term trajectory.
- Candidate-basis acceptance conditions therefore also need conditions for the adopted common solution-selection rule.
- Details: `GPU_OBJECTIVE_FIDELITY_REPORT_20260903.md`.

## 4. 優先順位の変更 (User-authorized change of order)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03 21:50 JST)

- Priority: beat CPU at matched accuracy, not complete five slow trials first.
- The supervised five-seed run was stopped by SIGTERM to the verified supervisor PID/start-time. Lifecycle record confirms requested signal 15 and worker exit -15 at 21:50:47 JST.
- Seed 20286201 passed; seed 20286202 is partial; the remaining seeds were not run.
- Keep all evidence. Do not count it as five passes.

## 5. 設計方針: 予測を答えにせず、検証付きの開始点にする (Design approach: verified starting points, not prediction-as-answer)

(出典: GPU_NEXT_SPEED_ARCHITECTURE_20260903.md, 2026-09-03 「優先案」; GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03 「Implementation sequence」)

Design-note approach:

1. Save basis and non-basic bound states from the current GEM / same 3-stage objective, including model hash, reaction IDs, stage, and preprocessing correspondence.
2. Select candidate bases on GPU and batch-solve the basis equations. Check not only original-problem feasibility but also reduced-cost sign conditions and objective value.
3. Adopt only candidates that satisfy numerical optimality for the original LP; closeness to predicted flux or dictionary distance alone is not an acceptance criterion.
4. If a candidate is unsuitable, send it to a GPU correction path for the same LP. Never convert numerical failure into culture death or negative reward.

This differs from the earlier "QP close to predicted flux" method: the prediction does not replace the final objective; acceptance is decided by the mechanistic LP side.

Implementation sequence (plan):

1. Compile candidate optimal bases offline. Apply exact coordinate scaling z_i = X_i v_i so that most constraint coefficients become fixed. This is an algebraic variable change, not a changed biological model.
2. GPU evaluation: reuse basis linear algebra, correct the few changing rows with a low-rank update, check full original primal residuals and numerical dual/KKT optimality. Reject unknown structure or invalid candidates.
3. Batch independent environments/queries on GPU. Report basis-building cost, GPU resident memory, transfer-inclusive timings and acceptance rate. A saved-state kernel speedup alone is not full dFBA speedup.
4. Connect the validated fast path with explicit GPU-only correction/rejection; no silent CPU LP substitution or accepting an inaccurate surrogate.
5. Short dynamic regression and CPU comparison, then competitive parallel CPU/batch GPU throughput. Only then run the larger final qualification.

- No claim of fully resident simulation until bounds, objectives, state updates and checks stay on device during the online loop. Initial offline CPU compilation must be disclosed and amortization reported.

## 6. 文献から採用する原理 (Literature principles)

(出典: GPU_NEXT_SPEED_ARCHITECTURE_20260903.md, 2026-09-03; GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03 also cites [Brunner & Chia 2020] and [Fan et al. 2023] as design motivation, not as measured speedups)

| Literature | Principle adopted | Additional verification needed here |
|---|---|---|
| [Brunner & Chia (2020)](https://arxiv.org/abs/2003.03638) | Reuse bases across successive dFBA states to reduce optimization | Handling of shared medium, 3-stage objective, bound switches in our model |
| [Fan et al., ICML (2023)](https://proceedings.mlr.press/v202/fan23d.html) | Predict starting bases with a neural model; linear algebra then verifies validity | Do not transfer accuracy to different GEMs; train/evaluate only for the current model |
| [Blin et al. (2026), preprint](https://arxiv.org/html/2601.21990v1) | Batch multiple LPs sharing a common constraint matrix as matrix–matrix products | Our dFBA matrices change with biomass; separate the shared constant part from the environment-dependent part |
| [Gleixner, Steffy & Wolter (2012)](https://desteffy.github.io/papers/lprefinement.pdf) | Treat floating-point LP errors including residual checks and correction | Current work is FP64 basis iterative refinement / feasibility recovery, NOT the paper's rational exact computation |

- Do not reuse literature speedup ratios for RTX 4060 GEM computation.
- A single-LP convergence problem is not solved just by batching.

## 7. バッチ化のための座標変換 (Coordinate transformation for batching)

(出典: GPU_NEXT_SPEED_ARCHITECTURE_20260903.md, 2026-09-03 「バッチ化に向けた行列の整理」; GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03 step 1)

Using the same B_i = max(1e-12, X_i) as the current code, set z_i = B_i v_i. This is a variable change in the linear problem; it does not change metabolism or medium conditions.

| Constraint / objective | After transformation | Shared across environments |
|---|---|---|
| Intracellular balance S_i v_i = 0 | S_i z_i = 0 | S_i fixed |
| Shared-medium exchange amounts | Exchange stoichiometric coefficients × z_i | Main exchange matrix fixed |
| Reaction bounds | B_i l_i ≤ z_i ≤ B_i u_i | Bound vectors change |
| Common growth lower bound g_i v_i ≥ μ | g_i z_i ≥ B_i μ | Biomass coefficient of the μ column changes |
| Live objective c_iᵀv_i | (c_i/B_i)ᵀ z_i | Objective coefficients change |
| Exchange absolute-value saving | Auxiliary variable ≥ ± exchange coefficient × z_i | Main constraint coefficients fixed |

- The common-growth column and stage-2 objective-keeping rows are still environment-dependent. Do not treat this as "exactly the same A can be shared across all environments"; separate fixed-matrix operations and a small number of variable terms into different kernels.
- At very small biomass, c_i/B_i becomes large and can be ill-conditioned; always check residuals after returning to the original variables. If the transformation is introduced, run equivalence tests first.

## 8. 第1段階の上限制約による証明 (Stage-1 upper-bound proof) — 未実装 (not implemented)

(出典: GPU_NEXT_SPEED_ARCHITECTURE_20260903.md, 2026-09-03)

- In the development warm-up log, 109 of 120 stage-1 instances reached the common growth upper bound 0.005 h⁻¹ within 1e-8.
- Stage 1 maximizes μ and has the known constraint μ ≤ 0.005. If a candidate satisfies all constraints at that time and μ reaches this upper bound, no further improvement is possible, so the stage-1 objective value can be certified.
- This is a special-case use of an optimality upper bound, not adoption of an approximation.
- The candidate must be re-checked against all GEM balances, bounds and shared-medium constraints in the current state; if unsuitable, fall back to the usual GPU LP solve.
- Candidate-creation success rate is unmeasured; 109/120 must NOT be converted into an LP-omission rate or speedup ratio.
- Stages 2 and 3 cannot be omitted this way because the upper-bound proof does not directly apply.
- If adopted, record "GPU proof solution" and "GPU iterative LP solve" with separate counters.

## 9. 実装済みコンポーネント (Implemented components)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03)

- `src/gpu_certified_basis.py`: offline HiGHS anchor solve, exact biomass coordinate change, sparse inverse, GPU Woodbury updates, two rounds of primal/dual iterative refinement, full original residual and numerical KKT checks. Original LP tolerances and biological objectives are not relaxed.
- `src/gpu_basis_bank.py`: multiple offline bases, candidate selection and acceptance masks on GPU. Invalid/missing values are NaN, never zero flux.
- `src/gpu_revised_basis.py`: experimental GPU bounded revised-simplex repair, including original-cost dual repair when dual feasible and a bound-distance feasibility phase otherwise. Final certificate always uses original cost. Fixed pivot budgets, no online CPU optimizer, fail closed when insufficient.
- `src/gpu_compiled_community_backend.py`: opt-in integration adapter; explicit optional GPU repair. CPU LP assembly, normalization, setup on cache misses, result transfer and simulator updates REMAIN HOST WORK. Do not call this whole-environment GPU residency.
- No default production solver was changed.

## 10. コンポーネント速度 (Component speed, not whole-dFBA speed)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03)

`results/pf_gpu_certified_basis_refined_comparison_20260903.json` contains the original three stages for 32 RHS perturbations per stage around saved step 16 (nonzero shared-pool RHS multiplied by 0.999–1.001; zero stays zero). Every one of these 96 GPU candidates was accepted. Three-stage totals below sum separate stage measurements, not a coupled dynamic trajectory.

| Comparator / scope | Time, s |
|---|---:|
| GPU resident, sum of five-repeat medians | 0.372279 |
| GPU host-adapter + transfers, sum of one measurement per stage | 0.424453 |
| CPU NumPy/SciPy, SAME batched basis/certificate algorithm, sum of medians | 1.459129 |
| CPU SciPy HiGHS, sequential cold solve per query, one measurement | 26.615284 |

- GPU vs the same CPU candidate algorithm: 3.92x for the resident component; this is not a benchmark against tuned parallel/warm-start HiGHS.
- The 71.49x cold-HiGHS/component ratio must NOT be advertised as training/dFBA speedup.
- Offline anchor compilation cost 20.489451 s; GEM/state/LP construction excluded.
- The saved step 17, 18, 113–116 queries were all rejected by the single-basis evaluator; this local acceptance rate does not establish broad coverage.

## 11. 長期認定前の短期統合 (Short integration before any long qualification)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03)

`results/pf_gpu_basis_bank_short_20260903.json`: offline train seed 20286311, 4 steps, 16 CPU LPs including fourth-policy anchors, 16 distinct bases. Offline training 5.971424 s; factorization/GPU setup 110.839284 s; GPU pool 1338.425 MiB. Test seed 20286312 was not supplied to the bank.

- The test stopped at the first step's aggregate LP (bank miss). It completed ZERO dynamic steps, so its 0.187980 s runtime is a rejection time, NOT a speedup against the CPU's 8-step 8.100660 s run. CPU LP calls online: zero.
- The next short probe enables GPU repair after a bank miss. This is a development seed once inspected, not an untouched final test.
- Do not enlarge the five-seed qualification until a complete short rollout is accurate and faster. Current obstacles are candidate coverage, repair iterations and remaining host simulation work, not insufficient validation duration.

## 12. 後続の短期プローブ (Subsequent short probes)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03)

- 24-pivot repair recovered the first test step's aggregate LP (3 pivots) and primary exchange LP (8 pivots), but the separate fourth LP remained invalid. 64 pivots did not fix it and amplified numerical error; neither run completed a dynamic step.
- Retained files: `pf_gpu_basis_bank_repair_short_20260903.json` and `pf_gpu_basis_bank_repair64_short_20260903.json`. No full speedup ratio.
- An additional opt-in route now reuses the primary exchange primal/dual/basis state, fixes nonzero reduced-cost variables, and solves the secondary objective. The resulting candidate must ALSO pass a reconstructed numerical KKT test for the original allowed-loss fourth LP; exact-face optimization alone is not sufficient. CPU reference/GEMs unchanged. Source: `src/gpu_optimal_face.py`. The reduced-cost-fixing principle is documented in [Gurobi's hierarchical LP reference](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/multiobjective.html); no Gurobi dependency or its performance numbers are used here.
- The cached-bank probe `pf_gpu_basis_face_short_20260903.json` completed step 1, then rejected step 2 aggregate LP. Its 5.427419 s cannot be compared as an 8-step speedup to CPU 8.842965 s. Online CPU LP calls: 0. All probes finished. Source snapshots are retained beside the last two probe JSONs.

## 13. 段階的アーキテクチャの文脈 (Restored tiered-architecture context)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03)

- The user's reminder of the earlier tiered design is correct. The prior `cooperative_gpu_service.py` path tries 128 candidates and retries failed rows with 2,048, with per-species block-composed candidates in the projector.
- `gpu_qp_parallel_training_smoke_retry2048.json` recorded 5 retries, all 5 recovered, and 512/512 feasible requests. These are historical-model results.
- The newer basis-bank experiments do NOT yet reconnect that neural/convex-hull front end. Preserve it as the inexpensive tier; assess a nested larger-hull tier and a full-original-variable GPU correction tier for rejected requests.
- Large dictionary candidate counts alone cannot prove LP objective fidelity or the same long-term exchange trajectory. Per-tier gates and escalation rates must distinguish physical feasibility, objective error and rollout error.

## 14. 実装継続: 修復されたウォーム状態と疎な向き (Implementation continuation: repaired warm states and sparse orientation)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03)

- The 24-pivot dual-priority probe completed one step and rejected step 2; all available aggregate anchors lacked a dual-feasible warm start there. No end-to-end speedup has been established by this probe.

Changes now under short testing:

1. Compute nonbasic reduced-cost sign feasibility separately from the primal residual / complementarity score. Prefer a dual-feasible repair start when one exists; do not change acceptance tolerances.
2. Cache explicit CSR transposes of the inverse and original matrix to avoid repeatedly using scatter-style transposed sparse multiplication. Apply the same orientation improvement to the CPU algebra comparator.
3. Reuse a previously certified GPU repair basis across time. Apply exact low-rank changes on its CURRENT columns and recertify the new LP; do not reuse old fluxes or silently assume unchanged medium/biomass.
4. Preserve original cache hashes. A caller may provide the hash-matched old compiler source; reuse is allowed only if ASTs of ALL module-level definitions/imports other than the online evaluator are unchanged.

Rejected alternatives / diagnostics:

- Eliminating explicitly fixed columns removed only 13/7373 variables on a representative LP; not an effective size reduction here.
- Direct CuPy stream capture rejected cuBLAS calls in installed CuPy 14.2.0. No environment package was replaced and no unsafe bypass applied. See `scripts/probe_basis_graph.py` and CuPy stream-capture documentation.

Long five-condition validation remains deferred until a complete short dynamic comparison passes accuracy and speed. Production CPU/GEM defaults and the historical QP frontend have not been replaced.

## 15. デバイス制御とマイクロバッチ実装 (Device control and microbatch implementation)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03)

- Added capture-safe FP64 operations through a private documented cuBLAS C API handle, pivoted GPU small solves, and GPU CSR products. A persistent cuBLAS workspace is required: automatic allocation nodes are not allowed inside CUDA conditional graph bodies. No installed package was patched.
- Added GPU conditional control. An initially unrolled IF graph consumed excessive VRAM; the 128-pivot integration completed one step then rejected step 2 after 257.47 s including failed work. It finished BEFORE an attempted stop signal. Its result is retained, and no speedup is claimed.
- Replaced the unrolled capture with one device WHILE body, one iteration counter and reusable arrays. This is GPU-side stopping, not polling each pivot from Python. Bound/primal/dual/gap acceptance remains unchanged.
- Replaced the dense full-matrix column gather with a CSC-column GPU kernel; the LP only needs the entering and leaving columns. Sharing an existing bank evaluator also avoids duplicated inverse buffers.
- Added `gpu_batched_compiled_backend.py` and a greenlet-based benchmark that runs actual independent environments to their next LP barrier and batches that stage. Host assembly, dispatch, transfers, four GPU LPs, and state updates are included in online wall time; bank/environment setup is separate. The baseline is explicitly sequential single-thread HiGHS environments, not a tuned parallel CPU implementation. One-step throughput is not a long-trajectory accuracy qualification.
- The first 8-environment wiring probe found a missing per-environment backend history bridge after stage 1. This was fixed with YieldingLPBackend; its 2.126 s failure runtime is NOT a speedup versus 7.441 s CPU.

Engineering references (these motivate the architecture, not measured speed/accuracy of this repository; all performance conclusions require the local run results):

- [NVIDIA CUDA conditional graph nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/)
- [CUDA Python graph/runtime API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html)
- [Brunner and Chia: basis reuse in community dFBA](https://arxiv.org/abs/2003.03638)

## 16. 2026-09-04: マッチドマイクロバッチ比較 (Matched microbatch comparison)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-04)

- `pf_gpu_microbatch2_step1_fixed_20260903.json` completed two independent environments, one step each, without online CPU LP calls.
- Cold CPU/GPU times were 1.9050/3.0515 s (GPU slower). The one prepared-cache repeat took 2.4153/0.9472 s (2.550x); all endpoint gates passed.
- This is a limited actual dFBA result, not a statistical or long-horizon claim.
- Earlier CPU thread limits applied to OpenMP/BLAS, not explicit HiGHS options.

Next checks, in order:

1. Eight environments, three repetitions including cold setup. Explicitly set HiGHS threads=1 and parallel=False; compare four concurrent CPU LP workers. Only SciPy LP arrays enter threads, never COBRA/environment objects.
2. Use independently deep-copied/reset pristine environments for faster untimed setup (clone probe matched a full CPU step exactly); preserve model fingerprints and offer fresh construction as an option.
3. Extend to multiple steps; investigate stage-1 feasible basis reuse in stage 2 if cold anchors cannot repair a changed LP. Do not relax gates.
4. Only after complete short trajectories pass speed and accuracy, widen held-out conditions and reconnect the inexpensive surrogate/QP tier.

LP assembly, dispatch, transfers, graph compilation during an online call, GPU correction and state integration remain in the timer. Offline bank construction/loading and environment cloning are separately disclosed.

## 17. 8環境の結果と改訂 (Eight-environment results and revisions)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-03/09-04 continuation)

- Four concurrent CPU LP workers finished one step in 3.430 s. The initial 32-pivot GPU run rejected one environment at the exchange LP (residual 1.93248e-5 > 1e-5); no speedup is reported for that failed run.
- Raising the GPU budget to 64 completed all eight environments and all endpoint gates, but prepared GPU times 4.575/4.516 s were slower than four-worker CPU 3.478/3.382 s. Sequential-CPU gains alone do not establish an advantage over an appropriate parallel CPU baseline.
- Implemented small initial budgets (maxmin 8, later LPs 32) with larger GPU correction only for rejected rows; recertify the full original LP. Added a zero-pivot fourth-stage certificate before additional work and increased bounded graph cache capacity to avoid four signatures thrashing a two-entry cache. The prepared GPU repeat decreased to 3.828 s versus CPU 3.505 s, still not a win.
- Added device-counter-sized elimination of the EXACT identity-padded unused update block. This is not truncation or lower precision: all actual basis updates remain, and full original numerical gates are unchanged. Added tests for identity padding with nonzero padded RHS and continuation from a rejected short-budget state. 25 targeted tests pass.
- Next: measure two complete steps across eight environments; diagnose cross-stage warm starts if a changed LP fails. Do not launch long training or present failures as speedups.

## 18. 継続結果: クロスステージウォームスタートと独立カバレッジ (Continuation outcome: cross-stage warm start and independent coverage)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-04)

- Eight environments/two steps still reject step-2 aggregate LPs. A stage-1 feasible warm basis removes primal infeasibility (~1e-12 residual), but six rows remain nonoptimal after 64 pivots. Do not count partial work as a win.
- Sixteen environments/one step completes with all endpoint gates passing, but prepared GPU 7.552 s is slower than four-worker CPU 6.773 s (0.897x).
- Added reproducible offline aggregate-bank extension: independent training seeds 20286501–20286504, two steps each, eight new bases. Training/assembly 18.63 s plus factorization 52.15 s (70.78 s total), 24 offline CPU LP calls. Loader verifies completion, model fingerprints, compiler AST/hash, artifact hashes and training/evaluation seed separation. Nothing is fit to test LPs.
- Expanded aggregate bank (4→12) reduces step-2 aggregate rejections from 6/8 to 2/8. No environment completes the second step because the barrier stops when any LP is rejected. Full-run speedup remains unestablished.
- 26 targeted tests pass, including cross-objective warm-state recertification, rejected-state continuation, active identity padding, and parallel CPU LPs.

Next development priorities: diagnose the two remaining stage-2 rejections; expand independent training coverage together with inexpensive basis routing (not an ever-growing exhaustive bank scan); then validate the later exchange LPs and complete short rollouts. Keep the old neural/QP front-end connection and fully device-resident environment as separate uncompleted integration tasks. Do not restart long validation/training yet. All current probes ended.

## 19. AMN inspired continuation (2026-09-04)

(出典: GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md, 2026-09-04)

- See `AMN_GUIDED_ACCELERATION_PLAN_20260904.md` for the paper/code review, adaptation limits, complete measurements and next priorities.
- Implemented learned starting-basis proposals followed by full original-LP certification, GPU repair and exhaustive fallback only when needed.
- Actual independent training: 32 seeds, two steps, 192 CPU LPs; offline total 121.81 s.
- With eight environments/one step, two prepared neural repeats passed all gates and exceeded four-worker CPU by 1.127/1.122x (GPU 3.158/3.231 s).
- A constant-basis repair-first control also won by 1.071x: not all of the gain is due to learning.
- Neural initialization reduced aggregate pivots 55→15 on the matched one-step inputs.
- Initial GPU setup still makes the cold run slower.
- Two-step validation continues to reject two stage-2 LPs; no claim of long-horizon or whole-training speedup.
- 29 targeted tests passed; all jobs ended; production defaults unchanged.

## 20. 当初の評価順序 (Original evaluation order — superseded)

(出典: GPU_NEXT_SPEED_ARCHITECTURE_20260903.md, 2026-09-03)

The design note explicitly states that this order is 過去の経緯 (past history) and no longer the current work order, because the user-authorized speed-first plan (Section 4) takes precedence. It is retained here as the validation/regression discipline to follow once the speed-up path is matured:

1. Resolve exchange-flux solution selection; after regression tests, confirm overall accuracy on new independent five conditions.
2. Compare basis candidate selection with feasibility/reduced-cost checks on same-state LPs.
3. Add basis cache, CUDA Graph, and batching one at a time, with accuracy regression after each change.
4. Compare environment counts 1/8/32/128 within real-device memory. Metrics: environment-steps per second, total completion time across all environments, repair rate, maximum error, memory peak.
5. CPU side: same problem, same accuracy; compare not only the sequential version but also parallel and basis-reuse versions. Do not compute GPU ratios only against a slow CPU implementation.
6. The five conditions used for final evaluation become regression tests thereafter; improved versions use unused conditions.

- Do not make 100% GPU utilization or VRAM occupancy the goal. Resource use is considered improved only when accuracy, success rate and elapsed time improve.

## 食い違いに関する注記 (Discrepancy notes)

1. **"Independent first seed passed" vs "first independent condition failed".**
   - `GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md` (Frozen reference): "independent first seed passed."
   - `GPU_NEXT_SPEED_ARCHITECTURE_20260903.md` (更新): first independent condition was rejected — PHA 3.11865%, biomass 0.0247287 g/L.
   - Both files are dated 2026-09-03; the design note's update appears to be the later record. The statements may refer to different acceptance levels (e.g., solver completion vs biological output tolerance), but the source files do not explicitly reconcile them. This integrated document retains both statements verbatim; the failure is the one that motivated the exchange-flux solution-selection requirement (Section 3).

2. **GPU exact-form solver runtime and CPU reference differ between the files.**
   - Plan: first independent seed CPU 115.005 s, GPU 2447.095 s.
   - Design note: development 24 h trajectory GPU 3212.30 s vs CPU reference 129.17 s (at a different time point); 90.83% Phase I; host preprocessing 19.34 s.
   - These are different measurements (independent seed vs development trajectory), not the same benchmark. Both are retained; neither is a tuned/fair performance result, and neither supports a GPU speedup claim.

3. **Work order conflict.**
   - The design note's evaluation order (Section 20) begins with resolving solution selection and confirming new five-condition accuracy before acceleration work.
   - The plan (Section 4), issued later the same day with user authorization, reverses this: acceleration takes priority and the five-seed completion is deferred.
   - The design note itself marks the old order as past history, so this is a historical ordering difference rather than an unresolved contradiction.

## 制限・未達事項 (Limitations and unachieved status — consolidated)

- **End-to-end dFBA speedup: NOT established in any probe.** All batch/GPU-vs-CPU comparisons that completed are either slower, or faster only in prepared-cache single-step gates without a completed full trajectory.
- **GPU residency: NOT achieved.** Host LP assembly, normalization, setup on cache misses, result transfer and simulator updates remain host work in the integration adapter (`gpu_compiled_community_backend.py`).
- **Two-step validation:** still rejects two stage-2 aggregate LPs (as of the AMN continuation).
- **Cold start:** initial GPU setup makes cold runs slower than CPU in the matched microbatch comparison (cold CPU/GPU 1.9050/3.0515 s).
- **Complete short rollout:** not yet accurate and faster; long five-condition validation and long training remain deferred.
- **Stage-1 upper-bound proof:** 未実装 (not implemented); candidate success rate unmeasured.
- **Neural/QP surrogate front-end:** NOT yet reconnected to the basis-bank path; preserved as the historical inexpensive tier.
- **Fully device-resident environment:** separate, uncompleted integration task.
- **Article speedups are not reused as measurements** of this implementation.
- **One-step throughput is not a long-trajectory accuracy qualification.**
- **Overall status:** 設計・実装中 (design/implementation in progress). Frozen regression sources, production CPU/GEM defaults and the historical QP frontend remain unchanged.

## 構成元ファイル

- `docs/GPU_CPU_SPEED_PRIORITY_PLAN_20260903.md` — 2026-09-03 (with continuation sections dated 2026-09-04)
- `docs/GPU_NEXT_SPEED_ARCHITECTURE_20260903.md` — 2026-09-03
