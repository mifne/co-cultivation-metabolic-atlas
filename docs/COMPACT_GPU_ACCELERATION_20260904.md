# GPU acceleration implementation plan — 2026-09-04

## Acceptance criteria

- Compare identical current GEMs, initial conditions, action sequences, environment count and completed steps.
- CPU reference: unchanged three-stage HiGHS, 4 concurrent LP workers, one thread per LP.
- Online GPU path must not call CPU optimization. Host environment assembly remains measured; this is not yet a fully device-resident simulator.
- Original-LP feasibility <= 1e-5, dual violation <= 1e-7, relative KKT gap <= 1e-7. Endpoint PHA relative error <= 1%, biomass absolute error <= 0.01 g/L, PHV fraction absolute error <= 0.01. Do not loosen to obtain speed.
- Report speedup only for completed matched work passing every gate. Separate cold setup, offline preparation and repeated execution. A short result is not a 120-step result.

## Revision 1: compact exact basis maps

Evidence: the previous 8-environment attempt completed only one of two steps; exchange optimization at step two rejected 7/8 environments. CPU 7.206 s for two steps and GPU 17.689 s for incomplete work are NOT comparable speedups. Full inverse/repair caches approached the 8 GB VRAM capacity.

1. Fix and validate offline compact artifact creation. Project sparse-LU inverse action onto actual RHS, bound and cost directions instead of storing a full inverse. Keep runtime input-family guards and full original-LP certificates.
2. Add certificate and artifact-integrity tests, including secondary-objective allowed-loss rejection.
3. Build compact banks from the declared independent four 120-step CPU training trajectories and existing training bases. Training and benchmark seeds remain separate.
4. Measure matched 2-step / 8-environment execution; classify time into assembly, routing, numerical evaluation and transfer. Revise the plan using measured failures, not utilization alone.
5. If valid, extend to 8 then 120 steps and repeated independent evaluations. If misses persist, improve coverage/certified GPU repair without adding evaluation LP solutions to training.
6. Reduce host synchronization and use graph replay where profiling supports it. Move additional state operations only after correctness and matched throughput are established.

Current status: experimental compact code exists; builder metadata collision fixed. No compact-bank speedup established. No model chemistry, control timestep or production solver defaults changed.

## Revision 2: compact screening followed by small certified GPU repair

The completed bank has 44 maxmin, 52 aggregate and 44 exchange bases (327,733,060 projected bytes; 110.69 s build). Four compact-map/certificate tests pass. Independent 8x2 test v2 rejects 6/8 rows at maxmin before completing step one; no speed ratio is reported. Input-family guards pass, so the primary issue is primal infeasibility of the proposed active sets, not a missing parameter direction. Example first-anchor residuals range from 1.9e-5 to 0.94 in rejected rows. Basis expansion alone does not ensure feasibility between training samples.

- Keep compact screening and full numerical certificates.
- Restore the existing full-variable GPU correction only for misses, selecting the most nearly feasible, dual-feasible compact candidate.
- Precompute repair linear operators OFFLINE from the same declared training bases, without solving any validation input on CPU or fitting validation solutions.
- Load repair operators on demand under the existing bounded GPU LRU cache. Measure cold load/capture and warm execution separately.
- Reuse the existing original-LP optimal-face certificate and repair for the fourth tie-break stage.

## Revision 3: rank by number of violated constraints; reuse candidate evaluation

23 regression tests passed before capture work. Offline repair-operator build completed in 265.66 s with zero new CPU LP solves. At 48 pivots, 8x2 completes step one but rejects 4/8 at step-two aggregate; at 128 pivots only one row remains rejected. Neither run is a speedup (CPU ~7.1–7.4 s for complete work, GPU ~16.3 s for incomplete work).

Diagnosis on the recorded rejected input (development diagnosis, NOT added to training): maximum-residual ranking chose aggregate basis 4, with 34 violated constraints. Basis 11 has a similar maximum residual (0.754 vs 0.740) but only five violated constraints. Residual magnitude alone is a poor repair-cost proxy.

- Add an optional dual-feasibility-first, violation-count-then-L1 ranking. This changes proposals only, not acceptance tolerances or the LP.
- Evaluate each candidate once for the entire environment batch and reuse its result for routing, avoiding repeated small evaluations of the same candidate.
- Add fixed-shape CUDA graph replay for compact maps with bounded graph caches and shared capture-safe math per bank. Record graph memory and compilation time.
- Six compact tests pass, including changed-input graph replay and online CPU factorization/optimizer prohibition.

## Revision 4: collect a GPU-generated training prefix

The count/capture 8x2 run passes all rows through step-two aggregate. Step-two exchange still rejects 6/8 at 128 pivots. Compact exchange screening now takes 0.228 s in step two, but exchange repair costs ~8.07 s; the bottleneck is no longer simply dispatch. No matched speedup is reported. Graph pools total ~1.87 GB, and compilation totals ~2.03 s.

The offline full-horizon samples were generated by the three-stage CPU reference. The GPU uses an additional optimal-face tie-break, which can select different exchange fluxes among alternate optima and change the next input state. This is a concrete potential distribution mismatch; it is not yet proven to explain every miss.

- Generate four NEW independent training trajectories with a GPU first-step prefix; use the same fourth-objective policy in subsequent offline CPU teacher steps.
- Explicitly count and label offline CPU fallback/teacher work. Do not enable CPU fallback in the benchmark or production GPU path.
- Keep development benchmark seeds excluded. Record parent-bank training seeds in derived artifacts.
- Rebuild a small transition-focused compact bank and test the same two-step development case, then use fresh independent seeds for confirmation.

## Revision 5: stabilize and compress the actual GPU pivot updates

The new four-prefix trajectories complete in 26.63 s; first-step GPU-only solves succeed in every stage, and 48 subsequent CPU teacher LPs are explicitly offline. The rebuilt short bank improves step-two aggregate to ~2.70 s but still rejects four exchange rows at 128 pivots. Thus prefix coverage alone is insufficient. No speedup has been achieved yet.

An isolated failed exchange input showed enormous numerical error after a long update sequence. Add optional original-matrix iterative refinement during primal, dual and pivot-direction computation, not just at termination. Do not relax acceptance gates.

The prior Woodbury representation appended one column per pivot, including repeated changes to the same basis position. Combine those updates by position (the column differences telescope). Replace products with unit-vector V rows by gathers/scatters. Store only the few general V rows; unit rows are represented by indices. Trim retained warm U arrays to actual used rank, with an explicitly measured host size read. Changing the warm matrix is deliberately unsupported in this optional mode; current benchmark stage continuation keeps that matrix fixed.

On the isolated recorded failure (not an independent benchmark), 384-budget compact updates plus per-pivot refinement pass the original LP certificate in 156 pivots / 123 distinct update positions, ~6.39 s. Without refinement, this case remains rejected. This is correctness evidence, NOT an end-to-end speedup. Warm-continuation/capture tests and existing solver tests pass (23 tests in the selected subset). Next: verify complete paired trajectories, then increase batch size to test amortization of GPU repair and operator setup.

## Revision 6: coalesced small solve, short portfolio, memory-bounded continuation

The 384-pivot complete-trajectory attempt still rejects one exchange row; GPU 67.54 s for incomplete work vs CPU 7.15 s for two steps is not a speedup. A different starting basis solves that recorded hard row in 39 pivots (~0.38 s isolated), while the original basis does not pass at 384 pivots.

- Rewrite the small dense solve's elimination kernel to cache the pivot row/factors in shared memory and distribute contiguous elements across threads. Avoid processing identity-padded inactive columns. Keep FP64, pivoting, and original acceptance tolerances. The same isolated 156-pivot solution and residuals are obtained in 2.40 s instead of 6.39 s (single observations, not a confidence interval).
- Try two alternate bases briefly and continue the candidate with the best measured residual progress. This is a proposal heuristic, not an approximation accepted without certification.
- The first portfolio attempt reached ~7949 MiB and was explicitly terminated; its report is marked `terminated_gpu_memory_pressure` and has no GPU completion or speedup claim.
- Detach rejected warm states from large evicted GPU operators. Rebind them only after matching the stage, candidate and immutable operator provenance. Keep only improving candidate states.
- Cache reusable host operator assembly (up to 32 entries), independently of the small GPU LRU. Count preparation time and cache hits in benchmark reports.
- Add complete core solver/math/capture source snapshots to subsequent benchmarks. Earlier prototypes with incomplete core snapshots are development evidence only.
- Current selected regression suite: 30 tests pass. Next measurement uses unchanged 8x2 conditions; no CPU speedup established yet.

## Revision 7: routing correctness and matching the original three-stage problem

- Found and fixed a routing bug: adding 1e20 to deprioritize non-dual-feasible candidates erased their relative floating-point scores. Use a two-pass ordinal/score ordering instead. A regression test checks ordering inside each feasibility class.
- A 32-pivot probe stopped the useful alternative before it converged at 39 pivots. Add a configurable 64-pivot alternate probe and retain the initial candidate for continuation when alternatives do not pass.
- With this change, all eight environments pass the original three FBA stages through step two. The extra fourth tie-break still rejects six rows on complementarity gaps around 1.2e-7–3.8e-7. These were not reported as completed speedups.
- Tightening the auxiliary primary-objective allowance from 1e-10 to zero does NOT resolve the problem: diagnostics show the complementarity term, not allowance, dominates. Keep this negative result.
- Add `original3` comparison mode: use exactly the CPU reference's three objectives, omit the experimental fourth selection, and STILL enforce all original LP and endpoint accuracy gates. This is not permission to ignore accuracy failures. The original CPU formulation and GEMs remain unchanged.
- Remove unused GPU copies of the active inverse from repair input adapters; cache host transposes as well as expanded operators. The selected suite reached 31 passing tests before the original3 test.

The 8x2 original-three-stage comparison completed both repetitions. CPU: 7.093 / 7.233 s; GPU: 31.604 / 23.128 s; CPU/GPU ratios: 0.224 / 0.313. Both pass the endpoint gates and have zero online CPU LP calls. This is a valid comparison, but GPU is still slower.

## Revision 8: bound memory before increasing environment count

The first 64x2 comparison completed CPU in 55.418 s, but GPU exceeded ten minutes and occupied 7939 MiB of the 8188 MiB device. The worker was terminated and its report explicitly marked incomplete due to memory pressure; there is no valid GPU timing or speedup.

- Maxmin-to-aggregate warm states still retained their large GPU solver owners, defeating LRU eviction. Detach these states and recover them only under matching basis/operator provenance, as already done for portfolio continuation. Preserve the actual basis index instead of a placeholder zero.
- Bound repair groups independently of the number of environments. Screening can remain batched, while expensive correction uses memory-bounded chunks. Keep original LP certificates and endpoint gates unchanged.
- Print stage completion/timing so a stalled phase is identifiable. Measure 16 environments before retrying 64; do not infer throughput from GPU utilization or incomplete work.

No CPU speedup established. Host dFBA state/assembly is still not fully GPU-resident.

## Revision 9: reuse the small-system factorization within each GPU pivot

Memory-bounded 16x2 repeat zero completes with all endpoint gates, zero online CPU LP calls, CPU 14.271 s and GPU 54.605 s (ratio 0.261). The second-step exchange repair alone takes 31.068 s. Thus memory retention is not the only bottleneck; more parallel environments have not yet produced a speedup.

Each revised-basis iteration applies the same Woodbury small matrix to multiple primal, dual and refinement right-hand sides. Previously each application performed a fresh Gauss-Jordan solve. Add optional, capture-safe FP64 partial-pivoted LU computed once per iteration, followed by forward/transpose triangular solves. Recompute after every basis update and at final certification; do not reuse a stale factor across changed matrices.

Verify forced row-pivoting, transpose permutations, exact identity padding, changed-input graph replay, original-LP tests, then the difficult recorded repair and a complete paired benchmark. Do not change tolerances or accept an invalid factor/flux.

38 selected tests pass. The recorded difficult exchange repair passes in 1.581 s (153 pivots) vs prior 2.397 s (156 pivots); small numerical differences can change the degenerate pivot path. This is a single diagnostic observation, not an independent throughput claim. The first 16x2 paired repetition passes in GPU 50.246 s vs CPU 14.642 s; still no CPU speedup.

## Revision 10: assess repair-cost routing, not only dual feasibility

On a recorded exchange miss the dual-feasible-first route chooses a candidate with about 326 violated constraints even though another candidate has about six. The latter requires a primal feasibility/optimization phase but may be substantially cheaper overall. Add an explicitly optional exchange-only `count_only` routing ablation. This changes proposal order only; all original KKT, bounds, matrix and endpoint gates still apply. Compare against unchanged dual-first routing and retain negative results.

Record per-repair-group elapsed time and basis index. Freeze retained GPU graph outputs even when an entire chunk passes, so another chunk cannot overwrite an earlier environment's warm state. Chunk direct maxmin warm states as well, preventing an uncapped aggregate fallback.

The recorded row passes from the low-primal-violation candidate in 52 pivots / 0.599 s, compared with 153 pivots / 1.581 s from the dual-first candidate (single diagnostic observations). A complete independent-from-training 8x2 development comparison is required; these proposals have been inspected on development inputs, so subsequent confirmation must use fresh seeds.

## Revision 11: skip unused dual work entirely on the device

Primal/merit-phase iterations were also computing the full dual pivot row even though it was not used. Add an opt-in nested GPU IF inside the existing GPU WHILE. Only execute dual inverse/refinement work when at least one active environment in the repair group needs dual simplex. Initialize all shared output buffers outside the IF, retain identical final certification, and compare eager/replayed pivot sequences.

This control-flow mechanism is supported by [NVIDIA's conditional-node documentation](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/). It avoids a CPU decision/readback; it does not make host dFBA state GPU-resident. Benchmark independently of the routing change where practical.

39 regression tests pass. The same recorded candidate-13 repair takes 0.438 s instead of 0.599 s, with the same 52 pivots and final residuals. The `count_only` full 8x2 comparison FAILS three exchange rows and is not a speedup. Keep dual-first initial candidates; use primal-count ordering only for a short alternative probe, followed by continuation of the initial candidate. The next 64x2 test caps repair groups at four and GPU operator cache at two. Its cold setup/solve work remains charged to GPU timing as before.

## Revision 12: audit structural reduction; make iterative refinement conditional

Read-only audit of all 480 cached full-horizon training inputs finds only 11 variables fixed at zero by bounds throughout the observations. Homogeneous singleton equalities force a further 429 variables to zero and could remove 510 equality rows. This reduces exchange variables from 7373 to 6933, only about 6%; it is not evidence that trivial zero elimination will produce the required several-fold speedup. No GEM reactions or runtime LPs were removed. Any future reduction must guard observed fixed bounds and lift/certify the original LP.

Add an opt-in GPU-only trigger for per-pivot iterative-refinement solves: first compute the original-matrix residual, and skip the correction solve only when its internal residual is below the specified arithmetic threshold. Keep the final two refinement passes and all acceptance thresholds unchanged. Test nested IF/WHILE replay, changed inputs, infeasible rejection, and difficult recorded LPs before using this option in an end-to-end comparison.

Current best matched completed 16x2 second repetition: CPU 14.588 s, GPU 39.464 s, ratio 0.370 (GPU still 2.71x slower). No whole-RL or 120-step speedup established.

## Revision 13: cap private graph pools and reuse prior certified GPU bases

The second 64x2 attempt passes every row through five LP stages, but again reaches 7947 MiB at the second exchange stage. It was terminated without a completed GPU time or speed ratio. The two-operator LRU alone does not bound the four private graph pools per operator. Add an explicit per-operator graph-cache size; subsequent benchmarks use one.

Implement exact temporal matrix updates for compact Woodbury states. Replace the physical matrix-change rows and correct each changed-basis-position column by the difference between the new and old matrix effects. Do not append more update slots merely because time advanced. Changed RHS and bounds are solved anew; final original-matrix KKT checks still gate acceptance.

Retain only certified GPU basis states as proposals for the next dFBA step. Detach large operator owners; rebind only to the same immutable operator provenance. Freeze replay outputs. Try a bounded temporal GPU correction on dictionary misses, then use existing dictionary/portfolio correction if it fails. Clear temporal solutions before each new benchmark repetition, avoiding validation-state leakage across repeats. No offline teacher or online CPU LP is added.

43 selected tests pass, including changed-matrix temporal replay after pivots, backend temporal acceptance, mismatched provenance rejection, and infeasible fail-closed behavior. Adaptive refinement also passes both recorded difficult candidates; these are development diagnostics only. Next: complete 8x2 temporal comparison, then fresh-seed scaling/longer horizons if it is faster and accurate.

The temporal 8x2 comparison completes and passes both repetitions: CPU 7.198 / 7.276 s, GPU 27.878 / 20.470 s (ratios 0.258 / 0.355). The second exchange stage still takes 11.157 s in the repeated run. This is progress against the former 23.128 s GPU 8x2 repetition, but NOT a CPU speedup. No benchmark worker remains running after this comparison.

## Revision 14: user-proposed adaptive reduced-space optimization (design, not implemented)

The user asks whether low-dimensional data can be retained and only necessary high-dimensional directions expanded. Distinguish the existing compact parametric dictionary maps from a genuinely small online optimization problem: misses still enter a full-variable revised-basis solve, which dominates current timings.

Next architectural prototype should represent fluxes as v = v_ref + D z with a small, adaptively enlarged set of directions. Construct v_ref and D to preserve the structural stoichiometric equality constraints (or explicitly correct them), retain medium/bounds/coexistence and all three original objectives, and optimize z on GPU. Full original-LP residuals and dual optimality checks must determine acceptance and identify missing directions; no PCA explained-variance threshold can replace the biochemical/LP or trajectory gates.

- Start with small candidate dimensions (e.g. 32/64/128 as experimental sweep values, not established sufficient ranks).
- Include extracellular exchange, biomass and PHA-sensitive directions, not only high-variance flux components. Use previous certified GPU trajectories and independent offline training for proposals, without fitting validation solutions.
- Expand only the failing environments and missing directions, using a restricted-master/column-generation or equivalent certified enrichment strategy. Reconstruct full vectors for sparse original-LP checks, not a full dense inverse for every environment.
- Retain the existing full-variable GPU solver as a final safety tier during development, then measure and reduce its incidence. Never label this as fully GPU-resident dFBA while host state/assembly remains.
- Measure reduced-space acceptance, rank reached, enrichment/repair rates, original primal/dual/gap residuals, trajectory endpoint errors, total wall time and memory on fresh seeds; progress from two-step transition tests to longer/120-step verification.

Relevant foundations: [Fan et al., ICML 2023, Smart Initial Basis Selection for Linear Programs](https://proceedings.mlr.press/v202/fan23d.html), including learned construction of restricted master problems; [MIT column-generation treatment](https://web.mit.edu/15.053/www/AMP-Chapter-12.pdf); [Faure et al., Nature Communications 2023](https://www.nature.com/articles/s41467-023-40380-0), supporting neural-mechanistic integration but not proving this proposed reduced GPU solver's speed or accuracy.

## Revision 15: restricted-column tableau implementation

Implement an M-by-K reduced tableau: choose K nonbasic directions using full-space feasibility/optimality pricing, algebraically eliminate the M basic variables, and pivot only the reduced tableau on GPU. Unlike full revised-basis correction, sparse inverse applications occur at projection and final lifting, not at every pivot. Canonicalize the final basis using at most K replacements and apply the unchanged full original-LP certificate. Tiny tests verify full-span CPU agreement, changed RHS/bounds/matrix/objectives, captured replay, rejection of infeasible problems and rejection of a restricted optimum that is not globally optimal.

The first recorded hard exchange probe from candidate 2 rejects K=16/32/64/128 (2/4/7/53 restricted pivots, 0.11–0.28 s per cold call). These are invalid partial solutions, not speedups. Investigate direction selection and numerical stability; compare a lower-primal-violation starting basis. Do not integrate an unverified reduced optimum into dFBA.

## Revision 16: residual-priced enrichment and integration

The lower-primal-violation candidate reaches primal feasibility at K=32, but not full-space dual optimality. Simply enlarging the original direction ranking still fails at K=256. Add pricing of excluded directions at the lifted restricted solution: use the original objective when primal-feasible, and bound-distance Phase I otherwise. Rejected rows alone receive additional directions.

With adaptive pricing, the recorded candidate-13 case passes the full original LP at K=256 (41 reduced pivots, ~0.437 s for that tier; earlier tiers also consume time). This is a diagnostic case, not a speedup ratio. Candidate 2 remains rejected through K=256, confirming that the initial basis still matters.

Integrate reduced attempts before full-variable repair. Optionally attempt a short full-GPU polish only after a reduced solution passes original primal feasibility. Log reduced-only acceptance separately from polish and larger fallback; never present the combined route as reduced-only. Preserve every original numerical/endpoint gate and source snapshot. 47 regression tests pass, including a two-environment backend test that expands only the rejected environment and prohibits online CPU optimization.

Next comparison: 8 environments x 2 steps, K=32/64/128, at most 512 cheap reduced pivots, 8-pivot GPU polish, original full-GPU repair safety tier. Both cold and repeated end-to-end timings include all failed proposal/correction work.

## Revision 17: continue reduced bases instead of restarting enrichment

The first integrated 8x2 run passes accuracy but is slower than CPU: CPU 7.198 / 7.105 s, GPU 33.784 / 101.519 s. The repeated exchange stage slows severely. Fix missing destruction of private cuBLAS handles and break retired restricted-solver cycles; the leak is real, but is not yet proven to explain the entire slowdown. Replace full sparse-inverse-times-zero-padded-column projection with direct sparse selected-column products, and replace K host-generated scatter operations with one GPU kernel.

Add a diagnostic: evaluate reduced-cost sign violations of the restricted nonbasic set after lifting into the original LP. On a difficult recorded case the restricted violation is ~2e-13 while the full LP dual violation is ~221: the main issue is missing directions, not a falsely optimal reduced tableau. Enlarging a fresh initial pool to 1024 still fails that case.

Implement continued column generation. Keep the latest GPU basis, reprice a small new nonbasic pool, and reconstruct only its M-by-K tableau. Preserve prior basis replacements in a compact Woodbury state. The tableau width remains K; its full-LP lifting rank may grow independently. Rejected rows alone enter another round. Same-LP warm-state guards reject changed RHS/matrix/scales. No CPU optimization or validation-solution fitting is added.

Recorded development probes at K=64 now pass the full certificate: row 2 / candidate 1 in 12 rounds, and row 3 / candidate 13 in 2 rounds (~0.272 s total). These isolated diagnostics are not an end-to-end CPU speedup. Next: regression tests and complete matched 8x2 timings including all rounds, graph capture, host dispatch and any full-GPU safety repair.

## Revision 18: release retired graph-pool temporaries explicitly

Continued-column-generation 8x2 passes all LP and endpoint gates in both repetitions, with zero online CPU LP and **no full-variable correction calls**. Nevertheless, CPU 7.177 / 7.465 s vs GPU 35.649 / 73.721 s remains slower. Memory again approaches 8 GB.

A controlled memory audit finds that Python garbage collection does not resolve the growth; CUDA graph-allocation-pool reserved bytes are zero. Retired private CuPy pools can remain referenced by externally retained output allocations. Destroying a graph alone does not explicitly discard its cached free temporary blocks. Make close terminal: check CUDA destruction status, drop replay-owned input/result/solver references, and free only unused blocks of its private pool. External result allocations remain valid. Never free buffers of a live executable graph.

On the identical 12-round recorded probe, final device-used memory drops from 2.699 GB to 1.719 GB after this change. This is a diagnostic single-LP observation, not a peak-memory or speed claim for complete dFBA. Add an explicit regression test retaining an external result after graph close; repeat the complete paired benchmark next.

The memory-fixed 8x2 benchmark passes both repetitions: CPU 6.993 / 8.781 s, GPU 21.685 / 17.168 s. The severe repeated-run slowdown is removed in this comparison, but GPU remains slower than CPU. 53 tests pass, including explicit private-pool release with external outputs retained.

## Revision 19: distinguish warm-state routing from reusable graph shapes

An optional early stage-1-to-aggregate reduced warm-start path passes a changed-objective/tightened-bound regression test. However, the full 8x2 development comparison becomes slower (GPU 48.706 s vs CPU 7.044 s). Do NOT adopt it by default or use it in the next scaling test. Preserve its opt-in flag and negative result as an ablation. The selected test suite has 54 passes.

Run 64x2 with the memory-fixed continued solver, stage-1 shortcut disabled and at most 8 environments per correction group. Do not infer scaling until the whole trajectory completes and passes.

In parallel with that already-snapshotted benchmark, implement an opt-in rank-bucket prototype for the next run. Pad compact update arrays to bounded power-of-two capacities so nearby ranks reuse a CUDA graph. All inactive update slots are exactly zero; only the resulting exact identity padding may be skipped by the dense solver. A capacity overflow fails closed. Also avoid repeating the initial full-space reconstruction when the previous lifted state has identical RHS, matrix, bounds, scales and cost. Check all these guards and keep the final original-LP certificate unchanged. This prototype is not yet performance evidence.

The 64x2 / correction-batch-8 trial again approached 7900 MiB at second-step exchange and was terminated; its report is explicitly incomplete, with CPU 55.849 s but no completed GPU time or speed ratio. Do not present it as scaling evidence.

The rank-bucket implementation passes 55 tests, including changed-input fail-closed guards, fixed-width continued solves, graph reuse and external result lifetime. Skip unused Phase-I pricing inside a GPU conditional when all rows are primal-feasible. The recorded hard row passes in the same 12 rounds with substantially cheaper repeated graph calls (about 0.74 s across tiers; still not an end-to-end comparison).

The bucketed 8x2 comparison completes and passes: CPU 7.064 / 9.333 s, GPU 15.735 / 10.862 s; ratios 0.449 / 0.859. Even the repeated run remains slower than its paired CPU reference. Investigate proposal coverage with the EXISTING larger bank trained on seeds 20287101–20287104 and earlier training seeds, without appending any validation solutions. Its 44/52/44 maxmin/aggregate/exchange entries and model hashes are verified before a 16x2 comparison. Keep stage-1 warm shortcut disabled and correction groups capped at four.

The larger-bank 16x2 test completes and passes both repetitions, but does not establish a speedup: CPU 14.672 / 15.129 s, GPU 55.742 / 41.770 s. A larger bank does not guarantee a cheaper repair route. Online CPU LP calls remain zero. Do not promote this configuration on the strength of dictionary size. The best completed configuration in this turn remains the smaller-bank bucketed 8x2 development run; CPU speedup and full GPU-resident dFBA are still unachieved. No benchmark worker remains running.

Next priorities: identify which initial-basis/direction choices cause expensive second-step repairs; keep correction working sets memory-bounded; measure optional temporal basis reuse separately from same-LP column-generation continuation. The elapsed-time sequence is not independent: each step consumes the preceding biomass/metabolite state. Independent environments and pre-specified offline LP queries can be batched, but unknown future dFBA inputs cannot be treated as independent LPs.
