# GPU-first repeated LP acceleration: implementation and measurement plan

## Evidence and scope

The previous 8-environment, two-time-step repeated run took CPU 9.333 s versus GPU 10.862 s. GPU LP stages consumed 7.172 s, including 85 restricted-repair groups (54 single-row groups; mean 1.6 environments), 2,936 pivots and 4.991 s of restricted repair. GPU hardware capacity is not proof that this implementation is faster. A larger dictionary was slower. These are development measurements, not independent publication results.

Literature-informed design:

- [SurfinFBA, Brunner & Chia 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007786): reuse a basis until a change requires reoptimization. Our community matrix/cost also change, so retain full original primal/dual/gap checks, not just bound feasibility.
- [Fan et al. 2023](https://proceedings.mlr.press/v202/fan23d.html): predict useful bases/restricted master problems, then optimize; proposals are not accepted solutions.
- [BatchLP 2026 preprint](https://arxiv.org/abs/2601.21990): common-matrix GPU batching, active-column compaction, hardware-specific batch calibration, and a warm-start CPU comparison. A guarantee for this dFBA system does not follow.
- [DFBAlab 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4279678/): alternative LP optima can affect dynamic exchange fluxes. Raw internal-flux dispersion is not prediction error.
- [AMN, Faure et al. 2023](https://www.nature.com/articles/s41467-023-40380-0): motivates mechanistic/learned hybrid models, but does not establish our CPU-equivalent trajectory accuracy or GPU speed.
- [Distributed PDLP 2026 preprint](https://arxiv.org/abs/2601.07628): large-LP multi-GPU communication differs from independent-environment sharding; no three-GPU speed factor is claimed from one laptop GPU.

## Revision 1: implemented experimental architecture

1. Assemble exchange LP constraints in bulk CSR, preserving every coefficient and row/column ordering. This shared optimization benefits CPU and GPU equally.
2. Add persistent HiGHS simplex models and basis reuse per environment and LP stage. Update actual coefficients, RHS, bounds and cost; rebuild on sparsity changes. All preparation, solve and original-LP checks are timed.
3. Enable GPU candidate-ensemble monitoring, optionally. Project original specific rates to species growth, species-resolved exchange and PHA-related reactions; compute online sample variance on GPU. Explicit unit scales are an initial routing convention, NOT measured biological uncertainties. Fewer than two valid candidates yields unknown dispersion.
4. GPU-first hybrid path: compact dictionary -> original LP certificate -> bounded reduced GPU repair for low-dispersion misses; high/unknown dispersion goes to CPU concurrently. Uncertified rows after the GPU budget go to CPU as a separately counted safety route. Even high-dispersion certified rows stay on GPU. CPU fallback data never updates the frozen offline bank.
5. Optional fixed-shape CUDA graph replay and full-batch candidate evaluation reduce dispatch overhead. Candidate budget and preceding accepted dictionary index can reduce repeat searches. No future time-step independence is assumed.

Acceptance remains original primal <= 1e-5, dual <= 1e-7, relative KKT gap <= 1e-7; endpoint PHA relative <= 1%, biomass max absolute <= 0.01 g/L, PHV fraction absolute <= 0.01. CPU and GPU must use the same original three hierarchical objectives. Full simulator GPU residency remains unimplemented.

## Measurement sequence and promotion rules

- First small development two-step transitions: both cold first use and graph-reused runs; include all candidate, repair, CPU fallback, host assembly and transfer costs. No concurrent competing benchmarks.
- Ablations: GPU-only, GPU bank plus CPU safety, bounded repair plus dispersion routing; fresh CPU and persistent warm-start CPU. Record fallbacks and routing reasons, not only average GPU utilization.
- Freeze promising routing settings before fresh-seed validation. Unit-scale thresholds are experimental hyperparameters, not evidence that a given variance is safe. Do not tune on held-out results or add them to training.
- If accurate and faster, increase parallel environments and trajectory length, then 120 steps (24 simulated hours), with independent seeds and repeated timing. A two-step speedup is not a learning-time speedup.
- Multi-GPU design: persistent independent environment shards, each owning a bank and LP buffers; batch within each GPU. Do not split these small LPs across three PCIe devices or imply shared VRAM.

Current status at implementation: new architecture not yet benchmarked; CPU speedup is not claimed.

## Revision 2: first hybrid measurement and next bottleneck

`results/pf_hybrid_v1_8x2_20260904.json`: persistent CPU 6.098 / 6.123 s versus hybrid 8.584 / 6.323 s, cold / repeated. Both endpoint checks pass, but ratios 0.710 / 0.968 remain below one. Of 48 LPs, 16 pass the bank, three pass bounded GPU repair, and 29 use CPU. The repeated run's PHA relative discrepancy is below 5e-12. These development seeds were already inspected earlier; this is not independent validation.

Candidate dispersion reveals its limitations empirically: first-step aggregate candidates have observable spread around 3.39e-10 in the explicit unit scales, yet one failed both GPU repair rounds and needed the CPU safety route. Small spread is not an error certificate. Conversely, some high-spread rows are certified and stay on GPU.

Next ablation: 16 independent development environments x 8 steps, no reduced repair, to isolate bank-plus-CPU costs. In parallel, add a common candidate-cohort path: evaluate each candidate once for the whole batch and use its certified result for every environment, not merely the rows whose nearest-neighbor list selected it. This removes repeated per-rank host synchronization and reuses work already computed by full-batch candidate replay. Preserve first-certified selection and original LP checks.

## Revision 3: longer development ablations and dispatch capture

The smaller-bank 16x8 bank-plus-CPU ablation passes accuracy but remains slower: CPU 33.168 / 34.007 s, hybrid 38.508 / 38.024 s. CPU is still needed for 331/384 LPs; bank coverage falls on later time steps. This is not evidence of end-to-end speedup.

The existing larger training bank plus full candidate-cohort evaluation also passes, but takes 47.863 / 44.415 s versus CPU 32.775 / 32.534 s. CPU calls fall to 303/384, insufficient to offset screening. The aggregate stage has 52 entries but the per-entry graph LRU holds 48, causing repeated compilation (5.60 s total aggregate compilation in the first trajectory). Its full candidate ensemble also contains wildly extrapolated, uncertified fluxes: resulting enormous spread is not a calibrated uncertainty measure. Do not represent these values as error bars or biological variability.

Next implement ONE replay graph for the whole fixed candidate cohort, including candidate evaluation, first-certified selection and dispersion statistics. Avoid nested per-entry graph launches and keep one fixed-shape replay per stage/cohort. This also bounds cache working sets and avoids a cache capacity below the candidate count. Test changed inputs and buffer lifetimes before measuring. A whole-cohort graph does not change the LP or candidate acceptance conditions.

## Revision 4: whole-cohort replay measured; initial-basis handoff next

Whole-cohort replay passes the new unit tests and the full 16x8 development trajectories. CPU 32.445 / 34.521 s, hybrid 38.505 / 37.603 s; still no CPU speedup. The three candidate-graph private pools fall from 3.843 GB to 0.138 GB (decimal; this is NOT total device memory or peak VRAM). The first trajectory's entire cohort compilation is 1.536 s. Aggregate repeated screening drops from about 0.8 s/stage to 0.08–0.10 s/stage.

A CPU-fallback penalty remains: an environment previously solved by the GPU has no retained CPU basis. Its first CPU miss can need about 2,000 simplex pivots, whereas the CPU-only comparator already owns a previous basis and often needs only tens. Transfer the selected dictionary BASIS as an initialization to a newly built CPU model, never an uncertified flux as an accepted result. HiGHS must still optimize and pass the original-LP certificate. Existing persistent CPU bases take precedence over fresh dictionary proposals. No training data or CPU LP is added during GPU screening.

For attribution, also add a CPU-only nearest-centroid dictionary-initialization comparator using the same offline bank. It must perform only NumPy/SciPy/HiGHS work online; this separates the benefit of a good initial basis from GPU candidate screening itself.

Shared host optimization: `_objective_vector` previously decomposed the same COBRA objective expression once per reaction (6,733 calls per step). Read the current expression once per species using `linear_reaction_coefficients`, without caching across time. Verify exact legacy output under modified objectives and min/max directions. Apply identically to every CPU/GPU comparison.

## Revision 5: limited development speedup and frozen independent check

Adding GPU-selected initial bases to the larger-bank 16x8 hybrid reduces CPU initialization pivots (first aggregate CPU fallbacks average about 2.7 rather than roughly 2,200), but still does not beat its CPU comparator: CPU 31.060 / 29.742 s versus hybrid 34.599 / 33.590 s. Do not promote the larger bank.

The smaller-bank 32x2 development test includes the stronger CPU-only dictionary + persistent warm-basis comparator. It passes all accuracy checks in both trials: CPU 14.760800 / 14.981904 s; hybrid 14.318594 / 13.405811 s; ratios 1.030883 / 1.117568. GPU dictionary certificates cover 82/192 LP requests; 110/192 still use CPU. Reduced GPU repair is disabled in this selected ablation (`hybrid_rounds=0`). Thus its speed difference cannot be attributed to dispersion-based GPU repair.

Shared community inequality construction now uses a sparse template/value update instead of hundreds of dense zero rows. Tests compare the complete legacy CSR representation and RHS, including duplicate-term summation order and cache invalidation. This is applied to BOTH comparators. The selected regression suites pass 158 tests in two non-overlapping invocations (148 core plus 4 inequality and 6 schedule tests).

Freeze the following independent check BEFORE running it:

- Smaller 20/28/20-entry bank; all-candidate whole-cohort replay, count ranking, no reduced repair; CPU fallback gets GPU-selected initial basis only when a CPU model is new/rebuilt.
- Original three LP objectives and unchanged original-LP/endpoint gates.
- CPU reference: CPU-only nearest-centroid initialization from the same bank, then persistent HiGHS; four workers, one thread each.
- 32 environments × 2 time steps, six repetitions. Repetition zero measures cold graph construction; the other five use fresh, non-overlapping seed groups with only immutable graphs/maps reused. No CPU/GPU solutions cross repetitions.
- Seeds start at 20289801 with stride 32; all are checked against the offline training seeds before any run. Alternating CPU-first/GPU-first execution.
- Report the cold pair separately, and paired log-ratio geometric mean with two-sided 95% Student-t interval for the five warm pairs. Report every pair, failures and CPU fallback counts. No exclusion of unfavorable completed pairs.
- No changes to routing, dictionaries, tolerance or numerical kernels after inspecting these held-out results. If the interval includes one, do not call the speedup statistically established.

This validates short transition throughput only. It is not a 120-step, RL-training, full-GPU, or multi-GPU qualification.

## Revision 6: frozen check complete; narrower candidate batching next

The frozen six-pair run completed without failure or endpoint-gate relaxation in `results/pf_hybrid_frozen_32x2_independent_20260904.json`. The first-use pair is CPU 14.249679 s / hybrid 13.966668 s. The five subsequent, non-overlapping seed batches have CPU/hybrid ratios 1.074559, 1.088183, 1.103159, 1.012962 and 1.039630. All 192 trajectories (six batches of 32) complete both steps and pass the endpoint gates. The statistical sample size is five warm paired batches, not 160 independent timing observations. Warm CPU fallbacks total 594/960 LP requests (61.875%); GPU restricted repair remains disabled. Source snapshots and negative development runs remain preserved.

There is still substantial redundant candidate work: whole-cohort evaluation applies every selected basis to every environment. Implement an OPT-IN heterogeneous evaluator, selecting K bases separately per environment and reconstructing/certifying only B×K proposals in one shared-A sparse matrix–dense matrix pipeline. Use indexed projection kernels with flat per-basis weights to avoid a B×K×basis-size×parameter-size weight gather. Preserve all input-family, finite-value, original-coordinate primal/dual/gap checks, duplicate-aware dispersion statistics and first-certified selection. Only actual candidates receive scores; no valid proposal means no CPU initial-basis proposal, not a negative Python index. Fixed-shape replay must accept changing candidate IDs as GPU data rather than static graph keys.

Also combine all CPU misses into one fallback batch when the configured GPU repair budget is zero. Preserve per-row route labels, stable environment IDs, persistent bases and CPU failure handling. The positive-repair-budget route still overlaps CPU work with GPU repair. These are new development variants; they do not retroactively change or tune the frozen result. Run toy LP correctness/changed-input replay tests first, then fresh-seed end-to-end timings and longer horizons. Do not promote a prototype on kernel time alone.

## Revision 7: heterogeneous 32x8 result and preparation bottleneck

The opt-in B×K implementation, dynamic-candidate replay and single-wave zero-budget CPU fallback pass 211 combined regression tests, including mixed basis dimensions/low-rank update counts, invalid candidate IDs, original LP certificates and graph-buffer lifetimes. Seven additional plotting-validation tests pass separately.

`results/pf_hybrid_heterogeneous_k4_32x8_dev_20260904.json`: K=4, 32 environments × 8 steps, fresh development seeds 20290201–20290232, two repetitions with alternating execution order. CPU uses the same dictionary and persistent HiGHS (four workers). CPU 52.337611 / 52.443157 s; hybrid 54.060590 / 52.789274 s; ratios 0.968129 / 0.993443. Every trajectory passes the unchanged accuracy gates. Maximum endpoint discrepancies are PHA relative 1.919646e-7, biomass absolute 4.857598e-8 g/L, PHV fraction absolute 3.207403e-7. This variant still does NOT beat CPU over eight steps.

Only 87/768 LPs pass the GPU bank; 681/768 (88.67%) use CPU. By the third step, no exchange LP is accepted by the four-candidate bank; aggregate coverage becomes zero from step five. This does not mean the LPs are infeasible: the CPU solves and certifies them. Candidate availability/selection and reoptimization are distinct from model feasibility.

The second repetition partitions the 52.789274 s workflow into host LP preparation 2.422832 s, candidate screening 2.827593 s, remaining LP service 25.964458 s, and other host dFBA work 21.574391 s. Remaining LP service includes CPU fallback, dispatch and result handling; it is not a GPU timer. Likewise screening includes its host work/synchronization, not exclusively GPU kernels. Graph pools total 805,986,304 bytes; copied flat weights total 117,570,632 bytes. Neither is total or peak device usage.

Diagnostic plot: `results/pf_hybrid_heterogeneous_k4_32x8_diagnostics_20260904.svg` / `.png`; generated from the explicitly selected LAST repeat, not the fastest run. It shows per-stage bank acceptance across time and the workflow-time partition. Preserve this negative result alongside the positive frozen 32x2 check.

Next audit the arrays -> normalized LP -> GPU batch preparation path. It costs about 0.1 s per stage. Remove only demonstrably redundant copies/sparse assembly by a batch-equivalent transformation, with elementwise comparisons against the existing path. Keep CPU requests and original-coordinate certificates unchanged. Separately, later-state dictionary coverage still needs fresh TRAINING trajectories or a calibrated cost gate; never insert frozen validation solutions into the bank. Purely increasing the candidate count is not justified by the negative larger-bank experiments.

## Revision 8: exact sparse preparation fast path

Read-only microprofiling on the stored real maxmin/exchange matrix shapes, using synthetic positive nontrivial scaling (32 matrices, median of five repetitions), found two candidate savings. Row-diagonal @ A @ column-diagonal normalization took about 36.6–37.9 ms versus 5.6–5.7 ms for direct canonical CSR coefficient scaling in the same row-then-column order. Full sparse difference plus repeated row slicing in bank preparation took 9.3–10.8 ms versus 1.8–2.2 ms for a same-pattern coefficient check and direct extraction of variable rows. These are local microbenchmarks, NOT measured dFBA speedups, and the projected ~0.9 s per 24-stage workflow is not a result.

Implement only a narrow equivalence-checked path, retaining the old sparse operations for unsupported shapes, noncanonical data, changed sparsity and other exceptional cases. Test all coefficients/indices/order, signs, bounds, costs, row/column scales, explicit zeros, duplicate indices, underflow and unsupported fixed-row changes. The normalization improvement is shared by CPU and GPU. Do not hide that benefit in the CPU comparator. After tests, run a new 32x8, K=4, original3, CPU dictionary comparison on development seeds starting at 20290401, two disjoint groups (stride 32), CPU-first then GPU-first. Preserve failed or slower runs; the fixed 32x2 publication check remains untouched.

Implementation and safety checks are complete: 246 combined regression tests pass (50 harmless SciPy option-forwarding warnings, no skips). Saved real maxmin/aggregate/exchange matrix stress tests also match every original CSR/vector field exactly. Production-method microprofiles, including finite guards, are recorded separately in `docs/HOST_PREPARATION_EQUIVALENCE_20260904.md`; they are not E2E results. An inherited NaN hole in the unsupported fixed-row check is deliberately closed: nonfinite root/current/difference coefficients now raise rather than disappearing during variable-row extraction. Source files are frozen during `results/pf_hybrid_heterogeneous_fastprep_32x8_dev_20260904.json` measurement.

## Revision 9: failed CPU reference is not a speedup result

The first fresh 20290401–20290432 CPU reference stopped with `An LP was rejected; no fallback` before the GPU trajectory ran. Therefore the fast-preparation experiment is not a completed comparison, and no speed ratio can be calculated from it. The original incomplete artifact is retained. Its old CPU exception path failed to save the last LP history; fix this reporting gap rather than excluding the failed seeds.

The benchmark now records CPU progress, full service history, failure traceback and diagnostic-only original failed LP arrays, with hashes. Capture writes only a new artifact and refuses overwrite. The retry uses the same 32 seeds and the same numerical implementation, not easier initial conditions: `results/pf_hybrid_fastprep_cpu_diagnostic_retry_20260904.json`. Diagnostic LPs are excluded from dictionary training.

If the failure is HiGHS `Optimal` with failed independent certification, inspect the actual residual and complementarity components first. Any bounded stricter numerical retry must be shared by CPU-only and hybrid CPU fallback, charged to online time, and checked against unchanged original LP tolerances. Infeasible, unbounded, limited or uncertified answers must never be relabeled successful. Do not infer the cause from the exception string alone.

The same-seed diagnostic retry reproduced one rejected exchange LP at environment index 23 (seed 20290424), after seven completed steps. Its original primal residual was 5.0945e-9 and dual violation 8.6458e-12, but relative KKT gap was 1.9667e-7, above the unchanged 1e-7 threshold. The original sparse LP was saved as `results/pf_hybrid_fastprep_cpu_diagnostic_retry_20260904.cpu_failure_23.npz`. An isolated ordinary cold solve of that same LP passes with gap 7.50e-10 (about 0.352 s); minimum nonzero matrix coefficient is 5e-5, excluding tiny-coefficient dropping in this case. The exact failing trajectory basis was not saved, so success from its strict warm restart is not yet established. See `results/cpu_certificate_isolated_20260904.json`.

Add a bounded numerical recovery only after an Optimal-but-uncertified CPU result: strict 1e-9 basis refactor/reoptimization, then at most one strict 1e-10 fresh no-presolve solve if needed. Keep original certificate thresholds and all objectives/coefficients/bounds unchanged. Restore solver options before the next LP, record every attempt/status/certificate, and distinguish distinct CPU LP requests from actual optimizer run attempts. After tests, rerun the SAME planned seed groups; do not substitute a successful group.

The common CPU recovery is implemented and all 256 regression tests pass. A real two-variable warm-start counterexample (HiGHS Optimal, independent gap 2e-6) is repaired by one strict refactor in the regression suite. Tightening `kkt_tolerance` alone was insufficient in that example; each individual primal/dual/residual/optimality tolerance is explicitly tightened too, then restored. Nonoptimal outcomes are not retried as success; retry caps and exception-time restoration are tested. The new same-seed comparison is `results/pf_hybrid_fastprep_certified_retry_32x8_dev_20260904.json`. Executor teardown is now outside CPU online time, as for the hybrid service; source snapshots preserve this change. No historical timing is rewritten.

## Revision 10: test capacity independently of candidate-evaluation count

After the smaller-bank rerun finishes, evaluate the existing larger `pf_compact120_v3_20260904` bank with the SAME heterogeneous K=4 budget and exact preparation/recovery implementation. Unlike the old all-cohort method, adding stored bases no longer multiplies the reconstructed candidate count by the entire bank size. The router and weight memory still grow, so constant cost is not assumed.

Use its verified matching repair manifest, original3 and unchanged gates, CPU initialization from that same larger bank, four CPU workers. Evaluate 32×8 with two disjoint fresh development seed groups starting at 20290601 (stride 32), alternating CPU-first/GPU-first. This is a separate development experiment, not a modification of the fixed five-pair 32×2 check. No evaluation solution is added to the larger bank. Compare full elapsed time, coverage, CPU request/optimizer attempt counts and graph memory; retain failures and slowdowns. This tests whether longer-trajectory bank coverage can now improve without the old all-candidate dispatch penalty.

## Revision 11: completed eight-step measurements and remaining work

The small-bank fast-preparation/recovery rerun completed both originally planned seed groups: CPU 48.393580 / 52.239334 s, hybrid 51.187342 / 52.280558 s. Both accuracy checks pass, but neither pair beats CPU. The previously failed seed 20290424 reproduces its initial gap 1.9667e-7 and passes after one strict refactor (additional 0.005708 s, gap 1.4972e-9). There are 768 distinct CPU requests and 769 optimizer runs in that first CPU trajectory; original options are restored. No acceptance threshold is relaxed.

The larger-bank K=4 experiment also completed both planned groups in `results/pf_hybrid_largebank_k4_32x8_dev_20260904.json`. The bank has 44/52/44 entries for maxmin/aggregate/exchange; CPU and hybrid use that same bank. Seeds 20290601–20290632 give CPU 53.878433 s / hybrid 51.182578 s (ratio 1.052671). Seeds 20290633–20290664 give CPU 51.993043 s / hybrid 53.005983 s (ratio 0.980890). Both pass all unchanged gates. Thus this is a MIXED development result, not a consistent eight-step speedup; the cold/reused graph conditions and seed groups differ. Do not compute a confirmatory interval from these two selected development pairs or merge them with the frozen 32×2 check.

CPU fallback is 633/768 (82.42%) and 664/768 (86.46%). All CPU requests in this experiment need one optimizer run; no numerical retry occurs. Maximum endpoint discrepancies across the two pairs are PHA relative 3.92715e-12, biomass absolute 9.90276e-6 g/L, PHV fraction absolute 3.89022e-7. These compare numerical trajectories, not experimental biology.

The explicitly last repetition takes 53.005983 s: LP preparation 1.585431 s, screening 2.939108 s, other LP work 27.268067 s, other host dFBA work 21.213378 s. Exchange dictionary acceptance is zero from step two; aggregate acceptance is zero in most later steps. The diagnostic SVG/PNG `results/pf_hybrid_largebank_k4_32x8_diagnostics_20260904` records these intervals and stage-specific coverage. Neither interval timing nor graph-pool allocation is GPU utilization or peak VRAM. Raw JSON SHA-256: `0de37670f2d11eaeaa221e303d6c03becb0c985a706588dfcfa9c6bc64d7d214`. Missing/nonfinite diagnostic scalars in newly saved JSON are null, not zero; historical reports remain untouched.

Next priorities, in order:

1. Improve later-state aggregate/exchange candidate coverage and selection using NEW TRAINING trajectories only. Do not enlarge all-candidate evaluation blindly or train on frozen/diagnostic evaluation LPs. Evaluate useful certificates per unit total time, not raw bank size or GPU occupancy.
2. Profile and batch the remaining host state/LP assembly work with exact-equivalence checks shared by both comparators. About 21.2/53.0 s remains outside LP service in the latest repeat; GPU arithmetic throughput alone cannot remove it.
3. Develop heterogeneous warm GPU correction and a cost-aware routing gate. Compare its complete cost against persistent CPU reoptimization. Candidate dispersion alone is neither an accuracy certificate nor sufficient evidence that CPU is necessary; full certificates always take precedence and unresolved low-spread cases still need a bounded safety route.
4. After consistently faster eight-step development runs, freeze a new protocol and independently validate 120-step accuracy, elapsed time and memory stability. Then evaluate actual multi-GPU independent shards and RL throughput. Do not extrapolate the current two-step result to those claims.

Current bounded result: the frozen 32×2 warm comparison remains about 6% faster overall, while eight-step consistency, full GPU residency and multi-GPU scaling remain unachieved or unverified. All timing runs have finished; no further benchmark is left running by this report.

Final regression run: 257 selected related tests pass in 11.38 s, with 50 SciPy-to-HiGHS option-forwarding warnings and no failures or skips. This is the related regression suite, not an assertion that every unrelated repository test was run.
