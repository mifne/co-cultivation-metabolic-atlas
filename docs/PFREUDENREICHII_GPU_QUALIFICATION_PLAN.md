# GPU-only PHBV qualification plan (2026-09-03)

Latest follow-up: `GPU_MULTIOUTPUT_PLAN_20260903.md` and
`GPU_MULTIOUTPUT_REPORT_20260903.md`. Joint growth/PHB/PHV/exchange QP and
retraining reduced one development-seed PHA error to 0.5125%, with 120/120 GPU
accepts and no online CPU LP. Biomass error is still 0.028451 g/L; formal
qualification remains pending and the final five seeds remain untouched.

Scope: OR16 + curated NS21 (separate PHB/PHV pathways) + P. freudenreichii.
Keep current GEMs, medium, live objectives and CPU lexicographic reference fixed.
The old legacy-three-species result (maximum PHA error 0.861%) is not evidence
for this updated model.

## Acceptance criteria (unchanged)

- 120 actual steps per independent trajectory; initial NH4 0.05 mM.
- Maximum terminal PHB+PHV mass error <= 1%, not just the mean.
- Maximum species biomass absolute error <= 0.01 g/L.
- Maximum terminal 3HV mole-fraction difference <= 0.01.
- All online FBA requests accepted on GPU; zero online CPU LP solves.
- Infeasible GPU returns / skipped dynamics are failures, not speed-ups.
- Compare CPU and GPU with identical actions, model hashes and objectives.
  Time both serially under comparable machine load before claiming speed-up.
- GPU-only refers to the online FBA solver. Python environment integration,
  file I/O and offline exact training-label generation can remain on CPU.

## Baseline

32768-anchor PHBV/P. freudenreichii dictionary + neural reranker, K=512,
rerank pool=2048, total-PHA target interpolation, CPU fallback enabled:
three 120-step seeds 20260911--13; mean/max PHA errors 5.769% / 11.881%;
GPU acceptance 72.22%, so neither accuracy nor GPU-only qualification passes.
Existing CPU times used three concurrent workers and are not fair speed tests.

## Implementation sequence

1. DONE: fix active/DAgger data collection to use the selected consortium,
   initial NH4 and every live objective coefficient. Test exact context replay.
   Capture rejected GPU query contexts; do not reuse stale snapshots.
2. DONE: compute QP diagnostics on the actual blended output, not on an
   unrelated pre-interpolation anchor. Keep physical tolerances unchanged.
3. DONE: add independent long-trajectory exact anchors; DAgger relabel states
   induced by GPU dynamics. Preserve original production artifacts.
4. DONE FOR THIS ITERATION: compare local candidate selection and retrained/rebased rerankers;
   reject changes that improve feasibility while worsening PHBV trajectories.
5. PENDING: GPU-only independent validation with untouched seeds. If finite
   dictionary coverage remains insufficient, investigate exact GPU LP correction
   rather than suppressing failures or loosening the 1% criterion.

Final diagnostic B with 34,344 anchors, K=2048 / pool=4096: 120/120 GPU accepts,
zero CPU LP stages, PHA error 13.0530%, biomass error 0.11613 g/L, 3HV fraction
error 0.01990. All artifacts remain experimental_not_qualified. Final detailed
report: `docs/PFREUDENREICHII_GPU_QUALIFICATION_20260903.md`. Final scoped tests:
33 passed. Do not launch RL with an assumed 1% guarantee from these results.

Additional runtime optimization: cache normalized decision columns and squared
context norms on GPU. Reranking gathers 155 columns instead of all 6733 reaction
columns; full fluxes are materialized only for the selected QP candidates. This
does not change the ranking formula or the scientific model.

Verification so far: 28 scoped tests pass; a three-step real PHBV/P. freudenreichii
GPU-state replay restored all 20599 context values and produced 6733-flux exact
labels, including one rejected GPU query. Rejected queries now have their own
snapshots instead of stale flux labels. Direct counters distinguish cooperative
CPU solves from their individual (up to three) LP stage invocations.

## Iteration findings (not qualification)

- Added 960 exact rows from eight 120-step calibration trajectories, then 360
  exact labels at independent GPU-induced states (including 22 rejected queries).
  Total dictionary: 34,088. No prior validation seeds were used for labels.
- Retraining with the 960 active rows for 500 epochs reduced sample-held-out
  PHA flux RMSE from 0.630 to 0.347. This is not a rollout accuracy certificate.
- At diagnostic seed 20260901's first state, the maximum PHA mass rate among
  155 individually feasible anchors was 7.705, versus the CPU reference 8.021.
  However, an offline CPU oracle over all 512 retrieved anchors plus species
  block compositions reached 8.468 within the *unchanged* physical tolerances.
  Therefore, convex mixtures of individually infeasible anchors can fill this
  accuracy gap; it is not sufficient to search only feasible single anchors.
- The neural prediction there was 7.983 (0.46% below CPU), but the former
  projection stopped at 7.705. A GPU ADMM target-band correction now searches
  the full retrieved hull. It removes constraints that every anchor already
  satisfies at the exact bounds and audits the full
  physical system afterwards. It uses GPU float64 Cholesky and a Woodbury
  constraint-space update (or the smaller candidate-space system).
- New 24-step diagnostic: GPU accepted 24/24; online CPU LP stages 0; terminal
  PHA error 0.6904%, but biomass error 0.03530 g/L and 3HV mole-fraction error
  0.01465 still fail the additional gates. This is **not qualified** and not
  a five-seed result. Timing overlapped offline collection, so no speed claim.
- Before ADMM, a 120-step diagnostic with the augmented dictionary still had
  20.04% PHA error and two GPU failures at decision strength 0. Increasing the
  neural rank weight to 4 worsened this to 95.87% error / 87 failures; reject it.
- Additional boundary-only labels (256 independently seeded first-step queries)
  have been added, giving 34,344 anchors. They address sparse coverage at the initial low-N state,
  not a change in the medium or the biological model.
- An aggressive tolerance-based row-removal ablation lost the short-trajectory
  target correction; it was reverted. Only strictly hull-redundant rows are
  pruned in the final implementation, with unchanged final physical tolerances.
- Final serial repeat of the 24-step smoke test: CPU 23.4133 s, GPU 6.8840 s,
  PHA error 0.6904%, 24/24 accepted GPU requests, zero CPU LP stages.
  This 3.40x latency ratio is a single paired smoke test, not a qualified
  accuracy-preserving training speed-up or a confidence interval.

The remaining architectural limitation is objective fidelity: matching one
total-PHA rate and linear feasibility does not reproduce the CPU's max-min
growth / live-objective / parsimonious-exchange lexicographic solution. Small
NH4 and exchange-flux errors change later production-phase switching. Next
work must jointly constrain growth, PHB, PHV and key environmental exchanges,
and/or certify an optimality gap with an exact GPU LP correction. Increasing
candidate counts alone is not an accuracy guarantee.

ADMM reference: Boyd et al., *Distributed Optimization and Statistical Learning
via the Alternating Direction Method of Multipliers* (2011),
https://web.stanford.edu/~boyd/papers/admm_distr_stats.html . The implementation
is a project-specific compact GPU hull QP, not a claim of a new ADMM algorithm.

## Provenance / literature

DAgger labels the states visited by the approximate policy, mitigating rollout
distribution shift: https://proceedings.mlr.press/v15/ross11a.html .
Feasible convex combinations do not certify an optimality gap to CPU LP.
cuOpt provides GPU LP and PDLP warm starts, but requires separate verification
of solver tolerances and CPU-method exclusion for this problem:
https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/lp-qp-milp/lp-qp-milp-api.html .

Calibration seeds start at 20275001; prior diagnostic seeds 20260911--13 must
not be added to training. Reserve 20286001, 20286002, 20286003, 20286004,
20286005 for a final independent test (never add their states to the dictionary).
