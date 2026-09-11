# Literature-guided refinement, iteration 2 (2026-09-03)

Scope: current OR16 + NS21 PHBV + P. freudenreichii. No GEM/feed/CPU objective
changes. Preserve maximum PHA error <=1%, biomass <=0.01 g/L, 3HV fraction
<=0.01, all steps GPU accepted and zero online CPU LP. Prior legacy-model
qualification and throughput cannot be transferred to this model.

## Primary literature and exact scope of adoption

- Faure et al. (2023), AMN: https://www.nature.com/articles/s41467-023-40380-0
  Neural preprocessing plus a metabolic mechanistic layer. Retain stoichiometry
  and physical bounds; add training loss for current live reaction intervals.
  Our anchor-QP is an adaptation, NOT a reproduction of the paper's end-to-end
  differentiated mechanistic solver. Matching a CPU GEM is not experimental
  biological calibration.
- Amos & Kolter (2017), OptNet: https://proceedings.mlr.press/v70/amos17a.html
  Batched GPU QP layers. Restore the shared microbatch architecture for the
  current live-objective PHBV model; investigate independent species mixing.
  Our ADMM implementation is not OptNet's interior-point solver and currently
  does not backpropagate through the full QP.
- Ross et al. (2011), DAgger: https://proceedings.mlr.press/v15/ross11a.html
  Sequential predictions induce distribution shift. Adapt dataset aggregation
  to a simulator surrogate: relabel GPU-visited states with offline exact LPs,
  aggregate, retrain. This is not a proof of a 1% dFBA error guarantee.
- Hallmann et al. (2026), gsMOBO: https://spj.science.org/doi/full/10.34133/csbj.0072
  Outer-loop medium design trades off production, growth and medium cost. It
  does not replace the inner FBA solver. Keep this role separate from GPU FBA.

## Implementation sequence

1. COMPLETE: collect independent GPU trajectories using the ACTUAL current
   multi-output configuration (lambda=30, K=2048, 2,000 iterations), then label
   all visited states offline. Save row seeds and raw labels for provenance.
2. COMPLETE / not qualified: update training to emphasize aggregated trajectories and penalize
   violations of live decision-flux intervals; reject dataset/model mismatch.
   Compare data-only retraining and mechanistic-loss ablations.
3. IMPLEMENTED / not qualified: independent-species simplex option in GPU QP; same-state tests first.
   Keep original joint simplex as a control and preserve physical certification.
4. COMPLETE: update shared microbatch service to support current live objectives
   and the same multi-output QP, including retry. Test parallel requests.
5. DEVELOPMENT COMPARISON COMPLETE / joint criteria failed: 120-step CPU/GPU comparisons. Only after ALL
   development criteria pass use reserved final seeds 20286001--20286005.

Baseline: lambda=30, 120 steps, seed20260913, PHA error0.5125%, biomass0.028451
g/L, 3HV fraction0.001307, 120/120 GPU accepts, 0 online CPU LP. Not qualified.
No artificial endpoint correction, relaxed tolerances, or test-state training.

## Evidence-driven revision during implementation

- Collected 360 states on seeds20294001,20295010,20296019, including two GPU
  rejections. Relabeled offline; dictionary now34,704. Raw labels/row seeds saved.
- Service fixed: no longer hard-codes the legacy maintenance objective mode.
  It validates/uses the current live-objective artifact and supports multi-output
  projection also on retries. Large retries are split to control peak memory.
- Added a mathematically exact boundary-face presolve for independent mixing.
  If all anchor fluxes are nonnegative but a reaction is closed, positive-flux
  anchors cannot receive a positive weight in an exact feasible mixture.
- Scratch retraining with augmentation weight4 worsened the development rollout.
  Added a frozen-head dictionary control and warm-start calibration, retaining
  the original feature/output normalization and checking model/layout identity.
- Additional bounded prototype: signed affine anchor weights, inspired by the
  previous affine mechanistic representation. It removes the artificial convex-
  hull restriction, but requires full varying-bound checks and a GPU sparse
  stoichiometric residual audit. This prototype remains offline diagnostic only:
  tight inequalities still prevent useful convergence in some late states.
- Separate physical feasibility, CPU optimality/trajectory accuracy, and wall
  time in every result; no failed experimental branch is promoted by default.

## Final checks and decision

- Related regression suite: 54 passed, 8 warnings (sparse tensor API and fork),
  results/pf_literature_pytest.log. This is not the entire repository suite.
- Current-model shared CUDA service: 4 concurrent requests x2 rounds, 8/8
  feasible, maximum batch4, zero online CPU LP. Not a PPO scaling benchmark.
- Warm calibration lowered PHA endpoint error from0.5125% to0.2762%, but
  3HV fraction error worsened from0.001307 to0.030078. Maximum biomass error
  remained0.026461 g/L. Do not promote this model on total PHA alone.
- Frozen-head dictionary expansion, scratch retraining, boundary-loss training,
  and independent species mixing also failed the joint development criteria.
  Preserve all controls and failures in the report and machine-readable results.
- Reserved final seeds remain untouched. Next iteration should address QP
  convergence/active constraints and projected multi-output trajectory learning,
  not merely increase dictionary size or relax qualification thresholds.
- Deliverables: docs/GPU_LITERATURE_REFINEMENT_REPORT_20260903.md and
  results/pf_literature_refinement.{png,pdf}, generated reproducibly by
  scripts/report_literature_gpu_refinement.py.
