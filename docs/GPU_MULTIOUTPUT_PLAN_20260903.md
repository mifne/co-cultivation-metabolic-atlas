# Multi-output GPU projection: implementation plan

Scope: OR16 + curated NS21 PHBV + P. freudenreichii. Keep GEMs, feed, objective
definitions and existing accuracy/physical tolerances unchanged. Online FBA
must make zero CPU LP calls; offline exact labels are permitted.

1. COMPLETE: compare CPU and GPU at identical GPU-visited states. Separate
   network prediction error, candidate/projection error, and trajectory drift.
   Audit growth, separate PHB/PHV, and biomass-weighted net medium transfers.
2. COMPLETE: add a multi-output soft quadratic objective over growth/product/
   exchange predictions, subject to the full physical constraints. Do not force
   inconsistent neural predictions as hard equalities. Preserve scalar mode.
3. COMPLETE (diagnostic, not a pass): test with exact targets as an offline oracle, then neural targets;
   never use diagnostic CPU targets as online inference or as held-out training.
4. COMPLETE (diagnostic; full criteria FAILED): GPU-only 120-step comparison against identical-action CPU reference.
   Keep maximum PHA error <=1%, biomass error <=0.01 g/L, 3HV fraction error
   <=0.01, all steps accepted, no early termination. Do not promote an artifact
   until the independent five-seed qualification also passes.

Previously used diagnostic seeds may be used for tuning, but their states must
not become training data. Reserved final seeds 20286001--20286005 remain held out.
Baseline: 34,344 anchors, K=2048/pool=4096, 120/120 GPU accepts, PHA error13.053%.

## Findings and plan revision

- Same-state audit: 26 sampled states from a 120-step rollout. Large late-state
  neural errors and reversals of the net NH4 transfer direction were observed.
  A physical feasibility certificate alone does not certify CPU LP optimality.
- Six-state neural/oracle comparison: correct targets improve early states,
  but late-state QP iterates may not meet all bounds within 2,000 iterations.
  Added an analytical feasible line search from the audited GPU baseline;
  bounds and tolerances are unchanged, and worse soft-loss outputs are rejected.
- Re-trained the head using all 1,576 previously collected augmentation rows
  (not the diagnostic states). It remains an experimental artifact. Compare
  on the same held-out states/rollout, not different random validation splits.
- Local GPU QP path now supports multi-output matching; the separate GPU
  service is explicitly rejected with this option until it is implemented.
- Final related CPU/CUDA unit and regression tests: 44 passed (6 fork warnings).

## Final diagnostic results and remaining work

- Retrained head + multi-output strength 30: PHA error 0.5125%, 3HV fraction
  error 0.001307, biomass error 0.028451 g/L. All 120 steps GPU accepted;
  no online CPU LP. This is one tuning seed, not five-seed qualification.
- Strength 100: PHA 1.6135%, biomass 0.038616 g/L. Strength 300: PHA 3.0416%,
  biomass 0.040016 g/L. Stronger projection did not improve trajectory accuracy.
- The offline hull-capacity LP proved late-state target limitations even with
  all 34,344 anchors plus block compositions. Independent species mixing
  improved the limit but did not eliminate it on the three audited states.
- Next priority: independently collected long-horizon, nitrogen-switching
  examples, plus species-independent GPU mixing. More iterations or retrieval
  candidates alone cannot remove the measured finite-hull limitation.
- Five-seed final qualification remains PENDING, intentionally not consumed
  while the development biomass criterion fails. Production remains unchanged.
- Report: `docs/GPU_MULTIOUTPUT_REPORT_20260903.md`.

## Subsequent literature-guided iteration (2026-09-03)

The current-model microbatch service, DAgger collection, boundary-aware
training, warm-start controls and independent mixing were implemented and
tested in `docs/GPU_LITERATURE_REFINEMENT_PLAN_20260903.md`.
See `docs/GPU_LITERATURE_REFINEMENT_REPORT_20260903.md` for all positive and
negative results. The earlier service limitation above is historical: the
service itself now supports multi-output configuration. Combined accuracy
qualification is still NOT achieved; no experimental model was promoted.
