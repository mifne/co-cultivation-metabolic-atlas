# AMNを参考にした高速化計画

Latest iteration: [State-transition scaling and 120-step data](AMN_TRANSITION_SCALING_20260904.md).
The report below is historical one-step evidence. The newer iteration resolves
the second aggregate stage on eight development environments, but the following
exchange stage remains unqualified; no complete two-step or 120-step GPU
speedup is established.

Reference: Faure et al., Nature Communications 14, 4669 (2023),
https://www.nature.com/articles/s41467-023-40380-0
Author implementation: https://github.com/brsynth/amn_release (MIT).

The paper's neural layer predicts initial fluxes, followed by LP/QP or Wt
mechanistic iterations. It motivates amortizing initial guesses, not claiming
that a GEM surrogate automatically beats CPU LP. AMN-QP fitting and LP growth
optimization are different tasks; phenotype regression is not a certificate
of our three-stage community optimum or of a 120-step dFBA trajectory.

## Current change (adaptation, not AMN reproduction)

1. Learn a compact neural proposal of a compiled LP basis from offline LP
   parameters. The mechanistic layer reconstructs fluxes from that basis and
   certifies the FULL original LP. Neural confidence never accepts a flux.
2. On rejection, retain the existing exhaustive bank and GPU repair path.
   Do not reduce tolerances, change biological bounds, or call a CPU LP online.
3. Train only on paid offline data / declared synthetic perturbations of
   offline anchors. No evaluation seeds or failed evaluation LPs become data.
   Initial small-data routing is an engineering pilot, not a biological model.
4. Compare neural routing with no routing on matched independent environments,
   including transfers and dispatch. Report cold and prepared times, acceptance
   and repair fractions, total CPU LP calls, and complete endpoint accuracy.
5. Expand training on actual independent dFBA trajectories if routing improves
   coverage/cost. A learned full-flux/dual warm start is a later separate
   integration; it is not implemented by calling a basis classifier an AMN.

The existing source model / compiler fingerprints and numerical gates remain
authoritative. Production CPU/GEM defaults are unchanged. The host still
assembles LPs and integrates dFBA states; this is not full GPU residency.

## Implemented pilot details

- Small 64-hidden-unit tanh networks select a starting basis; inference is
  GPU matrix arithmetic, with no online PyTorch dependency or CPU LP call.
- Original full-variable mechanistic reconstruction and numerical primal,
  dual and gap tests remain mandatory. GPU repair is attempted before the
  expensive exhaustive bank scan; only rejected rows escalate to that scan.
- Synthetic data interpolate normalized LP parameters from the existing
  offline anchors, not from benchmark rollouts. Only samples independently
  certified by the full bank provide classification labels. This is neither
  additional phenotype evidence nor a neural full-flux predictor.
- Pilot used 512 attempted examples per stage; 210/130/177 were certified.
  Three models contain 4,484/5,324/4,868 parameters. Synthetic validation
  accuracy is not evidence of dFBA accuracy; endpoint tests are still required.
- GPU teaching and training took 47.59 s, with no new CPU LP solves. Earlier
  CPU costs for constructing the anchor banks remain part of provenance.
- The first teaching run hit an optional CUB/header compilation failure.
  Rerun used per-process CUPY_ACCELERATORS='' (standard CuPy GPU kernels).
  No packages or installed CUDA headers were modified. Matched benchmarks
  must use the same setting; historical timings are not a controlled ablation.

Initial code tests passed (11 targeted checks), including deliberately wrong
neural proposals, nonfinite inputs, artifact identity, GPU repair and exact
secondary-LP certification. Full matched dFBA comparison is underway.

## Actual-trajectory training and ablation

Synthetic routing did not transfer sufficiently: an eight-environment,
one-step run passed all endpoint gates but was slower than four-worker CPU;
two steps still rejected two stage-2 LPs. Do not interpret synthetic top-1
accuracy as rollout performance.

Collected 32 independent training trajectories (seeds 20286701–20286732),
two steps each: 192 CPU LP calls, 110.74 s including preparation. Teaching
and neural training brought the total to 121.81 s. Each stage has 64 actual
LP examples. Entire trajectories, not adjacent steps, are split into
training/validation groups; benchmark seeds are separately excluded.
Uncertified examples label the best available repair start, NOT an accepted
flux. Stage-2 routing validation top-1 is 57.1%; full LP certification remains
necessary. The majority-class comparator is also saved in the manifest.

Engineering fixes shared by both ablation branches:

- Increase the bounded repair cache from four to eight entries: neural
  routing activated more than four bases, which otherwise caused repeated
  graph/solver construction even on repeated identical workloads.
- Remove a dual solve and transpose product inside each GPU pivot that was
  immediately overwritten after the feasibility/phase decision. Primal
  feasibility and final certificates are unchanged. 28 targeted tests pass.

Matched no-neural control (`pf_amn_ablation_plain_20260904.json`): eight
environments, one step each, same expanded bank and CUPY_ACCELERATORS setting.
Prepared CPU/GPU: 3.5633/3.6077 s. This is one prepared repetition, not a
significance claim. The actual-trajectory neural branch is evaluated next.

## Measured outcome

All comparisons below use eight environments, one 0.2 h step each, the same
three GEMs, the same expanded bank, four concurrent CPU HiGHS LP workers,
and CUPY_ACCELERATORS=''. Assembly, transfers, dispatch and state integration
are timed. Environment/bank preparation and training are separately reported.

| Prepared run | CPU seconds | GPU seconds | CPU / GPU |
| --- | ---: | ---: | ---: |
| Exhaustive-bank control (one prepared repeat) | 3.5633 | 3.6077 | 0.9877 |
| Constant first basis + same repair-first schedule | 3.6167 | 3.3766 | 1.0711 |
| Actual-data neural proposal, repeat 1 | 3.5595 | 3.1580 | 1.1271 |
| Actual-data neural proposal, repeat 2 | 3.6239 | 3.2309 | 1.1217 |

The constant-proposal control demonstrates that scheduling alone contributes
to the gain. Neural routing is modestly faster in these few timings, not a
statistically established advantage. More directly, aggregate-stage pivot
counts drop from 55 (constant basis) to 15 (learned bases), summed over the
same eight environments. Its prepared stage time drops from 0.4176 to
0.1990 s. This is evidence of better initialization on these inputs, not a
claim of generalization across all media or long-horizon trajectories.

Every one-step endpoint gate passed; online CPU LP calls were zero. Maximum
PHA relative error was 3.85e-13. Initial neural-run CPU/GPU times were
3.6050/4.5394 s: GPU is still slower when first-use costs are included.
Offline actual-data training cost was 121.81 s, including 192 CPU LP calls.

Two-step test (`pf_amn_actual8_step2_20260904.json`) still stops at the second
aggregate LP, with two of eight rows uncertified. All environments completed
one step only. Its GPU failure time 9.241 s is NOT comparable to the full
CPU two-step time 7.028 s as a successful speedup. Production training remains
on its previous default; no long validation or PPO training has been started.

### Additional source and structural audit

The authors' current `Library/Build_Model.py` explicitly hides GPU devices
from TensorFlow. Therefore copying that implementation is not evidence of
GPU speedup: https://raw.githubusercontent.com/brsynth/amn_release/main/Library/Build_Model.py
Our GPU inference/certification code is a separate adaptation.

A read-only exact sign-propagation audit of the cached LP equality/bound
systems identifies 201 zero-forced columns versus 11 explicitly fixed-zero
columns, leaving 6,533 columns in the first two LPs and 7,172 in stage 3.
This is only about a 3% reduction, not the dramatic network reduction one
might hope for. It is query-specific and has NOT been applied to GEMs or
assumed safe for other media. No reactions or gene associations were deleted.

Final targeted regression run: 29 passed (`pf_amn_final_tests_20260904.log`).
All probes and training jobs finished. Reproduction scripts, manifests,
artifacts and benchmark source snapshots remain in the workspace.

## Next priorities (not yet achieved)

1. Repair the remaining stage-2 nonconvergence on independent trajectories;
   train on actual transitions and GPU correction cost, not only the label
   of a numerically close basis. Preserve all acceptance thresholds.
2. Measure longer complete rollouts before estimating training throughput.
3. Reduce cache-shape churn and host environment work; both still cost time.
4. Investigate a learned primal/dual flux warm start with mechanistic loss
   as a separate, closer AMN-style model. The present classifier must not
   be described as that model or as fully device-resident dFBA.
