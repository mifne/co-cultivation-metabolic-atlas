# State-transition coverage and GPU repair plan

## Question and diagnosis

The previous router had only 64 LP examples per stage (32 trajectories x 2
steps). Increasing PPO timesteps does not train this separate initial-basis
network. Increase offline state coverage first, keeping the small network
until held-out repair cost justifies more capacity.

The failed second-step aggregate rows had original primal residuals about
1e-12, but dual violations 0.374/1.449 and large KKT gaps after exhausting
64 pivots. These are unresolved optimization problems, not evidence that
the biological constraints are infeasible. Do not relax acceptance gates.

## Work sequence

1. Expand independent training to 64 trajectories x 8 steps = 512 LPs/stage,
   1,536 offline CPU LP solves (8x the previous examples per stage).
2. Fix feature selection: choose varying inputs from training trajectories,
   not only the small anchor bank, with held-out trajectories excluded.
3. Add short alternative-basis GPU repairs before spending a long budget on
   one poor start; retain original constraints/certification and fail closed.
4. Compare two-step and longer complete rollouts with four-worker CPU;
   separate cold setup/offline training from prepared execution.
5. Record failures and costs as well as improvements. No PPO run or claim of
   full device-resident dFBA until the complete environment path is qualified.
6. Eight steps is an intermediate debugging interval, NOT the final horizon.
   Collect complete 120-step offline trajectories to cover late depletion and
   accumulation. Final accuracy/speed validation must use disjoint complete
   120-step trajectories; passing a short interval cannot substitute for this.

Status: implementation and experiments in progress. Evaluation seeds are
excluded from training. No GEM, objective or tolerance modifications planned.

## Interim experiments

- 64 x 8 training completed: 1,536 CPU LP calls; 692.11 s total while other
  diagnostic processes also ran. The aggregate bank directly certified only
  12/512 inputs. More classifier examples alone cannot add missing bases.
- A short alternative-basis portfolio did not resolve the final two rows;
  kept as an opt-in diagnostic, not enabled by default.
- Saved original normalized failed LPs for solver debugging only, never for
  training. A failed query converged in 163 Dantzig-priced pivots; a
  Devex-style approximate weighted rule required 71 pivots. This rules out
  interpreting this particular failure as biological infeasibility. It is
  not proof of cycling or of general speedup.
- Pricing changes do not change primal/dual/gap gates. The approximate
  implementation is NOT a full reproduction of HiGHS exact steepest-edge
  or Devex reference-reset machinery. Sources:
  https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html
  https://github.com/ERGO-Code/HiGHS/blob/master/docs/src/options/definitions.md
- Capture/replay tests caught and fixed an in-place state-update issue in
  the new edge weights. Eager and replayed weights/pivots now agree in tests.
- In the eight-environment diagnostic with 128 pivots, aggregate rejection
  fell from two rows to one. All 12 existing bases still failed to solve that
  remaining held-out LP within 128 pivots in a separate exhaustive probe.
  Training and diagnostics overlapped: none of these runtimes is a clean
  hardware speed comparison.
- Independent reserve dictionary: eight new seeds x four steps, 32 aggregate
  anchors. Only unresolved aggregate rows consult this wider dictionary.
- Full-horizon offline data: four new seeds x 120 steps (1,440 CPU LP calls),
  with normalized inputs cached for future relabeling without recollection.
  This is initial late-state coverage, not sufficient statistical validation.

## Memory and dataset integration

All four offline CPU trajectories completed 120 steps. The 64 x 8 dataset
and the 4 x 120 dataset have disjoint seeds. A joint trainer preserves each
dataset's trajectory-held-out splits and uses the intersection of recorded
feature IDs; missing features are never fabricated. The combined proposal
training set contains 992 LP examples per stage before splitting.

The 32-anchor reserve benchmark reached approximately 7.9 GB VRAM, with a
large execution slowdown. That benchmark was explicitly terminated and
marked interrupted; it is not a speed comparison. Physical VRAM was nearly
full; the specific transfer/paging mechanism was not profiled. The retained
offline dictionary is unchanged. A time-stratified load of just step-2
anchors (eight candidates, declared before the rerun) bounds device storage.
This is explicit configured loading, not automatic GPU-resident paging.

Regression tests: 36 passed, three expected warnings, before the joint-model
benchmark. New tests cover feature alignment, trajectory split preservation,
reserve/portfolio certification, infeasible rejection, and eager/replay edge
weight agreement. Scientific GEMs and production training defaults unchanged.

The remaining aggregate LP converged from a new reserve basis in 82 pivots.
The first reserve implementation discarded its partial 32-pivot progress;
selective warm continuation now retains it before returning to the old bank.
The joint early/long router was trained on 992 examples/stage, with held-out
top-1 accuracies 0.960/0.746/0.906; these are routing-label metrics, not flux
or trajectory accuracy.

An eight-reserve/eight-repair-cache run also approached 7.95 GB and was
terminated before returning a GPU timing. Explicit graph-cache release was
added on solver eviction to break solver/graph/warm-owner reference cycles.
Referenced warm-state arrays remain valid. A bounded three-solver repair
cache is tested next; cache capacity is not an acceptance-tolerance change.

## Final measured status for this iteration

Offline work completed:

| Dataset / artifact | Size | New offline CPU LP calls | Recorded elapsed seconds |
| --- | ---: | ---: | ---: |
| Independent early trajectories | 64 x 8 steps | 1,536 | 692.11 |
| Independent complete trajectories | 4 x 120 steps | 1,440 | 576.45 |
| Reserve aggregate dictionary | 8 x 4 steps / 32 bases | 96 | 280.63 |
| Joint routing model | 992 examples per LP stage | 0 | 3.22 |

Elapsed times include different preparation workloads, and some offline jobs
overlapped. Do not add them as wall-clock duration or compare them as speedups.
120 steps x 0.2 h = 24 h simulated culture per complete trajectory. This is
initial-basis model training, not PPO policy training. The saved 120-step GPU
input cache permits subsequent offline relabeling without rerunning dFBA.

Final isolated development regression:
`results/pf_amn_joint_reserve8_cache3_step2_20260904.json`.
Eight environments, two requested steps, four concurrent CPU LP workers,
96-pivot maximum, eight step-2 reserve bases and three cached repair solvers.
No concurrent training process ran during this benchmark.

- First-step four GPU LP stages passed all eight environments.
- Second-step maxmin AND aggregate LPs passed all eight environments.
  The prior aggregate failure is resolved on these development inputs.
  Second aggregate-stage time was 2.611 s including cold work.
- Second-step exchange LP then failed seven of eight rows. All environments
  therefore completed only ONE full step; no two-step endpoint certification.
- CPU completed its two steps in 7.206 s. GPU stopped after 17.689 s. These
  do NOT constitute a completed-work timing pair; no speedup is reported.
- Online CPU LP calls remained zero; failures stop the run. No PPO learning
  or complete 120-step GPU qualification has been started or claimed.
- These inputs have been used for development diagnosis, so future final
  qualification must also use fresh seeds, not only these regression seeds.

Final tests: 39 passed, three expected warnings,
`results/pf_amn_transition_final_tests_20260904.log`.
All jobs in this iteration finished or were explicitly terminated and marked.
No background training or validation job is left running.

## Updated next priorities

1. Apply long-horizon coverage/initialization and certified continuation to
   the exchange-minimization LP; its bank still contains only four bases.
   Investigate transporting the previous stage's feasible state across the
   added absolute-exchange variables/constraints rather than restarting far
   from feasibility. Do not equate objective agreement with exchange accuracy.
2. Bound dictionary and graph memory independently. Keep full-precision
   original constraint checks; never fill VRAM solely to demonstrate usage.
3. Retest complete 2/8-step rollouts, then a complete 120-step development
   rollout, then fresh independent 120-step accuracy and throughput runs.
4. Expand beyond four long trajectories if repair-rate learning curves show
   insufficient late-state coverage. Four trajectories provide late examples,
   not evidence that 120-step coverage is statistically sufficient.
