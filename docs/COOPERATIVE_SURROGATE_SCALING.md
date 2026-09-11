# Cooperative FBA surrogate: size and validity requirements

## Finding

The previous default of 512 exact labels is too small for this three-member
community. The current WCFS1, OR16 and NS21 models contain 6,585 reactions in
total and their existing per-species LP feature vectors contain 19,758 values.
The more important problem is not size alone: the old collector labels each
species with an independent FBA solve (`fba_mode="separate"`), whereas the
validated RL environment uses a single cooperative shared-medium problem.
Those labels therefore must not be used to approximate the current RL model.

## Required representation

One dictionary row must represent one complete community state and contain all
three aligned flux vectors from the same exact cooperative solve. Its context
must include, at minimum:

- all reaction lower and upper bounds and active objective coefficients;
- biomass of all three members;
- shared extracellular inventories and the integration interval;
- common and species-specific feed controls, oxygen-transfer control and phase;
- the common-growth optimum and any PHA/rubber objective switch.

GPU selection must rank whole-community candidates, not select one candidate
independently for each species. A selected row is accepted only after checking
reaction bounds, `S v = 0`, every shared-medium inequality, and the retained
minimum-growth requirement. Rejected or out-of-domain states fall back to the
exact cooperative solver. Final scientific evaluation always uses exact HiGHS.

## Dataset tiers

| Tier | Aligned states | Approx. raw flux dictionary | Purpose |
|---|---:|---:|---|
| Smoke | 1,024 | 26 MB | integration and guard tests |
| Pilot | 8,192 | 206 MB | coverage/fallback measurement on RTX 4060 Laptop |
| Production | 32,768 | 823 MB | RL rollout acceleration after pilot acceptance |

The estimates store 6,585 float32 fluxes per state and exclude feature indexes,
temporary tensors and model overhead. They nevertheless fit within 8 GB VRAM;
batch size and top-k workspace must be measured rather than inferred from the
raw dictionary alone.

## Sampling design

Sampling must be stratified rather than drawn only from a random policy:

- Defined-10 common-feed scale 0--0.02, densely sampling 0.003--0.012;
- oxygen-transfer action across the full 0--200 range;
- early, production and depletion phases;
- initial biomass ratios perturbed by at least +/-20%;
- glucose, isoleucine, pyridoxamine and ammonium limitation boundaries;
- both cooperative-growth and PHA/rubber production objective modes.

Training-policy rollouts may be added by active learning, but exact audit states
and the held-out test set must not be reused as dictionary candidates.

## Acceptance gates

The pilot is acceptable only if all of the following hold on held-out states:

- at least 95% of states pass the GPU guards (99% for production);
- zero accepted states violate reaction, mass-balance or shared-medium limits;
- common growth and active production objectives have <=2% relative error;
- 24 h biomass, PHA and rubber outcomes differ by <=1% from exact integration;
- exact-solver fallback is <=5% for pilot and <=1% for production.

Candidate count is increased only after failures are classified. Constraint
failures require a representation/guard fix; out-of-domain failures require
targeted active-learning samples. Blindly increasing a scientifically mismatched
dictionary is not an acceptable speed optimization.

## Measured smoke scaling on RTX 4060 Laptop

Equal-action, 120-step (24 h at `dt=0.2 h`) rollouts were measured against the
exact cooperative HiGHS path on 2026-09-01:

| Candidates | GPU-valid steps | End-to-end speedup | 24 h PHA relative error |
|---:|---:|---:|---:|
| 128 | 33/120 (27.5%) | 1.17x | 0.06% |
| 1,024 | 120/120 (100%) | 3.03x | 2.30% |

The low error at 128 candidates is caused by exact CPU fallback on 72.5% of
steps, not by an accurate small dictionary. Increasing to 1,024 candidates
removed fallback and exposed the remaining approximation error, which exceeds
the predeclared 1% 24 h gate. Therefore 1,024 is a speed/integration smoke tier,
not a scientific production tier. The planned 8,192-candidate pilot is retained.

Raw comparisons are stored in
`results/cooperative_surrogate_e2e_120.json` and
`results/cooperative_surrogate_e2e_1024_120.json`. The corresponding figure and
table are in `outputs/cooperative_surrogate_scaling/`.
