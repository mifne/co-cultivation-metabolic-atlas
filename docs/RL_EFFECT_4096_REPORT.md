# Three-species dFBA–PPO training result

## Outcome

The final 4,096-step PPO policy passed the predeclared held-out effect gate in
five independently perturbed in-silico scenarios. Initial biomass was varied by
±5% and nonzero medium concentrations by ±2%. Every reported validation
trajectory was recomputed with the exact cooperative HiGHS formulation; the GPU
dictionary was not used to generate the final endpoint values.

| Endpoint (24 h) | PPO, mean ± SD | Fixed-25, mean ± SD | PPO / Fixed-25 |
|---|---:|---:|---:|
| Raw episode return | 174.42 ± 6.12 | −15.57 ± 0.47 | — |
| Rubber degraded (g/L) | 2.297 ± 0.028 | 1.735 ± 0.017 | 1.324 |
| PHA (mmol) | 0.424 ± 0.012 | 0.0555 ± 0.0009 | 7.645 |
| Defined medium (g/L) | 0.01489 ± 0.00018 | 0.01591 ± 0.00000 | 0.936 |
| Species-specific feeds (mmol/L) | 0.289 ± 0.013 | 9.075 ± 0.000 | 0.0318 |
| Minimum final/initial biomass ratio | 1.121 ± 0.001 | 1.122 ± 0.000 | 0.999 |
| Solver success rate | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 |

The paired return improvement over Fixed-25 was 189.99, with a two-sided 95%
confidence interval of [182.35, 197.62]. All five gate components passed:
hard culture/solver constraints, positive paired-return confidence interval,
noninferior PHA, noninferior rubber degradation, and no increase in defined
medium mass.

Relative to Fixed-25, the PPO policy increased simulated PHA 7.65-fold and
rubber degradation by 32.4%, while reducing defined-medium use by 6.36% and
species-specific supplementation by 96.8%.

## Training architecture and corrections

The first 2,048 timesteps used four parallel environments with the 32,768-entry
GPU cooperative-FBA dictionary and an exact solve every eighth transition. The
policy was then fine-tuned from 2,048 to 4,096 timesteps using exact cooperative
HiGHS transitions in four CPU-parallel environments.

PPO's continuous Gaussian policy was corrected by exposing a symmetric
policy-space action range of [−1, 1] and mapping it to physical pump/kLa
fractions in [0, 1]. The action head was initialized near the screened
low-feed/high-aeration rule with an initial standard deviation of about 0.30.
This removed the mismatch in which stochastic training actions were positive
but deterministic deployment actions collapsed toward zero.

The common defined-medium command has a deployment safety cap of 0.25, matching
the Fixed-25 resource budget. The cap acts only on the PPO common-feed output;
species-specific feeds, kLa, and all comparator policies retain their original
ranges. A penalty for commands above the same predeclared threshold is also
present during training.

## GPU finding

The GPU dictionary accelerated isolated cooperative-FBA lookup, but its
single-transition error accumulated over recurrent dFBA rollouts. At 4,096
hybrid-only timesteps, the approximate training reward continued to improve
while exact endpoint PHA deteriorated. Consequently, GPU dictionary output is
retained for candidate generation and warm-start exploration, but it is not
accepted as a substitute for exact multi-step training or endpoint validation.

On this laptop, four-environment hybrid training ran at about 6 timesteps/s,
whereas four-environment exact training ran at about 5 timesteps/s. The small
end-to-end gain reflects CPU-side context construction and repeated small GPU
queries. This result does not support claiming that the present recurrent
training architecture is GPU-bound; batched, device-resident state integration
is still required before multi-GPU scaling can be justified.

## Artifacts

- Final policy: `outputs/rl_wcfs1_exact_capped_validation/ppo_wcfs1_effect_final.zip`
- Observation normalizer: `outputs/rl_wcfs1_exact_capped_validation/vecnormalize_final.pkl`
- Full machine-readable result: `outputs/rl_wcfs1_exact_capped_validation/rl_effect_results.json`
- Per-scenario endpoints: `outputs/rl_wcfs1_exact_capped_validation/evaluation_scenarios.csv`
- Nominal action trajectory: `outputs/rl_wcfs1_exact_capped_validation/ppo_action_trajectory.csv`
- Publication figure: `outputs/rl_wcfs1_exact_capped_validation/Figure_rl_effect_wcfs1.pdf`

## Interpretation boundary

These results demonstrate an effect inside the current repaired three-GEM
dFBA model. They do not establish biological coexistence, polymer degradation,
or PHA yield experimentally. Exchange bounds, polymer-degradation kinetics,
maintenance demand, oxygen transfer, and product measurements require wet-lab
calibration before the learned controller can be transferred to the 1 L
fermenter.
