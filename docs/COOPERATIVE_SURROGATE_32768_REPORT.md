# 32,768-candidate cooperative FBA dictionary report

Date: 2026-09-01

## Why 32,768

32,768 (`2^15`) was selected as an engineering production tier, not derived as
a unique statistical sample-size solution. The hierarchy was defined as 1,024
smoke states, 8,192 pilot states, and four times the pilot density for the
production candidate set. The choice provides power-of-two GPU batching,
multiple stratified repeats of feed/oxygen/biomass/nutrient states, and a
complete three-GEM flux dictionary that fits in the 8 GB RTX 4060 Laptop VRAM.
It is always conditional on held-out accuracy gates.

## Exact-label construction

- Formulation: exact cooperative shared-medium HiGHS with parsimonious exchange
- Species: OR16, NS21 and WCFS1, aligned in every row
- Reactions: 6,585 total
- Context: 20,200 values per state
- Exact states: 32,768, all finite and all raw flux rows unique
- Varying context columns: 97
- Varying reaction fluxes: 2,828
- Rounded common-growth levels: 6,120
- Exact-label collection: 4,333.6 s (72.2 min), 8 CPU workers
- Host dataset: 3,511,788,071 bytes
- GPU artifact: 1,001.2 MiB

## Equal-action validation

| Dictionary | Evaluation | GPU-valid | Speedup | PHA relative error | VRAM allocated / peak |
|---:|---|---:|---:|---:|---:|
| 8,192 | seed 20260901 | 95.0% | 2.92x | 0.88% | 217.1 / 220.1 MiB |
| 32,768 | held-out seed 20260902 | 100.0% | 2.87x | 2.27% | 844.3 / 856.6 MiB |

For the held-out 32,768-state comparison, exact HiGHS required 70.79 s and the
GPU dictionary path required 24.70 s for 120 steps. Rubber remaining differed
by 0.00194 g/L (0.00197%), while PHA differed by 0.001157 g/L (2.27%). Physical
reaction-bound, shared-medium and growth guards accepted all 120 GPU states.

## Decision

The requested 32,768-candidate dictionary has been created and gives a measured
2.87x end-to-end speedup, but it is **not yet qualified as the production
scientific solver** because the predeclared 24 h PHA error gate is <=1%.
Increasing candidate count solved coverage/fallback but did not by itself solve
PHA accuracy around objective/limitation transitions. Exact HiGHS remains the
final evaluator. The next improvement should target active-learning states near
NH4/PHA switching and integrate periodic exact audits; blindly adding more
uniform candidates is not justified by this result.

## Artifacts

- Exact dataset: `models/cooperative_surrogate/production_32768/`
- GPU dictionary: `models/cooperative_surrogate/cooperative_dictionary_32768.pt`
- Dataset audit: `models/cooperative_surrogate/production_32768/audit.json`
- Held-out benchmark: `results/cooperative_surrogate_e2e_32768_heldout_20260902.json`
