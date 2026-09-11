# Host-side LP preparation: equivalence and local timing

Date: 2026-09-04. These measurements are **CPU-only microprofiles**, not completed dFBA/RL runs and not evidence that the GPU backend exceeds the CPU comparator.

## Implementation scope

Only two existing methods changed:

- `CommunityCoordinates.normalize` in `src/gpu_certified_basis.py`: canonical, finite, nonzero CSR matrices are scaled directly through their data arrays. Multiplication order remains `row_scale * value`, then multiplication by the already computed reciprocal column scale. CSR indices/indptr are copied, not aliased to the caller. Explicit zeros, noncanonical storage, nonfinite entries, or a zero/nonfinite scaled result retain the original `diags(row_scale) @ A @ diags(1/col_scale)` implementation. No reaction, variable, row, RHS, objective, or bound is removed.
- `CompactBank.prepare_host` in `src/gpu_compact_basis.py`: when the current and reference matrices are canonical CSR with exactly equal indptr/indices, compare every numerical coefficient through their data arrays. All fixed-row differences are still checked against `2e-12`. Variable-row differences are scattered into the original dense delta layout. Reference layout metadata is cached, but the current reference indices/indptr and variable-row list are checked before reusing that cache. Different patterns/noncanonical matrices use the original sparse difference and row-slicing implementation.

The normalization change is shared by CPU and GPU callers. In particular, the CPU dictionary-initialized comparator receives the same optimization; it is not withheld to manufacture a GPU advantage.

### Intentional safety tightening

The old fixed-row check could miss NaN: `max(abs(change)) > tolerance` evaluates false when the maximum is NaN. Since only variable-row deltas are subsequently copied to the GPU, a NaN in a supposedly fixed row could otherwise disappear from the GPU input representation. Both preparation paths now explicitly reject nonfinite reference/current coefficients and nonfinite computed differences. This is an intentional fail-closed change for invalid inputs; normal finite inputs remain numerically equivalent.

## Tests

`tests/test_compact_host_preparation.py` contains the previous implementations as regression oracles. Tests compare all CSR data/indices/indptr, RHS, lower/upper bounds, objective, row/column scales, and prepared deltas by exact array equality.

Coverage includes:

- float64, float32-to-float64, and integer-to-float64 inputs; nontrivial positive scales;
- explicit zeros, underflow, duplicate entries, unsorted storage, and input nonmutation;
- zero/negative/NaN/infinite biomass rejection;
- unchanged and changed CSR patterns, a changed reference pattern, variable-row ordering, duplicate variable-row fallback;
- forbidden fixed-row value/new-column changes and shape/equality-count changes;
- NaN/Inf in fixed rows, nonfinite reference values, and overflowed finite-coefficient subtraction;
- CPU and GPU copies of all prepared input fields.

The following combined run passed **51 tests, zero skips**, in 2.35 s:

```text
CUPY_ACCELERATORS= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-cuopt-26.8/bin/python -m pytest \
  tests/test_compact_host_preparation.py \
  tests/test_gpu_compact_basis.py \
  tests/test_gpu_heterogeneous_compact.py \
  tests/test_cpu_dictionary_lp.py -q
```

## Saved real-matrix equivalence and microprofile

Matrices were loaded from `results/pf_compact_prefix4x4_20260904/{maxmin,aggregate,exchange}/root.npz`. No GEM, dictionary, or benchmark data were modified.

This is a **numerical stress test on saved real sparse structures**, not a reconstruction of a physical culture trajectory. Three positive synthetic scale values, 0.37, 1.29, and 2.71, were used through the production `normalize` method. Column and equality-row scale assignments cycled through these three values for this numerical test; these assignments are not biological species annotations. RHS, finite bounds, and costs were nontrivial arrays. Variable-row perturbations of `1e-4` were used for preparation tests.

For all three saved matrices:

- normalized CSR data, indices, indptr and all six vectors were exactly equal to the old implementation;
- every prepared field, including delta, was exactly equal to the old implementation;
- fixed-row guards remained active.

Each timing sample processed 32 matrices (normalization) or a 32-problem batch (preparation). Five samples were taken, with one CPU thread per OpenMP/OpenBLAS setting. Preparation used a warmed layout cache. The production method was timed, including its added finite checks. Preparation included NumPy stacking of all vector fields, but **excluded GPU transfer** by using NumPy as the test array namespace. LP solving, simulator state updates, and graph/kernel execution were not measured.

| Stage | Shape; nnz | Normalize old / new, median ms per 32 | Prepare old / new, median ms per batch |
|---|---|---:|---:|
| maxmin | 5051 × 6734; 28,941 | 41.864 / 12.365 | 14.672 / 6.863 |
| aggregate | 5051 × 6734; 28,941 | 38.594 / 12.907 | 13.482 / 6.032 |
| exchange | 6330 × 7373; 31,501 | 42.433 / 13.130 | 12.181 / 4.071 |

Raw samples in milliseconds:

```text
maxmin normalize old: 42.929904, 41.967117, 41.850898, 41.864075, 41.530406
maxmin normalize new: 13.131037, 12.611374, 12.155188, 12.336211, 12.364932
maxmin prepare old:   15.409358, 14.671872, 14.095429, 14.465961, 14.925191
maxmin prepare new:    6.803671,  6.701695,  6.863090,  7.074466,  7.182969
aggregate normalize old: 38.798442, 38.593657, 38.936320, 38.406534, 37.997135
aggregate normalize new: 12.711961, 12.906549, 12.579029, 14.307248, 16.654810
aggregate prepare old:   15.468553, 13.481625, 13.847724, 13.437008, 13.431216
aggregate prepare new:    6.256437,  6.101412,  5.909934,  5.830293,  6.031861
exchange normalize old: 42.809880, 42.432851, 44.950417, 41.201724, 40.648922
exchange normalize new: 14.059099, 13.385296, 13.004785, 13.129633, 13.039063
exchange prepare old:   15.630391, 12.180794, 12.011170, 11.939489, 12.317166
exchange prepare new:    4.070944,  4.086104,  3.837114,  4.082052,  3.840817
```

The arithmetic sum of these local median differences, repeated over eight three-stage time steps, is approximately 0.86 s. This is only a microprofile extrapolation: the CPU reference also benefits from normalization, real input distributions and allocations differ, and the complete end-to-end speed difference remains to be measured on new seeds.

## Provenance

SHA-256 at measurement:

```text
src/gpu_certified_basis.py
4b523eec902be5f2623a839d5c17f672114ec5aadf9beba1b3d47b07942bb05e
src/gpu_compact_basis.py
b1a9648075a4327b088a75e857223723699f4f52c79af7a63ce1e1dcc319b5cc
maxmin/root.npz and aggregate/root.npz
aed2ce2760dac0325f46f2199cae1a25764ec9c781c47b7a835d4ab08f6df8e2
exchange/root.npz
047636b65a5e4631771e4bf249648cd960d4b8cbaccf858f6f1c1fb9f7d82cf4
```
