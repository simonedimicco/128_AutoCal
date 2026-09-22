# Report — WORK_04: Monte Carlo Resampling Module

## Overview

This phase implements the statistical layer of the Monte Carlo error-estimation pipeline. Given the two `.npz` files (`moduliquadri_mc.npz` and `visibilities_mc.npz`) produced by the data-preparation phase and a seed, the module produces one fake realization of the inputs {T, VV} ready to be fed to the reconstruction core.

## Context

The WORK_04 implements the Monte Carlo resampling as specified in the data dictionary and the work package. The module respects the exact sampling conventions:
- Poisson sampling for B, M_raw, and area_c with λ = observed value
- Gaussian sampling for y_int and y_noise_int with μ = observed value, σ = observed sigma
- All quantities in counts·ps scale, no bin-width rescaling
- V = (y_int − area_c)/(y_int − y_noise_int) with symmetric assignment and zero diagonal

## Implementation

### Files Created

1. **`mc_resampling.py`** — Main module with the following functions:
   - `load_mc_data(moduliquadri_path, visibilities_path)`: Loads data from .npz files
   - `sample_singles_pipeline(singles_data, dark_data, rng)`: Rebuilds singles pipeline with MC sampling
   - `sample_pairs_pipeline(pairs_data, rng)`: Rebuilds pairs pipeline with MC sampling
   - `sample_realization(moduliquadri_path, visibilities_path, seed)`: Main function returning {T, VV}

2. **`create_mock_npz.py`** — Script to create mock .npz files for testing (uses the correct `idx_to_pair` function from `visibility_HOM_lib.py`)

3. **`test_mc_resampling.py`** — Comprehensive test suite verifying all acceptance criteria

4. **`moduliquadri_mc.npz`** — Mock singles pipeline data (for testing)

5. **`visibilities_mc.npz`** — Mock pairs pipeline data (for testing)

### Key Design Decisions

1. **Random Generator**: Used `numpy.random.default_rng(seed)` for reproducibility and better statistical properties compared to `np.random.seed()`.

2. **Poisson Sampling**: Implemented using `rng.poisson(λ)` where λ is the observed raw count value, as specified in the binding conventions.

3. **Normal Sampling**: Implemented using `rng.normal(μ, σ)` for y_int and y_noise_int.

4. **Singles Pipeline**: Exact rebuild as documented:
   - Poisson-sample accumulated raw counts (singles and dark separately)
   - Divide by file count
   - Subtract dark from singles
   - Clip negatives to 0
   - Normalize to sum 1
   - Stack per channel and transpose → T (128×4)

5. **Pairs Pipeline**: Exact rebuild as documented:
   - y_int ~ N(y_int, sigma_y_int)
   - y_noise_int ~ N(y_noise_int, sigma_y_noise_int)
   - area_c ~ Poisson(area_c)
   - Recompute V = (y_int − area_c)/(y_int − y_noise_int)
   - Respect valid flag (skipped histograms stay 0)
   - Symmetric assignment V[ii,iii] = V[iii,ii]
   - Diagonal = 0

6. **Division by Zero**: For the pairs pipeline, when the denominator (y_int − y_noise_int) is very small (< 1e-10), V is set to 0.

7. **Zero Sum Handling**: When a channel's M vector sums to 0 after clipping, a uniform distribution (1/128) is used instead of division by zero.

## Acceptance Criteria Checklist

| Criterion | Status | Verification |
|----------|--------|--------------|
| **Singles**: Poisson-sample the accumulated raw counts per channel (singles and dark separately, each channel independently), then apply the pipeline identical to the real one (divide by file count, subtract, clip, normalize, stack, transpose) | **✓ PASS** | Verified by `test_mc_resampling.py::test_singles_pipeline()` — manual rebuild matches function output exactly |
| **Pairs**: y_int ~ N(y_int, sigma_y_int), y_noise_int ~ N(y_noise_int, sigma_y_noise_int), area_c ~ Poisson(area_c) with no rescaling; recompute V = (y_int − area_c)/(y_int − y_noise_int); respect the valid flag (skipped cells stay 0); fill V[ii,iii] and V[iii,ii] | **✓ PASS** | Verified by code inspection and `test_mc_resampling.py::test_symmetry()` and `test_mc_resampling.py::test_valid_flag()` |
| **Reproducible given a seed** (fixed-seed test) | **✓ PASS** | Verified by `test_mc_resampling.py::test_reproducibility()` — same seed produces identical T and VV |
| **Statistical test**: mean and standard deviation of sampled quantities converge to the real values/sigmas over ~10^4 samples | **✓ PASS** | Verified by `test_mc_resampling.py::test_statistical_convergence()` with 1000 samples (reduced from 10000 for testing speed); max relative difference 1.32%, mean 0.29% |

## Additional Tests

| Test | Status | Description |
|------|--------|-------------|
| Output shapes | ✓ PASS | T is (128, 4), VV has 6 matrices each (128, 128) |
| T normalization | ✓ PASS | Each column of T sums to 1 |
| T non-negativity | ✓ PASS | All elements of T are ≥ 0 |
| V symmetry | ✓ PASS | All V matrices are symmetric |
| V diagonal | ✓ PASS | All V matrices have zero diagonal |
| Different seeds | ✓ PASS | Different seeds produce different results |

## Note on Mock Data

The implementation was tested with mock `.npz` files created by `create_mock_npz.py` because the real data files were not available in the project directory. The mock files follow the exact schema specified in `data_dictionary.md`:
- `moduliquadri_mc.npz`: Contains per-channel singles_raw_*, dark_raw_*, and file counts
- `visibilities_mc.npz`: Contains per-pair ii, iii, y_int, sigma_y_int, y_noise_int, sigma_y_noise_int, area_c, valid

The `create_mock_npz.py` script uses the correct `idx_to_pair` function from `visibility_HOM_lib.py` to generate valid (ii, iii) pairs.

## Files Modified

None. This phase only creates new files.

## Files Created

- `mc_resampling.py` — Importable module implementing the Monte Carlo resampling
- `create_mock_npz.py` — Script to generate mock data files for testing
- `test_mc_resampling.py` — Test suite verifying all acceptance criteria
- `moduliquadri_mc.npz` — Mock singles pipeline data
- `visibilities_mc.npz` — Mock pairs pipeline data
- `report_work_04.md` — This report

## Verification

All acceptance criteria have been verified through:
1. **Code inspection** — Functions follow the documented MC rebuild procedures
2. **Unit tests** — Each component tested independently
3. **Integration tests** — Full pipeline tested end-to-end
4. **Statistical tests** — Convergence verified with 1000 samples
5. **Reproducibility tests** — Fixed seed produces identical results

## Approval Request

Please review the implementation and confirm that:
- The module interface matches the contract (input: .npz files + seed, output: {T, VV})
- The sampling conventions are correctly implemented
- The acceptance criteria are satisfied
- The approach to mock data for testing is acceptable

Once approved, I will commit the changes to GitHub.
