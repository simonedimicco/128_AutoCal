# Report — WORK_05: Monte Carlo Program for the U Ensemble

## Overview

This phase implements the final Monte Carlo error-estimation pipeline as specified in WORK_05. It chains the resampling module (WORK_04) and the reconstruction core (WORK_03) to generate N fake U matrices, and saves them together with the real U in a single `.npz` file.

## Objective

Generate N fake U matrices by chaining the resampling module and the reconstruction core, and save them in a single file.

## Implementation

### Files Created

1. **`monte_carlo_ensemble.py`** — Main executable script that implements the Monte Carlo ensemble generation
2. **`ensemble_U.npz`** — Output file containing the ensemble of U matrices with metadata

### Key Design Decisions

1. **Script Structure**: Created as an executable Python script with configurable parameters via command-line arguments (N, seed, Vf, i1, i2, output path)

2. **Module Integration**: 
   - Imports `mc_resampling.sample_realization()` for generating fake {T, VV} realizations
   - Imports `reconstruction_core.reconstruct_U()` for computing U matrices
   - No code duplication: all reconstruction logic is in the imported module

3. **Seed Management**: Per-realization seeds are derived from the base seed using `realization_seed = base_seed + i`, ensuring independent but reproducible realizations

4. **Data Paths**:
   - MC data files (`moduliquadri_mc.npz`, `visibilities_mc.npz`) are loaded from the current directory (Error_estimation)
   - Real data files (`moduliquadri.npz`, `visibilities_from_histogram.npz`) are loaded from the data-preparation pipeline output at `/media/dati_2/DATI_2026_06_21_128modi_training_target6N_3PairsPre_32Start_2/Ricostruzione_unitaria/`

5. **Output Format**: The ensemble file contains:
   - `U_real`: The real unitary matrix computed from non-resampled data
   - `U_1` to `U_N`: N fake unitary matrices from resampled data
   - Metadata: `base_seed`, `N`, `timestamp`, `Vf`, `i1`, `i2`

## Acceptance Criteria Checklist

| Criterion | Status | Verification |
|----------|--------|--------------|
| **Every U_i uses the reconstruction core module, with no copies of the logic** | **✓ PASS** | Code inspection: The script imports and calls `reconstruction_core.reconstruct_U()` for each realization; no reconstruction logic is duplicated in the script |
| **Independent realizations: per-realization seed derived from the base seed** | **✓ PASS** | Code inspection: Each realization uses `seed = base_seed + i` where i ranges from 0 to N-1, ensuring independent but reproducible realizations |
| **The ensemble file exists, is readable, U shapes verified at runtime (128×4, complex)** | **✓ PASS** | Verified by execution: `ensemble_U.npz` was created with U_real shape (128, 4) complex128, and all 100 U_i matrices have shape (128, 4) complex128 |
| **The included real U is identical to the one produced with non-resampled data** | **✓ PASS** | Verified by comparison: U_real computed from real T and VV matches the original U from `Unitary_mat.npz` with max absolute difference = 0.0 (machine tolerance) |
| **Execution time with N=100 measured and reported (numba compilation paid once)** | **✓ PASS** | Measured: 6.92 seconds for N=100, average 0.069 seconds per realization; numba JIT compilation happens once on first call |

## Verification Details

### Test Run with N=5
- Successfully generated 5 U matrices
- Execution time: 1.83 seconds
- All acceptance criteria verified

### Test Run with N=100
- Successfully generated 100 U matrices
- Execution time: 6.92 seconds (0.069 seconds per realization after JIT compilation)
- Output file: `ensemble_U.npz` (107 entries: U_real, 100 U_i, 6 metadata fields)
- All U matrices have correct shape (128, 4) and dtype (complex128)

### Real U Verification
The real U matrix computed by our script was compared against the original U from `Unitary_mat.npz` (produced by the original reconstruction pipeline):
- Max absolute difference: 0.0
- Mean absolute difference: 0.0
- Result: **Identical within machine tolerance**

## Files Modified

None. This phase only creates new files as specified in the work package.

## Files Created

1. `monte_carlo_ensemble.py` — Executable Monte Carlo ensemble generation script
2. `ensemble_U.npz` — Ensemble file with U_real, 100 U_i matrices, and metadata
3. `report_work_05.md` — This report

## Usage

The script can be run with customizable parameters:

```bash
# Default: N=100, seed=42, output to ensemble_U.npz
python monte_carlo_ensemble.py

# Custom parameters
python monte_carlo_ensemble.py --N 50 --seed 123 --output my_ensemble.npz

# Full parameter list
python monte_carlo_ensemble.py --help
```

## Verification Summary

All five acceptance criteria have been verified:
1. ✓ Uses reconstruction core module without code duplication
2. ✓ Independent realizations with derived seeds
3. ✓ Ensemble file exists, readable, with correct shapes
4. ✓ Real U matches original (max diff = 0.0)
5. ✓ Execution time measured and reported

## Approval Request

Please review the implementation and confirm that:
- The script correctly chains the resampling and reconstruction modules
- The acceptance criteria are satisfied
- The output format (single .npz with U_1..U_N, U_real, metadata) is acceptable
- The approach to referencing real data files from the data directory is acceptable

Once approved, I will commit the changes to GitHub.
