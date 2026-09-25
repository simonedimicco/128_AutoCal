# Report — WORK_01: Data Dictionary and .npz Schema

## Summary

Executed work package `Error_estimation/work_01.md` to freeze the exact schema of the `.npz` files and the semantics of every field for the Monte Carlo error-estimation pipeline.

## Outputs Produced

- `Error_estimation/data_dictionary.md` — Complete data dictionary document containing:
  - Schema and semantics for `moduliquadri_mc.npz` (singles pipeline)
  - Schema and semantics for `visibilities_mc.npz` (pairs pipeline)
  - Field-by-field documentation with types, shapes, and units
  - Visibility formula V = (y_int − area_c)/(y_int − y_noise_int)
  - Monte Carlo rebuild procedures for both pipelines
  - Sampling conventions (binding for all downstream phases)
  - Reconstruction core interface documentation

## Decisions Made

- Used explicit per-channel naming pattern (e.g., `singles_raw_b`, `dark_raw_c`) as the primary format, with a documented compact 4×128 array alternative.
- Documented channel order as `['b', 'c', 'd', 'e']` for both formats.
- Structured the visibility formula section to explicitly show symmetric assignment and zero-filling for skipped histograms.
- Included code blocks showing the exact Monte Carlo resampling logic for both singles and pairs pipelines.
- Added a dedicated "Sampling Conventions" section to satisfy the binding requirement for downstream phases.

## Acceptance Criteria Checklist

- [x] **`moduliquadri_mc.npz` schema fully specified**: Per channel c, accumulated raw Singles counts (128,), accumulated raw dark counts (128,), and the file count for each of the two accumulations; arrays kept per channel (4 separate entries per quantity, or a 4×128 array with documented channel order), never merged.
  **Verification**: Section 1 of `data_dictionary.md` contains complete field tables for both per-channel and compact formats, with explicit channel order documentation.

- [x] **`visibilities_mc.npz` schema fully specified**: For each of the 6 pairs and each histogram index i: ii, iii, y_int, sigma_y_int, y_noise_int, sigma_y_noise_int, area_c, valid flag (0 for skipped histograms).
  **Verification**: Section 2 of `data_dictionary.md` contains the complete field structure per pair, with all required fields documented.

- [x] **Units documented**: area_c, y_int, y_noise_int in counts·ps, no bin-width division anywhere; Poisson sampling of area_c uses λ = area_c directly.
  **Verification**: Units column in all tables explicitly states "counts·ps"; the Units Convention subsection in Section 2 explicitly states "no bin-width division anywhere" and "Poisson parameter is λ = area_c directly".

- [x] **Formula V documented**: V = (y_int − area_c)/(y_int − y_noise_int) with symmetric assignment V[ii,iii] = V[iii,ii] and skipped histograms left at 0.
  **Verification**: The "Visibility Formula" subsection in Section 2 contains the exact formula, explicitly states symmetric assignment, and specifies that skipped histograms are left at 0.

- [x] **MC rebuild of singles pipeline documented**: Poisson-sample the accumulated raw counts, divide by file count, subtract dark from singles, clip, normalize, stack, transpose.
  **Verification**: The "Usage in Monte Carlo Rebuild" subsection in Section 1 contains a complete Python code block showing all steps: Poisson sampling, division by file count, subtraction, clipping, normalization, stacking, and transposing.

## Files Modified/Created

| File | Action | Location |
|------|--------|----------|
| `data_dictionary.md` | Created | `Error_estimation/` |
| `report_work_01.md` | Created | Project root |

## Notes

No interface gaps identified. All required information from the work package was present and sufficient to produce the complete data dictionary. No deviations from the specified contracts were necessary.
