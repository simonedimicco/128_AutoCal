# WORK_04 — Monte Carlo resampling module

## Project context
The project is a Monte Carlo error-estimation pipeline for unitary reconstruction from histograms (quantum optics, 4 channels `['b','c','d','e']`, 128 modes). The statistical inputs (raw counts, histogram-derived quantities with uncertainties) are saved by the data-preparation phase into two `.npz` files whose schema is frozen in the data dictionary. This phase implements the statistical layer: given those files and a seed, produce one fake realization of the inputs {T, VV}, ready to be fed to the reconstruction core by the final Monte Carlo phase.

## Objective
Implement the sampling of fake realizations of the inputs.

## Inputs
- `data_dictionary.md` (schema of `moduliquadri_mc.npz` and `visibilities_mc.npz`, semantics, units, MC rebuild of the singles pipeline).
- The two `.npz` files (`moduliquadri_mc.npz`, `visibilities_mc.npz`) produced by the previous saving phase.

## Binding domain facts (copied from the plan)
- Channels: C = 4, labels `['b','c','d','e']`; modes: 128; T is 128×4; VV is a list of 6 matrices 128×128, ordered by pair label `bc, bd, be, cd, ce, de`.
- Singles pipeline (real): per channel c, dark counts B(c) and singles M_raw(c) accumulated over the channel's files (np.bincount of the timestamp arrays, one division by file count at the end); then `M = M_raw − B`, clip negatives to 0, normalize M to sum 1, stack per channel and transpose → T (128×4).
- Visibility: V = (y_int − area_c)/(y_int − y_noise_int), assigned symmetrically V[ii,iii] = V[iii,ii], diagonal 0; skipped histograms have valid flag 0.
- Sampling conventions (binding): B, M and area_c drawn from Poisson with expected value equal to the observed value, no bin-width rescaling (all quantities are on the same counts·ps scale; V is a ratio, any common factor cancels); y_int and y_noise_int drawn from Gaussians with sigma equal to the already-computed intersection errors (sigma_y_int, sigma_y_noise_int).

## Interfaces to respect
- Input contract: the module consumes the two `.npz` files exactly as specified in the data dictionary (identical schema and field semantics).
- Output contract: a function that, given the `.npz` files and a seed, returns one realization of {T, VV} with the same shapes and semantics as the real ones (T 128×4, VV list of 6 matrices 128×128 in pair order). This contract is consumed by the Monte Carlo ensemble phase; do not change it.
- Documented MC rebuild of the singles pipeline (identical to the data dictionary): Poisson-sample the accumulated raw counts, divide by file count, subtract dark from singles, clip, normalize, stack, transpose.
- MC rebuild of the pairs: y_int ~ N(y_int, sigma_y_int), y_noise_int ~ N(y_noise_int, sigma_y_noise_int), area_c ~ Poisson(area_c) with no rescaling; recompute V = (y_int − area_c)/(y_int − y_noise_int); respect the valid flag (skipped cells stay 0); fill V[ii,iii] and V[iii,ii].

## Acceptance criteria (verify each one)
- Singles: Poisson-sample the accumulated raw counts per channel (singles and dark separately, each channel independently), then apply the pipeline identical to the real one (divide by file count, subtract, clip, normalize, stack, transpose).
- Pairs: y_int ~ N(y_int, sigma_y_int), y_noise_int ~ N(y_noise_int, sigma_y_noise_int), area_c ~ Poisson(area_c) with no rescaling; recompute V = (y_int − area_c)/(y_int − y_noise_int); respect the valid flag (skipped cells stay 0); fill V[ii,iii] and V[iii,ii].
- Reproducible given a seed (fixed-seed test).
- Statistical test: mean and standard deviation of sampled quantities converge to the real values/sigmas over ~10⁴ samples.

## Expected outputs
- A module (importable) that, given the two `.npz` files and a seed, returns one realization of {T, VV}.

## Final instruction
Verify every acceptance criterion before finishing. If a criterion cannot be met, stop and report why instead of improvising. If the statistical convergence test fails, report the measured deviations instead of loosening the tolerances silently.