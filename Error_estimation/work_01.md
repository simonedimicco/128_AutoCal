# WORK_01 — Data dictionary and .npz schema

## Project context
The project is a Monte Carlo error-estimation pipeline for unitary reconstruction from histograms (quantum optics, 4 channels, 128 modes). The overall goal is to extend an existing data-preparation pipeline so it saves all statistical inputs (raw counts, histogram-derived quantities with uncertainties), then build a separate Monte Carlo program that generates N realizations of the reconstructed unitary U by resampling the data and reusing the existing reconstruction code unchanged. This phase freezes the data contracts every later phase must respect, so that different agents can work with isolated context.

## Objective
Freeze the exact schema of the `.npz` files and the semantics of every field.

## Inputs
- The domain facts below (the only input; no previous phase exists yet).

## Binding domain facts (copied from the plan)
- Channels: C = 4, labels `['b','c','d','e']`; modes: 128. Unitaries are 128×4.
- Singles pipeline, per channel c: dark counts B(c) accumulated over all files in `Ricostruzione_unitaria/Buio/<c>/` (each file contains a timestamp array `c_tot`; accumulate `B += np.bincount(c_tot, minlength=128)`; divide by file count once, at the end). Singles counts M_raw(c): same accumulation over `Ricostruzione_unitaria/Singles/<c>/`, same single division by file count. Then `M = M_raw − B`; clip negatives to 0; normalize M to sum 1; stack per channel and transpose → T (128×4), the squared-modulus matrix. Each channel is an independent input source, so B and M must be handled separately per channel.
- Pairs pipeline, per pair label `bc, bd, be, cd, ce, de` (6 pairs, ordered): input one `.npz` per pair with arrays `hist_totals` (list of 2D histograms) and `bin_edges`. For each histogram index i, decode the mode pair with the triangular-index map: given n=128 and linear index iii, `i = int((2n − 1 − sqrt((2n−1)² − 8·iii)) // 2)`, then adjust with the two while-loops on the offsets `i*n − i(i+1)/2` until the index falls in the correct block, and `ii = iii − offset + i + 1`, yielding (ii, iii) with ii < iii.
- Peak finding (bin centers, min peak distance 7 bins): left peaks, right peaks, one central peak (closest to x=0), noise positions midway between adjacent peaks per side. If fewer than 2 left peaks, or fewer than 2 right peaks, or no central peak, the histogram is skipped (V entries stay 0, valid flag 0).
- Peak integration: symmetric window of 1800 ps, trapezoid rule → peak areas; central peak same way → area_c (counts·ps, no normalization).
- Linear fit (y = m·x + q) of left-peak and right-peak areas vs position, each returning coefficients and covariance; same for the two noise sequences. Intersections give (x_int, y_int) and (x_noise_int, y_noise_int); sigma_y_int and sigma_y_noise_int from standard propagation of the fit covariances through the intersection formula (full covariance terms included).
- Visibility: d_int = y_int − area_c; V = (y_int − area_c)/(y_int − y_noise_int); assign V[ii,iii] = V[iii,ii] (symmetric matrix, diagonal 0). One 128×128 V matrix per pair, stacked in order into the list VV.
- Reconstruction core (from `Ricostruzione_unitaria_histo_numba.py`, numba `@njit`): input `t = sqrt(T)` (128×4), VV (list of 6 matrices 128×128, ordered by pair), a 4×4 map com_index with `com_index[h,k]` = index in VV for channel pair h<k (−1 elsewhere), Vf=0.78, pivots i1=79, i2=0. It computes the phase matrix FF (128×4) by arccos of a visibility-derived ratio (clipped to [−1,1]) with sign fixing via the two pivot rows, then `U = t · exp(i·FF)` (128×4 complex). No I/O inside the core.
- Storage format: `.npz` (compressed numpy) in the same folder as the current outputs.

## Interfaces to respect (this phase defines them; state them exactly as below)
- Schema of `moduliquadri_mc.npz` (singles): for each of the 4 channels c: accumulated raw Singles counts (128,), accumulated raw dark counts (128,), and the file count for each of the two accumulations. Arrays kept per channel (4 separate entries per quantity, or a 4×128 array with a documented channel order), never merged: each channel is an independent statistical source.
- Schema of `visibilities_mc.npz` (pairs): for each of the 6 pairs and each histogram index i: ii, iii, y_int, sigma_y_int, y_noise_int, sigma_y_noise_int, area_c, valid flag (0 for skipped histograms).
- Units documented: area_c, y_int, y_noise_int all in counts·ps, no bin-width division anywhere; Poisson sampling of area_c uses λ = area_c directly.
- Documented formula V = (y_int − area_c)/(y_int − y_noise_int) with symmetric assignment V[ii,iii] = V[iii,ii] and skipped histograms left at 0.
- Documented MC rebuild of the singles pipeline: Poisson-sample the accumulated raw counts, divide by file count, subtract dark from singles, clip, normalize, stack, transpose.
- Sampling conventions (binding for all downstream phases): B, M and area_c drawn from Poisson with expected value equal to the observed value, no bin-width rescaling (all quantities are on the same counts·ps scale; V is a ratio, any common factor cancels); y_int and y_noise_int drawn from Gaussians with sigma equal to the already-computed intersection errors. N realizations configurable at runtime (default 100).

## Acceptance criteria (verify each one)
- The `moduliquadri_mc.npz` schema is fully specified: per channel c, accumulated raw Singles counts (128,), accumulated raw dark counts (128,), and the file count for each of the two accumulations; arrays kept per channel (4 separate entries per quantity, or a 4×128 array with documented channel order), never merged.
- The `visibilities_mc.npz` schema is fully specified: for each of the 6 pairs and each histogram index i: ii, iii, y_int, sigma_y_int, y_noise_int, sigma_y_noise_int, area_c, valid flag (0 for skipped histograms).
- Units are documented: area_c, y_int, y_noise_int in counts·ps, no bin-width division anywhere; Poisson sampling of area_c uses λ = area_c directly.
- The formula V = (y_int − area_c)/(y_int − y_noise_int) is documented with symmetric assignment V[ii,iii] = V[iii,ii] and skipped histograms left at 0.
- The MC rebuild of the singles pipeline is documented: Poisson-sample the accumulated raw counts, divide by file count, subtract dark from singles, clip, normalize, stack, transpose.

## Expected outputs
- A data dictionary document (Markdown) named `data_dictionary.md`, in the same folder as this work package, containing both `.npz` schemas, field semantics, units, the V formula, and the MC rebuild of the singles pipeline.

## Final instruction
Verify every acceptance criterion before finishing. If a criterion cannot be met, stop and report why instead of improvising. Do not add fields, formats, or conventions not listed above; if something seems missing, flag it as an interface gap.