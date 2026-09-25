# WORK_02 — Saving of the statistical inputs in preparation

## Project context
The project is a Monte Carlo error-estimation pipeline for unitary reconstruction from histograms (quantum optics, 4 channels `['b','c','d','e']`, 128 modes). The goal is to propagate errors by resampling the statistical inputs and regenerating N realizations of the reconstructed unitary U. The data contracts (`.npz` schemas, semantics, units) have been frozen in a data dictionary produced by a previous phase; this phase extends the existing data-preparation script to save all statistical inputs into those files, without changing any existing numerical output.

## Objective
Add the `.npz` saving of all statistical inputs to `preparazione_dati_histo.py`, without changing any existing numerical output.

## Inputs
- `data_dictionary.md` (output of the previous phase: schema of `moduliquadri_mc.npz` and `visibilities_mc.npz`, field semantics, units, V formula, MC rebuild of the singles pipeline).
- The current `preparazione_dati_histo.py` script and its input data.

## Binding domain facts (copied from the plan)
- Channels: C = 4, labels `['b','c','d','e']`; modes: 128.
- Singles pipeline, per channel c: dark counts B(c) accumulated over all files in `Ricostruzione_unitaria/Buio/<c>/` (each file contains a timestamp array `c_tot`; accumulate `B += np.bincount(c_tot, minlength=128)`; divide by file count once, at the end). Singles counts M_raw(c): same accumulation over `Ricostruzione_unitaria/Singles/<c>/`, same single division. Then `M = M_raw − B`; clip negatives to 0; normalize M to sum 1; stack per channel and transpose → T (128×4). Each channel is an independent statistical source, so B and M must be saved separately per channel.
- Pairs pipeline, per ordered pair label `bc, bd, be, cd, ce, de` (6 pairs): input one `.npz` per pair with arrays `hist_totals` and `bin_edges`; triangular-index decode of (ii, iii); peak finding (min peak distance 7 bins); integration of peaks over a symmetric 1800 ps window (trapezoid rule) → areas and area_c (counts·ps, no normalization); linear fits with covariances; intersections → (x_int, y_int) and (x_noise_int, y_noise_int) with errors sigma_y_int and sigma_y_noise_int from covariance propagation.
- Visibility: V = (y_int − area_c)/(y_int − y_noise_int), assigned symmetrically V[ii,iii] = V[iii,ii], diagonal 0; skipped histograms (fewer than 2 left peaks, or fewer than 2 right peaks, or no central peak) have V entries 0 and valid flag 0.
- Storage format: `.npz` (compressed numpy) in the same folder as the current outputs.

## Interfaces to respect
- `moduliquadri_mc.npz` schema (identical to the data dictionary): for each of the 4 channels c: accumulated raw Singles counts (128,), accumulated raw dark counts (128,), and the file count for each of the two accumulations; arrays kept per channel (4 separate entries per quantity, or a 4×128 array with documented channel order), never merged.
- `visibilities_mc.npz` schema (identical to the data dictionary): for each of the 6 pairs and each histogram index i: ii, iii, y_int, sigma_y_int, y_noise_int, sigma_y_noise_int, area_c, valid flag (0 for skipped histograms).
- Units: area_c, y_int, y_noise_int in counts·ps, no bin-width division anywhere.
- Existing outputs (T, VV) must remain unchanged.

## Acceptance criteria (verify each one)
- The two `.npz` files are produced with fields and shapes conforming to the data dictionary schema (runtime asserts).
- B and M raw counts are saved separately for each channel c, as accumulated sums over that channel's files, together with the per-channel file counts.
- The sigmas saved per histogram are exactly the intersection errors already computed by the script (`sigma_y_int`, `sigma_y_noise_int`).
- Skipped histograms appear with valid flag = 0 and placeholder values.
- Existing outputs (T, VV) are byte-identical to the current script's on the same data.

## Expected outputs
- Updated `preparazione_dati_histo.py`.
- The two `.npz` files (`moduliquadri_mc.npz`, `visibilities_mc.npz`) from a real run, in the same folder as the current outputs.

## Final instruction
Verify every acceptance criterion before finishing. If a criterion cannot be met, stop and report why instead of improvising. In particular, if the existing script cannot produce byte-identical T and VV after the modification, stop and report the discrepancy.