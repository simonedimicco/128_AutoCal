# Plan: Monte Carlo error estimation on U (unitary reconstruction from histograms)

## Goal
Extend the data-preparation pipeline to save all statistical inputs needed for error propagation (raw counts for M and B per channel; per-histogram y_int, y_noise_int, area_c with their uncertainties), then create a separate program that generates N (configurable, default 100) Monte Carlo realizations of U by resampling the data (Poisson for counts, Gaussians with the already-computed sigmas) and reusing the existing reconstruction code unchanged. Final output: a single `.npz` file with the N fake U matrices plus the real U.

## Constraints
- Intermediate and final storage format: `.npz` (compressed numpy) in the same folder as the current outputs.
- N realizations configurable at runtime (default 100).
- The reconstruction logic is reused identically, not rewritten; same pivots (i1=79, i2=0) and same filter visibility Vf=0.78.
- Sampling: B, M and area_c drawn from Poisson with expected value equal to the observed value, no bin-width rescaling (all quantities are on the same counts·ps scale; V is a ratio, any common factor cancels); y_int and y_noise_int drawn from Gaussians with sigma equal to the already-computed intersection errors.
- The physics of the V and U computation does not change: only additional saving and resampling.
- Each phase must be executable by an agent with isolated context using only this plan and the outputs of previous phases. All code-specific facts needed downstream are stated explicitly below; source scripts are named only for traceability.

## Domain facts (binding for all phases)
- Channels: C = 4, labels `['b','c','d','e']`; modes: 128. Unitaries are 128×4.
- Singles pipeline, per channel c (loop over channels; each channel is an independent input source, so B and M must be saved separately for each c):
  - Dark counts B(c): read every file in `Ricostruzione_unitaria/Buio/<c>/`, each containing a timestamp array `c_tot`; accumulate `B += np.bincount(c_tot, minlength=128)` over all files; divide by the file count only once, at the end. The file loop is only repeated measurement for more statistics, not a separate statistical source.
  - Singles counts M_raw(c): same accumulation over `Ricostruzione_unitaria/Singles/<c>/`, same single division by file count.
  - Then: `M = M_raw − B`; clip negatives to 0; normalize M to sum 1; stack per channel and transpose → T (128×4), the squared-modulus matrix.
- Pairs pipeline, per pair label `bc, bd, be, cd, ce, de` (from combinations of the 4 channels; 6 pairs, ordered):
  - Input: one `.npz` per pair with arrays `hist_totals` (a list of 2D histograms) and `bin_edges`.
  - For each histogram index i, decode the mode pair with the standard triangular-index map: given n=128 and linear index iii, `i = int((2n − 1 − sqrt((2n−1)² − 8·iii)) // 2)`, then adjust with the two while-loops on the offsets `i*n − i(i+1)/2` until the index falls in the correct block, and `ii = iii − offset + i + 1`. This yields (ii, iii) with ii < iii.
  - Peak finding (on bin centers, min peak distance 7 bins) splits peaks into: left peaks, right peaks, one central peak (closest to x=0), and noise positions midway between adjacent peaks on each side. If fewer than 2 left peaks, or fewer than 2 right peaks, or no central peak exists, the histogram is skipped (its V entries stay 0 and its valid flag is 0).
  - Integrate each peak over a symmetric window of 1800 ps (trapezoid rule on the histogram) → peak areas; integrate the central peak the same way → area_c (counts·ps, no normalization).
  - Linear fit (y = m·x + q) of left-peak areas vs position and of right-peak areas vs position, each returning coefficients and a covariance matrix; same for the two noise sequences.
  - Intersection of the left and right lines → (x_int, y_int); intersection of the two noise lines → (x_noise_int, y_noise_int). Errors sigma_y_int and sigma_y_noise_int come from standard propagation of the fit covariances through the intersection formula (y of the intersection of two fitted lines, full covariance terms included).
  - Visibility: d_int = y_int − area_c; V = (y_int − area_c)/(y_int − y_noise_int); assign V[ii,iii] = V[iii,ii] (symmetric matrix, diagonal 0). One 128×128 V matrix per pair, stacked in order into the list VV.
- Reconstruction core (from `Ricostruzione_unitaria_histo_numba.py`, numba `@njit`): input `t = sqrt(T)` (128×4), VV (list of 6 matrices 128×128, indexed by pair order), a 4×4 map com_index with `com_index[h,k]` = index in VV for channel pair h<k (−1 elsewhere), Vf=0.78, pivots i1=79, i2=0. It computes the phase matrix FF (128×4) by arccos of a visibility-derived ratio (clipped to [−1,1]) with sign fixing via the two pivot rows, then `U = t · exp(i·FF)` (128×4 complex). No I/O inside the core; all inputs are arguments.

## Deliverables
- Updated `preparazione_dati_histo.py` (singles and pairs sections): unchanged existing outputs plus `.npz` saving of all statistical inputs.
- A Monte Carlo resampling module (statistical inputs → fake realizations of T and VV).
- An importable module with the reconstruction core (logic unchanged, I/O-free).
- A Monte Carlo program producing N fake U matrices saved in a single `.npz` with the real U and metadata.
- A data dictionary document (schema of the `.npz` files) for the isolated agents.

## Phases

### Phase 1 — Data dictionary and .npz schema
- Objective: Freeze the exact schema of the `.npz` files and the semantics of every field.
- Inputs: the domain facts above.
- Outputs: data dictionary document.
- Acceptance criteria:
  - Schema of `moduliquadri_mc.npz` (singles): for each of the 4 channels c: accumulated raw Singles counts (128,), accumulated raw dark counts (128,), and the file count for each of the two accumulations. Arrays kept per channel (4 separate entries per quantity, or a 4×128 array with a documented channel order), never merged: each channel is an independent statistical source.
  - Schema of `visibilities_mc.npz` (pairs): for each of the 6 pairs and each histogram index i: ii, iii, y_int, sigma_y_int, y_noise_int, sigma_y_noise_int, area_c, valid flag (0 for skipped histograms).
  - Units documented: area_c, y_int, y_noise_int all in counts·ps, no bin-width division anywhere; Poisson sampling of area_c uses λ = area_c directly.
  - Documented formula V = (y_int − area_c)/(y_int − y_noise_int) with symmetric assignment V[ii,iii] = V[iii,ii] and skipped histograms left at 0.
  - Documented MC rebuild of the singles pipeline: Poisson-sample the accumulated raw counts, divide by file count, subtract dark from singles, clip, normalize, stack, transpose.
- Dependencies: none.

### Phase 2 — Saving in preparation
- Objective: Add the `.npz` saving of all statistical inputs to `preparazione_dati_histo.py`, without changing any existing numerical output.
- Inputs: Phase 1 (schema), current `preparazione_dati_histo.py`.
- Outputs: updated script + the two `.npz` files from a real run.
- Acceptance criteria:
  - The two `.npz` files are produced with fields and shapes conforming to the Phase 1 schema (runtime asserts).
  - B and M raw counts are saved separately for each channel c, as accumulated sums over that channel's files, together with the per-channel file counts.
  - The sigmas saved per histogram are exactly the intersection errors already computed by the script (`sigma_y_int`, `sigma_y_noise_int`).
  - Skipped histograms appear with valid flag = 0 and placeholder values.
  - Existing outputs (T, VV) are byte-identical to the current script's on the same data.
- Dependencies: Phase 1.

### Phase 3 — Extraction of the reusable reconstruction core
- Objective: Make the U computation invocable as a module, without duplicating logic.
- Inputs: Domain facts (core contract), `Ricostruzione_unitaria_histo_numba.py`.
- Outputs: importable module with contract (t, VV, com_index, Vf, i1, i2) → U, jit functions unchanged.
- Acceptance criteria:
  - With real data (current T and VV) the module returns U identical to the original script's output (element-wise, machine tolerance), with i1=79, i2=0, Vf=0.78.
  - No paths or I/O inside the core: only explicit inputs as arguments.
  - Functions remain `@njit` with the same signatures.
- Dependencies: Phase 1; real data from Phase 2 for the identity test.

### Phase 4 — Monte Carlo resampling module
- Objective: Implement the sampling of fake realizations of the inputs.
- Inputs: Phases 1 and 2 (schema and `.npz` files).
- Outputs: module that, given the `.npz` files and a seed, returns one realization of {T, VV}.
- Acceptance criteria:
  - Singles: Poisson-sample the accumulated raw counts per channel (singles and dark separately, each channel independently), then apply the pipeline identical to the real one (divide by file count, subtract, clip, normalize, stack, transpose).
  - Pairs: y_int ~ N(y_int, sigma_y_int), y_noise_int ~ N(y_noise_int, sigma_y_noise_int), area_c ~ Poisson(area_c) with no rescaling; recompute V = (y_int − area_c)/(y_int − y_noise_int); respect the valid flag (skipped cells stay 0); fill V[ii,iii] and V[iii,ii].
  - Reproducible given a seed (fixed-seed test).
  - Statistical test: mean and standard deviation of sampled quantities converge to the real values/sigmas over ~10⁴ samples.
- Dependencies: Phase 1, Phase 2.

### Phase 5 — Monte Carlo program for the U ensemble
- Objective: Generate N fake U matrices by chaining Phase 4 and Phase 3, and save them in a single file.
- Inputs: Phases 3 and 4, real `.npz` files.
- Outputs: program with configurable N (default 100) saving a single `.npz`: {U_1..U_N, U_real, metadata (base seed, N, timestamp, Vf, i1, i2)}.
- Acceptance criteria:
  - Every U_i uses the Phase 3 core, with no copies of the logic.
  - Independent realizations: per-realization seed derived from the base seed.
  - The ensemble file exists, is readable, U shapes verified at runtime (128×4, complex).
  - The included real U is identical to the one produced with non-resampled data.
  - Execution time with N=100 measured and reported (numba compilation paid once).
- Dependencies: Phase 3, Phase 4.

## Interfaces between phases
- Phase 1 → all: data dictionary (`.npz` schema, semantics, units, MC rebuild of the singles pipeline).
- Phase 2 → Phases 4, 5: `moduliquadri_mc.npz` and `visibilities_mc.npz` conforming to the schema.
- Phase 3 → Phase 5: module (t, VV, com_index, Vf, i1, i2) → complex U 128×4, certified identical to the original.
- Phase 4 → Phase 5: function (`.npz` + seed) → one realization {T, VV}.
- Phase 5 → user: ensemble file {U_1..U_N, U_real, metadata}, single `.npz`.

## Risks and open questions
- Residual correlations: y_int comes from fits on lateral peak areas, and area_c shares the same histogram with them; independent sampling (Gaussian y_int, Poisson area_c) ignores this correlation. The impact on V = (y_int − area_c)/(y_int − y_noise_int) is not quantified; if needed, the extension is to also save the line-fit parameters and covariances and resample upstream. Deferred: White decides after the first results.
- Degenerate intersections (nearly parallel fitted lines) can produce absurd V values; these histograms must be identifiable (valid flag) so the MC does not propagate garbage.
- C=4 channels and 128 modes are hardcoded as in the original; if the setup changes, the schema will need generalizing (decision: not for now).

## Definition of done
- `preparazione_dati_histo.py` produces its existing outputs unchanged plus the two `.npz` files conforming to the schema, with per-channel B/M raw counts and file counts saved separately.
- The reconstruction core is a reusable module, certified identical to the original on real data.
- The Monte Carlo program generates N reproducible U matrices (seed), with N configurable, and saves the ensemble in a single `.npz` with the real U and metadata.