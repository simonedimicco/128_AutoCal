# WORK_05 — Monte Carlo program for the U ensemble

## Project context
The project is a Monte Carlo error-estimation pipeline for unitary reconstruction from histograms (quantum optics, 4 channels, 128 modes). Two modules were built in previous phases: a resampling module that produces one fake realization of the inputs {T, VV} from the saved statistical `.npz` files and a seed, and an importable reconstruction core that maps (t, VV, com_index, Vf, i1, i2) to a complex 128×4 unitary U. This phase chains them N times and saves the resulting ensemble of U matrices together with the real U, delivering the final product of the plan.

## Objective
Generate N fake U matrices by chaining the resampling module and the reconstruction core, and save them in a single file.

## Inputs
- The resampling module (contract: function (`.npz` + seed) → one realization {T, VV}).
- The reconstruction core module (contract: (t, VV, com_index, Vf, i1, i2) → complex U 128×4, certified identical to the original on real data; defaults Vf=0.78, i1=79, i2=0).
- The real `.npz` statistical inputs (`moduliquadri_mc.npz`, `visibilities_mc.npz`) and the real T, VV from the data-preparation pipeline.

## Binding domain facts (copied from the plan)
- Channels: C = 4, labels `['b','c','d','e']`; modes: 128. Unitaries are 128×4 complex.
- VV is a list of 6 matrices 128×128, indexed by pair order `bc, bd, be, cd, ce, de`; com_index is a 4×4 map with `com_index[h,k]` = index in VV for channel pair h<k (−1 elsewhere).
- The reconstruction logic is reused identically, not rewritten; same pivots (i1=79, i2=0) and same filter visibility Vf=0.78.
- N realizations configurable at runtime (default 100).
- Storage format: `.npz` (compressed numpy) in the same folder as the current outputs.

## Interfaces to respect
- From the reconstruction core phase: module (t, VV, com_index, Vf, i1, i2) → complex U 128×4, certified identical to the original. Consume it via import; make no copies of the logic.
- From the resampling phase: function (`.npz` + seed) → one realization {T, VV}. Consume it via import.
- Final deliverable to the user: ensemble file {U_1..U_N, U_real, metadata}, single `.npz`.

## Acceptance criteria (verify each one)
- Every U_i uses the reconstruction core module, with no copies of the logic.
- Independent realizations: per-realization seed derived from the base seed.
- The ensemble file exists, is readable, U shapes verified at runtime (128×4, complex).
- The included real U is identical to the one produced with non-resampled data.
- Execution time with N=100 measured and reported (numba compilation paid once).

## Expected outputs
- A program (executable script) with configurable N (default 100) that saves a single `.npz` containing: {U_1..U_N, U_real, metadata (base seed, N, timestamp, Vf, i1, i2)}.

## Final instruction
Verify every acceptance criterion before finishing. If a criterion cannot be met, stop and report why instead of improvising. In particular, if the real U stored in the ensemble differs from the non-resampled one, stop and report the discrepancy rather than adjusting the core or the sampling.