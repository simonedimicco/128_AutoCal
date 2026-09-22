# WORK_03 — Extraction of the reusable reconstruction core

## Project context
The project is a Monte Carlo error-estimation pipeline for unitary reconstruction from histograms (quantum optics, 4 channels `['b','c','d','e']`, 128 modes). The goal is to generate N Monte Carlo realizations of the reconstructed unitary U by resampling the statistical inputs and reusing the existing reconstruction code unchanged. This phase turns the U computation into an importable, I/O-free module so that later Monte Carlo phases can call it without duplicating logic.

## Objective
Make the U computation invocable as a module, without duplicating logic.

## Inputs
- The core contract below (binding domain facts from the plan).
- `Ricostruzione_unitaria_histo_numba.py` (source of the existing logic, named for traceability; the logic must be reused, not rewritten).
- Real data from the previous saving phase: T and VV as currently produced by the pipeline (for the identity test), and the real `.npz` statistical inputs.

## Binding core contract (copied from the plan)
- Reconstruction core (numba `@njit`): input `t = sqrt(T)` (128×4), VV (list of 6 matrices 128×128, indexed by pair order `bc, bd, be, cd, ce, de`), a 4×4 map com_index with `com_index[h,k]` = index in VV for channel pair h<k (−1 elsewhere), Vf=0.78, pivots i1=79, i2=0.
- It computes the phase matrix FF (128×4) by arccos of a visibility-derived ratio (clipped to [−1,1]) with sign fixing via the two pivot rows, then `U = t · exp(i·FF)` (128×4 complex).
- No I/O inside the core; all inputs are arguments.

## Interfaces to respect
- Module signature: (t, VV, com_index, Vf, i1, i2) → complex U 128×4. This exact contract is consumed by the Monte Carlo ensemble phase; do not change names, order, or defaults (Vf=0.78, i1=79, i2=0).
- Functions remain `@njit` with the same signatures.

## Acceptance criteria (verify each one)
- With real data (current T and VV) the module returns U identical to the original script's output (element-wise, machine tolerance), with i1=79, i2=0, Vf=0.78.
- No paths or I/O inside the core: only explicit inputs as arguments.
- Functions remain `@njit` with the same signatures.

## Expected outputs
- An importable module (single Python file) containing the reconstruction core, satisfying the contract above.

## Final instruction
Verify every acceptance criterion before finishing. If a criterion cannot be met, stop and report why instead of improvising. If the identity test against the original script fails on any element beyond machine tolerance, stop and report the discrepancy rather than adjusting the logic.