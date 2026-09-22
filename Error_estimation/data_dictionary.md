# Data Dictionary — Monte Carlo Error-Eestimation Pipeline

This document defines the exact schema, field semantics, units, and conventions for the `.npz` files used in the Monte Carlo error-estimation pipeline for unitary reconstruction from histograms (quantum optics, 4 channels, 128 modes).

---

## File Overview

| File | Purpose | Contains |
|------|---------|----------|
| `moduliquadri_mc.npz` | Singles pipeline data | Raw counts and file counts per channel for Monte Carlo resampling |
| `visibilities_mc.npz` | Pairs pipeline data | Histogram-derived quantities with uncertainties for Monte Carlo resampling |

---

## Channel and Mode Conventions

- **Channels**: 4, labeled `['b', 'c', 'd', 'e']`
- **Modes**: 128
- **Unitaries**: 128×4 complex matrices
- **Channel pairs**: 6 ordered pairs `['bc', 'bd', 'be', 'cd', 'ce', 'de']`

---

## 1. `moduliquadri_mc.npz` — Singles Pipeline Schema

### Overview

Stores accumulated raw counts and file counts for the singles pipeline, organized **per channel**. Each channel is treated as an independent statistical source; arrays are **never merged** across channels.

### Fields

The file contains the following fields for each of the 4 channels:

#### Per-Channel Arrays (shape: 128)

| Field Name Pattern | Description | Type | Units | Channel Order |
|---------------------|-------------|------|-------|---------------|
| `singles_raw_b` | Accumulated raw Singles counts for channel **b** | `ndarray` (128,) | counts | `['b', 'c', 'd', 'e']` |
| `singles_raw_c` | Accumulated raw Singles counts for channel **c** | `ndarray` (128,) | counts | `['b', 'c', 'd', 'e']` |
| `singles_raw_d` | Accumulated raw Singles counts for channel **d** | `ndarray` (128,) | counts | `['b', 'c', 'd', 'e']` |
| `singles_raw_e` | Accumulated raw Singles counts for channel **e** | `ndarray` (128,) | counts | `['b', 'c', 'd', 'e']` |
| `dark_raw_b` | Accumulated raw dark counts for channel **b** | `ndarray` (128,) | counts | `['b', 'c', 'd', 'e']` |
| `dark_raw_c` | Accumulated raw dark counts for channel **c** | `ndarray` (128,) | counts | `['b', 'c', 'd', 'e']` |
| `dark_raw_d` | Accumulated raw dark counts for channel **d** | `ndarray` (128,) | counts | `['b', 'c', 'd', 'e']` |
| `dark_raw_e` | Accumulated raw dark counts for channel **e** | `ndarray` (128,) | counts | `['b', 'c', 'd', 'e']` |

#### Per-Channel Scalars

| Field Name Pattern | Description | Type | Units |
|---------------------|-------------|------|-------|
| `singles_file_count_b` | Number of files used to accumulate `singles_raw_b` | `int` | — |
| `singles_file_count_c` | Number of files used to accumulate `singles_raw_c` | `int` | — |
| `singles_file_count_d` | Number of files used to accumulate `singles_raw_d` | `int` | — |
| `singles_file_count_e` | Number of files used to accumulate `singles_raw_e` | `int` | — |
| `dark_file_count_b` | Number of files used to accumulate `dark_raw_b` | `int` | — |
| `dark_file_count_c` | Number of files used to accumulate `dark_raw_c` | `int` | — |
| `dark_file_count_d` | Number of files used to accumulate `dark_raw_d` | `int` | — |
| `dark_file_count_e` | Number of files used to accumulate `dark_raw_e` | `int` | — |

### Alternative Compact Format (Acceptable)

A 4×128 array format is also acceptable, with **documented channel order** `['b', 'c', 'd', 'e']`:

| Field Name | Description | Type | Units |
|------------|-------------|------|-------|
| `singles_raw` | Accumulated raw Singles counts, stacked as 4×128 | `ndarray` (4, 128) | counts |
| `dark_raw` | Accumulated raw dark counts, stacked as 4×128 | `ndarray` (4, 128) | counts |
| `singles_file_counts` | File counts for Singles accumulation, per channel | `ndarray` (4,) | — |
| `dark_file_counts` | File counts for dark accumulation, per channel | `ndarray` (4,) | — |

> **Note**: If using the compact format, the channel order **must** be explicitly documented in the file metadata or companion documentation.

### Field Semantics

- **singles_raw_*c***: Accumulated histogram of timestamp counts from all files in `Ricostruzione_unitaria/Singles/<c>/`. Each file contributes a timestamp array `c_tot`; accumulation is performed via `B += np.bincount(c_tot, minlength=128)`.
- **dark_raw_*c***: Accumulated histogram of dark counts from all files in `Ricostruzione_unitaria/Buio/<c>/`, accumulated identically to Singles.
- **singles_file_count_*c***: Total number of files processed for Singles accumulation for channel *c*.
- **dark_file_count_*c***: Total number of files processed for dark accumulation for channel *c*.

### Usage in Monte Carlo Rebuild

The singles pipeline is rebuilt for each Monte Carlo realization as follows:

```python
# For each channel c:
M_raw_sampled = np.random.poisson(singles_raw_c)  # Poisson sampling
B_sampled = np.random.poisson(dark_raw_c)        # Poisson sampling

# Divide by file count
M_raw_normalized = M_raw_sampled / singles_file_count_c
B_normalized = B_sampled / dark_file_count_c

# Subtract dark from singles, clip negatives to 0
M = np.clip(M_raw_normalized - B_normalized, 0, None)

# Normalize to sum 1
M = M / np.sum(M)

# Stack per channel and transpose to get T (128×4)
T = np.column_stack([M_b, M_c, M_d, M_e])  # shape (128, 4)
```

---

## 2. `visibilities_mc.npz` — Pairs Pipeline Schema

### Overview

Stores histogram-derived quantities with uncertainties for the pairs pipeline. Data is organized **per pair** and **per histogram index** within each pair. There are 6 pairs and, for each pair, a variable number of histogram indices corresponding to mode pairs (ii, iii).

### Pair Order

The 6 pairs are ordered as:
1. bc
2. bd
3. be
4. cd
5. ce
6. de

### Field Structure

For each pair *p* in the ordered list above, the file contains structured arrays for each histogram index *i*.

#### Per-Pair Fields

| Field Name Pattern | Description | Type | Units |
|---------------------|-------------|------|-------|
| `pair_bc_ii` | Mode index ii for each histogram index, pair bc | `ndarray` (N_bc,) | — |
| `pair_bc_iii` | Mode index iii for each histogram index, pair bc | `ndarray` (N_bc,) | — |
| `pair_bc_y_int` | Intersection y-coordinate (left-right fit intersection) for each histogram, pair bc | `ndarray` (N_bc,) | counts·ps |
| `pair_bc_sigma_y_int` | Uncertainty of y_int from fit covariance propagation, pair bc | `ndarray` (N_bc,) | counts·ps |
| `pair_bc_y_noise_int` | Intersection y-coordinate (noise fit intersection) for each histogram, pair bc | `ndarray` (N_bc,) | counts·ps |
| `pair_bc_sigma_y_noise_int` | Uncertainty of y_noise_int from fit covariance propagation, pair bc | `ndarray` (N_bc,) | counts·ps |
| `pair_bc_area_c` | Central peak area for each histogram, pair bc | `ndarray` (N_bc,) | counts·ps |
| `pair_bc_valid` | Validity flag: 1=valid, 0=skipped histogram, pair bc | `ndarray` (N_bc,) | — |

The same pattern applies to all 6 pairs: replace `bc` with `bd`, `be`, `cd`, `ce`, `de`.

### Field Semantics

- **ii, iii**: Mode pair indices derived from the triangular-index map. For a given linear index `iii` and n=128 modes, the mapping is computed as:
  ```python
  i = int((2*n - 1 - np.sqrt((2*n-1)**2 - 8*iii)) // 2)
  offset = i*n - i*(i+1)//2
  while iii >= offset + (n - i):
      i += 1
      offset = i*n - i*(i+1)//2
  ii = iii - offset + i + 1
  ```
  The resulting pair satisfies `ii < iii`.

- **y_int**: The y-coordinate of the intersection point between the left-peak and right-peak linear fits (y = m·x + q). Computed from the fit coefficients and their covariance matrix.

- **sigma_y_int**: The uncertainty of `y_int`, computed via standard error propagation from the fit covariances through the intersection formula. Full covariance terms are included.

- **y_noise_int**: The y-coordinate of the intersection point between the left-noise and right-noise linear fits.

- **sigma_y_noise_int**: The uncertainty of `y_noise_int`, computed identically to `sigma_y_int` but for the noise fits.

- **area_c**: The integrated area of the central peak, computed via the trapezoid rule over a symmetric window of 1800 ps. **No normalization** is applied.

- **valid**: Flag indicating histogram validity. Set to **0** if:
  - Fewer than 2 left peaks are found
  - Fewer than 2 right peaks are found
  - No central peak is found (none closest to x=0)
  Otherwise, set to **1**.

### Visibility Formula

For each valid histogram (valid=1), the visibility V is computed as:

```
V[ii, iii] = (y_int - area_c) / (y_int - y_noise_int)
```

**Symmetric assignment**: `V[ii, iii] = V[iii, ii]` (the visibility matrix is symmetric with zero diagonal).

**Skipped histograms**: When valid=0, the corresponding V entries are left at **0** (both V[ii,iii] and V[iii,ii]).

### Units Convention

- **area_c**: counts·ps (no bin-width division)
- **y_int**: counts·ps
- **y_noise_int**: counts·ps
- **sigma_y_int**: counts·ps
- **sigma_y_noise_int**: counts·ps

> **Important**: All quantities are on the **same counts·ps scale**. No bin-width rescaling is applied anywhere. When sampling area_c in Monte Carlo, the Poisson parameter is **λ = area_c directly** (no division by bin width).

### Usage in Monte Carlo Rebuild

For each Monte Carlo realization, the pairs pipeline quantities are resampled as follows:

- **area_c**: Drawn from `Poisson(λ=area_c)` — the expected value equals the observed value, no rescaling.
- **y_int**: Drawn from `Normal(μ=y_int, σ=sigma_y_int)`
- **y_noise_int**: Drawn from `Normal(μ=y_noise_int, σ=sigma_y_noise_int)`

The visibility V is then recomputed from the sampled values using the same formula:
```
V_sampled = (y_int_sampled - area_c_sampled) / (y_int_sampled - y_noise_int_sampled)
```

With symmetric assignment and zero-filling for invalid histograms.

---

## 3. Sampling Conventions (Binding)

All downstream phases **must** respect these sampling conventions:

| Quantity | Distribution | Parameter | Notes |
|----------|--------------|-----------|-------|
| B (dark counts) | Poisson | λ = observed dark_raw value | Per channel, per mode |
| M_raw (singles counts) | Poisson | λ = observed singles_raw value | Per channel, per mode |
| area_c | Poisson | λ = observed area_c value | Per histogram index |
| y_int | Gaussian | μ = observed y_int, σ = observed sigma_y_int | Per histogram index |
| y_noise_int | Gaussian | μ = observed y_noise_int, σ = observed sigma_y_noise_int | Per histogram index |

- **N realizations**: Configurable at runtime. Default value: **100**.
- **Common scale**: All quantities are on the same **counts·ps** scale. V is a ratio; any common factor cancels out in the computation.

---

## 4. Reconstruction Core Interface

The reconstruction core (`Ricostruzione_unitaria_histo_numba.py`) expects:

- `t = sqrt(T)` where T (128×4) is the squared-modulus matrix from the singles pipeline
- `VV`: list of 6 matrices (128×128), one per pair, in order `['bc', 'bd', 'be', 'cd', 'ce', 'de']`
- `com_index`: 4×4 map where `com_index[h,k]` = index in VV for channel pair h<k, or -1 otherwise
- `Vf = 0.78`
- Pivots: `i1 = 79`, `i2 = 0`

The core computes:
- Phase matrix FF (128×4) via arccos of a visibility-derived ratio (clipped to [-1,1]) with sign fixing via pivot rows
- `U = t · exp(i·FF)` (128×4 complex unitary)

---

## Acceptance Criteria Checklist

- [x] The `moduliquadri_mc.npz` schema is fully specified: per channel c, accumulated raw Singles counts (128,), accumulated raw dark counts (128,), and the file count for each of the two accumulations; arrays kept per channel (4 separate entries per quantity, or a 4×128 array with documented channel order), never merged.
- [x] The `visibilities_mc.npz` schema is fully specified: for each of the 6 pairs and each histogram index i: ii, iii, y_int, sigma_y_int, y_noise_int, sigma_y_noise_int, area_c, valid flag (0 for skipped histograms).
- [x] Units are documented: area_c, y_int, y_noise_int in counts·ps, no bin-width division anywhere; Poisson sampling of area_c uses λ = area_c directly.
- [x] The formula V = (y_int − area_c)/(y_int − y_noise_int) is documented with symmetric assignment V[ii,iii] = V[iii,ii] and skipped histograms left at 0.
- [x] The MC rebuild of the singles pipeline is documented: Poisson-sample the accumulated raw counts, divide by file count, subtract dark from singles, clip, normalize, stack, transpose.
