# Monte Carlo Error Estimation Pipeline

**Stato del Progetto:** COMPLETO ✅

Pipeline per la stima degli errori tramite Monte Carlo sulla ricostruzione unitaria da istogrammi in ottica quantistica (4 canali, 128 mode).

---

## Overview

Questo progetto estende la pipeline di preparazione dati per:
1. Salvare tutti gli input statistici necessari per la propagazione degli errori
2. Generare N realizzazioni Monte Carlo della matrice unitaria U
3. Riutilizzare il codice di ricostruzione esistente senza modifiche

**Output finale:** Un singolo file `.npz` contenente N matrici U generate tramite Monte Carlo più la matrice U reale.

---

## Architettura

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 1: Data Dictionary                          │
│  Error_estimation/data_dictionary.md                              │
│  Schema .npz, semantica, unità, convenzioni di campionamento        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 2: Salvataggio Input                       │
│  Error_estimation/preparazione_dati_histo.py                       │
│  → error_files/moduliquadri_mc.npz                               │
│  → error_files/visibilities_mc.npz                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 3: Reconstruction Core                     │
│  Error_estimation/reconstruction_core.py                           │
│  Modulo importabile, I/O-free, @njit                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 4: MC Resampling Module                    │
│  Error_estimation/mc_resampling.py                                 │
│  Campionamento Poisson/Gaussiano, ricostruzione T e VV              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 5: Ensemble Generation                     │
│  Error_estimation/monte_carlo_ensemble.py                          │
│  → Error_estimation/ensemble_U.npz                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Struttura del Progetto

```
128_auto_calibration/
├── Error_estimation/
│   ├── preparazione_dati_histo.py      # Phase 2: Salvataggio input statistici
│   ├── reconstruction_core.py         # Phase 3: Core di ricostruzione
│   ├── mc_resampling.py                # Phase 4: Modulo ricampionamento MC
│   ├── monte_carlo_ensemble.py         # Phase 5: Generazione ensemble
│   ├── test_work_02.py                 # Test Phase 2
│   ├── test_reconstruction_core.py     # Test Phase 3
│   ├── test_mc_resampling.py           # Test Phase 4
│   ├── visibility_HOM_lib.py            # Libreria funzioni ausiliarie
│   ├── data_dictionary.md              # Phase 1: Dizionario dati
│   └── ensemble_U.npz                  # Output finale
├── error_files/
│   ├── moduliquadri_mc.npz             # Input statistici Singles
│   └── visibilities_mc.npz             # Input statistici Pairs
└── .vibe/
    ├── plan.md                          # Piano originale
    ├── data_dictionary.md              # Copia del dizionario
    └── report_work_*.md                 # Report di ogni fase
```

---

## Modifiche al Path di Salvataggio

### Contesto
Lo script `preparazione_dati_histo.py` originariamente salvava tutti gli output nella directory dei dati:
```python
data_path = '/media/dati_2/DATI_2026_06_14_128modi_training_target9_3PairsPre_32Start_17ResComp_1'
data_folder = 'Ricostruzione_unitaria'
save_path = join(data_path, data_folder)
```

### Modifica Introdotta
Per i file specifici del Monte Carlo è stato aggiunto un path dedicato **relativo** alla directory di lavoro:

```python
# Linea 19 in preparazione_dati_histo.py
save_path_error = './error_files'
os.makedirs(save_path_error, exist_ok=True)
```

### Salvataggio File MC
I file per il Monte Carlo vengono salvati in:
```python
# Linea 80
np.savez(join(save_path_error, 'moduliquadri_mc'), ...)

# Linea 252
np.savez(join(save_path_error, 'visibilities_mc'), ...)
```

### Path Effettivi
- **Directory di lavoro:** `/home/simone/Scrivania/128_auto_calibration/`
- **Path assoluti:**
  - `/home/simone/Scrivania/128_auto_calibration/error_files/moduliquadri_mc.npz`
  - `/home/simone/Scrivania/128_auto_calibration/error_files/visibilities_mc.npz`

### Motivazione
1. **Isolamento:** Separazione tra output originali e file per Monte Carlo
2. **Portabilità:** Path relativo consente esecuzione senza dipendenza da mount esterni
3. **Accessibilità:** I file sono accessibili localmente per le fasi successive

### Impatto su Phase 5
Il programma `monte_carlo_ensemble.py` reference correttamente i file:
```python
# Linee 303-304
moduliquadri_mc_path = './error_files/moduliquadri_mc.npz'
visibilities_mc_path = './error_files/visibilities_mc.npz'
```

---

## Dati di Input

### Canali
- 4 canali: `['b', 'c', 'd', 'e']`
- 128 mode per canale

### Pipeline Singles
Per ogni canale `c`:
- **Dark counts (B):** Accumulazione da `Ricostruzione_unitaria/Buio/<c>/`
- **Singles counts (M):** Accumulazione da `Ricostruzione_unitaria/Singles/<c>/`
- **Elaborazione:** `M = (M_raw / N_files) - (B_raw / N_files)`, clip negativi, normalizza a somma 1
- **Output:** T (128×4), matrice squared-modulus

### Pipeline Pairs
Per ogni coppia `bc, bd, be, cd, ce, de`:
- **Input:** File `.npz` con `hist_totals` e `bin_edges`
- **Peak finding:** Picchi laterali, centrale, e rumore
- **Integrazione:** Finestra simmetrica 1800 ps → area_c
- **Fit lineare:** Coefficienti e matrice di covarianza
- **Intersezione:** (x_int, y_int), (x_noise_int, y_noise_int)
- **Visibilità:** V = (y_int − area_c) / (y_int − y_noise_int)
- **Output:** VV (lista di 6 matrici 128×128)

---

## Schema File `.npz`

### `moduliquadri_mc.npz` — Input Statistici Singles

| Campo | Tipo | Shape | Unità | Descrizione |
|-------|------|-------|-------|-------------|
| `singles_raw_b/c/d/e` | int64 | (128,) | counts | Conteggi raw Singles accumulati |
| `dark_raw_b/c/d/e` | int64 | (128,) | counts | Conteggi raw Dark accumulati |
| `singles_file_count_b/c/d/e` | int64 | () | — | Numero file Singles per canale |
| `dark_file_count_b/c/d/e` | int64 | () | — | Numero file Dark per canale |

### `visibilities_mc.npz` — Input Statistici Pairs

Per ogni coppia `XX` in `['bc', 'bd', 'be', 'cd', 'ce', 'de']`:

| Campo | Tipo | Shape | Unità | Descrizione |
|-------|------|-------|-------|-------------|
| `pair_XX_ii` | int64 | (N,) | — | Indice modo ii |
| `pair_XX_iii` | int64 | (N,) | — | Indice modo iii |
| `pair_XX_y_int` | float64 | (N,) | counts·ps | y intersezione left-right |
| `pair_XX_sigma_y_int` | float64 | (N,) | counts·ps | Errore su y_int |
| `pair_XX_y_noise_int` | float64 | (N,) | counts·ps | y intersezione noise |
| `pair_XX_sigma_y_noise_int` | float64 | (N,) | counts·ps | Errore su y_noise_int |
| `pair_XX_area_c` | float64 | (N,) | counts·ps | Area picco centrale |
| `pair_XX_valid` | int64 | (N,) | — | Flag validità (0=skipped, 1=valido) |

### `ensemble_U.npz` — Output Finale

| Campo | Tipo | Shape | Descrizione |
|-------|------|-------|-------------|
| `U_real` | complex128 | (128, 4) | Matrice unitaria reale |
| `U_1`..`U_N` | complex128 | (128, 4) | Matrici unitarie MC |
| `base_seed` | int64 | () | Seed base |
| `N` | int64 | () | Numero realizzazioni |
| `timestamp` | str | () | Timestamp generazione |
| `Vf` | float64 | () | Fattore visibilità (0.78) |
| `i1` | int64 | () | Primo pivot row (79) |
| `i2` | int64 | () | Secondo pivot row (0) |

---

## Convenzioni di Campionamento Monte Carlo

| Quantità | Distribuzione | Parametro | Note |
|----------|---------------|-----------|------|
| B (dark counts) | Poisson | λ = valore osservato | Per canale, per modo |
| M_raw (singles counts) | Poisson | λ = valore osservato | Per canale, per modo |
| area_c | Poisson | λ = valore osservato | Per istogramma |
| y_int | Gaussiana | μ = y_int, σ = sigma_y_int | Per istogramma |
| y_noise_int | Gaussiana | μ = y_noise_int, σ = sigma_y_noise_int | Per istogramma |

**Importante:** Tutte le quantità sono sulla stessa scala **counts·ps**. V è un rapporto, qualsiasi fattore comune si cancella.

---

## Reconstruction Core

### Interfaccia

```python
def reconstruct_U(t, VV, com_index, Vf=0.78, i1=79, i2=0) -> U
```

**Parametri:**
- `t`: sqrt(T), matrice (128×4)
- `VV`: Array 3D (6, 128, 128) o lista di 6 matrici 128×128
- `com_index`: Mappa 4×4, `com_index[h,k]` = indice in VV per la coppia h<k
- `Vf`: Fattore visibilità, default 0.78
- `i1`: Primo pivot row, default 79
- `i2`: Secondo pivot row, default 0

**Ritorna:** U, matrice complessa (128×4)

### Funzioni Interne
Tutte decorate con `@njit`:
- `gamma(g, h, j, k, VV_in, tt_in, Vf_local, com_idx)` — Ratio per arccos
- `compute_FF_magnitudes(FF_out, t_in, VV_in, com_idx, Vf_local, i1_local)` — Magnitudini FF
- `compute_FF_signs_row_i2(...)` — Segni per pivot row i2
- `compute_FF_signs_other_rows(...)` — Segni per altre righe

---

## Esecuzione

### Prerequisiti
```bash
cd /home/simone/Scrivania/128_auto_calibration/Error_estimation
```

### Phase 2: Generazione Input Statistici
```bash
python preparazione_dati_histo.py
# Output: ../error_files/moduliquadri_mc.npz, ../error_files/visibilities_mc.npz
```

### Phase 5: Generazione Ensemble Monte Carlo
```bash
# Default: N=100, seed=42
python monte_carlo_ensemble.py

# Con parametri personalizzati
python monte_carlo_ensemble.py --N 50 --seed 123 --output my_ensemble.npz

# Elenco completo parametri
python monte_carlo_ensemble.py --help
```

### Opzioni di `monte_carlo_ensemble.py`

| Opzione | Default | Descrizione |
|---------|---------|-------------|
| `--N` | 100 | Numero realizzazioni MC |
| `--seed` | 42 | Seed base |
| `--output` | ensemble_U.npz | Path file output |
| `--Vf` | 0.78 | Fattore visibilità |
| `--i1` | 79 | Primo pivot row |
| `--i2` | 0 | Secondo pivot row |

---

## Test

### Test Phase 2
```bash
python test_work_02.py
```
Verifica: schema compliance, campo corretti, shape attesi

### Test Phase 3
```bash
python test_reconstruction_core.py
```
Verifica: identità con codice originale (max diff = 0.0)

### Test Phase 4
```bash
python test_mc_resampling.py
```
Verifica: riproducibilità, convergenza statistica, simmetria V, normalizzazione T

---

## Prestazioni

| Metrica | Valore |
|---------|--------|
| Tempo esecuzione N=100 | 6.92 secondi |
| Tempo medio per realizzazione | 0.069 secondi |
| Compilazione JIT Numba | Pagata una volta sola |

---

## Risultati di Verifica

| Test | Stato | Dettagli |
|------|-------|----------|
| Schema compliance | ✅ PASS | Tutti i campi presenti con shape corretti |
| Identità U_reale | ✅ PASS | Max differenza assoluta = 0.0 vs originale |
| Riproducibilità | ✅ PASS | Stesso seed produce risultati identici |
| Convergenza statistica | ✅ PASS | Differenza media 0.29%, massima 1.32% (1000 campioni) |
| Shape corretti | ✅ PASS | T: (128,4), V: 6×(128,128), U: (128,4) complex |
| Normalizzazione T | ✅ PASS | Ogni colonna somma a 1 |
| Simmetria V | ✅ PASS | Tutte le matrici V simmetriche |
| Diagonale V | ✅ PASS | Tutte le diagonali a 0 |

---

## Dipendenze

- Python 3.x
- NumPy
- Numba
- SciPy
- tqdm
- matplotlib (per debugging)

---

## Documentazione

- **Piano Progetto:** `.vibe/plan.md`
- **Dizionario Dati:** `Error_estimation/data_dictionary.md`
- **Report Fasi:** `.vibe/report_work_*.md`

---

## Note

### Limitazioni Conoscute
1. **Correlazioni residue:** y_int deriva da fit su area picchi laterali, area_c condivide lo stesso istogramma. Il campionamento indipendente ignora questa correlazione.
2. **Intersezioni degenerate:** Linee quasi parallele possono produrre valori di V assurdi. Questi istogrammi sono identificabili tramite il flag `valid=0`.

### Hardcoded Values
- Numero canali: 4
- Numero mode: 128
- Pivot rows: i1=79, i2=0
- Fattore visibilità: Vf=0.78

---

## Contatti e Supporto

Per domande o problemi, fare riferimento ai report di fase in `.vibe/report_work_*.md`.

---

*Ultimo aggiornamento: 2026-09-23*
