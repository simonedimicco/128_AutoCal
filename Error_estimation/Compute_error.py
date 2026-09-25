import numpy as np
import os
from numba import njit
from tqdm import tqdm

@njit
def compute_tvd(A, B):
    P1 = np.abs(A)**2
    P2 = np.abs(B)**2

    return 0.5 * np.sum(np.abs(P1 - P2))

@njit
def compute_similarity(A, B):
    return np.sum(np.sqrt(np.abs(A)*np.abs(B)))**2

@njit
def Frobenius_norm(A):
    return 0.5*np.sqrt(np.sum(np.abs(A)**2))
@njit
def L1_norm(pred, target):
    return 0.5*(np.sum(np.abs(pred - target)))

# 1. Carica il file .npz
#path= '/media/dati_2/DATI_2026_06_07_128modi_training_target12_3PairsPre_32Start_17ResComp_1'
# path= '/media/dati_2/DATI_2026_06_13_128modi_training_target11_3PairsPre_32Start_17ResComp_1'
path= '/media/dati_2/DATI_2026_06_14_128modi_training_target9_3PairsPre_32Start_17ResComp_1'
with np.load(f'{path}/error_files/ensemble_U.npz') as data:
    # 2. Leggi la matrice reale
    U_trained_real = data['U_real']  # shape (128, 4), dtype complex128

    # 3. Leggi tutte le matrici Monte Carlo
    N = data['N']  # Numero di realizzazioni
    U_trained_monte = [data[f'U_{i}'] for i in range(1, N+1)]  # Lista di N matrici (128, 4)

    # # 4. (Opzionale) Leggi i metadata
    # base_seed = data['base_seed']
    # timestamp = data['timestamp']
    # Vf = data['Vf']
    # i1 = data['i1']
    # i2 = data['i2']

#Target with 18 resistances
#path_reference = f'/media/dati_2/DATI_2026_05_08_target_ripetuti/Target_12'
# path_reference = f'/media/dati_2/DATI_2026_05_08_target_ripetuti/Target_11/'
path_reference = f'/media/dati_2/DATI_2026_05_08_target_ripetuti/Target_9'
with np.load(f'{path_reference}/error_files/ensemble_U.npz') as data:
    # 2. Leggi la matrice reale
    U_18_real = data['U_real']  # shape (128, 4), dtype complex128

    # 3. Leggi tutte le matrici Monte Carlo
    N = data['N']  # Numero di realizzazioni
    U_18_monte = [data[f'U_{i}'] for i in range(1, N+1)]  # Lista di N matrici (128, 4)


#Target with 17 resistances
#path_new = '/media/dati_2/DATI_2026_05_27_misure_multiple/target_12_new'
# path_new = '/media/dati_2/DATI_2026_05_27_misure_multiple/target_11_new/'
path_new = '/media/dati_2/DATI_2026_05_27_misure_multiple/target_9_new'
with np.load(f'{path_new}/error_files/ensemble_U.npz') as data:
    # 2. Leggi la matrice reale
    U_17_real = data['U_real']  # shape (128, 4), dtype complex128

    # 3. Leggi tutte le matrici Monte Carlo
    N = data['N']  # Numero di realizzazioni
    U_17_monte = [data[f'U_{i}'] for i in range(1, N+1)]  # Lista di N matrici (128, 4)

#Start position 32
path_start = '/media/dati_2/DATI_2026_05_29_misure_multiple/all_32/'
#with np.load(f'{path_start}/error_files/ensemble_U.npz') as data:
with np.load(f'{path_start}/error_files/ensemble_U_for_9.npz') as data:
    # 2. Leggi la matrice reale
    U_start_real = data['U_real']  # shape (128, 4), dtype complex128

    # 3. Leggi tutte le matrici Monte Carlo
    N = data['N']  # Numero di realizzazioni
    U_start_monte = [data[f'U_{i}'] for i in range(1, N+1)]  # Lista di N matrici (128, 4)

list_L1_trained_18=[]
list_L1_17_18=[]
list_L1_start_18=[]
for i in tqdm(range(len(U_trained_monte)), desc='comparing with monte carlo realizations'):
    list_L1_trained_18.append(L1_norm(U_trained_monte[i], U_18_monte[i]))
    list_L1_17_18.append(L1_norm(U_17_monte[i], U_18_monte[i]))
    list_L1_start_18.append(L1_norm(U_start_monte[i], U_18_monte[i]))
    
array_L1_trained_18 = np.array(list_L1_trained_18)
array_L1_17_18 = np.array(list_L1_17_18)
array_L1_start_18 = np.array(list_L1_start_18)

mu_L1_trained_18 = L1_norm(U_trained_real, U_18_real)
std_L1_trained_18 = np.std(array_L1_trained_18)

mu_L1_17_18 = L1_norm(U_17_real, U_18_real)
std_L1_17_18 = np.std(array_L1_17_18)

mu_L1_start_18 = L1_norm(U_start_real, U_18_real)
std_L1_start_18 = np.std(array_L1_start_18)


print(f"Mean L1 norm between trained and 18 resistances: {mu_L1_trained_18:.2f} +/- {std_L1_trained_18:.2f}")
print(f"Mean L1 norm between 17 and 18 resistances: {mu_L1_17_18:.2f} +/- {std_L1_17_18:.2f}")
print(f"Mean L1 norm between start position and 18 resistances: {mu_L1_start_18:.2f} +/- {std_L1_start_18:.2f}")