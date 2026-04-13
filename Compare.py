import numpy as np
import os
from numba import njit

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

path  = '/media/dati_2'
folder_1 = 'DATI_2026_04_10_pasqua_training'
folder_2 = 'DATI_2026_03_27_target'
subfolder = 'Ricostruzione_unitaria'
file_name = 'Unitary_mat.npz'

data_1 = os.path.join(path, folder_1, subfolder, file_name)
data_2 = os.path.join(path, folder_2, subfolder, file_name)

with np.load(data_1) as data:
    U_1 = data['U']

with np.load(data_2) as data:
    U_2 = data['U']

print(f'size of matrices: {U_1.shape}, {U_2.shape}')
tmp = np.sum(np.abs(U_1), axis=0)
print(tmp.shape)
for i in range(4):
    print(f'matri U1 sum on column {i+1}: {np.sum(np.abs(U_1)**2, axis=0)[i]:.1f}')
    print(f'matri U2 sum on column {i+1}: {np.sum(np.abs(U_2)**2, axis=0)[i]:.1f}\n')


similarity = compute_similarity(U_1, U_2)

norm = Frobenius_norm(U_1 - U_2)
size_U_1 = Frobenius_norm(U_1)
size_U_2 = Frobenius_norm(U_2)

#print(f"Similarity: {similarity:.6f}")
for i in range(4):
    tvd = compute_tvd(U_1[:,i],U_2[:,i])
    print(f"Total Variation Distance column {i+1}: {tvd:.3f}")
print(f"Frobenius Norm of the difference: {norm:.6f}")
print(f"Size of U_1: {size_U_1:.6f}")
print(f"Size of U_2: {size_U_2:.6f}")