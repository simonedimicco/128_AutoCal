'''
Utilizzare questo codice per la ricostruzione unitaria a partire dagli istogrammi di coincidenze tra i canali, con l'ottimizzazione tramite numba.
Questo codice ricostruisce la matrice unitaria a partire dagli istogrammi di coincidenze tra i canali, utilizzando le funzioni jittate per velocizzare i calcoli.
Questo codice va utilizzato con i file generati da "preparazione_dati_histo.py"
'''

import numpy as np
from scipy.optimize import minimize
from os.path import join
from numba import njit
import time



@njit
def gamma(g,h,j,k, VV_in, tt_in, Vf_local, com_idx):
    idx_v = com_idx[h, k]
    if idx_v == -1:
        idx_v = com_idx[k,h]
    if idx_v == -1:
        return 0.0
    val = (-VV_in[idx_v, j, g]*(tt_in[j,h]**2+tt_in[j,k]**2)*(tt_in[g,h]**2+tt_in[g,k]**2) + tt_in[g, h] ** 2 * tt_in[j, h] ** 2 + tt_in[g, k] ** 2 * tt_in[j, k] ** 2)
    den = 2.0 * tt_in[g, h] * tt_in[j, k] * tt_in[j, h] * tt_in[g, k] * Vf_local
    #print('ok')
    ratio = val / den
    # clip
    if ratio > 1.0:
        ratio = 1.0
    elif ratio < -1.0:
        ratio = -1.0
    return ratio
# @njit
# def gamma(g,h,j,k, VV_in, tt_in, Vf_local, com_idx):
#     idx_v = com_idx[h, k]
#     if idx_v == -1:
#         idx_v = com_idx[k,h]
#     if idx_v == -1:
#         return 0.0
#     #val = (-VV_in[idx_v, j, g]*(tt_in[j,h]**2+tt_in[j,k]**2)*(tt_in[g,h]**2+tt_in[g,k]**2) + tt_in[g, h] ** 2 * tt_in[j, h] ** 2 + tt_in[g, k] ** 2 * tt_in[j, k] ** 2)
#     x = (tt_in[j,k]*tt_in[g,h])/(tt_in[j,h]*tt_in[g,k])
#     y = x + 1/x
#     #den = 2.0 * tt_in[g, h] * tt_in[j, k] * tt_in[j, h] * tt_in[g, k] * Vf_local
#     print('ok')
#     ratio = - 0.5
#     # clip
#     if ratio > 1.0:
#         ratio = 1.0
#     elif ratio < -1.0:
#         ratio = -1.0
#     return ratio
# Funzioni jittate per i loop pesanti
@njit
def compute_FF_magnitudes(FF_out, t_in, VV_in, com_idx, Vf_local, i1_local):
    rows = FF_out.shape[0]
    C_local = FF_out.shape[1]
    for g in range(rows):
        if g == i1_local:
            continue
        # trovo primo k con t[g,k] != 0
        k = 0
        found = False
        for kk in range(C_local):
            if t_in[g, kk] != 0:
                k = kk
                found = True
                break
        if not found:
            continue
        if t_in[g, k] == 0:
            continue
        for h in range(k + 1, C_local):
            if t_in[g, h] == 0:
                FF_out[g, h] = 0.0
            else:
                # indice di visibilità
                idx_v = com_idx[k, h]
                if idx_v == -1:
                    FF_out[g, h] = 0.0
                else:
                    ratio = gamma(g,h,i1_local,k, VV_in, t_in, Vf_local, com_idx)
                    # val = (-VV_in[idx_v, i1_local, g]*(t_in[i1_local,k]**2+t_in[i1_local,h]**2)*(t_in[g,k]**2+t_in[g,h]**2) + t_in[i1_local, k] ** 2 * t_in[g, k] ** 2 + t_in[i1_local, h] ** 2 * t_in[g, h] ** 2)
                    # den = 2.0 * t_in[i1_local, k] * t_in[g, h] * t_in[g, k] * t_in[i1_local, h] * Vf_local
                    # ratio = val / den
                    # # clip
                    # if ratio > 1.0:
                    #     ratio = 1.0
                    # elif ratio < -1.0:
                    #     ratio = -1.0
                    FF_out[g, h] = np.arccos(ratio)
    return FF_out

@njit
def compute_FF_signs_row_i2(FF_out, t_in, VV_in, com_idx, Vf_local, i1_local, i2_local):
    # calcolo i segni per la riga i2 (secondo pivot)
    C_local = FF_out.shape[1]
    k = 1
    for h in range(k + 1, C_local):
        idx_v = com_idx[k, h]
        if idx_v == -1:
            continue
        ratio = gamma(i2_local,h,i1_local,k, VV_in, t_in, Vf_local, com_idx)
        # val = (-VV_in[idx_v, i1_local, i2_local]*(t_in[i1_local,k]**2+t_in[i1_local,h]**2)*(t_in[i2_local,k]**2+t_in[i2_local,h]**2) + t_in[i1_local, k] ** 2 * t_in[i2_local, k] ** 2 + t_in[i1_local, h] ** 2 * t_in[i2_local, h] ** 2)
        # den = 2.0 * t_in[i1_local, k] * t_in[i2_local, h] * t_in[i2_local, k] * t_in[i1_local, h] * Vf_local
        # ratio = val / den
        # if ratio > 1.0:
        #     ratio = 1.0
        # elif ratio < -1.0:
        #     ratio = -1.0
        b = np.arccos(ratio)
        # s = sign(|b - |FF[i1,k] - FF[i1,h] - FF[i2,k] - FF[i2,h]|) - |b - |FF[i1,k] - FF[i1,h] - FF[i2,k] + FF[i2,h]|| )
        term1 = abs(b - abs(FF_out[i1_local, k] - FF_out[i1_local, h] - FF_out[i2_local, k] - FF_out[i2_local, h]))
        term2 = abs(b - abs(FF_out[i1_local, k] - FF_out[i1_local, h] - FF_out[i2_local, k] + FF_out[i2_local, h]))
        s = np.sign(term1 - term2)
        FF_out[i2_local, h] = FF_out[i2_local, h] * s
    return FF_out

@njit
def compute_FF_signs_other_rows(FF_out, t_in, VV_in, com_idx, Vf_local, i1_local, i2_local):
    rows = FF_out.shape[0]
    C_local = FF_out.shape[1]
    for g in range(rows):
        if g == i1_local or g == i2_local:
            continue
        # trovo primo k con t[g,k] != 0
        k = 0
        found = False
        for kk in range(C_local):
            if t_in[g, kk] != 0:
                k = kk
                found = True
                break
        if not found:
            continue
        if t_in[g, k] == 0:
            continue
        for h in range(k + 1, C_local):
            if t_in[g, h] == 0:
                FF_out[g, h] = 0.0
            else:
                idx_v = com_idx[k, h]
                if idx_v == -1:
                    FF_out[g, h] = 0.0
                else:
                    ratio = gamma(g,h,i2_local,k, VV_in, t_in, Vf_local, com_idx)
                    # val = (-VV_in[idx_v, i2_local, g]*(t_in[i2_local,k]**2+t_in[i2_local,h]**2)*(t_in[g,k]**2+t_in[g,h]**2) + t_in[i2_local, k] ** 2 * t_in[g, k] ** 2 + t_in[i2_local, h] ** 2 * t_in[g, h] ** 2)
                    # den = 2.0 * t_in[i2_local, k] * t_in[g, h] * t_in[g, k] * t_in[i2_local, h] * Vf_local
                    # ratio = val / den
                    # if ratio > 1.0:
                    #     ratio = 1.0
                    # elif ratio < -1.0:
                    #     ratio = -1.0
                    b = np.arccos(ratio)
                    term1 = abs(b - abs(FF_out[i2_local, k] - FF_out[i2_local, h] - FF_out[g, k] - FF_out[g, h]))
                    term2 = abs(b - abs(FF_out[i2_local, k] - FF_out[i2_local, h] - FF_out[g, k] + FF_out[g, h]))
                    s = np.sign(term1 - term2)
                    FF_out[g, h] = FF_out[g, h] * s
    return FF_out

@njit
def compute_U_from_MF(M_flat, F_flat, rows, cols):
    M = M_flat.reshape((rows, cols))
    F = F_flat.reshape((rows, cols))
    U = np.empty(M.shape, dtype=np.complex128)
    for i in range(rows):
        for j in range(cols):
            U[i, j] = M[i, j] * np.exp(1j * F[i, j])
    return U

@njit
def loss_jit(x_flat, rows, cols, t_in, VV_in, com_idx, Vf_local):
    # x_flat contains prima M.flatten() poi F.flatten()
    total_elements = rows * cols
    M_flat = x_flat[:total_elements]
    F_flat = x_flat[total_elements:]
    Ut = compute_U_from_MF(M_flat, F_flat, rows, cols)

    SS1 = 0.0
    # loss sulle visibilita'
    for h in range(cols):
        for k in range(h + 1, cols):
            idx_v = com_idx[h, k]
            if idx_v == -1:
                continue
            V = VV_in[idx_v]
            for i in range(rows):
                for j in range(i + 1, rows):
                    a = Ut[i, h] * Ut[j, k]
                    b = Ut[i, k] * Ut[j, h]
                    Vt1 = (abs(a + b) ** 2) * Vf_local
                    Vt2 = (1.0 - Vf_local) * (abs(a) ** 2 + abs(b) ** 2)
                    Vt = Vt1 + Vt2
                    diff = Vt - V[i, j]
                    SS1 += diff.real * diff.real  # diff is real but keep real part
    # loss sui moduli quadri
    M_mat = M_flat.reshape((rows, cols))
    SS2 = 0.0
    for i in range(rows):
        for j in range(cols):
            d = M_mat[i, j] - t_in[i, j]
            SS2 += d * d
    return SS1 + SS2 * 100.0


# Wrapper python per minimize che richiama la funzione jittata
def loss(x):
    return loss_jit(x, 128, C, t, VV, com_index, Vf)


def call(x):
    print(loss(x))


if __name__ == "__main__":

    # Percorsi e parametri
    data_path = '/media/dati_2/Progetto_128_modi_#01/'
    data_folder = 'Ricostruzione_unitaria_10_11_conf_03'
    save_path = join(data_path, data_folder)
    channels = ['b', 'c', 'd', 'e']
    C = len(channels)
    Vf = 0.78

    # Carico i dati
    with np.load(join(data_path, data_folder, 'moduliquadri.npz')) as f:
        t = np.sqrt(f['T'])

    with np.load(join(data_path, data_folder, 'visibilities_from_histogram.npz')) as f:
        VV = f['VV']

    # Costruisco una mappa (C x C) che da (h,k) ritorna l'indice in VV, oppure -1 se non esiste
    com_index = -1 * np.ones((C, C), dtype=np.int64)
    idx = 0
    for h in range(C):
        for k in range(h + 1, C):
            com_index[h, k] = idx
            idx += 1

    # Inizializzo FF
    FF = np.zeros((128, C))
    i1 = 79
    i2 = 0
    t_start = time.time()
    # Eseguo le fasi di inizializzazione usando le funzioni jittate
    FF = compute_FF_magnitudes(FF, t, VV, com_index, Vf, i1)
    print(FF)
    FF = compute_FF_signs_row_i2(FF, t, VV, com_index, Vf, i1, i2)
    FF = compute_FF_signs_other_rows(FF, t, VV, com_index, Vf, i1, i2)
    #print(FF)
    # Costruisco U iniziale
    U = t * np.exp(1j * FF)
    t_end = time.time()
    print(np.imag(U))
    print(np.sum(np.abs(U[:, 3])**2))
    print(f"Tempo inizializzazione: {t_end - t_start:.2f} secondi")
    '''
            # vettore iniziale
    x0 = np.append(t.flatten(), FF.flatten())

    # minimizzazione (nota: scipy.minimize non e' jittabile; chiamiamo la jitted loss dall'interno)
    res = minimize(loss, x0, callback=call)

    print(loss(x0), res.fun)

    # ricostruisco soluzioni
    rows = 128
    cols = C
    total = rows * cols
    M_opt = (res.x[:total]).reshape((rows, cols))
    F_opt = (res.x[total:]).reshape((rows, cols))

    Uo = M_opt * np.exp(1j * F_opt)
    '''
    #np.savez(join(save_path, 'Unitary_mat.npz'), U=U)