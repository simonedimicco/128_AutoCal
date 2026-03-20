import numpy as np
import os
from WhiteLib_no_setup import find_coincidences_numba# mantieni il tuo modulo
from itertools import combinations
import time
from numba import njit

@njit
def count_occurrences(shape, data):
    key = np.zeros(shape, dtype=np.int64)
    for row in data:
        a = row[0]
        b = row[1]
        key[a, b] += 1
        key[b, a] += 1
    return key





folder_path ='/media/dati_2/Progetto_128_modi_#01/Ricostruzione_unitaria_28_10/'
coppie =['bc', 'bd', 'be', 'cd', 'ce', 'de']
n=128
m=2
com= np.array(list(combinations(range(n), m)))
shape = (128,128)
for coppia in coppie:
    folder_name = f'complete_processed_dati_28_10_2ph_{coppia}/'
    filepath = os.path.join(folder_path, folder_name)
    npz_files = [f for f in os.listdir(filepath) if f.endswith('.npz')]
    npz_files.sort()
    print(npz_files)
    #saving_path = f'/media/dati_2/'
    saving_folder = f'Couple_coincidences_distributions/'

    savepath = os.path.join(folder_path, saving_folder)
    os.makedirs(savepath, exist_ok=True)

#%%
    tmp = []
    for i,file_name in enumerate(npz_files):
        file_name= npz_files[i]
        with np.load(filepath+file_name,allow_pickle=True) as data_load:
            t  = data_load['t_tot']
            c  = data_load['c_tot']

        mask = t != 0
        t = t[mask]
        c= c[mask]

        order = np.argsort(t)
        t_sorted = t[order]
        c_sorted = c[order]
        del t, c, order
        window_ps = 1800 # finestra di coincidenza in ps


        t_i=time.time()
        coincidences = find_coincidences_numba(t_sorted, c_sorted, window_ps, target_n=2)
        t_f=time.time()
        print(coincidences[0])
        print(f'coppia {coppia}, file {i}, coincidenze trovate in {t_f-t_i:.2f}s')
        tmp.append(coincidences)

    coincidences_tot = np.concatenate(tmp, axis=0)
   
    t_i = time.time()
    distribution = count_occurrences(shape, np.array(coincidences_tot))
    t_f = time.time()
    print(f'coppia {coppia}, distribuzione ottenuta in {t_f-t_i:.2f}s')
    print(distribution[20,60])
    print(distribution[60,20])
    np.savez(os.path.join(savepath,coppia), distribution=distribution)