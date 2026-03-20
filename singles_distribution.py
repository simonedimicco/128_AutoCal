import numpy as np
import os

folder_path ='/media/dati_2/Progetto_128_modi_#01/Ricostruzione_unitaria_28_10/'
inputs =['b', 'c', 'd', 'e']
save_folder = 'singles_distribution'
save_path = os.path.join(folder_path, save_folder)
os.makedirs(save_path, exist_ok=True)


for input in inputs:
    file_folder  = f'Singles/{input}'
    file_path = os.path.join(folder_path, file_folder)
    file_list = os.listdir(file_path)
    distribution = np.zeros(128, dtype=np.int64)
    for i,file in enumerate(file_list):
        with np.load(os.path.join(file_path, file),allow_pickle=True) as data_load:
            t  = data_load['t_tot']
            c  = data_load['c_tot']
        distribution += np.bincount(c, minlength=128)
        print(f'input {input}, file {i} di {len(file_list)} processed')
    np.savez(os.path.join(save_path, input), distribution=distribution)
