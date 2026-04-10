import numpy as np
import os 
from tqdm import tqdm
from WhiteLib_lite import find_coincidences_numba, count_occurrences


path = '/media/dati_2/DATI_2026_04_09_all20'
#path = 'C:/Users/ControlCenter/Desktop/128_AutoCal_dati/DATI_2026_03_30'
singles_folder = os.path.join(path, 'Singles')
dark_folder = os.path.join(path, 'Buio')

inputs = ['b', 'c', 'd', 'e']
S = np.zeros((4, 128), dtype=np.float64)
B = np.zeros((4, 128), dtype=np.float64)
for i, channel in tqdm(enumerate(inputs), desc="Processing singles distributions", total=len(inputs), ncols=100):
    files_signles = [f for f in os.listdir(os.path.join(singles_folder, channel)) if f.endswith('.npz')]
    for file in files_signles:
        with np.load(os.path.join(singles_folder, channel, file)) as data:
            c_tot = data['c_tot']
            S[i] += np.bincount(c_tot, minlength=128)
    S[i] = S[i]/len(files_signles)
    files_dark = [f for f in os.listdir(os.path.join(dark_folder, channel)) if f.endswith('.npz')]
    for file in files_dark:
        with np.load(os.path.join(dark_folder, channel, file)) as data:
            c_tot = data['c_tot']
            B[i] += np.bincount(c_tot, minlength=128)
    B[i] = B[i]/len(files_dark)
S = S - B
S[S < 0] = 0
for i in range(S.shape[0]):
    S[i] = S[i]/np.sum(S[i])
np.savez(os.path.join(path,'singles_distributions.npz'), distributions=S)

couples = ['bc', 'bd', 'be', 'cd', 'ce', 'de']
shape=(128,128)
C = np.zeros((6, 128, 128), dtype=np.float64)
for i, couple in tqdm(enumerate(couples), desc="Processing couples distributions", total=len(couples), ncols=100):
    files_couples= [f for f in os.listdir(os.path.join(path,'measurement_2ph_'+couple))]
    for file in files_couples:
        with np.load(os.path.join(path ,'measurement_2ph_'+couple, file)) as data:
            c_tot = data['c_tot']
            t_tot = data['t_tot']
            order = np.argsort(t_tot)
            t_sorted = t_tot[order]
            c_sorted = c_tot[order]
            coincidences = find_coincidences_numba(t_sorted, c_sorted, window_ps=1800, target_n=2)
            print(f"File: {file},\tCoincidences found: {len(coincidences)}")
            C[i] += count_occurrences(shape=shape,  data=coincidences)

for i in range(C.shape[0]):
    C[i] = C[i]/(np.sum(C[i])/2)
np.savez(os.path.join(path,'couples_distributions.npz'), distributions=C)
            
