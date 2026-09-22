import numpy as np
from scipy.optimize import minimize

from itertools import combinations
import matplotlib.pyplot as plt
from os.path import join
from os import listdir
from numba import njit, prange
import numba
from tqdm import tqdm
from visibility_HOM_lib import *
 
#%%

data_path = '/media/dati_2/DATI_2026_06_21_128modi_training_target6N_3PairsPre_32Start_2'
data_folder = 'Ricostruzione_unitaria'
save_path = join(data_path,data_folder)
channels = ['b','c','d','e']


#%%  Analisi singole

T = []

# Store raw accumulated counts for MC resampling
singles_raw = {}
dark_raw = {}
singles_file_counts = {}
dark_file_counts = {}

for c in tqdm(channels,'single'):
    # Accumulate dark counts
    dark_N = 0
    B_raw = np.zeros(128, dtype=int)
    for file in listdir(join(data_path,  data_folder,'Buio',c)):
        dark_N += 1
        with np.load(join(data_path,  data_folder,'Buio',c, file)) as f:
            S = f["c_tot"]
        B_raw += np.bincount(S, minlength=128)
    
    # Accumulate singles counts
    singles_N = 0
    M_raw = np.zeros(128, dtype=int)
    for file in listdir(join(data_path, data_folder, 'Singles', c)):
        singles_N += 1
        with np.load(join(data_path, data_folder, 'Singles', c, file)) as f:
            S = f["c_tot"]
        M_raw += np.bincount(S, minlength=128)
    
    # Store raw counts and file counts for MC
    singles_raw[c] = M_raw
    dark_raw[c] = B_raw
    singles_file_counts[c] = singles_N
    dark_file_counts[c] = dark_N
    
    # Original processing (unchanged)
    B = B_raw / dark_N
    M = M_raw / singles_N - B
    
    M = np.where(M>0,M,0)
    
    M = M/np.sum(M)
    non_zero = np.count_nonzero(M)

    print(non_zero)
    T.append(M)

T = np.array(T)

T = T.transpose()

#%% Salvataggio

# Save moduliquadri_mc.npz with raw counts for MC resampling
# Using per-channel field names as specified in data_dictionary.md
np.savez(
    join(save_path, 'moduliquadri_mc'),
    singles_raw_b=singles_raw['b'],
    singles_raw_c=singles_raw['c'],
    singles_raw_d=singles_raw['d'],
    singles_raw_e=singles_raw['e'],
    dark_raw_b=dark_raw['b'],
    dark_raw_c=dark_raw['c'],
    dark_raw_d=dark_raw['d'],
    dark_raw_e=dark_raw['e'],
    singles_file_count_b=singles_file_counts['b'],
    singles_file_count_c=singles_file_counts['c'],
    singles_file_count_d=singles_file_counts['d'],
    singles_file_count_e=singles_file_counts['e'],
    dark_file_count_b=dark_file_counts['b'],
    dark_file_count_c=dark_file_counts['c'],
    dark_file_count_d=dark_file_counts['d'],
    dark_file_count_e=dark_file_counts['e']
)

np.savez(join(save_path,'moduliquadri'), T = T)


#%%
com = [f'{a}{b}' for a,b in combinations(channels,2)]


VV = []

# Initialize storage for visibilities_mc.npz
visibilities_mc_data = {}

for c in tqdm(com, 'Doppie'):
    N = 0
    file_name = f'{c}.npz'
    with np.load(join(data_path,  data_folder,'histo_doppie',file_name)) as data:
        hist_totals = data['hist_totals']
        bin_edges = data['bin_edges']

    bin_centrale = 0  
    rmse_array=[]
    int_array=[]
    n_modes=128
    V=np.zeros((n_modes,n_modes))
    
    # Initialize lists to store histogram data for MC
    pair_ii_list = []
    pair_iii_list = []
    pair_y_int_list = []
    pair_sigma_y_int_list = []
    pair_y_noise_int_list = []
    pair_sigma_y_noise_int_list = []
    pair_area_c_list = []
    pair_valid_list = []
    
    for i in range(0,len(hist_totals)):
    #for i in range(1): #prova per stampare
        ii,iii=idx_to_pair(n_modes, i)
        bin_values = hist_totals[i]
        #plot_picchi_con_fit(bin_edges, bin_values, bin_centrale, distanza_bin=7)
        finestra=1800
        pos_sx, val_sx, pos_dx, val_dx, pos_centr, val_centr , pos_noise_sx, val_noise_sx, pos_noise_dx, val_noise_dx = trova_picchi(bin_edges, bin_values, bin_centrale, distanza_bin=7)
    
        if len(pos_sx) < 2 or len(pos_dx) < 2 or pos_centr is None:
            # Skipped histogram: save with valid=0 and placeholder values
            pair_ii_list.append(ii)
            pair_iii_list.append(iii)
            pair_y_int_list.append(0.0)
            pair_sigma_y_int_list.append(0.0)
            pair_y_noise_int_list.append(0.0)
            pair_sigma_y_noise_int_list.append(0.0)
            pair_area_c_list.append(0.0)
            pair_valid_list.append(0)
            continue
        
        # Integrare i picchi laterali con finestra di 2.5 ns (nella funzione in ps)
        pos_dx_int, area_dx = integra_picchi(bin_edges, bin_values, pos_dx, finestra=finestra)
        pos_sx_int, area_sx = integra_picchi(bin_edges, bin_values, pos_sx, finestra=finestra)

        pos_noise_sx, area_noise_sx = integra_picchi(bin_edges, bin_values, pos_noise_sx, finestra=finestra)
        pos_noise_dx, area_noise_dx = integra_picchi(bin_edges, bin_values, pos_noise_dx, finestra=finestra)
        area_noise_sx = val_noise_sx * finestra
        area_noise_dx = val_noise_dx * finestra

        # Integrare anche il centrale
        pos_c_int, area_c = integra_picchi(bin_edges, bin_values, [pos_centr], finestra=1800)

        # print("Picco centrale:", pos_centr, val_centr)
        # print("Picchi SX:", pos_sx, val_sx)
        # print("Picchi DX:", pos_dx, val_dx)
        
        m_sx, q_sx, rmse_sx, pcov_sx = fit_retta(pos_sx_int, area_sx)
        m_dx, q_dx, rmse_dx, pcov_dx = fit_retta(pos_dx_int, area_dx)

        m_noise_sx, q_noise_sx, rmse_noise_sx, pcov_noise_sx = fit_retta(pos_noise_sx, area_noise_sx)
        m_noise_dx, q_noise_dx, rmse_noise_dx, pcov_noise_dx = fit_retta(pos_noise_dx, area_noise_dx)

        # print("Fit SX:", m_sx, q_sx, "RMSE:", rmse_sx)
        # print("Fit DX:", m_dx, q_dx, "RMSE:", rmse_dx)

        # intersezione
        x_int, y_int = intersezione_rette(m_sx, q_sx, m_dx, q_dx)
        x_noise_int, y_noise_int = intersezione_rette(m_noise_sx, q_noise_sx, m_noise_dx, q_noise_dx)
        
        # errore sulla y di intersezione
        sigma_y_int = errore_intersezione_y(m_sx, q_sx, pcov_sx, m_dx, q_dx, pcov_dx)
        sigma_y_noise_int = errore_intersezione_y(m_noise_sx, q_noise_sx, pcov_noise_sx, m_noise_dx, q_noise_dx, pcov_noise_dx)
        # print("Intersezione rumore:", x_noise_int, y_noise_int)

        # print("Intersezione:", x_int, y_int)
        rmse_array.append((rmse_sx+rmse_dx)/2)
        int_array.append(y_int)
        # picco centrale (ad esempio)
        
        #########################################################################
        '''
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.figure(figsize=(9,6))
        plt.bar(bin_centers, bin_values, width=bin_edges[1]-bin_edges[0], alpha=0.5, color="skyblue", edgecolor="k")

        # picchi (x gialle)
        plt.scatter(pos_sx, val_sx, color="yellow", marker="x", s=100, label="Picchi SX")
        plt.scatter(pos_dx, val_dx, color="yellow", marker="x", s=100, label="Picchi DX")
        plt.scatter(pos_centr,  val_centr , color="red", marker="x", s=120, label="Picco centrale")
        plt.scatter(pos_noise_sx,   val_noise_sx , color="green", marker=".", s=80, label="Minimi rumore SX")
        plt.scatter(pos_noise_dx,   val_noise_dx,  color="green", marker=".", s=80, label="Minimi rumore DX")

        # integrali normalizzati (o blu)
        plt.scatter(pos_sx_int, area_sx/1800, color="blue", marker="o", s=80, label="Integrali norm. SX")
        plt.scatter(pos_dx_int, area_dx/1800, color="blue", marker="o", s=80, label="Integrali norm. DX")
        plt.scatter(pos_centr, area_c/1800, color="blue", marker="o", s=100, label="Integrale norm. centrale")
        plt.scatter(pos_noise_sx, area_noise_sx/1800, color="orange", marker=".", s=80, label="Integrali rumore SX")
        plt.scatter(pos_noise_dx, area_noise_dx/1800, color="orange", marker=".", s=80, label="Integrali rumore DX")

        plt.xlabel("x")
        plt.ylabel("Conteggi / Integrali normalizzati")
        plt.title("Istogramma con picchi e fit sugli integrali normalizzati")
        plt.legend()
        plt.show()
        '''
        #########################################################################
        # distanze
        d_int, d_r1, d_r2 = distanze_picco(m_sx, q_sx, m_dx, q_dx, x_int, y_int, pos_c_int, area_c)
        V[ii,iii]=d_int/(y_int-y_noise_int)
        V[iii,ii]=d_int/(y_int-y_noise_int)
        
        # Store data for MC (valid histogram)
        pair_ii_list.append(ii)
        pair_iii_list.append(iii)
        pair_y_int_list.append(y_int)
        pair_sigma_y_int_list.append(sigma_y_int)
        pair_y_noise_int_list.append(y_noise_int)
        pair_sigma_y_noise_int_list.append(sigma_y_noise_int)
        pair_area_c_list.append(area_c[0] if len(area_c) > 0 else 0.0)
        pair_valid_list.append(1)
    
    #V = V / (np.sum(V)/2)
    print(V.sum())
    
    # Store pair data for visibilities_mc.npz
    visibilities_mc_data[f'pair_{c}_ii'] = np.array(pair_ii_list, dtype=np.int64)
    visibilities_mc_data[f'pair_{c}_iii'] = np.array(pair_iii_list, dtype=np.int64)
    visibilities_mc_data[f'pair_{c}_y_int'] = np.array(pair_y_int_list, dtype=np.float64)
    visibilities_mc_data[f'pair_{c}_sigma_y_int'] = np.array(pair_sigma_y_int_list, dtype=np.float64)
    visibilities_mc_data[f'pair_{c}_y_noise_int'] = np.array(pair_y_noise_int_list, dtype=np.float64)
    visibilities_mc_data[f'pair_{c}_sigma_y_noise_int'] = np.array(pair_sigma_y_noise_int_list, dtype=np.float64)
    visibilities_mc_data[f'pair_{c}_area_c'] = np.array(pair_area_c_list, dtype=np.float64)
    visibilities_mc_data[f'pair_{c}_valid'] = np.array(pair_valid_list, dtype=np.int64)
        
    VV.append(V)


# Save visibilities_mc.npz with all histogram data for MC resampling
np.savez(join(save_path, 'visibilities_mc'), **visibilities_mc_data)

np.savez(join(save_path,'visibilities_from_histogram'), VV = VV)     

