#%%
import numpy as np
from numba import njit
import time 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from WhiteLib import T_sinc, merge_time_channel_arrays, compute_trigger_index, find_coincidences, Modes_separator, compact_results
from Spazzatura.WhiteDict import pulse_dict, retard_box, time_differences, fine_tuning
import pyvisa as visa
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime
strnow = lambda: datetime.now().strftime("%Y%m%d-%H%M%S")
strtoday = lambda: datetime.now().strftime("%Y_%m_%d")
strtimenow = lambda: datetime.now().strftime("%H:%M:%S")
#%%
def process_measurement(measurement, photons):
    trigger_channel=17
    trigger_channel_merged=trigger_channel +32
    sync_ch_1=3
    sync_ch_2=27

    times = [(t,c) for t,c in measurement]
    times_box_1,channels_box_1 = times[0]
    times_box_2,channels_box_2 = times[1]
    t_sync= T_sinc(TT1=times_box_1, 
                   CC1=channels_box_1, 
                   TT2=times_box_2, 
                   CC2=channels_box_2, 
                   ch1=sync_ch_1, 
                   ch2=sync_ch_2, 
                   window=100, 
                   mx=1e10)
    
    if t_sync>0:
        times_box_1 = times_box_1 + t_sync
    else:
        times_box_2 = times_box_2 - t_sync
    
    times_merged, channels_merged = merge_time_channel_arrays(times1 = times_box_1,
                                                              channels1 = channels_box_1,
                                                              times2 = times_box_2, 
                                                              channels2 = channels_box_2, 
                                                              ch_sync_1=sync_ch_1, 
                                                              ch_sync_2=sync_ch_2,
                                                              channel_offset=32)
    
    del times_box_1, channels_box_1, times_box_2, channels_box_2

    mask = times_merged != 0
    times_merged = times_merged[mask]
    channels_merged= channels_merged[mask]
    trigger_times = times_merged[channels_merged == trigger_channel_merged]
    window=np.mean(trigger_times[1:] - trigger_times[:-1]) 
    window_std=np.std(trigger_times[1:] - trigger_times[:-1])
    print(f"Estimated window: {window:.2f} ± {window_std:.2f} ps based on {len(trigger_times)} triggers.")
    # plt.plot(times_merged)
    # plt.show()
    # plt.plot(trigger_times)
    # plt.show()
    trig_idx = compute_trigger_index(channels_times=times_merged, 
                                     channels_list=channels_merged, 
                                     trigger_channel=trigger_channel_merged, 
                                     trigger_times=trigger_times, 
                                     window=725000)
    

    invalid_triggers = np.where(trig_idx == -1)[0]
    print(f"Found {len(invalid_triggers)} invalid triggers out of {len(times_merged)} total events.")
    t_tmp, c_tmp, valid = Modes_separator(channels_times=times_merged,
                                   channels_list=channels_merged,
                                   trig_idx=trig_idx,
                                   trigger_times=trigger_times,
                                   mio_dizionario=pulse_dict,
                                   Time_differences=time_differences,
                                   Time_differences_fine_tuning=fine_tuning,
                                   retards=retard_box,
                                   window=725000,
                                   offset=0)
    t_tot, c_tot = compact_results(t_tmp=t_tmp, c_tmp=c_tmp, valid=valid)
    print(len(t_tot))
    if photons==1:
        return t_tot, c_tot
    
    if photons==2:
        order = np.argsort(t_tot)
        t_sorted = t_tot[order]
        c_sorted = c_tot[order]
        window_ps = 1800
        
        coincidences = find_coincidences(t_sorted, c_sorted, window_ps, photons)
        return coincidences
    # Process the data as needed
    # For example, you could compute histograms, correlations, etc.
    return 
#%%
if __name__ == "__main__":
    # === CONFIGURAZIONE ===
    path = '/media/dati_2/'
    FOLDER_NAME = 'DATI_2026_02_12_noise_bcde'
    INPUT_FOLDER = os.path.join(path, FOLDER_NAME)
    # === RILEVA FILE ===
    files = [
        f for f in os.listdir(INPUT_FOLDER)
        if os.path.isfile(os.path.join(INPUT_FOLDER, f)) and f.endswith('.npz')
    ]
    files.sort()
    # === PROCESSA FILE ===
    for file in tqdm(files, desc="Processing files"):
        file_path = os.path.join(INPUT_FOLDER, file)
        trigger_channel=17
        trigger_channel_merged=trigger_channel +32
        sync_ch_1=3
        sync_ch_2=27
        with np.load(file_path,allow_pickle=True) as data_load:
            times_box_1     = data_load['times_box_1']
            channels_box_1  = data_load['channels_box_1']
            times_box_2     = data_load['times_box_2']
            channels_box_2  = data_load['channels_box_2']
        measurement = [(times_box_1, channels_box_1), (times_box_2, channels_box_2)]
        t_i= time.time()
        coincidences = process_measurement(measurement, photons=2)
        t_f = time.time()
        print(f"Processed {file} in {t_f - t_i:.2f} seconds, found {len(coincidences)} coincidences.")