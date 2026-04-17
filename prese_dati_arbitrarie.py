# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:32:11 2026

@author: ControlCenter
"""

from qlab.devices.tdc import QuTag
import qlab.counting.counting as counting
import qlab.counting.cocount as cocount
from auto_classical import PowerSupplies
from qlab.devices.KeithleyPowerSupply import KeithleyPowerSupply
import logging
from dmx_controller import DMXController
#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time 
import pyvisa as visa
from tqdm import tqdm
from WhiteLib import process_measurement, setloop, change_voltages, split_times_by_channel, all_inter_histograms, all_intra_histograms, data_collection, data_collection_new, data_collection_parallel
from datetime import datetime
strnow = lambda: datetime.now().strftime("%Y%m%d-%H%M%S")
strtoday = lambda: datetime.now().strftime("%Y_%m_%d")
strtimenow = lambda: datetime.now().strftime("%H:%M:%S")
import time
import os


#%%
'''
INIZIALIZATION KEYTHLEYS
'''
#%%
rm = visa.ResourceManager()
print(rm.list_resources())
list_conncted=[x for x in rm.list_resources()]
addresses = ['ASRL24::INSTR', 'ASRL25::INSTR', 'ASRL26::INSTR', 'ASRL27::INSTR', 'ASRL28::INSTR', 'ASRL29::INSTR','USB0::0x05E6::0x2230::9100018::INSTR', 'USB0::0x05E6::0x2230::9102515::INSTR', 'ASRL6::INSTR', 'ASRL8::INSTR']
print(f'You are connecting {len(addresses)} Keithleys')
volts = [[0,0] for _ in range(len(addresses))]
#%%
supply = PowerSupplies(addresses)
change_voltages(supply, volts)

#%%
# Target volts
volts[0]= [5.601,4.346]
volts[1]= [5.367,3.763]
volts[2]= [3.396,5.966]
volts[3]= [4.299,5.298]
volts[4]= [5.832,5.795]
volts[5]= [5.099,4.853]
volts[6]= [4.801,4.724]
volts[7]= [3.132,3.577]
volts[8]= [4.594,5.756]
volts[9]= [3.842,5.787]
print(f'You are going to set the following voltages:')
print(volts)
#%%
change_voltages(supply, volts)
#%%
'''
INIZIALIZATION QU-TAGS
'''
#%%
boxes_ind = QuTag.discover()

#%%
ind = [b'T 02 0010', b'T 02 0021']
boxes = []
for i in ind:
    for j in boxes_ind:
        if j._sn == i:
           boxes.append(j)

if len(ind) != len(boxes):
    raise ValueError('scatole non trovate')

#%%
'''
INIZIALIZATION QDMX
'''
#%%
channel_a= 0
channel_b= 1
channel_c= 2
channel_d= 3
channel_e= 4
channel_f= 5
#%%
dmx = DMXController(log_level=logging.DEBUG)
#%%
dmx.stop_looping()
#%%
print(dmx.get_data())
#%%

dmx.set_dwell_time(channel=channel_a, dwell_time=26)
dmx.set_dwell_time(channel=channel_b, dwell_time=26)
dmx.set_dwell_time(channel=channel_c, dwell_time=26)
dmx.set_dwell_time(channel=channel_d, dwell_time=26)
dmx.set_dwell_time(channel=channel_e, dwell_time=16)
dmx.set_dwell_time(channel=channel_f, dwell_time=16)
#%%
'''
SET WORKING DIRECTORY
'''
#%%
path='C:/Users/ControlCenter/Desktop/128_AutoCal_dati/'
dir_name = os.path.join(path,'DATI_' + strtoday())
#dir_name = path+'misure_cluce_classica'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
print(dir_name)
save_path = dir_name

#%%
with open(os.path.join(path,dir_name) + '/'+"readme.txt", "w") as file:
    file.write("trigger ch 17 box 2\n")
    file.write('sync channels: ch 3 box 1 and ch 27 box 2\n')
    file.write("voltages:\n")
    for voltage in supply.voltages_measure:
        file.write(f"{voltage:.3f}\n")
    file.write("currents:\n")
    for curr in supply.currents_measure:
        file.write(f'{curr:.3e}\n')
#%%
''''
EXPERIMENTAL SECTION
'''
#%%
#parametri istogrammi
n_channels = 128
bin_width = 400
histo_width = 4 * 1e5
n_bins = int(histo_width / bin_width)
half = (n_bins // 2) * bin_width
bin_edges = np.linspace(-half, half, n_bins + 1)
centers = (np.arange(n_bins) - n_bins // 2) * bin_width
n_pairs = n_channels // 2


#%%
esposizione = 0.1   #in secondi
durata= 60   #in secondi
ripetizioni= int(durata/esposizione)

#%%
names=['b','c', 'd', 'e']
#Voltages=[0 for _ in range(len(addresses))]
inputs = [(1,), (2,), (3,), (4,)]
inputs = [(2,)]
inputs = [(1,2,3,4)]

for sequence in inputs:
    
    photons = len(sequence)
    name = ''.join(names[i-1] for i in sequence)
    save_folder = os.path.join(dir_name,name)
    os.makedirs(save_folder, exist_ok=True)
    loop = setloop(sequence)
    dmx.set_active_outputs(loop)
    time.sleep(1)
    hist_totals_inter = np.zeros((n_pairs, n_bins), np.int64)
    hist_totals_intra = np.zeros((n_pairs, n_bins), np.int64)
    for i in range(30):
        print(f'Measurement channels {name} {i+1}/30 - started at {strtimenow()}')
        measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
        times = [(t,c) for t,c in measure]
        times_box_1,channels_box_1 = times[0]
        times_box_2,channels_box_2 = times[1]
        t_tot, c_tot = process_measurement(times, photons=0)
        times_by_ch = split_times_by_channel(t_tot, c_tot, n_channels)
        h_inter = all_inter_histograms(times_by_ch, n_channels, bin_width, n_bins)
        h_intra = all_intra_histograms(times_by_ch, n_channels, bin_width, n_bins)
        hist_totals_inter += h_inter
        hist_totals_intra += h_intra
        np.savez(os.path.join(save_folder, f'histo_intra_{name}.npz'), histo_totals=hist_totals_intra, bin_edges=bin_edges)
        np.savez(os.path.join(save_folder, f'histo_inter_{name }.npz'), histo_totals=hist_totals_inter,  bin_edges=bin_edges)
    dmx.stop_looping()
    time.sleep(1)
#%%
Nsupp = len(addresses)
volts_list = [x for coppia in volts for x in coppia]
volts = volts_list
#%%
logging.disable(logging.DEBUG)
inputs = [(1,2)]
exposition = 0.1
duration= 6
repetitions_singles = 1
repetitions_doubles = 10

if __name__== "__main__":
    supply = PowerSupplies(addresses)
    boxes_ind = QuTag.discover()
    ind = [b'T 02 0010', b'T 02 0021']
    boxes = []
    for i in ind:
        for j in boxes_ind:
            if j._sn == i:
               boxes.append(j)

    if len(ind) != len(boxes):
        raise ValueError('scatole non trovate')
    dmx = DMXController(log_level=logging.DEBUG)
    t1 = time.time()
    results_new = data_collection_fallita(inputs, volts, supply, Nsupp, dmx, boxes, exposition, duration, repetitions_singles, repetitions_doubles)
    t2 = time.time()
    results = data_collection(inputs, volts, supply, Nsupp, dmx, boxes, exposition, duration, repetitions_singles, repetitions_doubles)
    
    t3 = time.time()
    print(f"Data collection new completed in {t2-t1:.2f} seconds")
    print(f"Data collection completed in {t3-t2:.2f} seconds")
    
    diff = abs(results_new - results)/results
    plt.plot(diff.flatten(), 'o')
    plt.title('Difference between old and new data collection')
    plt.xlabel('Index')
    plt.ylabel('Difference')
    plt.show()
        

#%%
dmx.stop_looping()
#%%
logging.disable(logging.DEBUG)
path='C:/Users/ControlCenter/Desktop/128_AutoCal_dati/'



with open(path+'log_file.txt', "w") as log_file:
    t0=time.time()
    measure_currents= supply.currents_measure
    #measure_voltages= supply.voltages_measure
    log_file.write(f'{strnow()}\n')
    #log_file.write('Voltages: ' + str(measure_voltages) + '\n')
    log_file.write('Currents: ' + str(measure_currents) + '\n')
    t1=time.time()
    
print(f'{t1-t0:.2f}s')


