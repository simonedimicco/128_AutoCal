# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:59:20 2026

@author: ControlCenter
"""

'''
IMPORT LIBRARIES SECTION:
'''
#%%
from qlab.devices.tdc import QuTag
import qlab.counting.counting as counting
import qlab.counting.cocount as cocount
from auto_classical import PowerSupplies
from qlab.devices.KeithleyPowerSupply import KeithleyPowerSupply
import logging
from dmx_controller import DMXController
#%%
import numpy as np
from numba import njit
import time 
import pyvisa as visa
from tqdm import tqdm
from WhiteLib import process_measurement
from datetime import datetime
strnow = lambda: datetime.now().strftime("%Y%m%d-%H%M%S")
strtoday = lambda: datetime.now().strftime("%Y_%m_%d")
strtimenow = lambda: datetime.now().strftime("%H:%M:%S")
import time
#%%
'''
FUNCTIONS SECTION:
'''
#%%
def setloop(input):
    loop =[0,0,0,5]
    for channel in input:
        if channel != 0 and channel != 5:
            loop[channel-1]=channel 
        else:
            print('You are measuring the dark')

    return loop

def control_volts(pairs):
    """
    Controlla ogni coppia in `pairs`: se uno dei due valori 0 o > 8,
    stampa un errore e imposta la coppia a [0, 0].
    """
    for idx, (a, b) in enumerate(pairs):
        # verifica se a o b sono fuori dall'intervallo 0\Uffffffff8
        if not (0 <= a <= 8 and 0 <= b <= 8):
            print(f"Errore: elementi fuori range in pairs[{idx}] = {pairs[idx]}")
            return False
    
    return True
def flatten_list(pairs):
    # prende ogni coppia in pairs e ogni elemento x in quella coppia
    return [x for pair in pairs for x in pair]

def change_voltages(supply: PowerSupplies, volts) -> None:
    assert isinstance(supply, PowerSupplies)
    assert isinstance(volts, list)
    assert supply._num_supplies==len(volts)
    if control_volts(volts):  
        volts = flatten_list(volts)
        supply.voltages = volts
        print('Volts changed ->' + str(supply.voltages))
    else:
        volts = [[0, 0] for _ in range(supply._num_supplies)]
        supply.voltages = flatten_list(volts)
        print('Volts reset to 0 due to error ->' + str(supply.voltages))

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
supply = PowerSupplies(addresses)
change_voltages(supply, volts)

#%%
raw = np.random.uniform(0, 54, size=len(addresses)*2)
# 2) trasformali con la radice quadrata → [0,8]
volt_vals = np.sqrt(raw)
# 3) rimodella in 6x2 e ottieni la list-of-lists
volts = volt_vals.reshape((len(addresses), 2)).tolist()
print(f'You are going to set the following voltages:')
print(volts)
#%%
change_voltages(supply, volts)

#%%
#pretraining voltages
volts[0]= [26.91770056, 20.25482259]
volts[1]= [25.13671606, 12.33392859] 
volts[2]= [14.39218764 ,38.11764358]
volts[3]= [15.06129933, 26.79019464] 
volts[4]= [32.93635505, 34.19253223]
volts[5]= [26.58369102, 22.61000908]
volts[6]= [24.09356394, 23.83949231]
volts[7]= [12.50908916, 17.35610151] 
volts[8]= [23.04587543, 28.30849853] 
volts[9]= [11.9300337 , 30.21736944]
volts_array = list( np.sqrt(volts))
volts = [[float(x), float(y)] for x, y in volts_array]
print(f'You are going to set the following voltages:')
print(volts)

#%%
#aftertraining voltages
volts[0]=[28.52133494, 19.29275721]
volts[1]=[26.45806454, 12.33392859]
volts[2]=[14.07755464, 39.44090354]
volts[3]=[15.34390228, 28.40254952]
volts[4]=[32.2095023 , 32.60211475]
volts[5]=[26.81896368, 21.92135263]
volts[6]=[23.63006376, 24.02677582]
volts[7]=[12.10942377, 15.74827279]
volts[8]=[22.66837316, 29.22583959]
volts[9]=[13.56544436, 30.54044808]
volts_array = list( np.sqrt(volts))
volts = [[float(x), float(y)] for x, y in volts_array]
print(f'You are going to set the following voltages:')
print(f'You are going to set the following voltages:')
print(volts)
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
#volts[8]= [4.594,5.756]
volts[8]= [4.594,0]
#volts[9]= [3.842,5.787]
volts[9]= [3.842,0]

print(f'You are going to set the following voltages:')
print(volts)
#%%
4.982, 6.744
5.936, 4.612
6.481, 5.619
6.817, 4.076
4.217, 2.074
4.483, 5.363
5.077, 1.618
4.077, 7.307
5.451, 1.067
6.470, 6.944
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
dwell_time=24
dmx.set_dwell_time(channel=channel_a, dwell_time=dwell_time)
dmx.set_dwell_time(channel=channel_b, dwell_time=dwell_time)
dmx.set_dwell_time(channel=channel_c, dwell_time=dwell_time)
dmx.set_dwell_time(channel=channel_d, dwell_time=dwell_time)
dmx.set_dwell_time(channel=channel_e, dwell_time=dwell_time)
dmx.set_dwell_time(channel=channel_f, dwell_time=dwell_time)

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
dir_name = path+'DATI_' + strtoday() + '_target_2'
#dir_name = path+'misure_cluce_classica'
import os
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
esposizione = 0.1   #in secondi
durata= 60   #in secondi
ripetizioni= int(durata/esposizione)

#%%
names=['b','c', 'd', 'e']
#Voltages=[0 for _ in range(len(addresses))]
inputs = [(1,), (2,), (3,), (4,), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]


for sequence in inputs:
    
    photons = len(sequence)
    
    if photons == 1:
        folder_dark= os.path.join(save_path, f'measurement_dark_{names[sequence[0]-1]}')
        os.makedirs(folder_dark, exist_ok=True )
        folder_signal = os.path.join(save_path, f'measurement_1ph_{names[sequence[0]-1]}')
        os.makedirs(folder_signal, exist_ok=True )
        loop = setloop((0,))
        dmx.set_active_outputs(loop)
        time.sleep(1)
        
        for i in range(5):
            print(f'Dark measurement channel {names[sequence[0]-1]} {i+1}/5 - started at {strtimenow()}')
            save_name_dark = os.path.join(folder_dark, f'misura_{i+1}')
            measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
            times = [(t,c) for t,c in measure]
            times_box_1,channels_box_1 = times[0]
            times_box_2,channels_box_2 = times[1]
            t_tot, c_tot = process_measurement(times, photons=0)
            np.savez_compressed(save_name_dark, t_tot=t_tot, c_tot=c_tot)
        dmx.stop_looping()
        time.sleep(1)
        loop = setloop(sequence)
        dmx.set_active_outputs(loop)
        time.sleep(1)
        
        for i in range(5):
            print(f'Singles measurement channel {names[sequence[0]-1]} {i+1}/5 - started at {strtimenow()}')
            save_name_signal = os.path.join(folder_signal, f'misura_{i+1}')
            measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
            times = [(t,c) for t,c in measure]
            times_box_1,channels_box_1 = times[0]
            times_box_2,channels_box_2 = times[1]
            t_tot, c_tot = process_measurement(times, photons=0)
            np.savez_compressed(save_name_signal, t_tot=t_tot, c_tot=c_tot)
        dmx.stop_looping()
        time.sleep(1)
        
    elif photons==2:
        folder_couples= os.path.join(save_path, f'measurement_2ph_{names[sequence[0]-1]}{names[sequence[1]-1]}')
        os.makedirs(folder_couples, exist_ok=True )
        loop = setloop(sequence)
        dmx.set_active_outputs(loop)
        time.sleep(1)
        for i in range(20):
            print(f'Couples measurement channels {names[sequence[0]-1]}{names[sequence[1]-1]} {i+1}/20 - started at {strtimenow()}')
            save_name_couples = os.path.join(folder_couples, f'misura_{i+1}')
            measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
            times = [(t,c) for t,c in measure]
            times_box_1,channels_box_1 = times[0]
            times_box_2,channels_box_2 = times[1]
            t_tot, c_tot = process_measurement(times, photons=0)
            np.savez_compressed(save_name_couples, t_tot=t_tot, c_tot=c_tot)
        dmx.stop_looping()
        time.sleep(1)
        
    else:
        raise ValueError('Invalid number of chosen inputs')
            
#%%

inputs = [(1,2,3)]
for sequence in inputs:
    photons = len(sequence)
    folder_txt = os.path.join(dir_name, 'threefold_coincidences')
    os.makedirs(folder_txt, exist_ok=True)
    loop = setloop(sequence)
    dmx.set_active_outputs(loop)
    time.sleep(1)
    for i in range(100):
        print(f'Boson sampling measurement {i+1}/100 - started at {strtimenow()}')
        file_txt = os.path.join(folder_txt, f'misura_{i+1}.txt')
        measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
        times = [(t,c) for t,c in measure]
        times_box_1,channels_box_1 = times[0]
        times_box_2,channels_box_2 = times[1]
        coincidences = process_measurement(times, photons=photons)
        with open(file_txt, 'w') as f:
            for group in coincidences:
                f.write(f"{tuple(int(x) for x in group)}\n")
    dmx.stop_looping()
    time.sleep(1)
#%%
volts = [[0,0] for _ in range(len(addresses))]
change_voltages(supply, volts)

#%%

del dmx