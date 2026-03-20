

#%%
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
volts = [[0,0] for _ in range(len(addresses))]
supply = PowerSupplies(addresses)
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
channel_f= 4
channel_e= 5

dmx = DMXController(log_level=logging.DEBUG)
dmx.set_dwell_time(channel=channel_a, dwell_time=41)
dmx.set_dwell_time(channel=channel_b, dwell_time=41)
dmx.set_dwell_time(channel=channel_c, dwell_time=41)
dmx.set_dwell_time(channel=channel_d, dwell_time=41)
dmx.set_dwell_time(channel=channel_e, dwell_time=31)
dmx.set_dwell_time(channel=channel_f, dwell_time=31)
#%%
'''
SET WORKING DIRECTORY
'''
#%%
path='C:/Users/ControlCenter/Desktop/Auto_calibration/'
dir_name = path+'DATI_' + strtoday()
#dir_name = path+'misure_cluce_classica'
import os
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
print(dir_name)
save_path = os.path.join(dir_name, 'misura_' + strtimenow())
#%%
''''
EXPERIMENTAL SECTION
'''
#%%
esposizione = 0.1   #in secondi
durata= 10   #in secondi
ripetizioni= int(durata/esposizione)
n_epochs=10**3
#%%

Voltages=[0 for _ in range(len(addresses))]
inputs = [(1,), (2,), (3,), (4,), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]

if __name__ == '__main__':
    for input in inputs:
        loop = setloop(input)
        dmx.set_active_outputs(loop)
        #change_voltages(supply, loop)
        photons = len(input)
        time.sleep(1)
        save_folder_processed = os.path.join(save_path, f'misura_{str(input)}_processed')
        if not os.path.exists(save_folder_processed):
            os.mkdir(save_folder_processed)
        save_folder_raw = os.path.join(save_path, f'misura_{str(input)}_raw')
        if not os.path.exists(save_folder_raw):
            os.mkdir(save_folder_raw)
        for i in range(2):
            save_name_processed = os.path.join(save_folder_processed, f'misura__{i+1}.npz')
            save_name_raw= os.path.join(save_folder_raw, f'misura__{i+1}.npz')
            measurement = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
            np.savez(save_name_raw, measurement=measurement)
            result = process_measurement(measurement, photons=photons, name=save_name_processed)
        dmx.stop_looping()
