# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:44:28 2026

@author: ControlCenter
"""

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
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
#%%
def setloop(input):
    loop =[0,0,0,5]
    for channel in input:
        if channel != 0 and channel != 5:
            loop[channel-1]=channel  

    return loop

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
dwell_time=24
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
path='C:/Users/ControlCenter/Desktop/128_AutoCal/'
dir_name = path+'DATI_' + strtoday()
#dir_name = path+'misure_cluce_classica'
import os
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
print(dir_name)
save_path = dir_name

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
inputs = (1,2,3,4)

if __name__ == '__main__':
    loop = setloop(inputs)
    dmx.set_active_outputs(loop)
    time.sleep(1)
    save_folder_processed = os.path.join(save_path, '4ph_processed')
    if not os.path.exists(save_folder_processed):
        os.mkdir(save_folder_processed)
    for i in range(5):
        save_name_processed = os.path.join(save_folder_processed, f'misura_{i+1}')
        measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
        #print(len(measure))
        times = [(t,c) for t,c in measure]
        t_i= time.time()
        #t_tot, c_tot = process_measurement(times, photons=0, name=save_name_processed)
        t_tot, c_tot = process_measurement(times, photons=0)
        t_f = time.time()
        print(f'dati iterazione {i+1} processati in {t_f-t_i:.2f}')
        np.savez(save_name_processed, t_tot= t_tot, c_tot=c_tot)
    dmx.stop_looping()
#%%

del dmx