# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:06:06 2026

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
from WhiteLib import process_measurement, setloop, change_voltages, split_times_by_channel, all_inter_histograms, all_intra_histograms, data_collection, data_collection_new, data_collection_fallita
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
    print(f"Data collection new completed in {t2-t1-duration*repetitions_doubles:.2f} seconds")
    print(f"Data collection completed in {t3-t2-duration*repetitions_doubles:.2f} seconds")
    
    diff = abs(results_new - results)/results
    plt.plot(diff.flatten(), 'o')
    plt.title('Difference between old and new data collection')
    plt.xlabel('Index')
    plt.ylabel('Difference')
    plt.show()
        

#%%
dmx.stop_looping()
