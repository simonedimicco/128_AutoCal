# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:23:12 2026

@author: ControlCenter
"""

#%%
from qlab.devices.tdc import QuTag
import qlab.counting.counting as counting
import qlab.counting.cocount as cocount
from auto_classical import PowerSupplies
from qlab.devices.KeithleyPowerSupply import KeithleyPowerSupply
import logging
from dmx_controller import DMXController
import numpy as np
import time
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

dmx = DMXController(log_level=logging.DEBUG)
#%%

#%%
print(dmx.get_data())
#%%
dmx.set_dwell_time(channel=0, dwell_time=41)
dmx.set_dwell_time(channel=1, dwell_time=41)
dmx.set_dwell_time(channel=2, dwell_time=41)
dmx.set_dwell_time(channel=3, dwell_time=41)
dmx.set_dwell_time(channel=4, dwell_time=41)
dmx.set_dwell_time(channel=5, dwell_time=41)
#%%
dmx.set_dwell_time(channel=0, dwell_time=41)
dmx.set_dwell_time(channel=1, dwell_time=23)
dmx.set_dwell_time(channel=2, dwell_time=23)
dmx.set_dwell_time(channel=3, dwell_time=20)
dmx.set_dwell_time(channel=4, dwell_time=20)
dmx.set_dwell_time(channel=5, dwell_time=31)

#%%
dmx.set_active_outputs([1,2,3,4])

#%%
dmx.stop_looping()


#%%
dwell_times=np.linspace(23, 24, 20)
#dwell_times=[x for x in range(20,31)]
#%%
esposizione = 0.1   #in secondi
durata= 3  #in secondi
ripetizioni= int(durata/esposizione)

#%%
list_au=[]
list_RR=[]
for dwell_time in dwell_times:
    print(f'dwell time set to {dwell_time} a.u.')
    dmx.set_dwell_time(channel=1, dwell_time=dwell_time)
    dmx.set_dwell_time(channel=2, dwell_time=dwell_time)
    dmx.set_dwell_time(channel=3, dwell_time=dwell_time)
    dmx.set_dwell_time(channel=4, dwell_time=dwell_time)
    time.sleep(1)
    dmx.set_active_outputs([1,2,3,4])
    
    measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
    times = [(t,c) for t,c in measure]
    times_box_1,channels_box_1 = times[0]
    times_box_2,channels_box_2 = times[1]
    trigger_times=times_box_2[channels_box_2==17]
    list_RR.append(np.mean(np.diff(trigger_times)))
    list_au.append(np.mean(np.diff(trigger_times))/(4*dwell_time))
    dmx.stop_looping()
    time.sleep(1)

#%%
import matplotlib.pyplot as plt

plt.scatter(dwell_times, list_RR)
plt.xlabel('dwell time')
plt.ylabel('cycle duration')
plt.grid(ls='--')
plt.show()

#%%

p = np.polyfit(dwell_times[:], list_RR[:], deg=1)
print(p)
x = np.linspace(20, 30, 100)
para = lambda x :p[0]*x+p[1]
y = para(x)

plt.scatter(dwell_times, list_RR, label='measured values')
plt.plot(x, y, label='fit')
plt.xlabel('dwell time')
plt.ylabel('a.u. value')
plt.grid(ls='--')
plt.show()
#%%
coeff= [p[0], p[1] -725000]
roots = np.roots(coeff)
print(roots)
real_roots = roots[np.isreal(roots)].real
print(real_roots)
#%%
dmx.set_dwell_time(channel=0, dwell_time=100)
dmx.set_dwell_time(channel=1, dwell_time=26)
dmx.set_dwell_time(channel=2, dwell_time=26)
dmx.set_dwell_time(channel=3, dwell_time=26)
dmx.set_dwell_time(channel=4, dwell_time=16)
dmx.set_dwell_time(channel=5, dwell_time=23)
#%%
time.sleep(1)
dmx.set_active_outputs([1,2,3,4])
#%%
measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
times = [(t,c) for t,c in measure]
times_box_1,channels_box_1 = times[0]
times_box_2,channels_box_2 = times[1]
trigger_times=times_box_2[channels_box_2==17]
trigger_RR= np.mean(np.diff(trigger_times))
trigger_RR_std = np.std(np.diff(trigger_times))
print(f'trigger RR : {trigger_RR}+/-{trigger_RR_std} ps')
dmx.stop_looping()

#%%
dmx.stop_looping()













