

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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from WhiteLib import change_voltages, data_collection, myTrainingLoopExp
import pyvisa as visa
from tqdm import tqdm

from datetime import datetime
strnow = lambda: datetime.now().strftime("%Y%m%d-%H%M%S")
strtoday = lambda: datetime.now().strftime("%Y_%m_%d")
strtimenow = lambda: datetime.now().strftime("%H:%M:%S")
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
path='C:/Users/ControlCenter/Desktop/Auto_calibration/'
dir_name = path+'DATI_' + strtoday()
#dir_name = path+'misure_cluce_classica'
import os
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
print(dir_name)

#%%
'''
TARGET DEFINITION
'''
#%%

targetList = []
targetList.append(np.load("128_auto_calibration/dati_test/conf_03/singles_distribution/b.npz")["distribution"])
targetList.append(np.load("128_auto_calibration/dati_test/conf_03/singles_distribution/c.npz")["distribution"])
targetList.append(np.load("128_auto_calibration/dati_test/conf_03/singles_distribution/d.npz")["distribution"])
targetList.append(np.load("128_auto_calibration/dati_test/conf_03/singles_distribution/e.npz")["distribution"])
targetList.append(np.load("128_auto_calibration/dati_test/conf_03/Couple_coincidences_distributions/bc.npz")["distribution"])
targetList.append(np.load("128_auto_calibration/dati_test/conf_03/Couple_coincidences_distributions/bd.npz")["distribution"])
targetList.append(np.load("128_auto_calibration/dati_test/conf_03/Couple_coincidences_distributions/be.npz")["distribution"])
targetList.append(np.load("128_auto_calibration/dati_test/conf_03/Couple_coincidences_distributions/cd.npz")["distribution"])
targetList.append(np.load("128_auto_calibration/dati_test/conf_03/Couple_coincidences_distributions/ce.npz")["distribution"])
targetList.append(np.load("128_auto_calibration/dati_test/conf_03/Couple_coincidences_distributions/de.npz")["distribution"])

for i in range(len(targetList)):
    targetList[i] = targetList[i].flatten()

targetSingles = np.array(targetList[0:4])
targetDoubles = np.array(targetList[4:])

#%%
''''
EXPERIMENTAL SECTION
'''
currentParamsTrainable = [0 for _ in range(len(addresses))]


input_states_one = [(1,), (2,), (3,), (4,)]
input_states_two_full = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
firstNeighbourList = input_states_two_full
inputsFull = input_states_one + input_states_two_full

numParams = len(addresses)

parameterValueMin = 0
parameterValueMax = 64
parameterValueMaxReset = 62
parameterValueMinReset = 2
#shotSize1 = int(1e4)
#shotSize2 = int(1e4)
duration=60
repetitions_singles=1
repetitions_doubles=2
#typeTraining = "proportional"
typeTraining = "absolute"
#typeOrder = "allRandom"
typeOrder = "listRandom"
useTwoPhotons = True
LR = 0.05
#LR = 0.005
#LR = 0.01
LR_check = LR
#LR_move = LR * 5
LR_move = LR
epochsNum = 100
trainingRepetitions = 10
printProgress = "all"                     # "off", "last", "all"
checkPairsNum = 6
avoidBoundary = True
 
Nsupp = len(addresses)


### end of tweakable parameters
trainingParams = {"epochsNum" : epochsNum, "LR_check" : LR_check, "LR_move" : LR_move, "useTwoPhotons" : useTwoPhotons, "typeTraining" : typeTraining, "typeOrder" : typeOrder, "printProgress" : printProgress, "checkPairsNum" : checkPairsNum, "firstNeighbourList": firstNeighbourList, "avoidBoundary": avoidBoundary, "supply": supply, "Nsupp": Nsupp, "boxes": boxes, "exposition": exposition, "parameterValueMin": parameterValueMin, "parameterValueMax": parameterValueMax, "parameterValueMinReset": parameterValueMinReset, "parameterValueMaxReset": parameterValueMaxReset}


#%%
esposizione = 0.1   #in secondi
durata= 60   #in secondi

#%%

#Voltages=[0 for _ in range(len(addresses))]
#inputs = ['b', 'c', 'd', 'e', 'bc', 'bd', 'be', 'cd', 'ce', 'de']

currentParamsTrainable, lossHistory, bestParams, bestLoss = myTrainingLoopExp(currentParamsTrainable, duration, repetitions_singles, repetitions_doubles, numParams, input_states_one, targetSingles, input_states_two_full, targetDoubles, trainingParams)

#for epoch in range(n_epochs):
    #distributions = data_collection(inputs, Voltages, supply, len(addresses), boxes, dmx, exposition= 0.1, duration=60, repetitions_singles=1, repetitions_doubles=2)

    

        
    
        
    
    