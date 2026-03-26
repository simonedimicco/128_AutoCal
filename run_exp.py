

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
from WhiteLib import change_voltages, data_collection
from PurpleLib import myTrainingLoopExp
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
#set voltages to 0
volts = [[0,0] for _ in range(len(addresses))]
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
path='C:/Users/ControlCenter/Desktop/128_AutoCal_dati/'
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
target_path = 'C:/Users/ControlCenter/Desktop/128_AutoCal_dati/Target'
with np.load(os.path.join(target_path, 'singles_distributions.npz')) as data:
    targetSingles = data['distributions']
with np.load(os.path.join(target_path, 'couples_distributions.npz')) as data:
    targetDoubles_tmp= data['distributions']
#targetList = []
#targetList.append(distribution for distribution in targetSingles)
#targetList.append(distribution for distribution in targetDoubles)
#print(typetargetSingles)
# targetList.append(np.load("128_auto_calibration/dati_test/conf_03/singles_distribution/b.npz")["distribution"])
# targetList.append(np.load("128_auto_calibration/dati_test/conf_03/singles_distribution/c.npz")["distribution"])
# targetList.append(np.load("128_auto_calibration/dati_test/conf_03/singles_distribution/d.npz")["distribution"])
# targetList.append(np.load("128_auto_calibration/dati_test/conf_03/singles_distribution/e.npz")["distribution"])
# targetList.append(np.load("128_auto_calibration/dati_test/conf_03/Couple_coincidences_distributions/bc.npz")["distribution"])
# targetList.append(np.load("128_auto_calibration/dati_test/conf_03/Couple_coincidences_distributions/bd.npz")["distribution"])
# targetList.append(np.load("128_auto_calibration/dati_test/conf_03/Couple_coincidences_distributions/be.npz")["distribution"])
# targetList.append(np.load("128_auto_calibration/dati_test/conf_03/Couple_coincidences_distributions/cd.npz")["distribution"])
# targetList.append(np.load("128_auto_calibration/dati_test/conf_03/Couple_coincidences_distributions/ce.npz")["distribution"])
# targetList.append(np.load("128_auto_calibration/dati_test/conf_03/Couple_coincidences_distributions/de.npz")["distribution"])
targetDoubles = np.zeros((6, 16384), dtype=np.float64)
#targetDoubles = np.reshape(targetDoubles, (6, 16384))
for i in range(len(targetDoubles)):
    targetDoubles[i] = targetDoubles_tmp[i].flatten()
    
print(targetDoubles[0, 0:20])

'''targetSingles = np.array(targetList[0:4])
targetDoubles = np.array(targetList[4:])'''

#%%
''''
EXPERIMENTAL SECTION
'''
currentParamsTrainable = [0 for _ in range(len(addresses))]

#logging.disable()


input_states_one = [(1,), (2,), (3,), (4,)]
input_states_two_full = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
firstNeighbourList = input_states_two_full
inputsFull = input_states_one + input_states_two_full

numParams = len(addresses) * 2

parameterValueMin = 0
parameterValueMax = 64
parameterValueMaxReset = 62
parameterValueMinReset = 2
#shotSize1 = int(1e4)
#shotSize2 = int(1e4)
duration=6
exposition = 0.1
repetitions_singles=1
repetitions_doubles=10
#typeTraining = "proportional"
typeTraining = "absolute"
#typeOrder = "allRandom"
typeOrder = "listRandom"
useTwoPhotons = False
LR = 1
#LR = 0.05
#LR = 0.005
#LR = 0.01
LR_check = LR
#LR_move = LR * 5
LR_move = LR
epochsNum = 100
trainingRepetitions = 10
printProgress = "all"                     # "off", "last", "all"
#checkPairsNum = 6
checkPairsNum = 0
avoidBoundary = True
 
Nsupp = len(addresses)


### end of tweakable parameters
trainingParams = {"epochsNum" : epochsNum, "LR_check" : LR_check, "LR_move" : LR_move, "useTwoPhotons" : useTwoPhotons, "typeTraining" : typeTraining, "typeOrder" : typeOrder, "printProgress" : printProgress, "checkPairsNum" : checkPairsNum, "firstNeighbourList": firstNeighbourList, "avoidBoundary": avoidBoundary, "supply": supply, "Nsupp": Nsupp, "boxes": boxes, "dmx": dmx, "exposition": exposition, "parameterValueMin": parameterValueMin, "parameterValueMax": parameterValueMax, "parameterValueMinReset": parameterValueMinReset, "parameterValueMaxReset": parameterValueMaxReset}

#%%

#volts[0]= [5.601,4.346]
#volts[1]= [5.367,3.763]
#volts[2]= [3.396,5.966]
#volts[3]= [4.299,5.298]
#volts[4]= [5.832,5.795]
#volts[5]= [5.099,4.853]
#volts[6]= [4.801,4.724]
#volts[7]= [3.132,3.577]
#volts[8]= [4.594,5.756]
#volts[9]= [3.842,5.787]

tempArr = np.array((5.601,4.346, 5.367,3.763,3.396,5.966,4.299,5.298,5.832,5.795,5.099,4.853,4.801,4.724,3.132,3.577,4.594,5.756,3.842,5.787))
#tempArr2 = tempArr + (np.random.rand(len(tempArr))) - 0.5
tempArr2 = tempArr**2
print(tempArr2)
currentParamsTrainable = tempArr2 + (np.random.rand(len(tempArr)) * 10) - 5
print(currentParamsTrainable)
#%%

strnow_DS = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fileName = path + "logs/" + strnow_DS + "_128modi_training_target1.txt"
logFile = open(fileName, 'w', encoding="utf-8")

logFile.write("Training Start \n")
logFile.write("Starting parameters: ")
logFile.write(str(currentParamsTrainable))
logFile.write("\n")

logging.disable(logging.DEBUG)

currentParamsTrainable, lossHistory, bestParams, bestLoss = myTrainingLoopExp(currentParamsTrainable, duration, repetitions_singles, repetitions_doubles, numParams, input_states_one, targetSingles, input_states_two_full, targetDoubles, logFile, trainingParams)

#for epoch in range(n_epochs):
    #distributions = data_collection(inputs, Voltages, supply, len(addresses), boxes, dmx, exposition= 0.1, duration=60, repetitions_singles=1, repetitions_doubles=2)

#set voltages to 0
volts = [[0,0] for _ in range(len(addresses))]
change_voltages(supply, volts)
    
logFile.close()
    
#%%
#%%

logFile.close()
logging.disable(logging.DEBUG)
dmx.stop_looping()
#set voltages to 0
volts = [[0,0] for _ in range(len(addresses))]
change_voltages(supply, volts)
    
        
#%%
print(targetSingles[1])
    
    