'''
IMPORT LIBRARIES SECTION:
'''
from qlab.devices.tdc import QuTag
import qlab.counting.counting as counting
import qlab.counting.cocount as cocount
from auto_classical import PowerSupplies
from qlab.devices.KeithleyPowerSupply import KeithleyPowerSupply
import logging
from dmx_controller import DMXController

import numpy as np
from numba import njit
import time 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from WhiteLib import change_voltages, data_collection
from PurpleLib128Modes import myTrainingLoopExp, lossEvalExp
import pyvisa as visa
from tqdm import tqdm
import os

from datetime import datetime
strnow = lambda: datetime.now().strftime("%Y%m%d-%H%M%S")
strtoday = lambda: datetime.now().strftime("%Y_%m_%d")
strtimenow = lambda: datetime.now().strftime("%H:%M:%S")

if __name__=="__main__":

    '''
    INIZIALIZATION KEYTHLEYS
    '''
    rm = visa.ResourceManager()
    print(rm.list_resources())
    list_conncted=[x for x in rm.list_resources()]
    addresses = ['ASRL24::INSTR', 'ASRL25::INSTR', 'ASRL26::INSTR', 'ASRL27::INSTR', 'ASRL28::INSTR', 'ASRL29::INSTR','USB0::0x05E6::0x2230::9100018::INSTR', 'USB0::0x05E6::0x2230::9102515::INSTR', 'ASRL6::INSTR', 'ASRL8::INSTR']
    volts = [[0,0] for _ in range(len(addresses))]
    supply = PowerSupplies(addresses)
    change_voltages(supply, volts)

    '''
    INIZIALIZATION QU-TAGS
    '''

    boxes_ind = QuTag.discover()
    ind = [b'T 02 0010', b'T 02 0021']
    boxes = []
    for i in ind:
        for j in boxes_ind:
            if j._sn == i:
                boxes.append(j)

    if len(ind) != len(boxes):
        raise ValueError('scatole non trovate')

    '''
    INIZIALIZATION QDMX
    '''
    channel_a= 0
    channel_b= 1
    channel_c= 2
    channel_d= 3
    channel_f= 4
    channel_e= 5

    dmx = DMXController(log_level=logging.DEBUG)
    time.sleep(0.1)
    dmx.stop_looping()
    time.sleep(0.1)
    dmx.set_dwell_time(channel=channel_a, dwell_time=26)
    dmx.set_dwell_time(channel=channel_b, dwell_time=26)
    dmx.set_dwell_time(channel=channel_c, dwell_time=26)
    dmx.set_dwell_time(channel=channel_d, dwell_time=26)
    dmx.set_dwell_time(channel=channel_e, dwell_time=16)
    dmx.set_dwell_time(channel=channel_f, dwell_time=16)
    
    '''
    SET WORKING DIRECTORY
    '''
    path='C:/Users/ControlCenter/Desktop/128_AutoCal_dati/'

    '''
    TARGET DEFINITION
    '''
    target_path = 'C:/Users/ControlCenter/Desktop/128_AutoCal_dati/Target_2'
    # target_path = 'C:/Users/ControlCenter/Desktop/128_AutoCal_dati/Target_all20'
    with np.load(os.path.join(target_path, 'singles_distributions.npz')) as data:
        targetSingles = data['distributions']
    with np.load(os.path.join(target_path, 'couples_distributions.npz')) as data:
        targetDoubles_tmp= data['distributions']
 
    targetDoubles = np.zeros((6, 16384), dtype=np.float64)
    for i in range(len(targetDoubles)):
        targetDoubles[i] = targetDoubles_tmp[i].flatten()
        
    print(targetDoubles[0, 0:20])

    ''''
    EXPERIMENTAL SECTION
    '''
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
    chipType = "128Modi"
    skippedParameters = [19, 17]


    duration=6
    exposition = 0.1
    repetitions_singles=1
    repetitions_doubles=10
    #typeTraining = "proportional"
    typeTraining = "absolute"
    #typeOrder = "allRandom"
    typeOrder = "listRandom"
    useTwoPhotons = False
    #checkPairsNum = 3
    checkPairsNum = 0
    LR = 8
    #LR = 1
    #LR = 0.05
    #LR = 0.005
    #LR = 0.01
    LR_check = LR
    #LR_move = LR * 5
    LR_move = LR
    epochsNum = 90
    trainingRepetitions = 10
    printProgress = "all"                     # "off", "last", "all"
    avoidBoundary = True
    
    Nsupp = len(addresses)

    ### end of tweakable parameters
    trainingParams = {"epochsNum" : epochsNum, "LR_check" : LR_check, "LR_move" : LR_move, "useTwoPhotons" : useTwoPhotons, "typeTraining" : typeTraining, "typeOrder" : typeOrder, "printProgress" : printProgress, "checkPairsNum" : checkPairsNum, "firstNeighbourList": firstNeighbourList, "avoidBoundary": avoidBoundary, "supply": supply, "Nsupp": Nsupp, "boxes": boxes, "dmx": dmx, "exposition": exposition, "parameterValueMin": parameterValueMin, "parameterValueMax": parameterValueMax, "parameterValueMinReset": parameterValueMinReset, "parameterValueMaxReset": parameterValueMaxReset, "chipType": chipType, "duration": duration, "repetitions_singles": repetitions_singles, "repetitions_doubles": repetitions_doubles, "skippedParameters": skippedParameters}           



    #target1 params
    #tempArr = np.array((5.601,4.346, 5.367,3.763,3.396,5.966,4.299,5.298,5.832,5.795,5.099,4.853,4.801,4.724,3.132,3.577,4.594,5.756,3.842,5.787))

    #target2 params
    #tempArr = np.array((4.982, 6.744, 5.936, 4.612, 6.481, 5.619, 6.817, 4.076, 4.217, 2.074, 4.483, 5.363, 5.077, 1.618, 4.077, 7.307, 5.451, 1.067, 6.470, 6.944))



    #tempArr2 = tempArr**2
    #print(tempArr2)
    #currentParamsTrainable = tempArr2
    #currentParamsTrainable = tempArr2 + (np.random.rand(len(tempArr)) * 10) - 5
    
    currentParamsTrainable = np.ones(20) * 32
    
    currentParamsTrainable[19] = 0.0
    currentParamsTrainable[17] = 0.0
    print(currentParamsTrainable)

    strnow_DS = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trainingName = "_128modi_training_target2_3PairsPre_32Start_1"
    fileName = path + "logs/" + strnow_DS + trainingName + ".txt"
    #fileName = path + "logs/" + strnow_DS + "_128modi_test.txt"
    logFile = open(fileName, 'w', encoding="utf-8")
    fileNameExtended = path + "logs/" + strnow_DS + trainingName + "_extended.txt"
    logFileExtended = open(fileNameExtended, 'w', encoding="utf-8")

    outputString = "Training Phase 1 Start \n" + "Starting parameters: " + str(currentParamsTrainable) + "\n"

    logFile.write(outputString)
    logFileExtended.write(outputString)
    
    logFileExtended.write(str(trainingParams))
    logFileExtended.write("\n")
    

    logFile.flush()
    logFileExtended.flush()

    logging.disable(logging.DEBUG)

    currentParamsTrainable, lossHistory, bestParams, bestLoss, lossUpHistory, lossDownHistory = myTrainingLoopExp(currentParamsTrainable, numParams, input_states_one, targetSingles, input_states_two_full, targetDoubles, logFile, logFileExtended, trainingParams)


    #for epoch in range(n_epochs):
        #distributions = data_collection(inputs, Voltages, supply, len(addresses), boxes, dmx, exposition= 0.1, duration=60, repetitions_singles=1, repetitions_doubles=2)


    savefileName = path + strnow_DS + "_128modi_training_target2_3PairsPre_32Start_intermediate_1.npz"
    np.savez(savefileName, currentParamsTrainable, lossHistory, bestParams, bestLoss, lossUpHistory, lossDownHistory)


    # Second Step training

    LR = 2
    LR_check = LR
    LR_move = LR
    epochsNum = 90
    useTwoPhotons = True
    checkPairsNum = 3
    
    trainingParams = {"epochsNum" : epochsNum, "LR_check" : LR_check, "LR_move" : LR_move, "useTwoPhotons" : useTwoPhotons, "typeTraining" : typeTraining, "typeOrder" : typeOrder, "printProgress" : printProgress, "checkPairsNum" : checkPairsNum, "firstNeighbourList": firstNeighbourList, "avoidBoundary": avoidBoundary, "supply": supply, "Nsupp": Nsupp, "boxes": boxes, "dmx": dmx, "exposition": exposition, "parameterValueMin": parameterValueMin, "parameterValueMax": parameterValueMax, "parameterValueMinReset": parameterValueMinReset, "parameterValueMaxReset": parameterValueMaxReset, "chipType": chipType, "duration": duration, "repetitions_singles": repetitions_singles, "repetitions_doubles": repetitions_doubles, "skippedParameters": skippedParameters}           

    
    outputString = "Training Phase 2 Start \n" + "Starting parameters: " + str(currentParamsTrainable) + "\n"

    logFile.write(outputString)
    logFileExtended.write(outputString)
    
    logFileExtended.write(str(trainingParams))
    logFileExtended.write("\n")
    
    logFile.flush()
    logFileExtended.flush()
    

    currentParamsTrainable, lossHistory, bestParams, bestLoss, lossUpHistory, lossDownHistory = myTrainingLoopExp(currentParamsTrainable, numParams, input_states_one, targetSingles, input_states_two_full, targetDoubles, logFile, logFileExtended, trainingParams)


    #for epoch in range(n_epochs):
        #distributions = data_collection(inputs, Voltages, supply, len(addresses), boxes, dmx, exposition= 0.1, duration=60, repetitions_singles=1, repetitions_doubles=2)


    savefileName = path + strnow_DS + "_128modi_training_target2_3PairsPre_32Start_result_1.npz"

    np.savez(savefileName, currentParamsTrainable, lossHistory, bestParams, bestLoss, lossUpHistory, lossDownHistory)


    outputString = "Training Finished \n" + "Final parameters: " + str(currentParamsTrainable) + "\n"

    logFile.write(outputString)
    logFileExtended.write(outputString)

    logFile.flush()
    logFileExtended.flush()

    #set voltages to 0
    volts = [[0,0] for _ in range(len(addresses))]
    change_voltages(supply, volts)
        
    logFile.close()
    logFileExtended.close()
        
    
    # Da far girare se non si e' salvato prima

    #savefileName = path + strnow_DS + "_128modi_training_target1_result_1.npz"

    #np.savez(savefileName, currentParamsTrainable, lossHistory, bestParams, bestLoss)

    

    #logFile.close()
    #logging.disable(logging.DEBUG)
    #dmx.stop_looping()
    ##set voltages to 0
    #volts = [[0,0] for _ in range(len(addresses))]
    #change_voltages(supply, volts)
    


        












