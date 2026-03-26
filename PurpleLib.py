import logging

import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

from tqdm import tqdm
from WhiteDict import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from WhiteLib import change_voltages, data_collection


'''
FUNZIONI DENIS
'''

def UpdateParameter(currentValue, shiftValue, avoidBoundary, parameterValueMin, parameterValueMax, parameterValueMaxReset, parameterValueMinReset):
    currentValue = currentValue + shiftValue
    if (avoidBoundary == False):
        if (currentValue > parameterValueMax):
            currentValue = parameterValueMax
        if (currentValue < parameterValueMin):
            currentValue = parameterValueMin
    if (avoidBoundary == True):
        if (currentValue > parameterValueMax):
            currentValue = parameterValueMaxReset
        if (currentValue < parameterValueMin):
            currentValue = parameterValueMinReset
    return currentValue

# Funzioni modificate per il caso sperimentale.

def lossEvalExp(parameters, inputs, target, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition, dmx):
    #parameters_reshaped = np.reshape(parameters, (Nsupp, 2))
    tempPrediction = data_collection(inputs, np.sqrt(parameters), supply, Nsupp, dmx, boxes, exposition, duration, repetitions_singles, repetitions_doubles)
    tempLoss = MyMaeExp(tempPrediction, target)
    #print(tempLoss)
    return(tempLoss)

def MyMaeExp(y_predicted, y_true):
    total_error_arr = 0
    for yp, yt in zip(y_predicted, y_true):
        print("YP = ", yp)
        print("YT = ", yt)
        yp = (yp/np.sum(yp))
        yt = (yt/np.sum(yt))
        print("YP = ", yp)
        print("YT = ", yt)
        #print("yp: ", yp, "yt: ", yt)
        total_error_arr += abs(yp - yt)
    total_error = np.sum(total_error_arr)
    #print("Total error is:",total_error)
    mae = total_error/len(y_predicted)
    #print("Mean absolute error is:",mae)
    return mae
# Funzione di training per il caso sperimentale.

def myTrainingLoopExp(currentParamsTrainable, duration, repetitions_singles, repetitions_doubles, numParams, input_states_one, targetState1, input_states_two_full, targetState2_full, trainingParams, paramsUnitary = []):
    epochsNum = trainingParams["epochsNum"]
    LR_check = trainingParams["LR_check"]
    LR_move = trainingParams["LR_move"]
    useTwoPhotons = trainingParams["useTwoPhotons"]
    typeTraining = trainingParams["typeTraining"]
    typeOrder = trainingParams["typeOrder"]
    printProgress = trainingParams["printProgress"]
    checkPairsNum = trainingParams["checkPairsNum"]
    firstNeighbourList = trainingParams["firstNeighbourList"]
    avoidBoundary = trainingParams["avoidBoundary"]
    supply = trainingParams["supply"]
    Nsupp = trainingParams["Nsupp"]
    boxes = trainingParams["boxes"]
    dmx = trainingParams["dmx"]
    exposition= trainingParams["exposition"]
    parameterValueMin = trainingParams["parameterValueMin"]
    parameterValueMax = trainingParams["parameterValueMax"]
    parameterValueMaxReset = trainingParams["parameterValueMaxReset"]
    parameterValueMinReset = trainingParams["parameterValueMinReset"]
    paramsOrderList = np.arange(numParams)
    bestLoss = 1000
    maxPairs = len(input_states_two_full)
    bestParams = np.zeros_like(currentParamsTrainable)
    lossHistory = np.empty(epochsNum + 1)
    #fidelityHistory = np.empty(epochsNum + 1)
    
    colorStart = '\033[92m'
    colorStop = '\033[0m'
    
    for epoch in range(epochsNum):
        if (typeOrder == "allRandom"):
            chosenParam = np.random.choice(numParams)
        elif (typeOrder == "listRandom"):
            if (epoch%numParams == 0):
                np.random.shuffle(paramsOrderList)
            chosenParam = paramsOrderList[epoch%numParams]
        else:
            print("ERROR, NO VALID ORDER TYPE SELECTED")

        chosenPairs = np.random.choice(maxPairs, checkPairsNum, replace=False)
        input_states_two = list( input_states_two_full[j] for j in chosenPairs )
        targetState2 = np.zeros((checkPairsNum, len(targetState2_full[0])))
        for j in range(checkPairsNum):
            targetState2[j] = targetState2_full[chosenPairs[j]]

        #print(input_states_two)
        
        #currentFidelity = MyFidelity(currentParamsTrainable, paramsNotTrainable, baseParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, paramsUnitary, paramsUnitary)
        
        print(colorStart, "Epoch:", epoch, "Measure 1", colorStop)
        tempLoss1 = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition, dmx)     #can be skipped if training step is equal to check step.
        prevLoss = tempLoss1
        if (useTwoPhotons == True):
            tempLoss2 = lossEvalExp(currentParamsTrainable, input_states_two, targetState2, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition, dmx)
            prevLoss = prevLoss + tempLoss2
        if (printProgress == "all"):
            print(colorStart, "Epoch:", epoch, colorStop)
            print(colorStart, "Current loss is:", prevLoss,  "    Changed param:", chosenParam,  "    Changed param value:", currentParamsTrainable[chosenParam], colorStop)
            print(colorStart, "Loss Singles:", tempLoss1, colorStop)
            if (useTwoPhotons == True):
                print(colorStart, "Loss Doubles:", tempLoss2, colorStop)
            #print("Current loss is:", prevLoss, "    Current fidelity is:", currentFidelity, "    Changed param:", chosenParam)
            print(colorStart, chosenPairs, colorStop)
        #print("Current fidelity is:", currentFidelity) 
        #print("Changed param:", chosenParam)
        if (prevLoss < bestLoss):
            bestLoss = prevLoss
            bestParams = currentParamsTrainable.copy()
        lossHistory[epoch] = prevLoss
        #fidelityHistory[epoch] = currentFidelity
        
        
        checkShift = prevLoss * LR_check
        tempStore = currentParamsTrainable[chosenParam]
        currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, checkShift, avoidBoundary, parameterValueMin, parameterValueMax, parameterValueMaxReset, parameterValueMinReset)
        #print(currentParamsTrainable)
        print(colorStart, "Epoch:", epoch, "Measure 2", colorStop)
        upLoss = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition, dmx)
        if (useTwoPhotons == True):
            upLoss = upLoss + lossEvalExp(currentParamsTrainable, input_states_two, targetState2, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition, dmx)
        currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (-checkShift), avoidBoundary, parameterValueMin, parameterValueMax, parameterValueMaxReset, parameterValueMinReset)
        print(colorStart, "Epoch:", epoch, "Measure 3", colorStop)
        downLoss = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition, dmx)
        if (useTwoPhotons == True):
            downLoss = downLoss + lossEvalExp(currentParamsTrainable, input_states_two, targetState2, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition, dmx)
        # calculating which direction to move
        if ((upLoss < prevLoss) & (downLoss < prevLoss)):
            if (downLoss < upLoss):
                proportion = -1 * ((prevLoss - downLoss)/prevLoss)
            else:
                proportion = ((prevLoss - upLoss)/prevLoss)
        elif(upLoss < prevLoss):
            proportion = ((prevLoss - upLoss)/prevLoss)
        elif(downLoss < prevLoss):
            proportion = -1 * ((prevLoss - downLoss)/prevLoss)
        else:
            if (printProgress == "all"):
                print(colorStart, "Changing parameter value does not improve loss", colorStop)
            proportion = 0
        # moving in the decided direction based on training type
        if (typeTraining == "proportional"):
            currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (proportion*LR_move), avoidBoundary, parameterValueMin, parameterValueMax, parameterValueMaxReset, parameterValueMinReset)
        elif (typeTraining == "absolute"):
            currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (np.sign(proportion)*checkShift*LR_move/LR_check), avoidBoundary, parameterValueMin, parameterValueMax, parameterValueMaxReset, parameterValueMinReset)
        else:
            print("ERROR, NO VALID TRAINING TYPE SELECTED")
        print(colorStart,  "Changed param value:", currentParamsTrainable[chosenParam], colorStop)
    
    print(colorStart, "Epoch:", epoch, "Measure 4", colorStop)
    prevLoss = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition, dmx)
    if (useTwoPhotons == True):
            prevLoss = prevLoss + lossEvalExp(currentParamsTrainable, input_states_two, targetState2, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition, dmx)
    #currentFidelity = MyFidelity(currentParamsTrainable, paramsNotTrainable, baseParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, paramsUnitary, paramsUnitary)
    lossHistory[-1] = prevLoss
    #fidelityHistory[-1] = currentFidelity
    if (printProgress == "last" or printProgress == "all"):
        #print("Last loss is:", prevLoss, "    Last fidelity is:", currentFidelity)
        print(colorStart, "Last loss is:", prevLoss, colorStop)
    #return currentParamsTrainable, lossHistory, fidelityHistory, bestParams, bestLoss
    return currentParamsTrainable, lossHistory, bestParams, bestLoss

# Varie funzioni training.