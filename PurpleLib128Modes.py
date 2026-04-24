import logging

import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

from tqdm import tqdm
from WhiteDict import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from WhiteLib import change_voltages, data_collection_parallel

#from script import intensities


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

def lossEvalExp(parameters, inputs, target, trainingParams, chipType):
    if (chipType == "128Modi"):
        return (lossEvalExp128(parameters, inputs, target, trainingParams))
    if (chipType == "32Modi"):
        return (lossEvalExp32(parameters, inputs, target, trainingParams))

def lossEvalExp32(parameters, inputs, target, trainingParams):
    #parameters_reshaped = np.reshape(parameters, (Nsupp, 2))
    bounds = trainingParams["bounds"]
    n_spots = trainingParams["n_spots"]
    q = trainingParams["q"]
    switch = trainingParams["switch"]
    cam = trainingParams["cam"]

    tempPrediction = intensities(np.sqrt(parameters), inputs, bounds, n_spots, q, switch, cam)
    tempLoss = MyMaeExp(tempPrediction, target)
    #print(tempLoss)
    return(tempLoss)

def lossEvalExp128(parameters, inputs, target, trainingParams):
    #parameters_reshaped = np.reshape(parameters, (Nsupp, 2))
    duration = trainingParams["duration"]
    repetitions_singles = trainingParams["repetitions_singles"]
    repetitions_doubles = trainingParams["repetitions_doubles"]
    supply = trainingParams["supply"]
    Nsupp = trainingParams["Nsupp"]
    boxes = trainingParams["boxes"]
    exposition = trainingParams["exposition"]
    dmx = trainingParams["dmx"]

    tempPrediction = data_collection_parallel(inputs, np.sqrt(parameters), supply, Nsupp, dmx, boxes, exposition, duration, repetitions_singles, repetitions_doubles)
    tempLoss = MyMaeExp(tempPrediction, target)
    #print(tempLoss)
    return(tempLoss)

def MyMaeExp(y_predicted, y_true):
    total_error_arr = 0
    for yp, yt in zip(y_predicted, y_true):
        #print("YP = ", yp)
        #print("YT = ", yt)
        yp = (yp/np.sum(yp))
        yt = (yt/np.sum(yt))
        #print("YP = ", yp)
        #print("YT = ", yt)
        #print("yp: ", yp, "yt: ", yt)
        total_error_arr += abs(yp - yt)
    total_error = np.sum(total_error_arr)
    #print("Total error is:",total_error)
    mae = total_error/len(y_predicted)
    #print("Mean absolute error is:",mae)
    return mae


# Funzione di training per il caso sperimentale.

def myTrainingLoopExp(currentParamsTrainable, numParams, input_states_one, targetState1, input_states_two_full, targetState2_full, logFile, logFileExtended, trainingParams, paramsUnitary = []):
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
    parameterValueMin = trainingParams["parameterValueMin"]
    parameterValueMax = trainingParams["parameterValueMax"]
    parameterValueMaxReset = trainingParams["parameterValueMaxReset"]
    parameterValueMinReset = trainingParams["parameterValueMinReset"]
    chipType = trainingParams["chipType"]
    skippedParameters = trainingParams["skippedParameters"]
    paramsOrderListTemp = list(np.arange(numParams))
    paramsOrderListList = [ param for param in paramsOrderListTemp if param not in skippedParameters ]
    paramsOrderList = np.array(paramsOrderListList)
    numParamsWorking = len(paramsOrderList)
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
            if (epoch%numParamsWorking == 0):
                np.random.shuffle(paramsOrderList)
            chosenParam = paramsOrderList[epoch%numParamsWorking]
        else:
            print("ERROR, NO VALID ORDER TYPE SELECTED")

        if (useTwoPhotons == True):
            chosenPairs = np.random.choice(maxPairs, checkPairsNum, replace=False)
            input_states_two = list( input_states_two_full[j] for j in chosenPairs )
            targetState2 = np.zeros((checkPairsNum, len(targetState2_full[0])))
            for j in range(checkPairsNum):
                targetState2[j] = targetState2_full[chosenPairs[j]]

        #print(input_states_two)
        
        #currentFidelity = MyFidelity(currentParamsTrainable, paramsNotTrainable, baseParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, paramsUnitary, paramsUnitary)
        
        print(colorStart, "Epoch:", epoch, "Measure 1", colorStop)
        tempLoss1 = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, trainingParams, chipType)     #can be skipped if training step is equal to check step.
        prevLoss = tempLoss1
        if (useTwoPhotons == True):
            tempLoss2 = lossEvalExp(currentParamsTrainable, input_states_two, targetState2, trainingParams, chipType)
            prevLoss = prevLoss + tempLoss2
        if (printProgress == "all"):
            print(colorStart, "Epoch:", epoch, colorStop)
            print(colorStart, "Current loss is:", prevLoss,  "    Changed param:", chosenParam,  "    Changed param value:", currentParamsTrainable[chosenParam], colorStop)
            print(colorStart, "Loss Singles:", tempLoss1, colorStop)
            if (useTwoPhotons == True):
                print(colorStart, "Loss Doubles:", tempLoss2, colorStop)
                print(colorStart, chosenPairs, colorStop)
            #print("Current loss is:", prevLoss, "    Current fidelity is:", currentFidelity, "    Changed param:", chosenParam)
            
        strnow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outputString = strnow + "\n Epoch: " + str(epoch) + "\n    Current loss is: " + str(prevLoss) + "    Changed param: " + str(chosenParam) + "    Changed param start value: " + str(currentParamsTrainable[chosenParam]) + "    Loss Singles: " + str(tempLoss1)
        logFile.write(outputString)
        logFileExtended.write(outputString)

        if (useTwoPhotons == True):
            logFile.write("    Loss Doubles: ")
            logFile.write(str(tempLoss2))
            logFileExtended.write("    Loss Doubles: ")
            logFileExtended.write(str(tempLoss2))
            logFileExtended.write("\n")
            logFileExtended.write(str(chosenPairs))

        logFile.write("\n")
        logFileExtended.write("\n")
        
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
        tempLoss1 = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, trainingParams, chipType)
        upLoss = tempLoss1
        logFileExtended.write("upLoss Singles: ")
        logFileExtended.write(str(tempLoss1))
        if (useTwoPhotons == True):
            tempLoss2 = lossEvalExp(currentParamsTrainable, input_states_two, targetState2, trainingParams, chipType)
            upLoss = upLoss + tempLoss2
            logFileExtended.write("    upLoss Doubles: ")
            logFileExtended.write(str(tempLoss2))
        logFileExtended.write("    upLoss Total: ")
        logFileExtended.write(str(upLoss))
            
        currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (-checkShift), avoidBoundary, parameterValueMin, parameterValueMax, parameterValueMaxReset, parameterValueMinReset)
        print(colorStart, "Epoch:", epoch, "Measure 3", colorStop)
        tempLoss1 = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, trainingParams, chipType)
        downLoss = tempLoss1
        logFileExtended.write("    downLoss Singles: ")
        logFileExtended.write(str(tempLoss1))
        if (useTwoPhotons == True):
            tempLoss2 = lossEvalExp(currentParamsTrainable, input_states_two, targetState2, trainingParams, chipType)
            downLoss = downLoss + tempLoss2
            logFileExtended.write("    downLoss Doubles: ")
            logFileExtended.write(str(tempLoss2))
        logFileExtended.write("    downLoss Total: ")
        logFileExtended.write(str(downLoss))
        logFileExtended.write("\n")
        
        
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
            logFile.write("Changing parameter value does not improve loss \n")
            logFileExtended.write("Changing parameter value does not improve loss \n")
            proportion = 0
        # moving in the decided direction based on training type
        if (typeTraining == "proportional"):
            currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (proportion*LR_move), avoidBoundary, parameterValueMin, parameterValueMax, parameterValueMaxReset, parameterValueMinReset)
        elif (typeTraining == "absolute"):
            currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (np.sign(proportion)*checkShift*LR_move/LR_check), avoidBoundary, parameterValueMin, parameterValueMax, parameterValueMaxReset, parameterValueMinReset)
        else:
            print("ERROR, NO VALID TRAINING TYPE SELECTED")
        print(colorStart,  "Changed param value:", currentParamsTrainable[chosenParam], colorStop)
        logFile.write("    Changed param end value: ")
        logFile.write(str(currentParamsTrainable[chosenParam]))
        logFile.write("\n")
        logFile.flush()

        logFileExtended.write("    Changed param end value: ")
        logFileExtended.write(str(currentParamsTrainable[chosenParam]))
        logFileExtended.write("\n")
        logFileExtended.flush()
    
    print(colorStart, "Epoch:", epoch, "Measure 4", colorStop)
    prevLoss = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, trainingParams, chipType)
    if (useTwoPhotons == True):
            prevLoss = prevLoss + lossEvalExp(currentParamsTrainable, input_states_two, targetState2, trainingParams, chipType)
    #currentFidelity = MyFidelity(currentParamsTrainable, paramsNotTrainable, baseParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, paramsUnitary, paramsUnitary)
    lossHistory[-1] = prevLoss
    #fidelityHistory[-1] = currentFidelity
    if (printProgress == "last" or printProgress == "all"):
        #print("Last loss is:", prevLoss, "    Last fidelity is:", currentFidelity)
        print(colorStart, "Last loss is:", prevLoss, colorStop)
    logFile.write("Last Loss is: ")
    logFile.write(str(prevLoss))
    logFile.write("\n")
    logFile.flush()
    logFileExtended.write("Last Loss is: ")
    logFileExtended.write(str(prevLoss))
    logFileExtended.write("\n")
    logFileExtended.flush()
    #return currentParamsTrainable, lossHistory, fidelityHistory, bestParams, bestLoss
    return currentParamsTrainable, lossHistory, bestParams, bestLoss

# Varie funzioni training.

def StabilityMeasure(currentParamsTrainable, iterations, input_states_one, targetState1, input_states_two, targetState2, logFile, logFileExtended, trainingParams, paramsUnitary = []):
    epochsNum = trainingParams["epochsNum"]
    useTwoPhotons = trainingParams["useTwoPhotons"]
    printProgress = trainingParams["printProgress"]
    avoidBoundary = trainingParams["avoidBoundary"]
    parameterValueMin = trainingParams["parameterValueMin"]
    parameterValueMax = trainingParams["parameterValueMax"]
    parameterValueMaxReset = trainingParams["parameterValueMaxReset"]
    parameterValueMinReset = trainingParams["parameterValueMinReset"]
    chipType = trainingParams["chipType"]
    costFluctuationSingles = np.zeros(iterations)
    costFluctuationDoubles = np.zeros(iterations)
    
    colorStart = '\033[92m'
    colorStop = '\033[0m'
    
    for i in range(iterations):    
        print(colorStart, "Iteration: ", i, colorStop)
        costFluctuationSingles[i] = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, trainingParams, chipType)
        costFluctuationDoubles[i] = lossEvalExp(currentParamsTrainable, input_states_two, targetState2, trainingParams, chipType)
        
        strnow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outputString = strnow + "\n Iteration: " + str(i) + "\n    Loss Singles: " + str(costFluctuationSingles[i]) + "    Loss Doubles: " + str(costFluctuationDoubles[i]) +  "\n"
        logFile.write(outputString)
        logFileExtended.write(outputString)
        logFile.flush()
        logFileExtended.flush()
    
    
    return costFluctuationSingles, costFluctuationDoubles
    
    


