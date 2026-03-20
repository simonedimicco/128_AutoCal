from qlab.devices.tdc import QuTag
import qlab.counting.counting as counting
import qlab.counting.cocount as cocount
from auto_classical import PowerSupplies
from qlab.devices.KeithleyPowerSupply import KeithleyPowerSupply
import logging
from dmx_controller import DMXController

import numpy as np
from numba import njit, prange
import time
from WhiteDict_old import pulse_dict, time_differences, fine_tuning, retard_box
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


@njit(parallel=True, cache=True)
def T_sinc(TT1, CC1, TT2, CC2, ch1, ch2, window=100, mx=1e10):
    # 1) Pre-filtraggio: estraiamo solo i tempi utili per CC1==ch1 e CC2==ch2
    # -------------------------------------------------------------
    # Contiamo prima quanti ne servono
    n1 = 0
    for i in range(len(CC1)):
        if CC1[i] == ch1:
            n1 += 1
    T1 = np.empty(n1, dtype=TT1.dtype)
    idx = 0
    for i in range(len(CC1)):
        if CC1[i] == ch1:
            T1[idx] = TT1[i]
            idx += 1

    n2 = 0
    for i in range(len(CC2)):
        if CC2[i] == ch2:
            n2 += 1
    T2 = np.empty(n2, dtype=TT2.dtype)
    idx = 0
    for i in range(len(CC2)):
        if CC2[i] == ch2:
            T2[idx] = TT2[i]
            idx += 1

    # 2) Istogramma dei ritardi tra -mx e +mx, suddiviso in 2*bins
    bins = int(mx / window)
    vals = np.zeros(2 * bins, dtype=np.int64)

    # 3) Scorrimento con due puntatori per ogni t1 in parallelo
    for i in prange(len(T1)):
        t1 = T1[i]
        # Mantengo i due puntatori come statici all'interno del thread
        left = 0
        right = 0

        # Sposto 'left' fino a che T2[left] < t1 - mx
        while left < n2 and T2[left] < t1 - mx:
            left += 1
        # Sposto 'right' fino a che T2[right] <= t1 + mx
        while right < n2 and T2[right] <= t1 + mx:
            right += 1

        # Ora T2[left:right] rientra nella finestra [t1-mx, t1+mx]
        for j in range(left, right):
            delta = T2[j] - t1
            # via integer division, calcoliamo il bin direttamente
            b = int(delta // window) + bins
            vals[b] += 1

    # 4) Troviamo il ritardo con più conti
    max_idx = 0
    max_val = vals[0]
    for i in range(1, 2 * bins):
        if vals[i] > max_val:
            max_val = vals[i]
            max_idx = i

    # Ritorniamo il tempo relativo corrispondente
    return max_idx * window - mx



@njit(parallel=True, cache=True)
def Modes_separator(channels_times,
                                   channels_list,
                                   trigger_times,
                                   trigger_channel,
                                   mio_dizionario,
                                   Time_differences,
                                   Time_differences_fine_tuning,
                                   retards,
                                   window,
                                   offset):

    n = channels_list.size
    t_tmp = np.zeros(n, dtype=np.int64)
    c_tmp = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=np.int8)

    for j in prange(n):
        ch = channels_list[j]
        if ch == trigger_channel:
            continue

        # ---- trova trigger (binary search manuale) ----
        tj = channels_times[j]
        # i = np.searchsorted(trigger_times, tj - window, side="right") - 1
        # if i < 0:
        #     continue
        i = np.searchsorted(trigger_times, tj, side="right") - 1
        if i < 0:
            continue

        if tj > trigger_times[i] + window:
            #print("Trigger non valido trovato")
            continue

        relative_time = tj - trigger_times[i]
        c1 = mio_dizionario[ch, 0, 0]
        w1 = mio_dizionario[ch, 0, 1]
        c2 = mio_dizionario[ch, 1, 0]
        w2 = mio_dizionario[ch, 1, 1]

        is_bin1 = (
            abs(relative_time - c1) <= w1 or
            abs(relative_time - (c1 + window)) <= w1
        )
        is_bin2 = (
            abs(relative_time - c2) <= w2 or
            abs(relative_time - (c2 + window)) <= w2
        )

        if not (is_bin1 or is_bin2):
            continue

        # ---- mapping canale ----
        if ch <= 31:
            base = 2 * ch if ch <= 15 else 2 * (ch + 16)
        else:
            base = 2 * (ch - 16) if ch <= 47 else 2 * ch

        c_final = base if is_bin1 else base + 1

        # ---- tempo ----
        if is_bin1:
            t_final = tj - retards[ch] - (offset if ch <= 31 else 0)
        else:
            t_final = (
                tj + (window - Time_differences[ch])
                - retards[ch]
                - Time_differences_fine_tuning[ch]
                - (offset if ch <= 31 else 0)
            )

        t_tmp[j] = t_final
        c_tmp[j] = c_final
        valid[j] = 1

    return t_tmp, c_tmp, valid
 
@njit(cache=True)
def compact_results(t_tmp, c_tmp, valid):
    n_valid = np.sum(valid)

    t_out = np.empty(n_valid, dtype=np.int64)
    c_out = np.empty(n_valid, dtype=np.int64)

    k = 0
    for i in range(valid.size):
        if valid[i]:
            t_out[k] = t_tmp[i]
            c_out[k] = c_tmp[i]
            k += 1

    return t_out, c_out

@njit(cache=True)
def merge_time_channel_arrays(times1, channels1, times2, channels2, ch_sync_1, ch_sync_2, channel_offset=32):
    merged_times = np.empty(len(times1) + len(times2), dtype=np.int64)
    merged_channels = np.empty(len(times1) + len(times2), dtype=np.int64)
    

    i = 0
    j = 0
    k = 0

    # Merge stile merge sort, saltando gli eventi sync
    while i < len(times1) and j < len(times2):
        while i < len(times1) and channels1[i] == ch_sync_1:
            i += 1
        while j < len(times2) and channels2[j] == ch_sync_2:
            j += 1
        if i >= len(times1) or j >= len(times2):
            break

        if times1[i] <= times2[j]:
            merged_times[k] = times1[i]
            merged_channels[k] = channels1[i]
            i += 1
        else:
            merged_times[k] = times2[j]
            merged_channels[k] = channels2[j] + channel_offset
            j += 1
        k += 1

    # Aggiunta finale di eventuali elementi rimasti
    while i < len(times1):
        if channels1[i] != ch_sync_1:
            merged_times[k] = times1[i]
            merged_channels[k] = channels1[i]
            k += 1
        i += 1

    while j < len(times2):
        if channels2[j] != ch_sync_2:
            merged_times[k] = times2[j]
            merged_channels[k] = channels2[j] + channel_offset
            k += 1
        j += 1

    return merged_times[:k], merged_channels[:k]

@njit(cache=True)
def find_coincidences_numba(t, c, window_ps, target_n):
    N = t.shape[0]
    
    # Timestamp trick
    seen = np.zeros(128, dtype=np.int64)
    current_id = 1
    
    # Array temporaneo per canali distinti nella finestra
    used = np.empty(128, dtype=np.int64)
    
    # Overallocate output (worst case: N coincidenze)
    out = np.empty((N, target_n), dtype=np.int64)
    out_count = 0
    
    i = 0
    while i < N:
        t0 = t[i]
        
        used_count = 0
        
        # Primo evento
        ch0 = c[i]
        seen[ch0] = current_id
        used[used_count] = ch0
        used_count += 1
        
        cnt = 1
        j_max = i
        
        # Scan forward (identico alla tua logica)
        for j in range(i + 1, N):
            dt = t[j] - t0
            if dt > window_ps:
                break
            
            ch = c[j]
            if seen[ch] != current_id:
                seen[ch] = current_id
                used[used_count] = ch
                used_count += 1
                
                cnt += 1
                j_max = j
        
        # Se ho trovato esattamente target_n canali distinti
        if cnt == target_n:
            for k in range(target_n):
                out[out_count, k] = used[k]
            out_count += 1
            
            i = j_max + 1
        elif cnt > target_n:
            i = j_max + 1
        else:
            i += 1
        
        current_id += 1
    
    return out[:out_count]

 # Wrapper Python che converte List[int] in tuple

def find_coincidences(t, c, window_ps, target_n):
    raw = find_coincidences_numba(t, c, window_ps, target_n)
    result = []
    for grp in raw:
        result.append(tuple(grp))
    return result



@njit
def count_occurrences(shape, data):
    key = np.zeros(shape, dtype=np.int64)
    for row in data:
        a = row[0]
        b = row[1]
        key[a, b] += 1
        key[b, a] += 1
    return key


def measure(boxes, esposizione, durata):
    ripetizioni= int(durata/esposizione)
    mesure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
    return mesure

def process_measurement(times, photons, name=None):
    trigger_channel=17
    trigger_channel_merged=trigger_channel +32
    sync_ch_1=3
    sync_ch_2=27

    #times = [(t,c) for t,c in measurement]
    times_box_1,channels_box_1 = times[0]
    times_box_2,channels_box_2 = times[1]
    t_sync= T_sinc(TT1=times_box_1, 
                   CC1=channels_box_1, 
                   TT2=times_box_2, 
                   CC2=channels_box_2, 
                   ch1=sync_ch_1, 
                   ch2=sync_ch_2, 
                   window=100, 
                   mx=1e10)
    
    if t_sync>0:
        times_box_1 = times_box_1 + t_sync
    else:
        times_box_2 = times_box_2 - t_sync
    
    times_merged, channels_merged = merge_time_channel_arrays(times1 = times_box_1,
                                                              channels1 = channels_box_1,
                                                              times2 = times_box_2, 
                                                              channels2 = channels_box_2, 
                                                              ch_sync_1=sync_ch_1, 
                                                              ch_sync_2=sync_ch_2,
                                                              channel_offset=32)
    
    del times_box_1, channels_box_1, times_box_2, channels_box_2

    mask = times_merged != 0
    times_merged = times_merged[mask]
    channels_merged= channels_merged[mask]
    trigger_times = times_merged[channels_merged == trigger_channel_merged]
    
    t_tmp, c_tmp, valid = Modes_separator(channels_times=times_merged,
                                   channels_list=channels_merged,
                                   trigger_times=trigger_times,
                                   trigger_channel=trigger_channel_merged,
                                   mio_dizionario=pulse_dict,
                                   Time_differences=time_differences,
                                   Time_differences_fine_tuning=fine_tuning,
                                   retards=retard_box,
                                   window=725000,
                                   offset=0)
    t_tot, c_tot = compact_results(t_tmp=t_tmp, c_tmp=c_tmp, valid=valid)
    
    if photons == 0:
        return t_tot, c_tot
    
    if photons==1:
        distribution = np.bincount(c_tot, minlength=128)
    
    if photons==2:
        shape = (128,128)
        order = np.argsort(t_tot)
        t_sorted = t_tot[order]
        c_sorted = c_tot[order]
        window_ps = 1800
        
        coincidences = find_coincidences_numba(t_sorted, c_sorted, window_ps, photons)
        distribution = count_occurrences(shape=shape, data= coincidences)
    # Process the data as needed
    # For example, you could compute histograms, correlations, etc.
    return distribution.flatten()

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

def setloop(input):
    '''channel_dict = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
        'e': 4,
        'f': 5
    }'''
    loop =[0,0,0,5]
    for channel in input:
        if channel != 0 and channel != 5:
            loop[channel-1]=channel  

    return loop
        
def data_collection(inputs: list, Voltages: list, supply: PowerSupplies, n_supplies: int, dmx: DMXController, boxes, exposition = 0.1, duration = 60, repetitions_singles=1, repetitions_doubles = 2) -> list:

    if len(Voltages)!=n_supplies:
        raise ValueError('Number of voltages does not match the number of supplies')
    if type(Voltages)==np.ndarray:
        Voltages = np.reshape(Voltages, (len(Voltages)//2, 2))
    elif type(Voltages)== list:
        Voltages = np.reshape(np.array(Voltages), (len(Voltages)//2, 2))
    else:
        raise ValueError('Voltages must be either list or numpy array')
    
    change_voltages(supply, Voltages)
    output_list=[]

    for input in inputs:

       
        ########################################################
        #                                                      #
        loop = setloop(input)
        dmx.set_active_outputs(loop)
        #                                                      #
        ########################################################

        photons= len(input)
        if photons == 1:
            n_measurments = repetitions_singles
        elif photons == 2:
            n_measurments = repetitions_doubles
        partial_distribution = []
        
        for i in range(n_measurments):
            if i==0:
                measurement = measure(boxes, exposition, duration)
            else:
                with ProcessPoolExecutor() as executor:
                    futures = {
                            executor.submit(measure, boxes, exposition, duration): 'measure',
                            executor.submit(process_measurement, measurement, photons=photons): 'process_measurement'
                            }
                    for future in as_completed(futures):
                        task_name = futures[future]
                        try:
                            result = future.result()
                            if task_name == 'measure':
                                measurement_tmp = result
                            elif task_name == 'process_measurement':
                                partial_distribution.append(result)
                        except Exception as e:
                            print(f"Error in {task_name}: {e}")
                    measurement = measurement_tmp #VERIFICARE CHE SIA UNA COPIA EFFETTIVA

        result = process_measurement(measurement, photons=photons)
        partial_distribution.append(result)
        distribution = np.sum(np.array(partial_distribution), axis=0)
        #distribution = np.concatenate(partial_distribution, axis = 0)
        # QUI DEVI SOMMARE LE VARIE MISURE CONCATENATE PER LO STESSO INPUT
        output_list.append(distribution)
        dmx.stop_looping()
    return np.array(output_list)

'''
FUNZIONI DENIS
'''

def UpdateParameter(currentValue, shiftValue, avoidBoundary = False, parameterValueMin = 0, parameterValueMax = 64, parameterValueMaxReset = 62, parameterValueMinReset = 2):
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

def lossEvalExp(parameters, inputs, target, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition):
    tempPrediction = data_collection(inputs, np.sqrt(parameters), supply, Nsupp, boxes, exposition, duration, repetitions_singles, repetitions_doubles)
    tempLoss = MyMaeExp(tempPrediction, target)
    #print(tempLoss)
    return(tempLoss)

def MyMaeExp(y_predicted, y_true):
    total_error_arr = 0
    for yp, yt in zip(y_predicted, y_true):
        yp = (yp/np.sum(yp))
        yt = (yt/np.sum(yt))
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
    exposition= trainingParams["exposition"]
    paramsOrderList = np.arange(numParams)
    bestLoss = 1000
    maxPairs = len(input_states_two_full)
    bestParams = np.zeros_like(currentParamsTrainable)
    lossHistory = np.empty(epochsNum + 1)
    #fidelityHistory = np.empty(epochsNum + 1)
    
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
        
        prevLoss = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)     #can be skipped if training step is equal to check step.
        if (useTwoPhotons == True):
            prevLoss = prevLoss + lossEvalExp(currentParamsTrainable, input_states_two, targetState2, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
        if (printProgress == "all"):
            print("Epoch:", epoch)
            print("Current loss is:", prevLoss,  "    Changed param:", chosenParam)
            #print("Current loss is:", prevLoss, "    Current fidelity is:", currentFidelity, "    Changed param:", chosenParam)
            print(chosenPairs)
        #print("Current fidelity is:", currentFidelity) 
        #print("Changed param:", chosenParam)
        if (prevLoss < bestLoss):
            bestLoss = prevLoss
            bestParams = currentParamsTrainable.copy()
        lossHistory[epoch] = prevLoss
        #fidelityHistory[epoch] = currentFidelity
        
        
        checkShift = prevLoss * LR_check
        tempStore = currentParamsTrainable[chosenParam]
        currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, checkShift, avoidBoundary)
        #print(currentParamsTrainable)
        upLoss = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
        if (useTwoPhotons == True):
            upLoss = upLoss + lossEvalExp(currentParamsTrainable, input_states_two, targetState2, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
        currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (-checkShift), avoidBoundary)
        downLoss = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
        if (useTwoPhotons == True):
            downLoss = downLoss + lossEvalExp(currentParamsTrainable, input_states_two, targetState2, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
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
                print("Changing parameter value does not improve loss")
            proportion = 0
        # moving in the decided direction based on training type
        if (typeTraining == "proportional"):
            currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (proportion*LR_move), avoidBoundary)
        elif (typeTraining == "absolute"):
            currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (np.sign(proportion)*checkShift*LR_move/LR_check), avoidBoundary)
        else:
            print("ERROR, NO VALID TRAINING TYPE SELECTED")
    
    prevLoss = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
    if (useTwoPhotons == True):
            prevLoss = prevLoss + lossEvalExp(currentParamsTrainable, input_states_two, targetState2, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
    #currentFidelity = MyFidelity(currentParamsTrainable, paramsNotTrainable, baseParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, paramsUnitary, paramsUnitary)
    lossHistory[-1] = prevLoss
    #fidelityHistory[-1] = currentFidelity
    if (printProgress == "last" or printProgress == "all"):
        #print("Last loss is:", prevLoss, "    Last fidelity is:", currentFidelity)
        print("Last loss is:", prevLoss)
    #return currentParamsTrainable, lossHistory, fidelityHistory, bestParams, bestLoss
    return currentParamsTrainable, lossHistory, bestParams, bestLoss

# Varie funzioni training.

'''def myTrainingLoopPartial(currentParamsTrainable, paramsNotTrainable, baseParamsTrainable, timeCoupling, sizeHam, numParams, input_states_one, output_states_one, targetState1, shotSize1, input_states_two_full, output_states_two, targetState2_full, shotSize2, trainingParams, paramsUnitary = []):
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
    paramsOrderList = np.arange(numParams)
    bestLoss = 1000
    maxPairs = len(input_states_two_full)
    bestParams = np.zeros_like(baseParamsTrainable)
    lossHistory = np.empty(epochsNum + 1)
    fidelityHistory = np.empty(epochsNum + 1)
    
    for i in range(epochsNum):
        if (typeOrder == "allRandom"):
            chosenParam = np.random.choice(numParams)
        elif (typeOrder == "listRandom"):
            if (i%numParams == 0):
                np.random.shuffle(paramsOrderList)
            chosenParam = paramsOrderList[i%numParams]
        else:
            print("ERROR, NO VALID ORDER TYPE SELECTED")

        chosenPairs = np.random.choice(maxPairs, checkPairsNum, replace=False)
        input_states_two = list( input_states_two_full[i] for i in chosenPairs )
        targetState2 = targetState2_full[chosenPairs]
        #print(input_states_two)
        
        currentFidelity = MyFidelity(currentParamsTrainable, paramsNotTrainable, baseParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, paramsUnitary, paramsUnitary)
        prevLoss = lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_one, output_states_one, targetState1, shotSize1, paramsUnitary = paramsUnitary)     #can be skipped if training step is equal to check step.
        if (useTwoPhotons == True):
            prevLoss = prevLoss + lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_two, output_states_two, targetState2, shotSize2, paramsUnitary = paramsUnitary)
        if (printProgress == "all"):
            print("Epoch:", i)
            print("Current loss is:", prevLoss, "    Current fidelity is:", currentFidelity, "    Changed param:", chosenParam)
            print(chosenPairs)
        #print("Current fidelity is:", currentFidelity) 
        #print("Changed param:", chosenParam)
        if (prevLoss < bestLoss):
            bestLoss = prevLoss
            bestParams = currentParamsTrainable.copy()
        lossHistory[i] = prevLoss
        fidelityHistory[i] = currentFidelity
        
        
        checkShift = prevLoss * LR_check
        tempStore = currentParamsTrainable[chosenParam].copy()
        currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, checkShift, avoidBoundary)
        upLoss = lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_one, output_states_one, targetState1, shotSize1, paramsUnitary = paramsUnitary)
        if (useTwoPhotons == True):
            upLoss = upLoss + lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_two, output_states_two, targetState2, shotSize2, paramsUnitary = paramsUnitary)
        currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (-checkShift), avoidBoundary)
        downLoss = lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_one, output_states_one, targetState1, shotSize1, paramsUnitary = paramsUnitary)
        if (useTwoPhotons == True):
            downLoss = downLoss + lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_two, output_states_two, targetState2, shotSize2, paramsUnitary = paramsUnitary)
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
                print("Changing parameter value does not improve loss")
            proportion = 0
        # moving in the decided direction based on training type
        if (typeTraining == "proportional"):
            currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (proportion*LR_move), avoidBoundary)
        elif (typeTraining == "absolute"):
            currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (np.sign(proportion)*checkShift*LR_move/LR_check), avoidBoundary)
        else:
            print("ERROR, NO VALID TRAINING TYPE SELECTED")
    
    prevLoss = lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_one, output_states_one, targetState1, shotSize1, paramsUnitary = paramsUnitary)
    if (useTwoPhotons == True):
        prevLoss = prevLoss + lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_two_full, output_states_two, targetState2_full, shotSize2, paramsUnitary = paramsUnitary)    
    currentFidelity = MyFidelity(currentParamsTrainable, paramsNotTrainable, baseParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, paramsUnitary, paramsUnitary)
    lossHistory[-1] = prevLoss
    fidelityHistory[-1] = currentFidelity
    if (printProgress == "last" or printProgress == "all"):
        print("Last loss is:", prevLoss, "    Last fidelity is:", currentFidelity)
    return currentParamsTrainable, lossHistory, fidelityHistory, bestParams, bestLoss'''


'''def myTrainingLoop(currentParamsTrainable, paramsNotTrainable, targetParamsTrainable, timeCoupling, sizeHam, numParams, input_states_one, output_states_one, targetState1, shotSize1, input_states_two, output_states_two, targetState2, shotSize2, trainingParams, paramsUnitary = []):
    epochsNum = trainingParams["epochsNum"]
    LR_check = trainingParams["LR_check"]
    LR_move = trainingParams["LR_move"]
    useTwoPhotons = trainingParams["useTwoPhotons"]
    typeTraining = trainingParams["typeTraining"]
    typeOrder = trainingParams["typeOrder"]
    printProgress = trainingParams["printProgress"]
    paramsOrderList = np.arange(numParams)
    prevLoss = lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_one, output_states_one, targetState1, shotSize1, paramsUnitary = paramsUnitary)
    if (useTwoPhotons == True):
        prevLoss = prevLoss + lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_two, output_states_two, targetState2, shotSize2, paramsUnitary = paramsUnitary)
    bestLoss = 1000
    bestParams = np.zeros_like(targetParamsTrainable)
    lossHistory = np.empty(epochsNum)
    fidelityHistory = np.empty(epochsNum)
    for i in range(epochsNum):
        if (typeOrder == "allRandom"):
            chosenParam = np.random.choice(numParams)
        elif (typeOrder == "listRandom"):
            if (i%numParams == 0):
                np.random.shuffle(paramsOrderList)
            chosenParam = paramsOrderList[i%numParams]
        else:
            print("ERROR, NO VALID ORDER TYPE SELECTED")
        checkShift = prevLoss * LR_check
        tempStore = currentParamsTrainable[chosenParam].copy()
        currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, checkShift)
        upLoss = lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_one, output_states_one, targetState1, shotSize1, paramsUnitary = paramsUnitary)
        if (useTwoPhotons == True):
            upLoss = upLoss + lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_two, output_states_two, targetState2, shotSize2, paramsUnitary = paramsUnitary)
        currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (-checkShift))
        downLoss = lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_one, output_states_one, targetState1, shotSize1, paramsUnitary = paramsUnitary)
        if (useTwoPhotons == True):
            downLoss = downLoss + lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_two, output_states_two, targetState2, shotSize2, paramsUnitary = paramsUnitary)
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
                print("Changing parameter value does not improve loss")
            proportion = 0
        # moving in the decided direction based on training type
        if (typeTraining == "proportional"):
            currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (proportion*LR_move))
        elif (typeTraining == "absolute"):
            currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (np.sign(proportion)*checkShift*LR_move/LR_check))
        else:
            print("ERROR, NO VALID TRAINING TYPE SELECTED")
        prevLoss = lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_one, output_states_one, targetState1, shotSize1, paramsUnitary = paramsUnitary)     #can be skipped if training step is equal to check step.
        if (useTwoPhotons == True):
            prevLoss = prevLoss + lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_two, output_states_two, targetState2, shotSize2, paramsUnitary = paramsUnitary)
    
        currentFidelity = MyFidelity(currentParamsTrainable, paramsNotTrainable, targetParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, paramsUnitary, paramsUnitary)
        lossHistory[i] = prevLoss
        fidelityHistory[i] = currentFidelity
        if (printProgress == "all"):
            print("Epoch:", i)
            print("Current loss is:", prevLoss, "    Current fidelity is:", currentFidelity, "    Changed param:", chosenParam)
        #print("Current fidelity is:", currentFidelity) 
        #print("Changed param:", chosenParam)
        if (prevLoss < bestLoss):
            bestLoss = prevLoss
            bestParams = currentParamsTrainable.copy()
    if (printProgress == "last"):
        print("Last loss is:", prevLoss, "    Last fidelity is:", currentFidelity)
    return currentParamsTrainable, lossHistory, fidelityHistory, bestParams, bestLoss'''


'''def myTrainingLoopPartialUnitaryTarget(currentParamsTrainable, paramsNotTrainable, targetUnitary, timeCoupling, sizeHam, numParams, input_states_one, output_states_one, targetState1, shotSize1, input_states_two_full, output_states_two, targetState2_full, shotSize2, trainingParams, paramsUnitary = []):
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
    paramsOrderList = np.arange(numParams)
    bestLoss = 1000
    maxPairs = len(input_states_two_full)
    bestParams = np.zeros_like(currentParamsTrainable)
    lossHistory = np.empty(epochsNum + 1)
    fidelityHistory = np.empty(epochsNum + 1)
    
    for i in range(epochsNum):
        if (typeOrder == "allRandom"):
            chosenParam = np.random.choice(numParams)
        elif (typeOrder == "listRandom"):
            if (i%numParams == 0):
                np.random.shuffle(paramsOrderList)
            chosenParam = paramsOrderList[i%numParams]
        else:
            print("ERROR, NO VALID ORDER TYPE SELECTED")

        chosenPairs = np.random.choice(maxPairs, checkPairsNum, replace=False)
        input_states_two = list( input_states_two_full[i] for i in chosenPairs )
        targetState2 = targetState2_full[chosenPairs]
        #print(input_states_two)
        
        currentFidelity = MyFidelityUnitaryTarget(currentParamsTrainable, paramsNotTrainable, targetUnitary, timeCoupling, sizeHam, paramsUnitary, paramsUnitary)
        prevLoss = lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_one, output_states_one, targetState1, shotSize1, paramsUnitary = paramsUnitary)     #can be skipped if training step is equal to check step.
        if (useTwoPhotons == True):
            prevLoss = prevLoss + lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_two, output_states_two, targetState2, shotSize2, paramsUnitary = paramsUnitary)
        if (printProgress == "all"):
            print("Epoch:", i)
            print("Current loss is:", prevLoss, "    Current fidelity is:", currentFidelity, "    Changed param:", chosenParam)
            print(chosenPairs)
        #print("Current fidelity is:", currentFidelity) 
        #print("Changed param:", chosenParam)
        if (prevLoss < bestLoss):
            bestLoss = prevLoss
            bestParams = currentParamsTrainable.copy()
        lossHistory[i] = prevLoss
        fidelityHistory[i] = currentFidelity
        
        
        checkShift = prevLoss * LR_check
        tempStore = currentParamsTrainable[chosenParam].copy()
        currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, checkShift, avoidBoundary)
        upLoss = lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_one, output_states_one, targetState1, shotSize1, paramsUnitary = paramsUnitary)
        if (useTwoPhotons == True):
            upLoss = upLoss + lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_two, output_states_two, targetState2, shotSize2, paramsUnitary = paramsUnitary)
        currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (-checkShift), avoidBoundary)
        downLoss = lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_one, output_states_one, targetState1, shotSize1, paramsUnitary = paramsUnitary)
        if (useTwoPhotons == True):
            downLoss = downLoss + lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_two, output_states_two, targetState2, shotSize2, paramsUnitary = paramsUnitary)
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
                print("Changing parameter value does not improve loss")
            proportion = 0
        # moving in the decided direction based on training type
        if (typeTraining == "proportional"):
            currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (proportion*LR_move), avoidBoundary)
        elif (typeTraining == "absolute"):
            currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (np.sign(proportion)*checkShift*LR_move/LR_check), avoidBoundary)
        else:
            print("ERROR, NO VALID TRAINING TYPE SELECTED")
    
    prevLoss = lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_one, output_states_one, targetState1, shotSize1, paramsUnitary = paramsUnitary)
    if (useTwoPhotons == True):
        prevLoss = prevLoss + lossEval(currentParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, input_states_two_full, output_states_two, targetState2_full, shotSize2, paramsUnitary = paramsUnitary)    
    currentFidelity = MyFidelityUnitaryTarget(currentParamsTrainable, paramsNotTrainable, targetUnitary, timeCoupling, sizeHam, paramsUnitary, paramsUnitary)
    lossHistory[-1] = prevLoss
    fidelityHistory[-1] = currentFidelity
    if (printProgress == "last" or printProgress == "all"):
        print("Last loss is:", prevLoss, "    Last fidelity is:", currentFidelity)
    return currentParamsTrainable, lossHistory, fidelityHistory, bestParams, bestLoss'''