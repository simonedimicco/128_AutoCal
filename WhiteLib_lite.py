# from qlab.devices.tdc import QuTag
# import qlab.counting.counting as counting
# import qlab.counting.cocount as cocount
# from auto_classical import PowerSupplies
# from qlab.devices.KeithleyPowerSupply import KeithleyPowerSupply
# import logging
# from dmx_controller import DMXController


import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

from numba import njit, prange
from tqdm import tqdm
from WhiteDict import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

@njit(cache=True)
def T_sinc(TT1, CC1, TT2, CC2, ch1, ch2, window=100, mx=1e10):
    T1 = np.sort(TT1[CC1 == ch1])
    T2 = np.sort(TT2[CC2 == ch2])
    n1 = len(T1)

    bins   = int(mx / window)
    n_bins = 2 * bins
    vals   = np.zeros(n_bins, dtype=np.int64)

    for i in range(n1):           # seriale: niente race condition
        t1    = T1[i]
        left  = np.searchsorted(T2, t1 - mx)
        right = np.searchsorted(T2, t1 + mx, side='right')
        for j in range(left, right):
            b = int((T2[j] - t1 + mx) / window)
            if 0 <= b < n_bins:
                vals[b] += 1

    max_idx = np.argmax(vals)
    return (max_idx - bins + 0.5) * window


@njit(cache=True)
def merge_time_channel_arrays(times1, channels1, times2, channels2,
                               ch_sync_1, ch_sync_2, channel_offset=32):

    # 1) Pre-filtraggio: maschera booleana ? selezione vettoriale
    mask1 = channels1 != ch_sync_1
    mask2 = channels2 != ch_sync_2

    t1 = times1[mask1]
    c1 = channels1[mask1]
    t2 = times2[mask2]
    c2 = channels2[mask2] + channel_offset   # offset applicato una volta sola

    n1 = len(t1)
    n2 = len(t2)

    # 2) Alloca output della dimensione esatta
    merged_times    = np.empty(n1 + n2, dtype=np.int64)
    merged_channels = np.empty(n1 + n2, dtype=np.int64)

    # 3) Merge pulito senza controlli sync inline
    i = 0
    j = 0
    k = 0

    while i < n1 and j < n2:
        if t1[i] <= t2[j]:
            merged_times[k]    = t1[i]
            merged_channels[k] = c1[i]
            i += 1
        else:
            merged_times[k]    = t2[j]
            merged_channels[k] = c2[j]
            j += 1
        k += 1

    # 4) Tail: uno dei due array \Uffffffffsaurito, copia il resto
    while i < n1:
        merged_times[k]    = t1[i]
        merged_channels[k] = c1[i]
        i += 1
        k += 1

    while j < n2:
        merged_times[k]    = t2[j]
        merged_channels[k] = c2[j]
        j += 1
        k += 1

    return merged_times, merged_channels


@njit(parallel=True, cache=True)
def modes_separator(channels_times, channels_list,
                    centers1, widths1, centers2, widths2,
                    retards, td, tdf,
                    trigger_channel,
                    window=725000,
                    offset=0):

    n_total = channels_list.size

    # Estrai trigger times
    trig_mask = channels_list == trigger_channel
    trigger_times = channels_times[trig_mask]
    n_trig = trigger_times.size

    if n_trig == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Buffer con sentinella: t==-1 significa "evento scartato"
    t_out_buf = np.full(n_total, -1, dtype=np.int64)
    c_out_buf = np.empty(n_total,    dtype=np.int64)

    # --- FASE 1: parallela su j ---
    # Ogni iterazione \Uffffffffndipendente: searchsorted sostituisce trig_ptr
    for j in prange(n_total):
        ch = channels_list[j]
        if ch == trigger_channel:
            continue

        t_event = channels_times[j]

        # Trova il trigger pi\Uffffffffente con t_trig <= t_event
        # searchsorted(..., 'right') - 1 d\Uffffffff'indice del trigger immediatamente
        # precedente o coincidente con t_event
        idx = np.searchsorted(trigger_times, t_event, side='right') - 1

        # Nessun trigger prima di questo evento
        if idx < 0:
            continue

        # Controlla che il trigger trovato sia ancora dentro la finestra
        if trigger_times[idx] + window < t_event:
            continue

        relative_time = t_event - trigger_times[idx]

        c1 = centers1[ch]
        w1 = widths1[ch]
        c2 = centers2[ch]
        w2 = widths2[ch]

        in_bin1 = (abs(relative_time - c1)           <= w1 or
                   abs(relative_time - (c1 + window)) <= w1)
        in_bin2 = (abs(relative_time - c2)           <= w2 or
                   abs(relative_time - (c2 + window)) <= w2)

        if not in_bin1 and not in_bin2:
            continue

        if ch <= 15:
            c_modo = 2 * ch
        elif ch <= 31:
            c_modo = 2 * (ch + 16)
        elif ch <= 47:
            c_modo = 2 * (ch - 16)
        else:
            c_modo = 2 * ch

        if in_bin1:
            c_final = c_modo
            if ch <= 31:
                t_final = t_event - retards[ch] - offset
            else:
                t_final = t_event - retards[ch]
        else:
            c_final = c_modo + 1
            if ch <= 31:
                t_final = (t_event + (window - td[ch])
                           - retards[ch] - tdf[ch] - offset)
            else:
                t_final = (t_event + (window - td[ch])
                           - retards[ch] - tdf[ch])

        t_out_buf[j] = t_final
        c_out_buf[j] = c_final

    # --- FASE 2: compattazione seriale (rimuovi sentinelle) ---
    # Prima conta quanti validi ci sono
    valid = 0
    for j in range(n_total):
        if t_out_buf[j] != -1:
            valid += 1

    t_out = np.empty(valid, dtype=np.int64)
    c_out = np.empty(valid, dtype=np.int64)

    k = 0
    for j in range(n_total):
        if t_out_buf[j] != -1:
            t_out[k] = t_out_buf[j]
            c_out[k] = c_out_buf[j]
            k += 1

    return t_out, c_out

def process_measurement(times, photons, trigger_channel=17, sync_ch_1=3, sync_ch_2=27, window=725000):
    """
    Legge il file di input, applica un'operazione di esempio e scrive il risultato su output.
    """
    trigger_channel_merged=trigger_channel +32


    times_box_1,channels_box_1 = times[0]
    times_box_2,channels_box_2 = times[1]
    times_box_1 = times_box_1 - times_box_1[0]
    times_box_2 = times_box_2 - times_box_2[0]
    t_sync= T_sinc(TT1=times_box_1, CC1=channels_box_1, TT2=times_box_2, CC2=channels_box_2, ch1=sync_ch_1, ch2=sync_ch_2, window=100, mx=1e10)
    if t_sync>0:
        times_box_1 = times_box_1 + t_sync
    else:
        times_box_2 = times_box_2 - t_sync
    times_merged, channels_merged = merge_time_channel_arrays(times1 = times_box_1, channels1 = channels_box_1, times2 = times_box_2, channels2 = channels_box_2, ch_sync_1=sync_ch_1, ch_sync_2=sync_ch_2,channel_offset=32)
    #print(len(times_merged), len(times_box_1)+ len(times_box_2))
    del times_box_1, channels_box_1, times_box_2, channels_box_2
    mask = times_merged != 0
    times_merged = times_merged[mask]
    channels_merged= channels_merged[mask]
    t_tot, c_tot = modes_separator(times_merged, channels_merged,
                                            centers1, widths1, centers2, widths2,
                                            retards, td, tdf,
                                            trigger_channel=trigger_channel_merged,
                                            window=window,
                                            offset=0
                                            )
    del times_merged, channels_merged

    if photons==0:
        return t_tot, c_tot
    
    elif photons==1:
        distribution = np.bincount(c_tot, minlength=128)
    
    elif photons==2:
        shape = (128,128)
        order = np.argsort(t_tot)
        t_sorted = t_tot[order]
        c_sorted = c_tot[order]
        window_ps = 1800
        
        coincidences = find_coincidences(t_sorted, c_sorted, window_ps, photons)
        distribution = count_occurrences(shape=shape, data= coincidences)
    elif photons == 3:
        order = np.argsort(t_tot)
        t_sorted = t_tot[order]
        c_sorted = c_tot[order]
        window_ps = 1800
        coincidences = find_coincidences(t_sorted, c_sorted, window_ps, photons)
        coincidences = sort_coincidences_descending(coincidences)
        return coincidences
    # Process the data as needed
    # For example, you could compute histograms, correlations, etc.
    return distribution.flatten()
    
    # order = np.argsort(t_tot)
    # t_sorted = t_tot[order]
    # c_sorted = c_tot[order]
    # window_ps = 1800
    
    #coincidences = find_coincidences(t_sorted, c_sorted, window_ps, photons)

# def measure(boxes, esposizione, durata):
#     ripetizioni= int(durata/esposizione)
#     mesure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
#     return mesure


@njit(cache=True)
def find_coincidences_numba(t, c, window_ps, target_n):
    N = t.shape[0]

    # generation mask: mask[ch] == gen significa "ch visto in questa iterazione"
    mask = np.zeros(128, dtype=np.int64)
    gen  = 0

    max_out = N // target_n + 1
    out_arr = np.zeros((max_out, target_n), dtype=np.int64)
    touched = np.empty(target_n, dtype=np.int64)
    n_found = 0

    i = 0
    while i < N:
        gen += 1          # nuova generazione: tutti i canali risultano "non visti"
        t0   = t[i]
        ch0  = c[i]

        mask[ch0]  = gen
        touched[0] = ch0
        cnt        = 1
        j_max      = i

        for j in range(i + 1, N):
            if t[j] - t0 > window_ps:
                break
            ch = c[j]
            if mask[ch] != gen:
                mask[ch] = gen
                if cnt < target_n:
                    touched[cnt] = ch   # traccia solo i primi target_n
                cnt += 1
                j_max = j
                # non break mai: vogliamo contare tutti i canali distinti

        if cnt == target_n:
            for m in range(target_n):
                out_arr[n_found, m] = touched[m]
            n_found += 1
            i = j_max + 1
        else:
            i += 1

    return out_arr[:n_found]

@njit
def count_occurrences(shape, data):
    key = np.zeros(shape, dtype=np.int64)
    for row in data:
        a = row[0]
        b = row[1]
        key[a, b] += 1
        key[b, a] += 1
    return key

@njit(cache=True)
def sort_coincidences_descending(coincidences):
    n_found  = coincidences.shape[0]
    target_n = coincidences.shape[1]
    out = np.empty((n_found, target_n), dtype=np.int64)

    for i in range(n_found):
        # Copia la riga
        for m in range(target_n):
            out[i, m] = coincidences[i, m]
        # Insertion sort decrescente (target_n piccolo ? ottimale)
        for m in range(1, target_n):
            key = out[i, m]
            j = m - 1
            while j >= 0 and out[i, j] < key:
                out[i, j + 1] = out[i, j]
                j -= 1
            out[i, j + 1] = key

    return out
   
def find_coincidences(t, c, window_ps, target_n):
    coincidenses=find_coincidences_numba(t, c, window_ps, target_n)
    coincidenses=sort_coincidences_descending(coincidenses)
    return coincidenses

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

# def change_voltages(supply: PowerSupplies, volts) -> None:
#     assert isinstance(supply, PowerSupplies)
#     assert isinstance(volts, list)
#     assert supply._num_supplies==len(volts)
#     if control_volts(volts):  
#         volts = flatten_list(volts)
#         supply.voltages = volts
#         print('Volts changed ->' + str(supply.voltages))
#     else:
#         volts = [[0, 0] for _ in range(supply._num_supplies)]
#         supply.voltages = flatten_list(volts)
#         print('Volts reset to 0 due to error ->' + str(supply.voltages))

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

# def data_collection(inputs: list, Voltages: list, supply: PowerSupplies, n_supplies: int, dmx: DMXController, boxes, exposition = 0.1, duration = 60, repetitions_singles=1, repetitions_doubles = 2) -> list:

#     if len(Voltages)!=n_supplies:
#         raise ValueError('Number of voltages does not match the number of supplies')
#     if type(Voltages)==np.ndarray:
#         Voltages = np.reshape(Voltages, (len(Voltages)//2, 2))
#     elif type(Voltages)== list:
#         Voltages = np.reshape(np.array(Voltages), (len(Voltages)//2, 2))
#     else:
#         raise ValueError('Voltages must be either list or numpy array')
    
#     change_voltages(supply, Voltages)
#     output_list=[]

#     for input in inputs:

       
#         ########################################################
#         #                                                      #
#         loop = setloop(input)
#         dmx.set_active_outputs(loop)
#         #                                                      #
#         ########################################################

#         photons= len(input)
#         if photons == 1:
#             n_measurments = repetitions_singles
#         elif photons == 2:
#             n_measurments = repetitions_doubles
#         partial_distribution = []
        
#         for i in range(n_measurments):
#             if i==0:
#                 measurement = measure(boxes, exposition, duration)
#             else:
#                 with ProcessPoolExecutor() as executor:
#                     futures = {
#                             executor.submit(measure, boxes, exposition, duration): 'measure',
#                             executor.submit(process_measurement, measurement, photons=photons): 'process_measurement'
#                             }
#                     for future in as_completed(futures):
#                         task_name = futures[future]
#                         try:
#                             result = future.result()
#                             if task_name == 'measure':
#                                 measurement_tmp = result
#                             elif task_name == 'process_measurement':
#                                 partial_distribution.append(result)
#                         except Exception as e:
#                             print(f"Error in {task_name}: {e}")
#                     measurement = measurement_tmp #VERIFICARE CHE SIA UNA COPIA EFFETTIVA

#         result = process_measurement(measurement, photons=photons)
#         partial_distribution.append(result)
#         distribution = np.sum(np.array(partial_distribution), axis=0)
#         #distribution = np.concatenate(partial_distribution, axis = 0)
#         # QUI DEVI SOMMARE LE VARIE MISURE CONCATENATE PER LO STESSO INPUT
#         output_list.append(distribution)
#         dmx.stop_looping()
#     return np.array(output_list)


@njit
def split_times_by_channel(times, channels, n_channels):
    counts = np.zeros(n_channels, np.int64)
    for ch in channels:
        counts[ch] += 1

    times_by_ch = [np.empty(counts[ch], np.int64) for ch in range(n_channels)]
    idx = np.zeros(n_channels, np.int64)

    for i in range(len(times)):
        ch = channels[i]
        times_by_ch[ch][idx[ch]] = times[i]
        idx[ch] += 1

    return times_by_ch

@njit(parallel=True)
def all_intra_histograms(times_by_ch, n_channels, bin_width, num_bins):
    half = (num_bins // 2) * bin_width
    n_pairs = n_channels // 2
    hist_totals = np.zeros((n_pairs, num_bins), np.int64)

    for pair in prange(n_pairs):
        ch1 = pair * 2
        t1s = times_by_ch[ch1]
        n1 = t1s.shape[0]
        ch2 = ch1 + 1

        t2s = times_by_ch[ch2]
        n2 = t2s.shape[0]

        j_start = 0
        j_end = 0

        for i in range(n1):
            t1 = t1s[i]

            while j_start < n2 and t2s[j_start] < t1 - half:
                j_start += 1
            if j_end < j_start:
                j_end = j_start
            while j_end < n2 and t2s[j_end] <= t1 + half:
                j_end += 1

            for k in range(j_start, j_end):
                delay = t2s[k] - t1
                b = num_bins // 2 + int(delay // bin_width)
                if 0 <= b < num_bins:
                    hist_totals[pair, b] += 1
    return hist_totals
    
@njit(parallel=True)
def all_inter_histograms(times_by_ch, n_channels, bin_width, num_bins):
    half = (num_bins // 2) * bin_width
    n_pairs = n_channels // 2
    hist_totals = np.zeros((n_pairs, num_bins), np.int64)

    for pair in prange(n_pairs):
        ch1 = 32
        t1s = times_by_ch[ch1]
        n1 = t1s.shape[0]
        ch2 = 2 * pair

        t2s = times_by_ch[ch2]
        n2 = t2s.shape[0]

        j_start = 0
        j_end = 0

        for i in range(n1):
            t1 = t1s[i]

            while j_start < n2 and t2s[j_start] < t1 - half:
                j_start += 1
            if j_end < j_start:
                j_end = j_start
            while j_end < n2 and t2s[j_end] <= t1 + half:
                j_end += 1

            for k in range(j_start, j_end):
                delay = t2s[k] - t1
                b = num_bins // 2 + int(delay // bin_width)
                if 0 <= b < num_bins:
                    hist_totals[pair, b] += 1
    return hist_totals

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

# def lossEvalExp(parameters, inputs, target, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition):
#     tempPrediction = data_collection(inputs, np.sqrt(parameters), supply, Nsupp, boxes, exposition, duration, repetitions_singles, repetitions_doubles)
#     tempLoss = MyMaeExp(tempPrediction, target)
#     #print(tempLoss)
#     return(tempLoss)

# def MyMaeExp(y_predicted, y_true):
#     total_error_arr = 0
#     for yp, yt in zip(y_predicted, y_true):
#         yp = (yp/np.sum(yp))
#         yt = (yt/np.sum(yt))
#         #print("yp: ", yp, "yt: ", yt)
#         total_error_arr += abs(yp - yt)
#     total_error = np.sum(total_error_arr)
#     #print("Total error is:",total_error)
#     mae = total_error/len(y_predicted)
#     #print("Mean absolute error is:",mae)
#     return mae
# # Funzione di training per il caso sperimentale.

# def myTrainingLoopExp(currentParamsTrainable, duration, repetitions_singles, repetitions_doubles, numParams, input_states_one, targetState1, input_states_two_full, targetState2_full, trainingParams, paramsUnitary = []):
#     epochsNum = trainingParams["epochsNum"]
#     LR_check = trainingParams["LR_check"]
#     LR_move = trainingParams["LR_move"]
#     useTwoPhotons = trainingParams["useTwoPhotons"]
#     typeTraining = trainingParams["typeTraining"]
#     typeOrder = trainingParams["typeOrder"]
#     printProgress = trainingParams["printProgress"]
#     checkPairsNum = trainingParams["checkPairsNum"]
#     firstNeighbourList = trainingParams["firstNeighbourList"]
#     avoidBoundary = trainingParams["avoidBoundary"]
#     supply = trainingParams["supply"]
#     Nsupp = trainingParams["Nsupp"]
#     boxes = trainingParams["boxes"]
#     exposition= trainingParams["exposition"]
#     parameterValueMin = trainingParams["parameterValueMin"]
#     parameterValueMax = trainingParams["parameterValueMax"]
#     parameterValueMaxReset = trainingParams["parameterValueMaxReset"]
#     parameterValueMinReset = trainingParams["parameterValueMinReset"]
#     paramsOrderList = np.arange(numParams)
#     bestLoss = 1000
#     maxPairs = len(input_states_two_full)
#     bestParams = np.zeros_like(currentParamsTrainable)
#     lossHistory = np.empty(epochsNum + 1)
#     #fidelityHistory = np.empty(epochsNum + 1)
    
#     for epoch in range(epochsNum):
#         if (typeOrder == "allRandom"):
#             chosenParam = np.random.choice(numParams)
#         elif (typeOrder == "listRandom"):
#             if (epoch%numParams == 0):
#                 np.random.shuffle(paramsOrderList)
#             chosenParam = paramsOrderList[epoch%numParams]
#         else:
#             print("ERROR, NO VALID ORDER TYPE SELECTED")

#         chosenPairs = np.random.choice(maxPairs, checkPairsNum, replace=False)
#         input_states_two = list( input_states_two_full[j] for j in chosenPairs )
#         targetState2 = np.zeros((checkPairsNum, len(targetState2_full[0])))
#         for j in range(checkPairsNum):
#             targetState2[j] = targetState2_full[chosenPairs[j]]

#         #print(input_states_two)
        
#         #currentFidelity = MyFidelity(currentParamsTrainable, paramsNotTrainable, baseParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, paramsUnitary, paramsUnitary)
        
#         prevLoss = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)     #can be skipped if training step is equal to check step.
#         if (useTwoPhotons == True):
#             prevLoss = prevLoss + lossEvalExp(currentParamsTrainable, input_states_two, targetState2, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
#         if (printProgress == "all"):
#             print("Epoch:", epoch)
#             print("Current loss is:", prevLoss,  "    Changed param:", chosenParam)
#             #print("Current loss is:", prevLoss, "    Current fidelity is:", currentFidelity, "    Changed param:", chosenParam)
#             print(chosenPairs)
#         #print("Current fidelity is:", currentFidelity) 
#         #print("Changed param:", chosenParam)
#         if (prevLoss < bestLoss):
#             bestLoss = prevLoss
#             bestParams = currentParamsTrainable.copy()
#         lossHistory[epoch] = prevLoss
#         #fidelityHistory[epoch] = currentFidelity
        
        
#         checkShift = prevLoss * LR_check
#         tempStore = currentParamsTrainable[chosenParam]
#         currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, checkShift, avoidBoundary, parameterValueMin, parameterValueMax, parameterValueMaxReset, parameterValueMinReset)
#         #print(currentParamsTrainable)
#         upLoss = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
#         if (useTwoPhotons == True):
#             upLoss = upLoss + lossEvalExp(currentParamsTrainable, input_states_two, targetState2, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
#         currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (-checkShift), avoidBoundary, parameterValueMin, parameterValueMax, parameterValueMaxReset, parameterValueMinReset)
#         downLoss = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
#         if (useTwoPhotons == True):
#             downLoss = downLoss + lossEvalExp(currentParamsTrainable, input_states_two, targetState2, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
#         # calculating which direction to move
#         if ((upLoss < prevLoss) & (downLoss < prevLoss)):
#             if (downLoss < upLoss):
#                 proportion = -1 * ((prevLoss - downLoss)/prevLoss)
#             else:
#                 proportion = ((prevLoss - upLoss)/prevLoss)
#         elif(upLoss < prevLoss):
#             proportion = ((prevLoss - upLoss)/prevLoss)
#         elif(downLoss < prevLoss):
#             proportion = -1 * ((prevLoss - downLoss)/prevLoss)
#         else:
#             if (printProgress == "all"):
#                 print("Changing parameter value does not improve loss")
#             proportion = 0
#         # moving in the decided direction based on training type
#         if (typeTraining == "proportional"):
#             currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (proportion*LR_move), avoidBoundary, parameterValueMin, parameterValueMax, parameterValueMaxReset, parameterValueMinReset)
#         elif (typeTraining == "absolute"):
#             currentParamsTrainable[chosenParam] = UpdateParameter(tempStore, (np.sign(proportion)*checkShift*LR_move/LR_check), avoidBoundary, parameterValueMin, parameterValueMax, parameterValueMaxReset, parameterValueMinReset)
#         else:
#             print("ERROR, NO VALID TRAINING TYPE SELECTED")
    
#     prevLoss = lossEvalExp(currentParamsTrainable, input_states_one, targetState1, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
#     if (useTwoPhotons == True):
#             prevLoss = prevLoss + lossEvalExp(currentParamsTrainable, input_states_two, targetState2, duration, repetitions_singles, repetitions_doubles, supply, Nsupp, boxes, exposition)
#     #currentFidelity = MyFidelity(currentParamsTrainable, paramsNotTrainable, baseParamsTrainable, paramsNotTrainable, timeCoupling, sizeHam, paramsUnitary, paramsUnitary)
#     lossHistory[-1] = prevLoss
#     #fidelityHistory[-1] = currentFidelity
#     if (printProgress == "last" or printProgress == "all"):
#         #print("Last loss is:", prevLoss, "    Last fidelity is:", currentFidelity)
#         print("Last loss is:", prevLoss)
#     #return currentParamsTrainable, lossHistory, fidelityHistory, bestParams, bestLoss
#     return currentParamsTrainable, lossHistory, bestParams, bestLoss

# # Varie funzioni training.