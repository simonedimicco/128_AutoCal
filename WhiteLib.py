from qlab.devices.tdc import QuTag
import qlab.counting.counting as counting
import qlab.counting.cocount as cocount
from auto_classical import PowerSupplies
from qlab.devices.KeithleyPowerSupply import KeithleyPowerSupply
import logging
from dmx_controller import DMXController


import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

from numba import njit, prange
from tqdm import tqdm
from WhiteDict import *


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

    # 1) Pre-filtraggio: maschera booleana → selezione vettoriale
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

    # 4) Tail: uno dei due array è esaurito, copia il resto
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
    # Ogni iterazione è indipendente: searchsorted sostituisce trig_ptr
    for j in prange(n_total):
        ch = channels_list[j]
        if ch == trigger_channel:
            continue

        t_event = channels_times[j]

        # Trova il trigger più recente con t_trig <= t_event
        # searchsorted(..., 'right') - 1 dà l'indice del trigger immediatamente
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
    if photons==0:
        return t_tot, c_tot
    del times_merged, channels_merged
    
    
    # order = np.argsort(t_tot)
    # t_sorted = t_tot[order]
    # c_sorted = c_tot[order]
    # window_ps = 1800
    
    #coincidences = find_coincidences(t_sorted, c_sorted, window_ps, photons)

def measure(boxes, esposizione, durata):
    ripetizioni= int(durata/esposizione)
    mesure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
    return mesure


@njit(cache=True)
def find_coincidences(t, c, window_ps, target_n):
    N = t.shape[0]

    # generation mask: mask[ch] == gen significa "ch visto in questa iterazione"
    mask = np.zeros(128, dtype=np.int64)
    gen  = 0

    max_out = N // target_n + 1
    out_arr = np.empty((max_out, target_n), dtype=np.int64)
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


def hello():
    print("Hello, World!")
   