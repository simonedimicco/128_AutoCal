from qlab.devices.tdc import QuTag
import qlab.counting.counting as counting
import qlab.counting.cocount as cocount
from auto_classical import PowerSupplies
from qlab.devices.KeithleyPowerSupply import KeithleyPowerSupply


import numpy as np
from numba import njit, prange
import time
from Spazzatura.WhiteDict import pulse_dict, time_differences, fine_tuning, retards_box
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



@njit(cache=True)
def compute_trigger_index(channels_times,
                          channels_list,
                          trigger_channel,
                          trigger_times,
                          window):

    n = channels_times.size
    trig_idx = np.full(n, -1, dtype=np.int64)

    i = 0
    n_trig = trigger_times.size

    for j in range(n):

        if channels_list[j] == trigger_channel:
            continue

        tj = channels_times[j]

        while i < n_trig - 1 and trigger_times[i] + window < tj:
            i += 1

        #if trigger_times[i] <= tj:
        trig_idx[j] = i

    return trig_idx

from numba import prange

@njit(parallel=False, cache=True)
def Modes_separator(channels_times,
                                  channels_list,
                                  trig_idx,
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

        if trig_idx[j] == -1:
            continue

        ch = channels_list[j]
        if ch == trigger_channel:
            continue

        tj = channels_times[j]
        trig_time = trigger_times[trig_idx[j]]
        relative_time = tj - trig_time

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

        # ---- mapping ----
        if ch <= 31:
            base = 2 * ch if ch <= 15 else 2 * (ch + 16)
        else:
            base = 2 * (ch - 16) if ch <= 47 else 2 * ch

        c_final = base if is_bin1 else base + 1

        # ---- tempo ----
        if is_bin1:
            if ch <= 31:
                t_final = tj - retards[ch] - offset
            else:
                t_final = tj - retards[ch]
        else:
            if ch <= 31:
                t_final = (
                    tj + (725000 - Time_differences[ch])
                    - retards[ch]
                    - Time_differences_fine_tuning[ch]
                    - offset
                )
            else:
                t_final = (
                    tj + (725000 - Time_differences[ch])
                    - retards[ch]
                    - Time_differences_fine_tuning[ch]
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


@njit
def count_occurrences(shape, data):
    key = np.zeros(shape, dtype=np.int64)
    for row in data:
        a = row[0]
        b = row[1]
        key[a, b] += 1
    return key


def measure(boxes, esposizione, durata):
    ripetizioni= int(durata/esposizione)
    mesure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
    return mesure

def process_measurement(measurement, photons):
    trigger_channel=17
    trigger_channel_merged=trigger_channel +32
    sync_ch_1=3
    sync_ch_2=27

    times = [(t,c) for t,c in measurement]
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
    
    trig_idx = compute_trigger_index(channels_times=times_merged, 
                                     channels_list=channels_merged, 
                                     trigger_channel=trigger_channel_merged, 
                                     trigger_times=trigger_times, 
                                     window=725000)
    
    t_tot, c_tot = Modes_separator(channels_times=times_merged,
                                   channels_list=channels_merged,
                                   trig_idx=trig_idx,
                                   trigger_times=trigger_times,
                                   trigger_channel=trigger_channel_merged,
                                   mio_dizionario=pulse_dict,
                                   Time_differences=time_differences,
                                   Time_differences_fine_tuning=fine_tuning,
                                   retards=retards_box,
                                   window=725000,
                                   offset=0)
    
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
    return distribution

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

def data_collection(inputs: list, Voltages: list, supply: PowerSupplies, n_supplies: int, boxes, exposition = 0.1, duration = 60, repetitions_singles=1, repetitions_doubles = 2) -> list:

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
        #                   CODICE QDMX                        #
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
        distribution = np.concatenate(partial_distribution, axis = 0)
        output_list.append(distribution)
    return output_list

'''@njit(cache=True)
def Modes_separator(channels_times, channels_list, trigger_channel=63, mio_dizionario=typed_dict_tot, 
                    Time_differences= Time_differences_tot, Time_differences_fine_tuning=Time_differences_fine_tuning_tot,
                    retards=retards_box_tot, window=730000):
    # Trovo quanti trigger ci sono
    trig_mask = channels_list == trigger_channel
    num_triggers = np.sum(trig_mask)            # operazione veloce in Numba
    n_out = channels_list.size - num_triggers   # quanti voglio tenere

    # Pre-allocazione degli array di output filtrati
    t_filt = np.empty(n_out, dtype=np.int64)
    c_filt = np.empty(n_out, dtype=np.int64)

    # Se non ci sono eventi di trigger, ritorno vuoti
    if num_triggers == 0:
        return t_filt, c_filt

    # Prendo i tempi del trigger
    trigger_times = channels_times[trig_mask]

    i = 0    # indice per scorrere trigger_times
    k = 0    # indice di inserimento in t_filt/c_filt

    # Ciclo una sola volta su tutti gli eventi
    for j in range(channels_list.size):
        ch = channels_list[j]
        if ch == trigger_channel:
            # salta i trigger, non incrementa k
            continue

        # estraggo dal dizionario
        c1, w1, c2, w2 = mio_dizionario[ch]
        shift = retards[ch]

        # avanzo l'indice trigger fino al bin giusto
        while i < trigger_times.size - 1 and trigger_times[i] + window < channels_times[j]:
            i += 1

        relative_time = channels_times[j] - trigger_times[i]
        ##divido i  modi in base al tempo di arrivo rispetto al trigger
        ## Nel dividere i modi assegno anche i canali delle scatole ai giusti detector: 
        ## nella scatola 1: canali 1 a 16 -> modi 1 a 16; canali 17 a 32 -> modi 33 a 48
        ## nella scatola 2 :canali 1 a 16 (33 a 48) -> modi 17 a 32; canali 17 a 32 (49 a 64) -> modi 49 a 64
        ## Inoltre nella scatola 1 il modo pari è quello in ritardo (e va quindi riscalato), mentre nella scatola 1 il modo dispari è quello in ritardo 
    
        #caso time bin 1 (modo dispari)
        if abs(relative_time - c1) <= w1 or abs(relative_time - (c1 + window)) <= w1:
            if ch <= 31:
                if ch <= 15:
                    c_out = 2 * ch
                else:
                    c_out = 2 * (ch + 16)
                t_out = channels_times[j] 

            elif ch <= 63:
                if ch <= 47:
                    c_out = 2 * ( ch - 16 )
                else:
                    c_out = 2 * ch
                t_out = channels_times[j]  #+ (725000 - Time_differences[ch]) 
            else:
                print('channel error assignement')
        #caso time bin 2 (modo pari)
        elif abs(relative_time - c2) <= w2 or abs(relative_time - ( c2 + window))<= w2:
            if ch <= 31:
                if ch <= 15:
                    c_out = (2 * ch) + 1
                else:
                    c_out = (2 * (ch + 16)) + 1

                t_out = channels_times[j] + (725000 - Time_differences[ch]) 
            elif ch <= 63:
                if ch <= 47:
                    c_out = (2 * (ch - 16)) + 1
                else:
                    c_out = (2 * ch) + 1
                t_out = channels_times[j] + (725000 - Time_differences[ch]) 
            else:
                print('channel error assignement')

        else:
            continue
        c_filt[k] = c_out
        t_filt[k] = t_out
        k+=1
    # Filtro finale per rimuovere i t_out = 0 (non validi)
    final_len = 0
    for idx in range(k):
        if t_filt[idx] != 0:
            final_len += 1

    t_clean = np.empty(final_len, dtype=np.int64)
    c_clean = np.empty(final_len, dtype=np.int64)

    kk = 0
    for idx in range(k):
        if t_filt[idx] != 0:
            t_clean[kk] = t_filt[idx]
            c_clean[kk] = c_filt[idx]
            kk += 1

    return t_clean, c_clean

@njit(parallel=True, cache=True)
def apply_remaining_transformations(t_lite,
                                    c_lite,
                                    retards,
                                    Time_differences,
                                    Time_differences_fine_tuning,
                                    offset=0):
    """
    Prende in input:
      - t_lite, c_lite = output “grezzo” di Modes_separator_lite
      - retards, Time_differences, Time_differences_fine_tuning, channel_versions_np
    Restituisce:
      - t_full: array di tempi corretti come farebbe Modes_separator “completo”
      - c_full: stesso array di canali (uguale a c_lite)
    NB: per invertire la trasformazione “c_out” → “ch originale” usiamo la logica
        sui semicanali pari/dispari (half = floor(c_out/2)) e i quattro possibili range:
          0–15, 16–31, 32–47, 48–63
    """

    n = c_lite.size
    t_full = np.empty(n, dtype=np.int64)
    c_full = np.empty(n, dtype=np.int64)

    for idx in range(n):
        c_out = c_lite[idx]
        t0    = t_lite[idx]

        # ricavo semicanale “half” e parità per decidere bin1/bin2
        if c_out % 2 == 0:
            # time bin 1
            half = c_out // 2
            is_bin2 = False
        else:
            # time bin 2
            half = (c_out - 1) // 2
            is_bin2 = True

        # ---- trovo ch originale a partire da half ----
        # caso half 0–15
        if half <= 15:
            ch = half
        # caso half 16–31: corrisponde a ch = half+16 (range 32–47)
        elif half <= 31:
            ch = half + 16
        # caso half 32–47: corrisponde a ch = half-16 (range 16–31)
        elif half <= 47:
            ch = half - 16
        # caso half 48–63: corrisponde a ch = half (range 48–63)
        else:
            ch = half

        # ---- a questo punto ch ∈ [0..63], is_bin2 dice se era time bin 2 (True) o bin 1 (False) ----
        # ricopio il canale “finale” senza modifiche (è già c_out corretto)
        c_full[idx] = c_out

        # ora calcolo t_full in base a quanto manca rispetto al “lite”:
        if not is_bin2:
            
            if ch <= 31:
                t_full[idx] = t0 - retards[ch]-offset
            else:
                # ch ∈ [32..63]
                # formula completa: t_full = channels_times - retards[ch]
                #                       + (725000 - TD[ch]) + TDF[ch]
                # t_lite = channels_times + (725000 - TD[ch])
                # quindi delta = (-retards[ch] + TDF[ch])
                t_full[idx] = t0 - retards[ch] 

        else:
            if ch <= 31:
                t_full[idx] = t0 - retards[ch] - Time_differences_fine_tuning[ch] - offset
            else:
                # ch ∈ [32..63]
                # formula completa: t_full = channels_times - retards[ch]
                #                       + (725000 - TD[ch]) + TDF[ch]
                # t_lite = channels_times + (725000 - TD[ch])
                # quindi delta = (-retards[ch] + TDF[ch])
                t_full[idx] = t0 - retards[ch] - Time_differences_fine_tuning[ch] 
            
            # # === time bin 2 ===
            # if ch <= 31:
            #     t_full[idx] = t0 - retards[ch] - Time_differences_fine_tuning[ch]
            # else:
            #     # ch ∈ [32..63], bin2: 
            #     # full: t_full = channels_times - retards[ch]
            #     # t_lite = channels_times
            #     # delta = -retards[ch]
            #     t_full[idx] = t0 - retards[ch]

    return t_full, c_full
    '''

'''
@njit(cache=True)
def find_coincidences_numba(t, c, window_ps, target_n):
    N = t.shape[0]
    mask = np.zeros(128, dtype=np.uint8)
    out = List()  # List of List[int]
    
    i = 0
    while i < N:
        t0 = t[i]
        # azzera la maschera
        for k in range(128):
            mask[k] = 0
        mask[c[i]] = 1
        cnt = 1
        
        # inizializzo j_max all’indice corrente, così se non entra nessun “nuovo” canale rimane i
        j_max = i
        
        # scorro gli eventi successivi
        for j in range(i+1, N):
            dt = t[j] - t0
            if dt > window_ps:
                break
            ch = c[j]
            if mask[ch] == 0:
                mask[ch] = 1
                cnt += 1
                j_max = j  # aggiorno l’indice dell’ultimo canale nuovo
        # se ho trovato esattamente `target_n` canali distinti
        if cnt == target_n:
            grp = List()
            for k in range(128):
                if mask[k]:
                    grp.append(k)
            out.append(grp)
            # salto direttamente a j_max + 1
            i = j_max + 1
        else:
            # nessuna coincidenza, passo semplicemente all’evento successivo
            i += 1

    return out
'''


'''def process_data(times_box_1, channels_box_1, times_box_2, channels_box_2, photons = 3) -> None:
    
    #Legge il file di input, applica un'operazione di esempio e scrive il risultato su output.
    
    trigger_channel=17
    trigger_channel_merged=trigger_channel +32
    sync_ch_1=3
    sync_ch_2=27

    times_box_1 = times_box_1 - times_box_1[0]
    times_box_2 = times_box_2 - times_box_2[0]
    t_sync= T_sinc(TT1=times_box_1, CC1=channels_box_1, TT2=times_box_2, CC2=channels_box_2, ch1=sync_ch_1, ch2=sync_ch_2, window=100, mx=1e10)
    if t_sync>0:
        times_box_1 = times_box_1 + t_sync
    else:
        times_box_2 = times_box_2 - t_sync
    times_merged, channels_merged = merge_time_channel_arrays(times1 = times_box_1, channels1 = channels_box_1, times2 = times_box_2, channels2 = channels_box_2, ch_sync_1=sync_ch_1, ch_sync_2=sync_ch_2,channel_offset=32)
    print(len(times_merged), len(times_box_1)+ len(times_box_2))
    del times_box_1, channels_box_1, times_box_2, channels_box_2
    mask = times_merged != 0
    times_merged = times_merged[mask]
    channels_merged= channels_merged[mask]
    t_tot, c_tot = Modes_separator(channels_times = times_merged, channels_list = channels_merged, 
                                    trigger_channel= trigger_channel_merged, mio_dizionario=typed_dict_tot, 
                                    Time_differences= typed_td_tot, Time_differences_fine_tuning=typed_tdf_tot,
                                    retards=typed_retards_tot, window=725000)
    
    del times_merged, channels_merged
    t_0=time.time()
    mask = t_tot != 0
    t = t_tot[mask]
    c= c_tot[mask]
    t_tot, c_tot = apply_remaining_transformations(t_lite=t,c_lite=c,
                                                   retards=typed_retards_tot,Time_differences=typed_td_tot,
                                                   Time_differences_fine_tuning=typed_tdf_tot,
                                                   offset=0)
    t_1=time.time()
    print(f'Fine iterazione (tempo impiegato : {t_1-t_0:.1f} sec). Salvo i dati...\n')
    order = np.argsort(t_tot)
    t_sorted = t_tot[order]
    c_sorted = c_tot[order]
    window_ps = 1800
    
    coincidences = find_coincidences(t_sorted, c_sorted, window_ps, photons)'''
    