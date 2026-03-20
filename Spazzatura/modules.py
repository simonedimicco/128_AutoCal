import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

from numba import jit
from numba import njit, prange
from numba.core import types
#from numba.typed import Dict
from tqdm import tqdm
from WhiteDict import *
from matplotlib.ticker import FuncFormatter



#############################################################################################################################################################################################
#IN QUESTO BLOCCO  METTO LE FUNZIONI PER SEPARARE E SINCRONIZZARE I MODI DOPO LE MISURE, INOLTRE NUMERA CORRETTAMENTE I MODI



@njit
def Modes_separator(channels_times, channels_list, trigger_channel=63, mio_dizionario=typed_dict_tot, 
                    Time_differences= Time_differences_tot, Time_differences_fine_tuning=Time_differences_fine_tuning_tot,
                    retards=retards_box_tot, window=725000):
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

                t_out = channels_times[j] + (window - Time_differences[ch]) 
            elif ch <= 63:
                if ch <= 47:
                    c_out = (2 * (ch - 16)) + 1
                else:
                    c_out = (2 * ch) + 1
                t_out = channels_times[j] + (window - Time_differences[ch]) 
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




@njit
def  apply_remaining_transformations(t_lite,
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
            
            '''# === time bin 2 ===
            if ch <= 31:
                t_full[idx] = t0 - retards[ch] - Time_differences_fine_tuning[ch]
            else:
                # ch ∈ [32..63], bin2: 
                # full: t_full = channels_times - retards[ch]
                # t_lite = channels_times
                # delta = -retards[ch]
                t_full[idx] = t0 - retards[ch]'''

    return t_full, c_full
#############################################################################################################################################################################################à
# QUESTO BLOCCO SERVE PER VISUALIZZARE I CONTEGGI ALL'INTERNO DI UNA FINESTRA DATA DAL TRIGGER DEL DMX



# Prima funzione: calcola i tempi (compatibile con JIT)
@njit
def calculate_time_differences(channel_time, trigger_time, window):
    i = 0
    # Pre-allocare un array di dimensione massima possibile
    max_size = len(channel_time)
    time_diffs = np.zeros(max_size, dtype=np.float64)
    count = 0
    err = 0
    
    for j in range(len(channel_time)):
        while i < len(trigger_time) - 1 and trigger_time[i] + window < channel_time[j]:
            i += 1
        if i < len(trigger_time):
            if channel_time[j] - trigger_time[i] < 0:
                err += 1
            else:
                time_diffs[count] = channel_time[j] - trigger_time[i]
                count += 1
    
    # Restituisci solo gli elementi validi
    return time_diffs[:count], err

# Seconda funzione: calcola l'istogramma (compatibile con JIT)
@njit
def calculate_histogram(time_diffs, window, num_bins=100):
    bins = np.linspace(0, window, num_bins + 1)
    
    # Calcola l'istogramma e specifica esplicitamente il tipo
    hist, bin_edges = np.histogram(time_diffs, bins=bins)
    
    # Converti esplicitamente in float64 per garantire coerenza di tipo
    hist = hist.astype(np.float64)
    
    max_val = hist.max()
    if max_val > 0:
        hist_norm = hist #/ max_val
    else:
        hist_norm = hist
        
    return hist_norm, bin_edges

# Funzione principale che combina i calcoli e la visualizzazione
def Window_histo(channel_time, trigger_time, window = 730000):
    
    
    # Chiamate alle funzioni JIT per i calcoli
    time_diffs, err = calculate_time_differences(channel_time, trigger_time, window)
    hist_norm, bin_edges = calculate_histogram(time_diffs, window)
    
    # Parte di visualizzazione (non JIT)
    bin_width = window / 100
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist_norm, align='edge', width=bin_width)
    plt.xlim(0, window)
    plt.xlabel('Time difference')
    plt.ylabel('Normalized counts')
    plt.title(f'Histogram of time differences (errors: {err})')
    
    return hist_norm, bin_edges

#############################################################################################################################################################################################
#IN QUESTO BLOCCO LA FUNZIONE CHE UNICE I DUE ARRAY DELLE DUE SCATOLE IN UNO SINGOLO

from numba import njit
import numpy as np

@njit
def merge_time_channel_arrays(times1, channels1, times2, channels2, ch_sync_1, ch_sync_2, channel_offset=32):
    # Primo passaggio: conta gli eventi validi per determinare la lunghezza finale
    valid_count = 0
    for i in range(len(times1)):
        if channels1[i] != ch_sync_1:
            valid_count += 1
    for j in range(len(times2)):
        if channels2[j] != ch_sync_2:
            valid_count += 1

    # Alloca array finali della lunghezza giusta
    merged_times = np.empty(valid_count, dtype=np.int64)
    merged_channels = np.empty(valid_count, dtype=np.int64)

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

    return merged_times, merged_channels


#############################################################################################################################################################################################V
#IN QUESTO BLOCCO LA FUNZIONE PER REALIZZARARE ISTOGRAMMI BIDIREZIONALI DI COINCIDENZE

@njit
def coincidence_histogram_fast(times, channels, ch1, ch2, bin_width, num_bins):
    half = (num_bins//2) * bin_width
    hist = np.zeros(num_bins, np.int64)
    n = len(times)

    # 1) Estrai e filtra `times2` per ch2
    #    (assumo channels ordinato insieme a times)
    cnt2 = 0
    for x in channels:
        if x == ch2:
            cnt2 += 1
    times2 = np.empty(cnt2, np.int64)
    idx2 = 0
    for i in range(n):
        if channels[i] == ch2:
            times2[idx2] = times[i]
            idx2 += 1

    # 2) Due‐puntatori per la finestra [t1-half, t1+half]
    j_start = 0
    j_end = 0
    len2 = times2.shape[0]

    for i in range(n):
        if channels[i] != ch1:
            continue
        t1 = times[i]

        # avanza j_start finché times2[j_start] < t1-half
        while j_start < len2 and times2[j_start] < t1 - half:
            j_start += 1
        # assicurati che j_end >= j_start
        if j_end < j_start:
            j_end = j_start
        # avanza j_end finché times2[j_end] <= t1+half
        while j_end < len2 and times2[j_end] <= t1 + half:
            j_end += 1

        # ora [j_start, j_end) è la finestra valida
        for k in range(j_start, j_end):
            delay = times2[k] - t1
            b = num_bins//2 + int(delay // bin_width)
            if 0 <= b < num_bins:
                hist[b] += 1

    return hist

#############################################################################################################################################################################################
#IN QUESTO BLOCCO LA FUNZIONE PER TROVARE IL TEMPO PER SINCRONIZZARE LE DUE SCATOLE
@njit(parallel=True)
def T_sinc_fast(TT1, CC1, TT2, CC2, ch1, ch2, window=100, mx=1e10):
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

#############################################################################################################################################################################################

def map_merged_cannel_box_2(ch):
    if ch<16:
        return ch+16
    else:
        return ch+32
    
###########################################################################################################################################################################àà

@jit
def Corr_hist(TT1,CC1,bins,mx,ch1=0,ch2=0):

    vals = np.zeros(2*bins)
    TT2=TT1
    CC2=CC1
    k = 0
    
    for i in range(len(TT1)):
        if CC1[i] == ch1:
            
            while k<len(TT2) and TT2[k]<TT1[i]:
                k+=1
            if k == len(TT2):
                break
            
            j = k
            while j<len(TT2) and TT2[j]-TT1[i]<mx:
                if CC2[j] == ch2:
                    b = int((TT2[j]-TT1[i])/mx*bins)+bins
                    vals[b]+=1
                j+=1
            
            j = k-1
            while j>0 and TT1[i]-TT2[j]<mx:
                if CC2[j] == ch2:
                    b = bins-int((TT1[i]-TT2[j])/mx*bins)-1
                    vals[b]+=1
                j-=1
    return vals

##############################################################################################################################################################################################à
#IN QUESTO BLOCCO PROCESSIAMO I DATI IN TEMPO REALE IN MODO DA RISPARMIARE SPAZIO
def process_file(input_path: str, output_path: str, output_path_2: str, output_path_3: str,photons = 3) -> None:
    """
    Legge il file di input, applica un'operazione di esempio e scrive il risultato su output.
    """
    trigger_channel=17
    trigger_channel_merged=trigger_channel +32
    sync_ch_1=3
    sync_ch_2=27
    with np.load(input_path,allow_pickle=True) as data_load:
        times_box_1     = data_load['times_box_1']
        channels_box_1  = data_load['channels_box_1']
        times_box_2     = data_load['times_box_2']
        channels_box_2  = data_load['channels_box_2']
    times_box_1 = times_box_1 - times_box_1[0]
    times_box_2 = times_box_2 - times_box_2[0]
    t_sync= T_sinc_fast(TT1=times_box_1, CC1=channels_box_1, TT2=times_box_2, CC2=channels_box_2, ch1=sync_ch_1, ch2=sync_ch_2, window=100, mx=1e10)
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
    #np.savez_compressed(output_path, t_tot=np.array(t_tot), c_tot=np.array(c_tot))
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
    np.savez_compressed(output_path_2, t_tot=np.array(t_tot), c_tot=np.array(c_tot))
    order = np.argsort(t_tot)
    t_sorted = t_tot[order]
    c_sorted = c_tot[order]
    window_ps = 1800
    
    coincidences = find_coincidences(t_sorted, c_sorted, window_ps, photons)
    
    output_txt_path = output_path_3.replace(".npz", ".txt")
    with open(output_txt_path, 'w') as f:
        for group in coincidences:
            f.write(f"{group}\n")
    print(f"📄 Coincidenze salvate in: {output_txt_path}")
    print(f"✨ File processato: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")


def watch_folder(
    input_folder: str,
    processed_folder: str,
    processed_folder_2: str,
    processed_folder_3: str,
    check_interval: int = 60,
    timeout_minutes: int = 5,
    photons = 3
) -> None:
    """
    Monitora la cartella `input_folder`:
    - Controlla ogni `check_interval` secondi se ci sono file.
    - Se trova file, li processa uno ad uno, salva l'output in `processed_folder` e rimuove l'originale.
    - Se non trova file per `timeout_minutes`, si ferma automaticamente.
    """
    last_found_time = datetime.now()
    timeout = timedelta(minutes=timeout_minutes)

    print(f"🔍 Inizio monitoraggio: {input_folder}")
    while True:
        # Lista dei file presenti nella cartella di input
        files = [f for f in os.listdir(input_folder)
                 if (os.path.isfile(os.path.join(input_folder, f))and f.endswith('.npz'))]
        files.sort()  # Ordina i file per nome (opzionale)
        if files:
            # Aggiorna il timestamp dell'ultimo file trovato
            last_found_time = datetime.now()

            for filename in files:
                src = os.path.join(input_folder, filename)
                dest = os.path.join(processed_folder, f"processed_{filename}")
                dest_2 = os.path.join(processed_folder_2, f"processed_{filename}_v2")
                dest_3 = os.path.join(processed_folder_3, f"coincidences_{filename}")

                try:
                    process_file(src, dest, dest_2, dest_3, photons)
                    os.remove(src)
                    print(f"✅ Rimosso file originale: {filename}")
                except Exception as e:
                    print(f"❌ Errore con {filename}: {e}")
        else:
            # Controlla se è passato il timeout
            elapsed = datetime.now() - last_found_time
            if elapsed > timeout:
                print(f"⏰ Nessun file rilevato per {timeout_minutes} minuti. Programma terminato.")
                break

        # Aspetta prima del prossimo controllo
        time.sleep(check_interval)







###############################################################################################################################################################################################
#IN QUESTO BLOCCA ANDIAMO A SELEZIONARE SOLO LE N-UPLE DI COINCIDENZE
def reconstruct_coincidences_window(times, channels, window_ps=2500, target_n=None):
    """
    Identify groups of detectors that click within a sliding window after each click.

    Parameters:
    - times: 1D numpy array of event times (ps)
    - channels: 1D numpy array of detector IDs (0-127)
    - window_ps: float, window width in ps
    - target_n: int or None, if int only groups of that size; if None, all groups size>=2

    Returns:
    - list of tuples of sorted detector IDs
    """
    # assume times sorted in ascending order
    N = len(times)
    results = []
    seen = set()

    j_start = 0
    for i in range(N):
        t0 = times[i]
        # advance window start index
        j_start = max(j_start, i + 1)
        group = {int(channels[i])}
        # collect all events within window_ps
        for j in range(j_start, N):
            if times[j] - t0 > window_ps:
                break
            group.add(int(channels[j]))
        # record if meets target size
        if len(group) >= 2 and (target_n is None or len(group) == target_n):
            tpl = tuple(sorted(group))
            if tpl not in seen:
                seen.add(tpl)
                results.append(tpl)
    return results

###############################################################################################################################################################################################
#IN QUESTO BLOCCO TROVIAMO LE COINCIDENZE


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
 

 # Wrapper Python che converte List[int] in tuple
def find_coincidences(t, c, window_ps, target_n):
    raw = find_coincidences_numba(t, c, window_ps, target_n)
    result = []
    for grp in raw:
        result.append(tuple(grp))
    return result