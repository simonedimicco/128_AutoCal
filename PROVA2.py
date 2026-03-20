from numba import njit, prange
import numpy as np

@njit(parallel=True, cache=True)
def Modes_separator_parallel_pass1(channels_times,
                                   channels_list,
                                   trigger_times,
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
        if ch == 63:
            continue

        # ---- trova trigger (binary search manuale) ----
        tj = channels_times[j]
        i = np.searchsorted(trigger_times, tj - window, side="right") - 1
        if i < 0:
            return

        relative_time = tj - trigger_times[i]
        c1, w1, c2, w2 = mio_dizionario[ch]

        is_bin1 = (
            abs(relative_time - c1) <= w1 or
            abs(relative_time - (c1 + window)) <= w1
        )
        is_bin2 = (
            abs(relative_time - c2) <= w2 or
            abs(relative_time - (c2 + window)) <= w2
        )

        if not (is_bin1 or is_bin2):
            return

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
                tj + (725000 - Time_differences[ch])
                - retards[ch]
                - Time_differences_fine_tuning[ch]
                - (offset if ch <= 31 else 0)
            )

        t_tmp[j] = t_final
        c_tmp[j] = c_final
        valid[j] = 1

    return t_tmp, c_tmp, valid


@njit(cache=True)
def merge_time_channel_arrays(times1, channels1, times2, channels2, ch_sync_1, ch_sync_2, channel_offset=32):
    n1, n2 = len(times1), len(times2)
    merged_times = np.empty(n1 + n2, dtype=np.int64)
    merged_channels = np.empty(n1 + n2, dtype=np.int64)

    i = 0
    j = 0
    k = 0

    while i < n1 or j < n2:
        # Trova il prossimo evento valido da times1
        while i < n1 and channels1[i] == ch_sync_1:
            i += 1
        # Trova il prossimo evento valido da times2
        while j < n2 and channels2[j] == ch_sync_2:
            j += 1

        # Se uno dei due è finito, copia l’altro
        if i >= n1:
            while j < n2:
                if channels2[j] != ch_sync_2:
                    merged_times[k] = times2[j]
                    merged_channels[k] = channels2[j] + channel_offset
                    k += 1
                j += 1
            break
        if j >= n2:
            while i < n1:
                if channels1[i] != ch_sync_1:
                    merged_times[k] = times1[i]
                    merged_channels[k] = channels1[i]
                    k += 1
                i += 1
            break

        # Merge lineare
        if times1[i] <= times2[j]:
            merged_times[k] = times1[i]
            merged_channels[k] = channels1[i]
            i += 1
        else:
            merged_times[k] = times2[j]
            merged_channels[k] = channels2[j] + channel_offset
            j += 1
        k += 1

    return merged_times[:k], merged_channels[:k]