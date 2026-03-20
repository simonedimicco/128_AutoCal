from numba import njit
import numpy as np

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

        if trigger_times[i] <= tj:
            trig_idx[j] = i

    return trig_idx

from numba import prange

@njit(parallel=True, cache=True)
def Modes_separator_parallel_core(channels_times,
                                  channels_list,
                                  trig_idx,
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

        if trig_idx[j] == -1:
            continue

        ch = channels_list[j]
        if ch == 63:
            continue

        tj = channels_times[j]
        trig_time = trigger_times[trig_idx[j]]
        relative_time = tj - trig_time

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
