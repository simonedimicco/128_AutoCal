from numba import njit
import numpy as np

@njit(cache=True)
def Modes_separator_full(channels_times,
                         channels_list,
                         trigger_channel=63,
                         mio_dizionario=typed_dict_tot,
                         Time_differences=Time_differences_tot,
                         Time_differences_fine_tuning=Time_differences_fine_tuning_tot,
                         retards=retards_box_tot,
                         window=730000,
                         offset=0):

    trig_mask = channels_list == trigger_channel
    num_triggers = np.sum(trig_mask)
    n_out = channels_list.size - num_triggers

    t_out = np.empty(n_out, dtype=np.int64)
    c_out = np.empty(n_out, dtype=np.int64)

    if num_triggers == 0:
        return t_out[:0], c_out[:0]

    trigger_times = channels_times[trig_mask]

    i = 0
    k = 0

    for j in range(channels_list.size):
        ch = channels_list[j]
        if ch == trigger_channel:
            continue

        c1, w1, c2, w2 = mio_dizionario[ch]

        while i < trigger_times.size - 1 and trigger_times[i] + window < channels_times[j]:
            i += 1

        relative_time = channels_times[j] - trigger_times[i]

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
            if ch <= 15:
                base = 2 * ch
            else:
                base = 2 * (ch + 16)
        else:
            if ch <= 47:
                base = 2 * (ch - 16)
            else:
                base = 2 * ch

        c_final = base if is_bin1 else base + 1

        # ---- tempo finale ----
        if is_bin1:
            if ch <= 31:
                t_final = channels_times[j] - retards[ch] - offset
            else:
                t_final = channels_times[j] - retards[ch]
        else:
            if ch <= 31:
                t_final = (
                    channels_times[j]
                    + (725000 - Time_differences[ch])
                    - retards[ch]
                    - Time_differences_fine_tuning[ch]
                    - offset
                )
            else:
                t_final = (
                    channels_times[j]
                    + (725000 - Time_differences[ch])
                    - retards[ch]
                    - Time_differences_fine_tuning[ch]
                )

        t_out[k] = t_final
        c_out[k] = c_final
        k += 1

    return t_out[:k], c_out[:k]
