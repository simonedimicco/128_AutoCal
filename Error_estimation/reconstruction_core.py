"""
Reconstruction core module for unitary reconstruction from histograms.

This module extracts the reconstruction logic from Ricostruzione_unitaria_histo_numba.py
and provides it as an importable, I/O-free function that can be reused for Monte Carlo
estimations.

Interface: reconstruct_U(t, VV, com_index, Vf=0.78, i1=79, i2=0) -> U (128x4 complex)

Where:
- t: sqrt(T) matrix, shape (128, 4)
- VV: 3D numpy array of shape (6, 128, 128), containing visibility matrices
       for channel pairs in order [bc, bd, be, cd, ce, de]
- com_index: 4x4 map where com_index[h,k] = index in VV for channel pair h<k (-1 elsewhere)
- Vf: visibility factor, default 0.78
- i1: first pivot row index, default 79
- i2: second pivot row index, default 0
"""

import numpy as np
from numba import njit


@njit
def gamma(g, h, j, k, VV_in, tt_in, Vf_local, com_idx):
    """
    Compute the ratio for arccos in phase reconstruction.
    
    Parameters:
        g, h, j, k: mode and channel indices
        VV_in: list of visibility matrices
        tt_in: sqrt(T) matrix
        Vf_local: visibility factor
        com_idx: 4x4 channel pair index map
    
    Returns:
        ratio value clipped to [-1, 1]
    """
    idx_v = com_idx[h, k]
    if idx_v == -1:
        idx_v = com_idx[k, h]
    if idx_v == -1:
        return 0.0
    
    val = (-VV_in[idx_v, j, g] * (tt_in[j, h]**2 + tt_in[j, k]**2) * (tt_in[g, h]**2 + tt_in[g, k]**2) 
           + tt_in[g, h] ** 2 * tt_in[j, h] ** 2 + tt_in[g, k] ** 2 * tt_in[j, k] ** 2)
    den = 2.0 * tt_in[g, h] * tt_in[j, k] * tt_in[j, h] * tt_in[g, k] * Vf_local
    ratio = val / den
    
    # Clip to [-1, 1]
    if ratio > 1.0:
        ratio = 1.0
    elif ratio < -1.0:
        ratio = -1.0
    
    return ratio


@njit
def compute_FF_magnitudes(FF_out, t_in, VV_in, com_idx, Vf_local, i1_local):
    """
    Compute the magnitudes of the phase matrix FF.
    
    Parameters:
        FF_out: output phase matrix to be filled, shape (128, 4)
        t_in: sqrt(T) matrix, shape (128, 4)
        VV_in: list of visibility matrices
        com_idx: 4x4 channel pair index map
        Vf_local: visibility factor
        i1_local: first pivot row index
    
    Returns:
        FF_out with magnitudes filled
    """
    rows = FF_out.shape[0]
    C_local = FF_out.shape[1]
    
    for g in range(rows):
        if g == i1_local:
            continue
        # Find first k with t[g,k] != 0
        k = 0
        found = False
        for kk in range(C_local):
            if t_in[g, kk] != 0:
                k = kk
                found = True
                break
        if not found:
            continue
        if t_in[g, k] == 0:
            continue
        for h in range(k + 1, C_local):
            if t_in[g, h] == 0:
                FF_out[g, h] = 0.0
            else:
                # visibility index
                idx_v = com_idx[k, h]
                if idx_v == -1:
                    FF_out[g, h] = 0.0
                else:
                    ratio = gamma(g, h, i1_local, k, VV_in, t_in, Vf_local, com_idx)
                    FF_out[g, h] = np.arccos(ratio)
    
    return FF_out


@njit
def compute_FF_signs_row_i2(FF_out, t_in, VV_in, com_idx, Vf_local, i1_local, i2_local):
    """
    Compute the signs for the i2 pivot row of the phase matrix FF.
    
    Parameters:
        FF_out: phase matrix with magnitudes, will be modified in-place
        t_in: sqrt(T) matrix
        VV_in: list of visibility matrices
        com_idx: 4x4 channel pair index map
        Vf_local: visibility factor
        i1_local: first pivot row index
        i2_local: second pivot row index
    
    Returns:
        FF_out with signs applied to i2 row
    """
    C_local = FF_out.shape[1]
    k = 1
    
    for h in range(k + 1, C_local):
        idx_v = com_idx[k, h]
        if idx_v == -1:
            continue
        ratio = gamma(i2_local, h, i1_local, k, VV_in, t_in, Vf_local, com_idx)
        b = np.arccos(ratio)
        # sign determination
        term1 = abs(b - abs(FF_out[i1_local, k] - FF_out[i1_local, h] - FF_out[i2_local, k] - FF_out[i2_local, h]))
        term2 = abs(b - abs(FF_out[i1_local, k] - FF_out[i1_local, h] - FF_out[i2_local, k] + FF_out[i2_local, h]))
        s = np.sign(term1 - term2)
        FF_out[i2_local, h] = FF_out[i2_local, h] * s
    
    return FF_out


@njit
def compute_FF_signs_other_rows(FF_out, t_in, VV_in, com_idx, Vf_local, i1_local, i2_local):
    """
    Compute the signs for all rows except the pivot rows i1 and i2.
    
    Parameters:
        FF_out: phase matrix with magnitudes, will be modified in-place
        t_in: sqrt(T) matrix
        VV_in: list of visibility matrices
        com_idx: 4x4 channel pair index map
        Vf_local: visibility factor
        i1_local: first pivot row index
        i2_local: second pivot row index
    
    Returns:
        FF_out with signs applied to all non-pivot rows
    """
    rows = FF_out.shape[0]
    C_local = FF_out.shape[1]
    
    for g in range(rows):
        if g == i1_local or g == i2_local:
            continue
        # Find first k with t[g,k] != 0
        k = 0
        found = False
        for kk in range(C_local):
            if t_in[g, kk] != 0:
                k = kk
                found = True
                break
        if not found:
            continue
        if t_in[g, k] == 0:
            continue
        for h in range(k + 1, C_local):
            if t_in[g, h] == 0:
                FF_out[g, h] = 0.0
            else:
                idx_v = com_idx[k, h]
                if idx_v == -1:
                    FF_out[g, h] = 0.0
                else:
                    ratio = gamma(g, h, i2_local, k, VV_in, t_in, Vf_local, com_idx)
                    b = np.arccos(ratio)
                    term1 = abs(b - abs(FF_out[i2_local, k] - FF_out[i2_local, h] - FF_out[g, k] - FF_out[g, h]))
                    term2 = abs(b - abs(FF_out[i2_local, k] - FF_out[i2_local, h] - FF_out[g, k] + FF_out[g, h]))
                    s = np.sign(term1 - term2)
                    FF_out[g, h] = FF_out[g, h] * s
    
    return FF_out


def reconstruct_U(t, VV, com_index, Vf=0.78, i1=79, i2=0):
    """
    Reconstruct the unitary matrix U from transmission and visibility data.
    
    This is the main reconstruction function that orchestrates the computation
    of the phase matrix FF and combines it with the transmission matrix to form U.
    
    Parameters:
        t: sqrt(T) matrix, shape (128, 4) - the square root of the transmission matrix
        VV: 3D numpy array of shape (6, 128, 128) for channel pairs [bc, bd, be, cd, ce, de]
        com_index: 4x4 numpy array where com_index[h,k] = index in VV for channel pair h<k (-1 elsewhere)
        Vf: visibility factor, default 0.78
        i1: first pivot row index, default 79
        i2: second pivot row index, default 0
    
    Returns:
        U: complex unitary matrix, shape (128, 4)
    
    Note:
        All functions called internally remain @njit with the same signatures as
        in the original Ricostruzione_unitaria_histo_numba.py script.
        No I/O is performed; all inputs are explicit arguments.
    """
    # Initialize FF phase matrix
    rows = t.shape[0]
    cols = t.shape[1]
    FF = np.zeros((rows, cols))
    
    # Compute phase matrix FF
    FF = compute_FF_magnitudes(FF, t, VV, com_index, Vf, i1)
    FF = compute_FF_signs_row_i2(FF, t, VV, com_index, Vf, i1, i2)
    FF = compute_FF_signs_other_rows(FF, t, VV, com_index, Vf, i1, i2)
    
    # Compute unitary matrix U = t * exp(i * FF)
    U = t * np.exp(1j * FF)
    
    return U
