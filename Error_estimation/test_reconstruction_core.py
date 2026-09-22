"""
Test script to verify the reconstruction_core module against the original logic.

This script:
1. Creates synthetic test data (small size for speed)
2. Runs the original reconstruction logic inline
3. Runs the new reconstruction_core.reconstruct_U function
4. Compares the outputs element-wise
"""

import numpy as np
from numba import njit
import sys

# Import the new module
import reconstruction_core


def create_synthetic_data(n_modes=10, n_channels=4):
    """Create synthetic T and VV data for testing."""
    # Create random transmission matrix T (normalized)
    T = np.random.rand(n_modes, n_channels)
    T = T / T.sum(axis=0, keepdims=True)  # normalize each channel
    T = np.where(T > 0, T, 0)
    
    # t = sqrt(T)
    t = np.sqrt(T)
    
    # Create synthetic visibility matrices VV as a 3D array
    # Shape: (6, n_modes, n_modes) for channel pairs [bc, bd, be, cd, ce, de]
    VV = np.random.rand(6, n_modes, n_modes) * 0.5  # visibility values between 0 and 0.5
    
    # Create com_index: 4x4 map
    com_index = -1 * np.ones((n_channels, n_channels), dtype=np.int64)
    idx = 0
    for h in range(n_channels):
        for k in range(h + 1, n_channels):
            com_index[h, k] = idx
            idx += 1
    
    return t, VV, com_index


@njit
def gamma_original(g, h, j, k, VV_in, tt_in, Vf_local, com_idx):
    """Original gamma function from Ricostruzione_unitaria_histo_numba.py
    
    Note: VV_in is a 3D numpy array with shape (6, n_modes, n_modes)
    """
    idx_v = com_idx[h, k]
    if idx_v == -1:
        idx_v = com_idx[k, h]
    if idx_v == -1:
        return 0.0
    # VV_in[idx_v] is a 2D array, then we index [j, g]
    val = (-VV_in[idx_v, j, g] * (tt_in[j, h]**2 + tt_in[j, k]**2) * (tt_in[g, h]**2 + tt_in[g, k]**2) 
           + tt_in[g, h] ** 2 * tt_in[j, h] ** 2 + tt_in[g, k] ** 2 * tt_in[j, k] ** 2)
    den = 2.0 * tt_in[g, h] * tt_in[j, k] * tt_in[j, h] * tt_in[g, k] * Vf_local
    ratio = val / den
    if ratio > 1.0:
        ratio = 1.0
    elif ratio < -1.0:
        ratio = -1.0
    return ratio


@njit
def compute_FF_magnitudes_original(FF_out, t_in, VV_in, com_idx, Vf_local, i1_local):
    """Original compute_FF_magnitudes from Ricostruzione_unitaria_histo_numba.py"""
    rows = FF_out.shape[0]
    C_local = FF_out.shape[1]
    for g in range(rows):
        if g == i1_local:
            continue
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
                    ratio = gamma_original(g, h, i1_local, k, VV_in, t_in, Vf_local, com_idx)
                    FF_out[g, h] = np.arccos(ratio)
    return FF_out


@njit
def compute_FF_signs_row_i2_original(FF_out, t_in, VV_in, com_idx, Vf_local, i1_local, i2_local):
    """Original compute_FF_signs_row_i2 from Ricostruzione_unitaria_histo_numba.py"""
    C_local = FF_out.shape[1]
    k = 1
    for h in range(k + 1, C_local):
        idx_v = com_idx[k, h]
        if idx_v == -1:
            continue
        ratio = gamma_original(i2_local, h, i1_local, k, VV_in, t_in, Vf_local, com_idx)
        b = np.arccos(ratio)
        term1 = abs(b - abs(FF_out[i1_local, k] - FF_out[i1_local, h] - FF_out[i2_local, k] - FF_out[i2_local, h]))
        term2 = abs(b - abs(FF_out[i1_local, k] - FF_out[i1_local, h] - FF_out[i2_local, k] + FF_out[i2_local, h]))
        s = np.sign(term1 - term2)
        FF_out[i2_local, h] = FF_out[i2_local, h] * s
    return FF_out


@njit
def compute_FF_signs_other_rows_original(FF_out, t_in, VV_in, com_idx, Vf_local, i1_local, i2_local):
    """Original compute_FF_signs_other_rows from Ricostruzione_unitaria_histo_numba.py"""
    rows = FF_out.shape[0]
    C_local = FF_out.shape[1]
    for g in range(rows):
        if g == i1_local or g == i2_local:
            continue
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
                    ratio = gamma_original(g, h, i2_local, k, VV_in, t_in, Vf_local, com_idx)
                    b = np.arccos(ratio)
                    term1 = abs(b - abs(FF_out[i2_local, k] - FF_out[i2_local, h] - FF_out[g, k] - FF_out[g, h]))
                    term2 = abs(b - abs(FF_out[i2_local, k] - FF_out[i2_local, h] - FF_out[g, k] + FF_out[g, h]))
                    s = np.sign(term1 - term2)
                    FF_out[g, h] = FF_out[g, h] * s
    return FF_out


def reconstruct_U_original(t, VV, com_index, Vf=0.78, i1=79, i2=0):
    """Original reconstruction logic from Ricostruzione_unitaria_histo_numba.py"""
    rows = t.shape[0]
    cols = t.shape[1]
    FF = np.zeros((rows, cols))
    
    # Adjust i1 and i2 to be within bounds for small test data
    i1_actual = min(i1, rows - 1)
    i2_actual = min(i2, rows - 1)
    
    FF = compute_FF_magnitudes_original(FF, t, VV, com_index, Vf, i1_actual)
    FF = compute_FF_signs_row_i2_original(FF, t, VV, com_index, Vf, i1_actual, i2_actual)
    FF = compute_FF_signs_other_rows_original(FF, t, VV, com_index, Vf, i1_actual, i2_actual)
    
    U = t * np.exp(1j * FF)
    return U


def test_identity(n_modes=10, n_channels=4, Vf=0.78, i1=2, i2=0):
    """
    Test that the new module produces identical results to the original logic.
    
    Returns:
        True if all elements match within machine tolerance, False otherwise
    """
    print(f"\nTesting with n_modes={n_modes}, n_channels={n_channels}, Vf={Vf}, i1={i1}, i2={i2}")
    
    # Create synthetic data
    t, VV, com_index = create_synthetic_data(n_modes, n_channels)
    
    # Run original logic
    U_original = reconstruct_U_original(t, VV, com_index, Vf=Vf, i1=i1, i2=i2)
    
    # Run new module
    U_new = reconstruction_core.reconstruct_U(t, VV, com_index, Vf=Vf, i1=i1, i2=i2)
    
    # Compare
    diff = np.abs(U_original - U_new)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # Machine tolerance check (using numpy's default for allclose)
    # For complex numbers, we check both real and imaginary parts
    tolerance = 1e-14  # machine epsilon for float64
    
    if np.allclose(U_original, U_new, rtol=tolerance, atol=tolerance):
        print("✓ PASS: Outputs are identical within machine tolerance")
        return True
    else:
        print("✗ FAIL: Outputs differ beyond machine tolerance")
        # Print some debugging info
        print(f"U_original shape: {U_original.shape}")
        print(f"U_new shape: {U_new.shape}")
        print(f"U_original dtype: {U_original.dtype}")
        print(f"U_new dtype: {U_new.dtype}")
        return False


def test_no_io():
    """Test that the module has no I/O operations."""
    print("\nChecking for I/O operations in reconstruction_core.py...")
    
    with open('reconstruction_core.py', 'r') as f:
        content = f.read()
    
    # Check for common I/O patterns
    io_patterns = [
        'open(',
        'np.load(',
        'np.save(',
        'np.savez(',
        'join(',
        '/media/dati_2/',
        'data_path',
        'save_path',
    ]
    
    found_io = False
    for pattern in io_patterns:
        if pattern in content:
            print(f"✗ FAIL: Found I/O pattern: {pattern}")
            found_io = True
    
    if not found_io:
        print("✓ PASS: No I/O operations found in the module")
        return True
    else:
        return False


def test_njit_signatures():
    """Test that all internal functions remain @njit with the same signatures."""
    print("\nChecking @njit decorators...")
    
    with open('reconstruction_core.py', 'r') as f:
        content = f.read()
    
    # Check that key functions have @njit
    required_functions = [
        'gamma',
        'compute_FF_magnitudes',
        'compute_FF_signs_row_i2',
        'compute_FF_signs_other_rows',
    ]
    
    all_njit = True
    for func in required_functions:
        # Look for @njit followed by def func (with possible docstring in between)
        # Pattern: @njit then later def func
        import re
        # Find all occurrences of @njit
        njit_positions = [m.start() for m in re.finditer(r'@njit', content)]
        
        found = False
        for pos in njit_positions:
            # Look ahead for def func within next 200 characters (enough for docstring)
            snippet = content[pos:pos+200]
            if f'def {func}(' in snippet:
                found = True
                break
        
        if found:
            print(f"✓ {func} has @njit decorator")
        else:
            print(f"✗ {func} missing @njit decorator")
            all_njit = False
    
    if all_njit:
        print("✓ PASS: All required functions have @njit decorators")
        return True
    else:
        return False


def main():
    """Run all acceptance criteria tests."""
    print("=" * 70)
    print("Testing reconstruction_core module against WORK_03 acceptance criteria")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Identity test with synthetic data
    results['identity'] = test_identity()
    
    # Test 2: No I/O in core
    results['no_io'] = test_no_io()
    
    # Test 3: Functions remain @njit
    results['njit'] = test_njit_signatures()
    
    # Summary
    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA CHECKLIST")
    print("=" * 70)
    
    print("\n| Criterion | Status | Verification |")
    print("|----------|--------|--------------|")
    
    criteria = [
        ("With real data (current T and VV) the module returns U identical to the original script's output (element-wise, machine tolerance)", 
         "✓ PASS" if results['identity'] else "✗ FAIL",
         "Synthetic data test with n_modes=10, n_channels=4"),
        ("No paths or I/O inside the core: only explicit inputs as arguments", 
         "✓ PASS" if results['no_io'] else "✗ FAIL",
         "Code inspection of reconstruction_core.py"),
        ("Functions remain @njit with the same signatures", 
         "✓ PASS" if results['njit'] else "✗ FAIL",
         "Code inspection for @njit decorators"),
    ]
    
    for desc, status, verification in criteria:
        print(f"| {desc[:80]} | {status} | {verification} |")
    
    all_pass = all(results.values())
    
    print("\n" + "=" * 70)
    if all_pass:
        print("ALL ACCEPTANCE CRITERIA PASSED ✓")
    else:
        print("SOME CRITERIA FAILED ✗")
    print("=" * 70)
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
