#!/usr/bin/env python3
"""Create mock moduliquadri_mc.npz and visibilities_mc.npz for testing WORK_04"""
import numpy as np
import math


def idx_to_pair(n, iii):
    """Convert triangular index to pair (i, ii) with i < ii."""
    # From visibility_HOM_lib.py
    i = int((2*n - 1 - math.sqrt((2*n - 1)**2 - 8*iii)) // 2)
    
    # Verify and adjust i
    while iii < i*n - (i*(i+1))//2:
        i -= 1
    while iii >= (i+1)*n - ((i+1)*(i+2))//2:
        i += 1
    
    offset = i * n - (i * (i + 1)) // 2
    ii = (iii - offset) + (i + 1)
    
    return i, ii


# Create mock moduliquadri_mc.npz
# Per channel: singles_raw, dark_raw (128,), singles_file_count, dark_file_count
np.random.seed(42)

singles_raw_b = np.random.poisson(100, 128).astype(np.int64)
singles_raw_c = np.random.poisson(100, 128).astype(np.int64)
singles_raw_d = np.random.poisson(100, 128).astype(np.int64)
singles_raw_e = np.random.poisson(100, 128).astype(np.int64)

dark_raw_b = np.random.poisson(10, 128).astype(np.int64)
dark_raw_c = np.random.poisson(10, 128).astype(np.int64)
dark_raw_d = np.random.poisson(10, 128).astype(np.int64)
dark_raw_e = np.random.poisson(10, 128).astype(np.int64)

singles_file_count_b = 5
singles_file_count_c = 5
singles_file_count_d = 5
singles_file_count_e = 5

dark_file_count_b = 5
dark_file_count_c = 5
dark_file_count_d = 5
dark_file_count_e = 5

np.savez('moduliquadri_mc',
    singles_raw_b=singles_raw_b,
    singles_raw_c=singles_raw_c,
    singles_raw_d=singles_raw_d,
    singles_raw_e=singles_raw_e,
    dark_raw_b=dark_raw_b,
    dark_raw_c=dark_raw_c,
    dark_raw_d=dark_raw_d,
    dark_raw_e=dark_raw_e,
    singles_file_count_b=singles_file_count_b,
    singles_file_count_c=singles_file_count_c,
    singles_file_count_d=singles_file_count_d,
    singles_file_count_e=singles_file_count_e,
    dark_file_count_b=dark_file_count_b,
    dark_file_count_c=dark_file_count_c,
    dark_file_count_d=dark_file_count_d,
    dark_file_count_e=dark_file_count_e)

print("Created moduliquadri_mc.npz")

# Create mock visibilities_mc.npz
pairs = ['bc', 'bd', 'be', 'cd', 'ce', 'de']

visibilities_mc_data = {}
n_modes = 128
n_histograms = n_modes * (n_modes - 1) // 2  # 8128

for pair in pairs:
    # Generate all possible (ii, iii) pairs with ii < iii
    ii_list = []
    iii_list = []
    for idx in range(n_histograms):
        # Use the correct idx_to_pair function
        i_val, ii_val = idx_to_pair(n_modes, idx)
        ii_list.append(i_val)
        iii_list.append(ii_val)
    
    ii_list = np.array(ii_list, dtype=np.int64)
    iii_list = np.array(iii_list, dtype=np.int64)
    
    # Create mock data
    y_int = np.random.normal(1000, 10, n_histograms).astype(np.float64)
    sigma_y_int = np.random.uniform(5, 20, n_histograms).astype(np.float64)
    y_noise_int = np.random.normal(500, 5, n_histograms).astype(np.float64)
    sigma_y_noise_int = np.random.uniform(2, 10, n_histograms).astype(np.float64)
    area_c = np.random.poisson(200, n_histograms).astype(np.float64)
    
    # Valid flag: most are valid, some skipped
    valid = np.ones(n_histograms, dtype=np.int64)
    valid[:100] = 0  # First 100 histograms are skipped
    
    visibilities_mc_data[f'pair_{pair}_ii'] = ii_list
    visibilities_mc_data[f'pair_{pair}_iii'] = iii_list
    visibilities_mc_data[f'pair_{pair}_y_int'] = y_int
    visibilities_mc_data[f'pair_{pair}_sigma_y_int'] = sigma_y_int
    visibilities_mc_data[f'pair_{pair}_y_noise_int'] = y_noise_int
    visibilities_mc_data[f'pair_{pair}_sigma_y_noise_int'] = sigma_y_noise_int
    visibilities_mc_data[f'pair_{pair}_area_c'] = area_c
    visibilities_mc_data[f'pair_{pair}_valid'] = valid

np.savez('visibilities_mc', **visibilities_mc_data)
print("Created visibilities_mc.npz")

# Verify files
with np.load('moduliquadri_mc.npz') as f:
    print(f"moduliquadri_mc.npz keys: {list(f.keys())}")
    print(f"singles_raw_b shape: {f['singles_raw_b'].shape}")

with np.load('visibilities_mc.npz') as f:
    print(f"visibilities_mc.npz has {len(f.files)} keys")
    print(f"pair_bc_ii shape: {f['pair_bc_ii'].shape}")

print("Mock files created successfully!")
