"""
Monte Carlo Resampling Module

This module implements the statistical layer for the Monte Carlo error-estimation pipeline.
Given the .npz files from the data-preparation phase and a seed, it produces one fake 
realization of the inputs {T, VV} ready to be fed to the reconstruction core.

Contract:
- Input: moduliquadri_mc.npz, visibilities_mc.npz, seed
- Output: dict with T (128x4) and VV (list of 6 matrices 128x128 in pair order)

Sampling conventions (binding):
- B, M_raw, area_c: Poisson(λ=observed_value)
- y_int, y_noise_int: Normal(μ=observed_value, σ=observed_sigma)
- All quantities in counts·ps scale, no bin-width rescaling
"""

import numpy as np


def load_mc_data(moduliquadri_path, visibilities_path):
    """
    Load the Monte Carlo data from .npz files.
    
    Args:
        moduliquadri_path: Path to moduliquadri_mc.npz
        visibilities_path: Path to visibilities_mc.npz
        
    Returns:
        Tuple of (singles_data, dark_data, pairs_data)
    """
    # Load moduliquadri_mc.npz
    with np.load(moduliquadri_path) as f:
        singles_data = {}
        dark_data = {}
        for c in ['b', 'c', 'd', 'e']:
            singles_data[c] = {
                'raw': f[f'singles_raw_{c}'],
                'file_count': f[f'singles_file_count_{c}']
            }
            dark_data[c] = {
                'raw': f[f'dark_raw_{c}'],
                'file_count': f[f'dark_file_count_{c}']
            }
    
    # Load visibilities_mc.npz
    with np.load(visibilities_path) as f:
        pairs_data = {}
        pair_names = ['bc', 'bd', 'be', 'cd', 'ce', 'de']
        for pair in pair_names:
            pairs_data[pair] = {
                'ii': f[f'pair_{pair}_ii'],
                'iii': f[f'pair_{pair}_iii'],
                'y_int': f[f'pair_{pair}_y_int'],
                'sigma_y_int': f[f'pair_{pair}_sigma_y_int'],
                'y_noise_int': f[f'pair_{pair}_y_noise_int'],
                'sigma_y_noise_int': f[f'pair_{pair}_sigma_y_noise_int'],
                'area_c': f[f'pair_{pair}_area_c'],
                'valid': f[f'pair_{pair}_valid']
            }
    
    return singles_data, dark_data, pairs_data


def sample_singles_pipeline(singles_data, dark_data, rng):
    """
    Rebuild the singles pipeline with Monte Carlo sampling.
    
    For each channel c:
    1. Poisson-sample the accumulated raw counts (singles and dark separately)
    2. Divide by file count
    3. Subtract dark from singles
    4. Clip negatives to 0
    5. Normalize to sum 1
    6. Stack per channel and transpose -> T (128×4)
    
    Args:
        singles_data: Dict with 'raw' and 'file_count' for each channel
        dark_data: Dict with 'raw' and 'file_count' for each channel
        rng: numpy.random.Generator instance
        
    Returns:
        T: 128×4 matrix
    """
    channels = ['b', 'c', 'd', 'e']
    M_list = []
    
    for c in channels:
        # Poisson sampling of raw counts
        M_raw_sampled = rng.poisson(singles_data[c]['raw'])
        B_sampled = rng.poisson(dark_data[c]['raw'])
        
        # Divide by file count
        M_raw_normalized = M_raw_sampled / singles_data[c]['file_count']
        B_normalized = B_sampled / dark_data[c]['file_count']
        
        # Subtract dark from singles, clip negatives to 0
        M = np.clip(M_raw_normalized - B_normalized, 0, None)
        
        # Normalize to sum 1
        M_sum = np.sum(M)
        if M_sum > 0:
            M = M / M_sum
        else:
            # If sum is 0, create uniform distribution
            M = np.ones(128) / 128
        
        M_list.append(M)
    
    # Stack per channel and transpose to get T (128×4)
    T = np.column_stack(M_list)
    
    return T


def sample_pairs_pipeline(pairs_data, rng):
    """
    Rebuild the pairs pipeline with Monte Carlo sampling.
    
    For each pair and each histogram index:
    - y_int ~ N(y_int, sigma_y_int)
    - y_noise_int ~ N(y_noise_int, sigma_y_noise_int)
    - area_c ~ Poisson(area_c)
    - V = (y_int - area_c) / (y_int - y_noise_int)
    - Respect valid flag (skipped histograms stay 0)
    - Symmetric assignment: V[ii,iii] = V[iii,ii]
    - Diagonal = 0
    
    Args:
        pairs_data: Dict with data for each pair
        rng: numpy.random.Generator instance
        
    Returns:
        VV: List of 6 matrices (128×128) in pair order
    """
    pair_names = ['bc', 'bd', 'be', 'cd', 'ce', 'de']
    n_modes = 128
    
    # Initialize V matrices for each pair
    VV = [np.zeros((n_modes, n_modes)) for _ in range(6)]
    
    for pair_idx, pair in enumerate(pair_names):
        data = pairs_data[pair]
        n_histograms = len(data['ii'])
        
        # Sample all quantities
        y_int_sampled = rng.normal(data['y_int'], data['sigma_y_int'])
        y_noise_int_sampled = rng.normal(data['y_noise_int'], data['sigma_y_noise_int'])
        area_c_sampled = rng.poisson(data['area_c'])
        valid = data['valid']
        
        ii_arr = data['ii']
        iii_arr = data['iii']
        
        # Compute V for each valid histogram
        for h in range(n_histograms):
            if valid[h] == 0:
                # Skipped histogram, V stays 0 (both directions)
                continue
            
            ii = int(ii_arr[h])
            iii = int(iii_arr[h])
            
            # Avoid division by zero
            denominator = y_int_sampled[h] - y_noise_int_sampled[h]
            
            if abs(denominator) < 1e-10:
                # If denominator is too small, V is undefined -> set to 0
                V_val = 0.0
            else:
                V_val = (y_int_sampled[h] - area_c_sampled[h]) / denominator
            
            # Symmetric assignment
            VV[pair_idx][ii, iii] = V_val
            VV[pair_idx][iii, ii] = V_val
        
        # Ensure diagonal is 0
        np.fill_diagonal(VV[pair_idx], 0)
    
    return VV


def sample_realization(moduliquadri_path, visibilities_path, seed):
    """
    Produce one fake realization of the inputs {T, VV}.
    
    This is the main function that implements the statistical layer.
    Given the .npz files and a seed, it returns one realization of {T, VV}
    with the same shapes and semantics as the real ones.
    
    Args:
        moduliquadri_path: Path to moduliquadri_mc.npz
        visibilities_path: Path to visibilities_mc.npz
        seed: Random seed for reproducibility
        
    Returns:
        dict with keys:
            - 'T': 128×4 matrix (squared-modulus matrix)
            - 'VV': list of 6 matrices (128×128) in pair order [bc, bd, be, cd, ce, de]
    """
    # Create random generator with seed
    rng = np.random.default_rng(seed)
    
    # Load data
    singles_data, dark_data, pairs_data = load_mc_data(moduliquadri_path, visibilities_path)
    
    # Sample singles pipeline
    T = sample_singles_pipeline(singles_data, dark_data, rng)
    
    # Sample pairs pipeline
    VV = sample_pairs_pipeline(pairs_data, rng)
    
    return {
        'T': T,
        'VV': VV
    }


# For direct script execution (testing)
if __name__ == '__main__':
    # Test with mock data
    result = sample_realization('moduliquadri_mc.npz', 'visibilities_mc.npz', seed=42)
    
    print(f"T shape: {result['T'].shape}")
    print(f"Number of VV matrices: {len(result['VV'])}")
    print(f"VV[0] shape: {result['VV'][0].shape}")
    print(f"T sum per column: {np.sum(result['T'], axis=0)}")
    print("Test passed!")
