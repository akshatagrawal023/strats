import os
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Define the Log-Moneyness Grid (k = log(K/S))
# e.g., -5%, -2%, 0% (ATM), +2%, +5%
MONEYNESS_GRID = np.array([-0.05, -0.02, 0.0, 0.02, 0.05])

def get_moneyness_normalized_features(greeks_mat: np.ndarray, moneyness_arr: np.ndarray) -> np.ndarray:
    """
    Interpolates a single tick's greeks_mat onto solving for fixed Log-Moneyness buckets.
    
    Args:
        greeks_mat: shape (14, n_strikes) -> 12 channels + 2 
        moneyness_arr: shape (n_strikes,) which is channel 8 (S/K).
            We use log(K/S) = -log(S/K) to measure moneyness.
            
    Returns:
        flat_features: 1D array of interpolated features.
    """
    # k = log(K/S) = -log(S/K)
    k = -np.log(moneyness_arr)
    
    # We only care about strikes that have valid data
    valid = ~np.isnan(k) & ~np.isnan(greeks_mat[6]) # Ensure CE IV is valid
    if np.sum(valid) < 3:
        # Not enough data to interpolate safely
        return np.full(greeks_mat.shape[0] * len(MONEYNESS_GRID), np.nan)
        
    k_valid = k[valid]
    
    # Sort by k to ensure interp1d works correctly
    sort_idx = np.argsort(k_valid)
    k_sorted = k_valid[sort_idx]
    
    flat_features = []
    
    for ch_idx in range(greeks_mat.shape[0]):
        ch_data = greeks_mat[ch_idx, valid][sort_idx]
        
        # Linear interpolation, filling edge cases with the nearest value
        interpolator = interp1d(
            k_sorted, 
            ch_data, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(ch_data[0], ch_data[-1])
        )
        
        # Interp onto our fixed grid
        grid_vals = interpolator(MONEYNESS_GRID)
        flat_features.extend(grid_vals)
        
    return np.array(flat_features)


def load_and_prep_h5(filepath: str, lookahead_ticks: int = 600, target_profit_bps: float = 10.0, stop_loss_bps: float = 10.0) -> tuple:
    """
    Reads the HDF5 file, normalizes the 3D options chain onto a 2D tabular dataset,
    and calculates classification targets (Y) based on forward returns.
    
    Args:
        filepath: path to .h5 file
        lookahead_ticks: Number of future ticks to evaluate (600 ticks @ 3s/tick = 30 mins)
        target_profit_bps: Basis points of target profit (10.0 bps = 0.1%)
        stop_loss_bps: Basis points of stop loss (10.0 bps = 0.1%)
        
    Returns:
        (df_features, df_targets)
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return pd.DataFrame(), pd.Series()
        
    print(f"Loading data from {filepath}...")
    with h5py.File(filepath, 'r') as f:
        ticks = f['ticks'][:]
        df = pd.DataFrame(ticks)
        
        greeks_mat = f['greeks_matrices'][:]
    
    print(f"Loaded {len(df)} ticks. Processing Moneyness Grids...")
    
    # 1. Process 3D Matrix into Flat Moneyness Grid
    interpolated_rows = []
    for i in range(len(df)):
        mat = greeks_mat[i]
        moneyness_arr = mat[8]
        flat_feats = get_moneyness_normalized_features(mat, moneyness_arr)
        interpolated_rows.append(flat_feats)
        
    grid_df = pd.DataFrame(interpolated_rows)
    
    # Rename columns for clarity: e.g. ch0_grid0, ch0_grid1...
    col_names = []
    num_channels = greeks_mat.shape[1]
    for ch in range(num_channels):
        for g_idx, g_val in enumerate(MONEYNESS_GRID):
            col_names.append(f"ch{ch}_mny{g_val:.2f}")
    grid_df.columns = col_names
    
    # 2. Combine with Scalar features (Relative only - No absolute Spot or ATM_IV)
    scalar_features = ['skew', 'iv_z_score', 'spot_ewm_vol']
    
    # Check for 'momentum' if it exists in the logs
    if 'momentum' in df.columns:
        scalar_features.append('momentum')
    
    # Check if 'recent_drop' exists (it's a percentage drop from mean)
    if 'recent_drop' in df.columns:
        scalar_features.append('recent_drop')

    available_scalars = [col for col in scalar_features if col in df.columns]
    
    X = pd.concat([df[available_scalars], grid_df], axis=1)
    
    print("Computing Lookahead Targets (Y)...")
    
    # 3. Compute the path-dependent target variable
    # A successful Bullish Entry needs spot to rise by target_profit_bps 
    # BEFORE it drops by stop_loss_bps within the lookahead window.
    
    spot = df['spot'].values
    y_bullish = np.zeros(len(spot))
    
    # Target and stop absolute multipliers
    tp_mult = 1.0 + (target_profit_bps / 10000.0)
    sl_mult = 1.0 - (stop_loss_bps / 10000.0)
    
    for i in range(len(spot)):
        if i + lookahead_ticks >= len(spot):
            continue # End of day, indeterminate
            
        entry_price = spot[i]
        path = spot[i+1 : i+1+lookahead_ticks]
        
        bullish_success = 0
        for price in path:
            if price <= entry_price * sl_mult:
                break # Hit stop loss first
            if price >= entry_price * tp_mult:
                bullish_success = 1 # Hit take profit first!
                break
                
        y_bullish[i] = bullish_success

    # Cleanup any rows with NaNs in X
    valid_rows = ~X.isna().any(axis=1)
    
    X_clean = X[valid_rows]
    y_clean = y_bullish[valid_rows]
    
    # Make sure we don't include the absolute end of day where Y is undetermined
    total_valid = len(y_clean) - lookahead_ticks
    if total_valid <= 0:
        return pd.DataFrame(), pd.Series()
        
    X_clean = X_clean.iloc[:total_valid]
    y_clean = y_clean[:total_valid]
    
    print(f"Data Prep Complete. Final Shape: X={X_clean.shape}, Y={len(y_clean)} (Bullish %: {np.mean(y_clean):.2%})")
    
    return X_clean, pd.Series(y_clean, index=X_clean.index)
