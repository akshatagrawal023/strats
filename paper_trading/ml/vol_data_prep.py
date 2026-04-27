"""
vol_data_prep.py
----------------
Data preparation for Volatility Arbitrage / Iron Condor ML.

Philosophy: We NEVER predict where the underlying goes.
We predict whether the implied volatility is OVERPRICED relative to
realized volatility — i.e., whether there is a harvestable "Variance Risk Premium."

Dependent Variable (Y):
    Y = 1 if IV_sold - Realized_Vol_over_next_N_ticks > cost_threshold
    i.e., the premium we collected more than covers the actual gamma risk
    that materialized during the holding period.

Independent Variables (X):
    - IV vs RV Spread (Variance Risk Premium proxy)
    - Surface Tension: Vanna, Volga at ATM and wings (moneyness-normalized)
    - Skew Percentile: Current skew rank vs rolling history
    - Theta/Vega Ratio: Premium decay efficiency
    - IV Velocity: Rate of change of implied vol (mean-reversion signal)
    - IV Z-Score: How stretched current vol is vs recent history
"""

import os
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Fixed moneyness grid for wing analysis
# We care about ATM (0.0) and the wings where we sell/buy
MONEYNESS_GRID = np.array([-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04])

# Channel map from market_features.py
CH_DELTA = 0
CH_GAMMA = 1
CH_THETA = 2
CH_VEGA = 3
CH_VANNA = 4
CH_VOLGA = 5
CH_CE_IV = 6
CH_PE_IV = 7
CH_MONEYNESS = 8
CH_THETA_VEGA = 9
CH_SKEW = 10
CH_CHARM = 11
CH_CE_IV_VEL = 12
CH_PE_IV_VEL = 13


def interpolate_channel(greeks_mat, channel_idx, k_sorted, sort_idx, valid_mask):
    """Interpolate a single channel onto the fixed moneyness grid."""
    ch_data = greeks_mat[channel_idx, valid_mask][sort_idx]
    try:
        interpolator = interp1d(
            k_sorted, ch_data,
            kind='linear', bounds_error=False,
            fill_value=(ch_data[0], ch_data[-1])
        )
        return interpolator(MONEYNESS_GRID)
    except Exception:
        return np.full(len(MONEYNESS_GRID), np.nan)


def extract_vol_arb_features(greeks_mat: np.ndarray) -> dict:
    """
    Extract volatility arbitrage features from a single tick's greeks matrix.
    
    Returns a flat dict of features describing the current vol surface state.
    """
    moneyness = greeks_mat[CH_MONEYNESS]
    k = -np.log(np.clip(moneyness, 1e-6, None))  # log(K/S)
    
    valid = ~np.isnan(k) & ~np.isnan(greeks_mat[CH_CE_IV]) & ~np.isnan(greeks_mat[CH_PE_IV])
    if np.sum(valid) < 3:
        return None
    
    k_valid = k[valid]
    sort_idx = np.argsort(k_valid)
    k_sorted = k_valid[sort_idx]
    
    features = {}
    
    # --- 1. Moneyness-Normalized Surface Channels ---
    # We only interpolate the channels critical for vol arb
    vol_arb_channels = {
        'vega': CH_VEGA,
        'vanna': CH_VANNA,
        'volga': CH_VOLGA,
        'ce_iv': CH_CE_IV,
        'pe_iv': CH_PE_IV,
        'theta_vega': CH_THETA_VEGA,
        'skew': CH_SKEW,
        'gamma': CH_GAMMA,
        'theta': CH_THETA,
        'charm': CH_CHARM,
    }
    
    for name, ch_idx in vol_arb_channels.items():
        grid_vals = interpolate_channel(greeks_mat, ch_idx, k_sorted, sort_idx, valid)
        for g_idx, g_val in enumerate(MONEYNESS_GRID):
            features[f'{name}_mny{g_val:+.2f}'] = grid_vals[g_idx]
    
    # --- 2. ATM-Specific Derived Features ---
    atm_grid_idx = 3  # Index of 0.0 in MONEYNESS_GRID
    
    ce_iv_grid = interpolate_channel(greeks_mat, CH_CE_IV, k_sorted, sort_idx, valid)
    pe_iv_grid = interpolate_channel(greeks_mat, CH_PE_IV, k_sorted, sort_idx, valid)
    vega_grid = interpolate_channel(greeks_mat, CH_VEGA, k_sorted, sort_idx, valid)
    volga_grid = interpolate_channel(greeks_mat, CH_VOLGA, k_sorted, sort_idx, valid)
    
    atm_ce_iv = ce_iv_grid[atm_grid_idx]
    atm_pe_iv = pe_iv_grid[atm_grid_idx]
    
    # Wing richness: How much more expensive are the wings vs ATM?
    # Positive = wings overpriced (good for selling)
    left_wing_iv = pe_iv_grid[0]   # -4% OTM put
    right_wing_iv = ce_iv_grid[-1]  # +4% OTM call
    
    features['wing_richness_put'] = left_wing_iv - atm_pe_iv if not np.isnan(left_wing_iv) else np.nan
    features['wing_richness_call'] = right_wing_iv - atm_ce_iv if not np.isnan(right_wing_iv) else np.nan
    
    # Smile asymmetry  
    features['smile_asymmetry'] = features['wing_richness_put'] - features['wing_richness_call']
    
    # Volga ratio: Wing volga vs ATM volga (high = market overhedging tails)
    atm_volga = volga_grid[atm_grid_idx]
    if not np.isnan(atm_volga) and abs(atm_volga) > 1e-10:
        features['volga_wing_ratio'] = (volga_grid[0] + volga_grid[-1]) / (2 * atm_volga)
    else:
        features['volga_wing_ratio'] = np.nan
    
    # IV velocity at ATM (mean-reversion signal) — guard against older files
    n_channels = greeks_mat.shape[0]
    if n_channels > CH_CE_IV_VEL:
        ce_vel_grid = interpolate_channel(greeks_mat, CH_CE_IV_VEL, k_sorted, sort_idx, valid)
        features['atm_ce_iv_vel'] = ce_vel_grid[atm_grid_idx]
    else:
        features['atm_ce_iv_vel'] = np.nan
    
    if n_channels > CH_PE_IV_VEL:
        pe_vel_grid = interpolate_channel(greeks_mat, CH_PE_IV_VEL, k_sorted, sort_idx, valid)
        features['atm_pe_iv_vel'] = pe_vel_grid[atm_grid_idx]
    else:
        features['atm_pe_iv_vel'] = np.nan
    
    return features


def compute_realized_vol(spot_series: np.ndarray, window: int = 60) -> np.ndarray:
    """
    Compute rolling realized volatility from spot prices.
    Uses log returns, annualized.
    
    Args:
        spot_series: 1D array of spot prices
        window: lookback window in ticks
    
    Returns:
        rv: 1D array of annualized realized vol (same length, NaN-padded)
    """
    log_ret = np.diff(np.log(spot_series))
    log_ret = np.insert(log_ret, 0, 0.0)
    
    rv = np.full(len(spot_series), np.nan)
    for i in range(window, len(spot_series)):
        chunk = log_ret[i-window:i]
        # Annualize: assume 3s per tick, ~6.5 hrs trading = 7800 ticks/day, 252 days/year
        ticks_per_year = 7800 * 252
        rv[i] = np.std(chunk) * np.sqrt(ticks_per_year)
    
    return rv


def load_vol_arb_dataset(filepath: str, lookahead_ticks: int = 600, rv_window: int = 120) -> tuple:
    """
    Loads HDF5 data and prepares features + targets for Vol Arb ML.
    
    Target Y:
        Y = 1 if the Variance Risk Premium (IV_sold - RV_realized) > threshold
        over the next `lookahead_ticks` period. This means selling vol at
        the current IV would have been profitable after accounting for the
        actual gamma risk that materialized.
    
    Args:
        filepath: Path to .h5 file
        lookahead_ticks: Forward window to measure realized vol (600 ticks = 30 min @ 3s)
        rv_window: Lookback for realized vol calculation (120 ticks = 6 min)
    
    Returns:
        (X, y) DataFrames
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return pd.DataFrame(), pd.Series()
    
    print(f"Loading data from {filepath}...")
    with h5py.File(filepath, 'r') as f:
        ticks = f['ticks'][:]
        df = pd.DataFrame(ticks)
        greeks_3d = f['greeks_matrices'][:]
    
    n_ticks = len(df)
    print(f"Loaded {n_ticks} ticks. Extracting Vol Arb features...")
    
    # --- 1. Extract per-tick vol surface features ---
    feature_rows = []
    for i in range(n_ticks):
        feats = extract_vol_arb_features(greeks_3d[i])
        feature_rows.append(feats if feats else {})
    
    grid_df = pd.DataFrame(feature_rows)
    
    # --- 2. Add scalar context features ---
    scalar_cols = []
    for col in ['iv_z_score', 'skew', 'spot_ewm_vol', 'T']:
        if col in df.columns:
            scalar_cols.append(col)
    
    X = pd.concat([df[scalar_cols], grid_df], axis=1)
    
    # --- 3. Compute Realized Vol (backward-looking) as a feature ---
    spot = df['spot'].values
    rv_backward = compute_realized_vol(spot, window=rv_window)
    X['rv_backward'] = rv_backward
    
    # --- 4. Compute IV vs RV Spread (The Variance Risk Premium proxy) ---
    if 'atm_iv' in df.columns:
        X['vrp'] = df['atm_iv'].values - rv_backward  # Positive = IV overpriced
    
    # --- 5. Compute Target: Forward Realized Vol ---
    print("Computing Forward Realized Volatility (Target Y)...")
    
    # For each tick, compute realized vol over the NEXT lookahead_ticks
    rv_forward = np.full(n_ticks, np.nan)
    log_ret = np.diff(np.log(spot))
    log_ret = np.insert(log_ret, 0, 0.0)
    
    ticks_per_year = 7800 * 252  # 3s ticks, 6.5hr day, 252 days
    for i in range(n_ticks - lookahead_ticks):
        chunk = log_ret[i+1 : i+1+lookahead_ticks]
        rv_forward[i] = np.std(chunk) * np.sqrt(ticks_per_year)
    
    # Y = 1 if the IV we would sell is HIGHER than the realized vol
    # that actually materialized (i.e., selling vol was profitable)
    # We use ATM IV as proxy for what we'd sell
    y = np.zeros(n_ticks)
    if 'atm_iv' in df.columns:
        atm_iv = df['atm_iv'].values
        for i in range(n_ticks - lookahead_ticks):
            if np.isnan(atm_iv[i]) or np.isnan(rv_forward[i]):
                continue
            # Edge = IV_sold - RV_realized
            # We need the edge to exceed transaction costs (~1-2 vol points)
            edge = atm_iv[i] - rv_forward[i]
            y[i] = 1.0 if edge > 0.02 else 0.0  # 2 vol point threshold
    
    # --- 6. Clean and return ---
    valid_mask = ~X.isna().any(axis=1)
    # Also exclude the tail where forward RV is NaN
    valid_mask &= ~np.isnan(rv_forward)
    
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    # Remove last lookahead_ticks (indeterminate)
    cutoff = len(X_clean) - lookahead_ticks
    if cutoff <= 0:
        print("Not enough data.")
        return pd.DataFrame(), pd.Series()
    
    X_clean = X_clean.iloc[:cutoff]
    y_clean = y_clean[:cutoff]
    
    print(f"Vol Arb Data Prep Complete.")
    print(f"  X shape: {X_clean.shape}")
    print(f"  Y=1 (Edge exists): {np.mean(y_clean):.1%}")
    print(f"  Y=0 (No edge):     {1-np.mean(y_clean):.1%}")
    
    return X_clean, pd.Series(y_clean, index=X_clean.index)
