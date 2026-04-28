"""
market_features.py
------------------
Pure stateless signal functions for option chain analysis.
Reusable across Iron Butterfly, Iron Condor, Calendar Spreads, etc.

Greeks Matrix Channel Layout (11 channels):
  0: CE Delta    1: CE Gamma    2: CE Theta    3: CE Vega
  4: CE Vanna    5: CE Volga    6: CE IV       7: PE IV
  8: Moneyness (S/K per strike)
  9: Theta/Vega Ratio (CE Theta / CE Vega) — premium decay efficiency
  10: IV Skew per strike (PE_IV - CE_IV)
"""

import numpy as np
import time
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from greeks.greeks import calculate_iv_vectorized, calculate_greeks_vectorized

# Standardized risk-free rate across all strategies (RBI Repo Rate, Apr 2026)
RISK_FREE_RATE = 0.065


def compute_T_from_expiry(expiry_ts: float, min_seconds: float = 60.0) -> float:
    """
    Compute exact time to expiry in years from a Unix expiry timestamp.
    Mirrors the pattern used in vol_trade.py.
    Applies a 60-second floor to prevent division-by-zero on 0DTE at expiration.

    Args:
        expiry_ts: Unix timestamp of the option expiry (from buffer.expiry_dates[-1])
        min_seconds: Minimum seconds floor (default 60s for 0DTE safety)

    Returns:
        T: Time to expiry in years (float)
    """
    seconds_remaining = max(expiry_ts - time.time(), min_seconds)
    return seconds_remaining / (365.25 * 24 * 3600)


def compute_atm_iv_and_greeks(
    ce_bid: np.ndarray,
    ce_ask: np.ndarray,
    pe_bid: np.ndarray,
    pe_ask: np.ndarray,
    spot: float,
    strikes: np.ndarray,
    T: float,
    r: float = RISK_FREE_RATE,
) -> tuple:
    """
    Computes the full 8-channel greeks matrix for all strikes in one pass.
    IV is computed first (needed for greeks), then folded into channels 6 and 7.

    This eliminates all duplicate IV calculations downstream — Skew, ATM IV,
    and Z-Score all read from this single matrix.

    Args:
        ce_bid, ce_ask: CE bid/ask arrays  shape (n_strikes,)
        pe_bid, pe_ask: PE bid/ask arrays  shape (n_strikes,)
        spot: Current index spot price
        strikes: Strike price array        shape (n_strikes,)
        T: Time to expiry in years
        r: Risk-free rate

    Returns:
        greeks_mat: np.ndarray shape (12, n_strikes)
                    Channels: [Delta, Gamma, Theta, Vega, Vanna, Volga, CE_IV, PE_IV, ... Charm]
        valid_mask: bool array marking strikes where calculation succeeded
    """
    n = len(strikes)
    greeks_mat = np.full((12, n), np.nan, dtype=np.float64)

    ce_mid = (ce_bid + ce_ask) / 2.0
    pe_mid = (pe_bid + pe_ask) / 2.0

    # Valid strike mask — need non-NaN prices and positive strikes
    valid = (~np.isnan(ce_mid)) & (~np.isnan(pe_mid)) & (~np.isnan(strikes)) & (strikes > 0)

    if not np.any(valid):
        return greeks_mat, valid

    v_strikes = strikes[valid]
    v_spot = np.full(np.sum(valid), spot)

    # --- Step 1: IV (vectorized, numba-accelerated) ---
    t_start = time.perf_counter()
    ce_ivs = calculate_iv_vectorized(
        prices=ce_mid[valid], S=v_spot, K=v_strikes, T=T, r=r, is_call=True
    )
    pe_ivs = calculate_iv_vectorized(
        prices=pe_mid[valid], S=v_spot, K=v_strikes, T=T, r=r, is_call=False
    )
    t_iv = time.perf_counter()

    # Fold IV into channels 6 and 7
    greeks_mat[6, valid] = ce_ivs
    greeks_mat[7, valid] = pe_ivs

    # --- Step 2: Greeks (uses CE IV as sigma input) ---
    valid_iv = valid & ~np.isnan(greeks_mat[6])
    if np.any(valid_iv):
        v_strikes_g = strikes[valid_iv]
        v_spot_g = np.full(np.sum(valid_iv), spot)
        v_ce_ivs = greeks_mat[6, valid_iv]

        deltas, gammas, thetas, vegas, vannas, volgas, charms = calculate_greeks_vectorized(
            S=v_spot_g, K=v_strikes_g, T=T, r=r, sigma=v_ce_ivs, is_call=True
        )

        greeks_mat[0, valid_iv] = deltas
        greeks_mat[1, valid_iv] = gammas
        greeks_mat[2, valid_iv] = thetas
        greeks_mat[3, valid_iv] = vegas
        greeks_mat[4, valid_iv] = vannas
        greeks_mat[5, valid_iv] = volgas
        greeks_mat[11, valid_iv] = charms
    t_greeks = time.perf_counter()

    # --- Step 3: Derived Feature Channels (same pass, no extra function calls) ---

    # Ch 8: Moneyness (S/K) per strike
    # >1.0 = ITM call / OTM put, <1.0 = OTM call / ITM put, ~1.0 = ATM
    greeks_mat[8, valid] = spot / v_strikes

    # Ch 9: Theta/Vega Ratio per strike
    # Measures premium decay per unit vol risk — higher = more efficient Iron Butterfly
    # Only defined where both Theta and Vega are valid and Vega > epsilon
    vega_safe_mask = valid_iv & (np.abs(greeks_mat[3]) > 1e-8)
    greeks_mat[9, vega_safe_mask] = greeks_mat[2, vega_safe_mask] / greeks_mat[3, vega_safe_mask]

    # Ch 10: IV Skew per strike (PE_IV - CE_IV)
    # Positive = puts more expensive than calls (normal for indices)
    # Abnormal spike = potential panic signal (used by detect_panic)
    both_iv_valid = valid & ~np.isnan(greeks_mat[6]) & ~np.isnan(greeks_mat[7])
    greeks_mat[10, both_iv_valid] = (
        greeks_mat[7, both_iv_valid] - greeks_mat[6, both_iv_valid]
    )
    t_end = time.perf_counter()

    timings = {
        'iv': (t_iv - t_start) * 1000,
        'core': (t_greeks - t_iv) * 1000,
        'features': (t_end - t_greeks) * 1000
    }

    return greeks_mat, timings


def compute_smile_spread(greeks_mat: np.ndarray, atm_idx: int, wing_offset: int) -> float:
    """
    Compute ATM IV minus Wing IV average.
    Always negative for indices (smile curvature).
    If it becomes less negative, the ATM is relatively expensive (signal).
    """
    n = greeks_mat.shape[1]
    ci, pi = atm_idx + wing_offset, atm_idx - wing_offset
    if ci >= n or pi < 0:
        return np.nan
    
    # Ch 6/7 are CE/PE IVs
    atm_iv = (greeks_mat[6, atm_idx] + greeks_mat[7, atm_idx]) / 2.0
    wing_iv = (greeks_mat[6, ci] + greeks_mat[7, pi]) / 2.0
    
    if np.isnan(atm_iv) or np.isnan(wing_iv):
        return np.nan
    
    return float(atm_iv - wing_iv)


def compute_25delta_skew(greeks_mat: np.ndarray) -> float:
    """
    Compute the 25-delta risk-reversal skew.
    Finds the strikes closest to delta = +0.25 (OTM call) and delta = -0.25 (OTM put),
    then returns PE_IV - CE_IV at those strikes.

    Positive value = puts more expensive than calls (normal for equity indices).
    A spike indicates elevated downside fear or potential panic.

    Args:
        greeks_mat: np.ndarray shape (12, n_strikes)
                    Ch 0 = CE Delta, Ch 6 = CE_IV, Ch 7 = PE_IV

    Returns:
        skew: float (PE_IV at 25d put strike) - (CE_IV at 25d call strike), or np.nan
    """
    deltas = greeks_mat[0]   # CE Deltas
    ce_ivs = greeks_mat[6]   # CE IVs
    pe_ivs = greeks_mat[7]   # PE IVs

    valid = ~np.isnan(deltas) & ~np.isnan(ce_ivs) & ~np.isnan(pe_ivs)
    if not np.any(valid):
        return np.nan

    v_deltas = deltas[valid]
    v_ce_ivs = ce_ivs[valid]
    v_pe_ivs = pe_ivs[valid]

    # 25-delta call: CE delta closest to +0.25
    call_idx = int(np.argmin(np.abs(v_deltas - 0.25)))
    # 25-delta put: CE delta closest to -0.25 (put delta = CE delta - 1 by put-call parity)
    put_idx = int(np.argmin(np.abs(v_deltas - 0.75)))  # delta ~0.75 CE = ~-0.25 PE delta

    skew = float(v_pe_ivs[put_idx] - v_ce_ivs[call_idx])
    return skew


def compute_rolling_zscore(history: deque) -> float:
    """
    Generic rolling Z-Score calculation for any deque history.
    Returns 0.0 if insufficient history or zero variance.
    """
    if len(history) < 120:  # Require 120 ticks (1 min at 0.5s) for a stable signal
        return 0.0
    arr = np.array(history, dtype=np.float64)
    mu, sigma = np.mean(arr), np.std(arr)
    if sigma < 1e-8:
        return 0.0
    return float((arr[-1] - mu) / sigma)


def compute_spot_ewm_volatility(spot_history: deque, alpha: float = 0.06) -> dict:
    """
    Compute Exponentially Weighted Moving statistics of spot prices.
    Far superior to first-vs-last subtraction: captures path, not just endpoints.

    EWM Variance: ewm_var_t = alpha*(spot_t - ewm_mean_{t-1})^2 + (1-alpha)*ewm_var_{t-1}
    EWM Mean:     ewm_mean_t = alpha*spot_t + (1-alpha)*ewm_mean_{t-1}

    alpha=0.06 ≈ 30-tick (~90s) half-life at 3s intervals.

    Args:
        spot_history: deque of spot prices
        alpha: EWM smoothing factor

    Returns:
        dict with keys:
          'ewm_vol'    : float — rolling EWM std dev of spot
          'ewm_mean'   : float — rolling EWM mean
          'momentum'   : float — latest_spot minus ewm_mean (signed directional drift)
          'recent_drop': float — fractional drop from ewm_mean (positive = down move)
    """
    if len(spot_history) < 2:
        spot = spot_history[-1] if spot_history else 0.0
        return {'ewm_vol': 0.0, 'ewm_mean': spot, 'momentum': 0.0, 'recent_drop': 0.0}

    spots = list(spot_history)
    ewm_mean = spots[0]
    ewm_var = 0.0

    for s in spots[1:]:
        diff = s - ewm_mean
        ewm_var = alpha * diff * diff + (1.0 - alpha) * ewm_var
        ewm_mean = alpha * s + (1.0 - alpha) * ewm_mean

    ewm_vol = np.sqrt(ewm_var)
    latest = spots[-1]
    momentum = latest - ewm_mean
    recent_drop = (ewm_mean - latest) / ewm_mean if ewm_mean > 0 else 0.0

    # Track previous momentum to calculate acceleration (rate of change)
    prev_ewm_mean = spots[-2] if len(spots) > 1 else spots[0]
    prev_momentum = (spots[-2] - prev_ewm_mean) if len(spots) > 1 else 0.0
    acceleration = momentum - prev_momentum

    return {
        'ewm_vol': float(ewm_vol),
        'ewm_mean': float(ewm_mean),
        'momentum': float(momentum),
        'acceleration': float(acceleration),
        'recent_drop': float(recent_drop),  
    }


def detect_panic(skew_history: deque, skew: float, spot_stats: dict,
                 skew_z_threshold: float = 2.0,
                 spot_calm_threshold: float = 0.001) -> tuple:
    """
    Panic Detection Filter.
    Blocks strategy entry when skew spikes violently WITHOUT a corresponding spot move.

    Logic:
      - Compute skew Z-Score vs its 1-hour rolling history
      - If skew_z > threshold AND spot EWM vol is calm (< threshold), flag panic

    Args:
        skew_history: deque of historical skew values
        skew: current skew value
        spot_stats: output from compute_spot_ewm_volatility()
        skew_z_threshold: Z-Score level that constitutes a panic spike (default 2.0)
        spot_calm_threshold: Max recent_drop to consider spot "calm" (default 0.1%)

    Returns:
        (panic_detected: bool, skew_z: float)
    """
    if len(skew_history) < 2 or np.isnan(skew):
        return False, 0.0

    arr = np.array(skew_history, dtype=np.float64)
    mean = np.mean(arr)
    std = np.std(arr) or 1e-8
    skew_z = float((skew - mean) / std)

    recent_drop = abs(spot_stats.get('recent_drop', 0.0))
    panic = (skew_z > skew_z_threshold) and (recent_drop < spot_calm_threshold)

    return panic, skew_z
