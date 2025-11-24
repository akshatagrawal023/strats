"""
Feature Engineering Functions for Option Chain Data

Computes key features for predictive analysis:
- PCR Ratio: Put-Call sentiment
- OI Momentum: Institutional flow (multiple time horizons)
- Volume Skew: Retail/flow conviction
- Bid Pressure: Real-time order flow
- Max Pain: Price gravity
- Future Basis: Cost of carry
"""
import numpy as np
from typing import Tuple, Optional, Dict


# Channel indices (from features matrix: 11 channels)
CE_BID = 0
CE_ASK = 1
PE_BID = 2
PE_ASK = 3
CE_VOL = 4
PE_VOL = 5
CE_OI = 6
PE_OI = 7
CE_OICH = 8
PE_OICH = 9
STRIKE = 10

# Greeks channels (from greeks matrix: 10 channels)
CE_IV = 0
PE_IV = 1
CE_DELTA = 2
PE_DELTA = 3
CE_GAMMA = 4
PE_GAMMA = 5
CE_THETA = 6
PE_THETA = 7
CE_VEGA = 8
PE_VEGA = 9


def get_spot_price(features: np.ndarray) -> float:
    """
    Extract spot price from features matrix.
    
    Note: In saved HDF5 data, UNDERLYING_LTP is not stored separately.
    We approximate spot using the middle strike (typically ATM) or
    by finding the strike where CE and PE prices are closest (put-call parity).
    
    Args:
        features: Feature matrix (11, strikes)
    
    Returns:
        Approximate spot price
    """
    strikes = features[STRIKE, :]
    if len(strikes) == 0:
        return np.nan
    
    # Method 1: Use middle strike (typically ATM)
    middle_strike = strikes[len(strikes) // 2]
    
    # Method 2: Use put-call parity to find ATM strike
    # ATM strike minimizes |CE_mid - PE_mid|
    ce_bid = features[CE_BID, :]
    ce_ask = features[CE_ASK, :]
    pe_bid = features[PE_BID, :]
    pe_ask = features[PE_ASK, :]
    
    ce_mid = (ce_bid + ce_ask) / 2
    pe_mid = (pe_bid + pe_ask) / 2
    
    # Find strike where CE and PE prices are closest (ATM)
    price_diff = np.abs(ce_mid - pe_mid)
    if not np.all(np.isnan(price_diff)):
        atm_idx = int(np.nanargmin(price_diff))
        atm_strike = strikes[atm_idx]
        # Use average of middle strike and ATM strike for better approximation
        return (middle_strike + atm_strike) / 2
    
    return middle_strike


def get_future_price(features: np.ndarray, greeks: Optional[np.ndarray] = None) -> float:
    """
    Extract future price. If not directly available, approximate from spot.
    """
    # Future price would be in channel 12 of augmented matrix
    # For saved data, we may need to approximate
    return get_spot_price(features)  # Fallback to spot


def find_atm_index(strikes: np.ndarray, spot: float) -> int:
    """Find index of strike closest to ATM (spot price)."""
    if np.isnan(spot) or len(strikes) == 0:
        return len(strikes) // 2  # Middle strike as fallback
    return int(np.nanargmin(np.abs(strikes - spot)))


def compute_pcr_ratio(
    features: np.ndarray,
    greeks: Optional[np.ndarray] = None,
    spot: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute Put-Call Ratio (PCR) at different moneyness levels.
    
    Args:
        features: Feature matrix (11, strikes)
        greeks: Greeks matrix (10, strikes) - optional
        spot: Spot price - if None, inferred from features
    
    Returns:
        dict with PCR ratios: pcr_atm, pcr_total, pcr_otm
    """
    ce_oi = features[CE_OI, :]
    pe_oi = features[PE_OI, :]
    
    if spot is None:
        spot = get_spot_price(features)
    
    strikes = features[STRIKE, :]
    atm_idx = find_atm_index(strikes, spot)
    
    # PCR at ATM
    ce_oi_atm = ce_oi[atm_idx]
    pe_oi_atm = pe_oi[atm_idx]
    pcr_atm = pe_oi_atm / (ce_oi_atm + 1e-6)
    
    # PCR total
    ce_oi_total = np.nansum(ce_oi)
    pe_oi_total = np.nansum(pe_oi)
    pcr_total = pe_oi_total / (ce_oi_total + 1e-6)
    
    # PCR OTM (puts OTM vs calls OTM)
    if atm_idx < len(strikes) - 1:
        pe_oi_otm = np.nansum(pe_oi[atm_idx + 1:])  # OTM puts
        ce_oi_otm = np.nansum(ce_oi[:atm_idx])      # OTM calls
        pcr_otm = pe_oi_otm / (ce_oi_otm + 1e-6)
    else:
        pcr_otm = pcr_total
    
    return {
        'pcr_atm': float(pcr_atm),
        'pcr_total': float(pcr_total),
        'pcr_otm': float(pcr_otm)
    }


def compute_oi_momentum(
    features_window: np.ndarray,
    horizons: list = [10, 20, 60],
    spot: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute OI Momentum over different time horizons.
    
    Args:
        features_window: Feature matrices (time, 11, strikes)
        horizons: List of time horizons in steps (e.g., [10, 20, 60] = 30s, 1min, 3min)
        spot: Spot price - if None, inferred from latest features
    
    Returns:
        dict with OI momentum for each horizon: oi_momentum_10, oi_momentum_20, etc.
    """
    if len(features_window) < max(horizons) + 1:
        return {}
    
    latest = features_window[-1]
    if spot is None:
        spot = get_spot_price(latest)
    
    strikes = latest[STRIKE, :]
    atm_idx = find_atm_index(strikes, spot)
    
    ce_oi_latest = latest[CE_OI, atm_idx]
    pe_oi_latest = latest[PE_OI, atm_idx]
    
    results = {}
    
    for horizon in horizons:
        if len(features_window) <= horizon:
            continue
        
        # Get OI from horizon steps ago
        past = features_window[-horizon - 1]
        ce_oi_past = past[CE_OI, atm_idx]
        pe_oi_past = past[PE_OI, atm_idx]
        
        # Compute momentum (rate of change)
        ce_momentum = (ce_oi_latest - ce_oi_past) / (ce_oi_past + 1e-6)
        pe_momentum = (pe_oi_latest - pe_oi_past) / (pe_oi_past + 1e-6)
        
        # Net momentum (positive = bullish, negative = bearish)
        # PE OI increase = bullish, CE OI increase = bearish
        net_momentum = pe_momentum - ce_momentum
        
        results[f'ce_oi_momentum_{horizon}'] = float(ce_momentum)
        results[f'pe_oi_momentum_{horizon}'] = float(pe_momentum)
        results[f'oi_momentum_{horizon}'] = float(net_momentum)
    
    return results


def compute_volume_skew(
    features: np.ndarray,
    spot: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute Volume Skew - measures retail/flow conviction.
    
    Args:
        features: Feature matrix (11, strikes)
        spot: Spot price - if None, inferred
    
    Returns:
        dict with volume skew metrics
    """
    ce_vol = features[CE_VOL, :]
    pe_vol = features[PE_VOL, :]
    
    if spot is None:
        spot = get_spot_price(features)
    
    strikes = features[STRIKE, :]
    atm_idx = find_atm_index(strikes, spot)
    
    # Volume at ATM
    ce_vol_atm = ce_vol[atm_idx]
    pe_vol_atm = pe_vol[atm_idx]
    vol_ratio_atm = pe_vol_atm / (ce_vol_atm + 1e-6)
    
    # Total volume ratio
    ce_vol_total = np.nansum(ce_vol)
    pe_vol_total = np.nansum(pe_vol)
    vol_ratio_total = pe_vol_total / (ce_vol_total + 1e-6)
    
    # Volume skew (OTM vs ITM)
    if atm_idx > 0 and atm_idx < len(strikes) - 1:
        ce_vol_itm = np.nansum(ce_vol[:atm_idx])
        ce_vol_otm = np.nansum(ce_vol[atm_idx + 1:])
        pe_vol_itm = np.nansum(pe_vol[atm_idx + 1:])  # ITM puts are higher strikes
        pe_vol_otm = np.nansum(pe_vol[:atm_idx])      # OTM puts are lower strikes
        
        ce_skew = ce_vol_itm / (ce_vol_otm + 1e-6)
        pe_skew = pe_vol_itm / (pe_vol_otm + 1e-6)
        vol_skew = pe_skew - ce_skew
    else:
        vol_skew = 0.0
    
    return {
        'vol_ratio_atm': float(vol_ratio_atm),
        'vol_ratio_total': float(vol_ratio_total),
        'vol_skew': float(vol_skew)
    }


def compute_bid_pressure(
    features: np.ndarray,
    spot: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute Bid Pressure - real-time order flow indicator.
    
    Bid pressure > 0.5 = buying pressure, < 0.5 = selling pressure
    
    Args:
        features: Feature matrix (11, strikes)
        spot: Spot price - if None, inferred
    
    Returns:
        dict with bid pressure metrics
    """
    ce_bid = features[CE_BID, :]
    ce_ask = features[CE_ASK, :]
    pe_bid = features[PE_BID, :]
    pe_ask = features[PE_ASK, :]
    
    if spot is None:
        spot = get_spot_price(features)
    
    strikes = features[STRIKE, :]
    atm_idx = find_atm_index(strikes, spot)
    
    # Bid pressure = bid / (bid + ask)
    ce_bid_pressure = ce_bid / (ce_bid + ce_ask + 1e-6)
    pe_bid_pressure = pe_bid / (pe_bid + pe_ask + 1e-6)
    
    # At ATM
    ce_bid_pressure_atm = ce_bid_pressure[atm_idx]
    pe_bid_pressure_atm = pe_bid_pressure[atm_idx]
    
    # Average across strikes
    ce_bid_pressure_avg = np.nanmean(ce_bid_pressure)
    pe_bid_pressure_avg = np.nanmean(pe_bid_pressure)
    
    # Net bid pressure (PE - CE)
    net_bid_pressure = pe_bid_pressure_avg - ce_bid_pressure_avg
    
    return {
        'ce_bid_pressure_atm': float(ce_bid_pressure_atm),
        'pe_bid_pressure_atm': float(pe_bid_pressure_atm),
        'ce_bid_pressure_avg': float(ce_bid_pressure_avg),
        'pe_bid_pressure_avg': float(pe_bid_pressure_avg),
        'net_bid_pressure': float(net_bid_pressure)
    }


def compute_max_pain(
    features: np.ndarray,
    spot: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute Max Pain - strike with maximum total OI (price gravity).
    
    Args:
        features: Feature matrix (11, strikes)
        spot: Spot price - if None, inferred
    
    Returns:
        dict with max pain metrics
    """
    ce_oi = features[CE_OI, :]
    pe_oi = features[PE_OI, :]
    strikes = features[STRIKE, :]
    
    if spot is None:
        spot = get_spot_price(features)
    
    # Total OI per strike
    total_oi = ce_oi + pe_oi
    
    # Find max pain strike
    max_oi_idx = int(np.nanargmax(total_oi))
    max_pain_strike = strikes[max_oi_idx]
    
    # Distance to max pain
    max_pain_distance = (max_pain_strike - spot) / (spot + 1e-6)
    
    # OI concentration at max pain
    total_oi_sum = np.nansum(total_oi)
    max_pain_concentration = total_oi[max_oi_idx] / (total_oi_sum + 1e-6)
    
    return {
        'max_pain_strike': float(max_pain_strike),
        'max_pain_distance': float(max_pain_distance),
        'max_pain_concentration': float(max_pain_concentration)
    }


def compute_future_basis(
    features: np.ndarray,
    spot: Optional[float] = None,
    future_price: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute Future Basis - cost of carry (macro factor).
    
    Positive basis = contango (normal), negative = backwardation (stressed)
    
    Args:
        features: Feature matrix (11, strikes)
        spot: Spot price - if None, inferred
        future_price: Future price - if None, approximated from spot
    
    Returns:
        dict with future basis metrics
    """
    if spot is None:
        spot = get_spot_price(features)
    
    if future_price is None:
        future_price = get_future_price(features)
    
    if np.isnan(spot) or np.isnan(future_price):
        return {
            'future_basis': 0.0,
            'future_basis_pct': 0.0
        }
    
    # Basis = Future - Spot
    basis = future_price - spot
    
    # Basis percentage
    basis_pct = basis / (spot + 1e-6)
    
    return {
        'future_basis': float(basis),
        'future_basis_pct': float(basis_pct)
    }


def compute_all_features(
    features_window: np.ndarray,
    greeks_window: Optional[np.ndarray] = None,
    spot: Optional[float] = None,
    future_price: Optional[float] = None,
    oi_horizons: list = [10, 20, 60]
) -> Dict[str, float]:
    """
    Compute all features from a window of data.
    
    Args:
        features_window: Feature matrices (time, 11, strikes)
        greeks_window: Greeks matrices (time, 10, strikes) - optional
        spot: Spot price - if None, inferred
        future_price: Future price - if None, approximated
        oi_horizons: Time horizons for OI momentum
    
    Returns:
        dict with all computed features
    """
    latest_features = features_window[-1]
    
    if spot is None:
        spot = get_spot_price(latest_features)
    
    all_features = {}
    
    # PCR Ratio
    pcr_features = compute_pcr_ratio(latest_features, greeks_window[-1] if greeks_window is not None else None, spot)
    all_features.update(pcr_features)
    
    # OI Momentum
    oi_features = compute_oi_momentum(features_window, oi_horizons, spot)
    all_features.update(oi_features)
    
    # Volume Skew
    vol_features = compute_volume_skew(latest_features, spot)
    all_features.update(vol_features)
    
    # Bid Pressure
    bid_features = compute_bid_pressure(latest_features, spot)
    all_features.update(bid_features)
    
    # Max Pain
    max_pain_features = compute_max_pain(latest_features, spot)
    all_features.update(max_pain_features)
    
    # Future Basis
    basis_features = compute_future_basis(latest_features, spot, future_price)
    all_features.update(basis_features)
    
    return all_features

