"""
Feature Engineering for Option Chain Data

Adds derived features beyond raw Greeks:
- PCR (Put-Call Ratio)
- IV Skew
- OI Skew
- Max Pain
- OI Momentum
- Volume-Price Divergence
"""
import numpy as np
from typing import Tuple, Optional


def calculate_pcr(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate Put-Call Ratio (PE_OI / CE_OI) per strike.
    
    Args:
        matrix: (channels, strikes) - channels 6=CE_OI, 7=PE_OI
    
    Returns:
        PCR array (strikes,)
    """
    ce_oi = matrix[6, :]  # CE_OI
    pe_oi = matrix[7, :]   # PE_OI
    
    # Avoid division by zero
    pcr = np.where(ce_oi > 0, pe_oi / ce_oi, np.nan)
    return pcr


def calculate_iv_skew(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate IV Skew (PE_IV - CE_IV).
    Negative = bearish sentiment.
    
    Args:
        matrix: (channels, strikes) - channels 13=CE_IV, 14=PE_IV
    
    Returns:
        IV Skew array (strikes,)
    """
    ce_iv = matrix[13, :] if matrix.shape[0] > 13 else np.full(matrix.shape[1], np.nan)
    pe_iv = matrix[14, :] if matrix.shape[0] > 14 else np.full(matrix.shape[1], np.nan)
    
    return pe_iv - ce_iv


def calculate_oi_skew(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate OI Skew (PE_OI - CE_OI).
    Positive = bearish sentiment.
    
    Args:
        matrix: (channels, strikes) - channels 6=CE_OI, 7=PE_OI
    
    Returns:
        OI Skew array (strikes,)
    """
    ce_oi = matrix[6, :]  # CE_OI
    pe_oi = matrix[7, :]  # PE_OI
    
    return pe_oi - ce_oi


def calculate_max_pain(matrix: np.ndarray) -> float:
    """
    Calculate Max Pain strike (strike with maximum combined OI).
    
    Args:
        matrix: (channels, strikes) - channels 6=CE_OI, 7=PE_OI, 10=STRIKE
    
    Returns:
        Max Pain strike price
    """
    ce_oi = matrix[6, :]
    pe_oi = matrix[7, :]
    strikes = matrix[10, :]
    
    combined_oi = ce_oi + pe_oi
    max_idx = np.nanargmax(combined_oi)
    
    return strikes[max_idx] if not np.isnan(strikes[max_idx]) else np.nan


def calculate_oi_momentum(matrices: np.ndarray) -> np.ndarray:
    """
    Calculate OI momentum (rate of change) over time.
    
    Args:
        matrices: (time_steps, channels, strikes) - last few timesteps
    
    Returns:
        OI momentum (channels, strikes) - average rate of change
    """
    if len(matrices) < 2:
        return np.zeros((matrices.shape[1], matrices.shape[2]))
    
    # Calculate difference
    oi_ce = matrices[:, 6, :]  # (time, strikes)
    oi_pe = matrices[:, 7, :]
    
    # Rate of change (first derivative)
    momentum_ce = np.diff(oi_ce, axis=0)  # (time-1, strikes)
    momentum_pe = np.diff(oi_pe, axis=0)
    
    # Average momentum
    avg_momentum_ce = np.mean(momentum_ce, axis=0) if len(momentum_ce) > 0 else np.zeros(matrices.shape[2])
    avg_momentum_pe = np.mean(momentum_pe, axis=0) if len(momentum_pe) > 0 else np.zeros(matrices.shape[2])
    
    # Stack as new channels
    momentum = np.vstack([avg_momentum_ce, avg_momentum_pe])  # (2, strikes)
    
    return momentum


def add_derived_features(
    matrix: np.ndarray,
    matrices_history: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Add derived features to matrix as additional channels.
    
    Args:
        matrix: Current matrix (channels, strikes)
        matrices_history: Historical matrices (time_steps, channels, strikes) for momentum
    
    Returns:
        Extended matrix with derived features
    """
    features = []
    
    # PCR
    pcr = calculate_pcr(matrix)
    features.append(pcr)
    
    # IV Skew (if Greeks are present)
    if matrix.shape[0] > 14:
        iv_skew = calculate_iv_skew(matrix)
        features.append(iv_skew)
    else:
        features.append(np.full(matrix.shape[1], np.nan))
    
    # OI Skew
    oi_skew = calculate_oi_skew(matrix)
    features.append(oi_skew)
    
    # Max Pain distance (normalized)
    spot = matrix[11, 0] if matrix.shape[0] > 11 else np.nan
    max_pain = calculate_max_pain(matrix)
    if not np.isnan(spot) and not np.isnan(max_pain):
        gap_to_max_pain = (max_pain - spot) / spot if spot > 0 else np.nan
        # Broadcast to all strikes
        gap_array = np.full(matrix.shape[1], gap_to_max_pain)
    else:
        gap_array = np.full(matrix.shape[1], np.nan)
    features.append(gap_array)
    
    # OI Momentum (if history available)
    if matrices_history is not None and len(matrices_history) > 1:
        momentum = calculate_oi_momentum(matrices_history)
        features.append(momentum[0, :])  # CE momentum
        features.append(momentum[1, :])  # PE momentum
    else:
        features.append(np.zeros(matrix.shape[1]))
        features.append(np.zeros(matrix.shape[1]))
    
    # Stack new features
    new_features = np.vstack(features)  # (n_new_features, strikes)
    
    # Combine with original matrix
    extended_matrix = np.vstack([matrix, new_features])
    
    return extended_matrix


def normalize_matrix(
    matrix: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize matrix channels (z-score normalization).
    
    Args:
        matrix: (channels, strikes) or (time, channels, strikes)
        mean: Pre-computed mean (channels,)
        std: Pre-computed std (channels,)
    
    Returns:
        Normalized matrix, mean, std
    """
    if len(matrix.shape) == 2:
        # (channels, strikes)
        if mean is None:
            mean = np.nanmean(matrix, axis=1, keepdims=True)
        if std is None:
            std = np.nanstd(matrix, axis=1, keepdims=True)
        
        normalized = (matrix - mean) / (std + 1e-8)
        return normalized, mean.squeeze(), std.squeeze()
    
    elif len(matrix.shape) == 3:
        # (time, channels, strikes)
        if mean is None:
            mean = np.nanmean(matrix, axis=(0, 2), keepdims=True)  # (1, channels, 1)
        if std is None:
            std = np.nanstd(matrix, axis=(0, 2), keepdims=True)
        
        normalized = (matrix - mean) / (std + 1e-8)
        return normalized, mean.squeeze(), std.squeeze()
    
    else:
        raise ValueError(f"Unsupported matrix shape: {matrix.shape}")


# Example usage
if __name__ == "__main__":
    # Dummy matrix (23 channels, 7 strikes)
    matrix = np.random.randn(23, 7)
    matrix[6, :] = np.abs(matrix[6, :])  # CE_OI (positive)
    matrix[7, :] = np.abs(matrix[7, :])  # PE_OI (positive)
    matrix[10, :] = np.arange(100, 107)  # STRIKE
    matrix[11, :] = 103.0  # UNDERLYING_LTP
    
    # Add derived features
    extended = add_derived_features(matrix)
    print(f"Original shape: {matrix.shape}")
    print(f"Extended shape: {extended.shape}")
    print(f"Added {extended.shape[0] - matrix.shape[0]} feature channels")

