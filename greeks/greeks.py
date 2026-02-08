import numpy as np
from numba import jit, vectorize, float64, prange
import math
from typing import Tuple, Optional


@jit(nopython=True, fastmath=True, inline='always')
def norm_pdf_numba(x: float) -> float:
    """Fast standard normal PDF"""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

@jit(nopython=True, fastmath=True, inline='always')
def norm_cdf_numba(x: float) -> float:
    """Fast standard normal CDF using error function"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@jit(nopython=True, fastmath=True)
def implied_vol_newton(price: float, S: float, K: float, T: float, r: float, 
                       is_call: bool, max_iter: int = 50) -> float:
    """Calculate implied volatility using Newton-Raphson method."""
    # Edge cases
    if T <= 0 or S <= 0 or K <= 0 or price <= 0:
        return np.nan
    
    # Intrinsic value bounds
    intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
    if price < intrinsic * 0.99:  # Below intrinsic (with tolerance)
        return np.nan
    
    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2.0 * math.pi / T) * (price / S)
    sigma = max(0.01, min(sigma, 5.0))  # Bound between 1% and 500%
    
    # Newton-Raphson iterations
    for _ in range(max_iter):
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Price calculation
        if is_call:
            bs_price = S * norm_cdf_numba(d1) - K * math.exp(-r * T) * norm_cdf_numba(d2)
        else:
            bs_price = K * math.exp(-r * T) * norm_cdf_numba(-d2) - S * norm_cdf_numba(-d1)
        
        # Vega (same for call and put)
        vega = S * norm_pdf_numba(d1) * sqrt_T
        
        # Check convergence
        diff = bs_price - price
        if abs(diff) < 1e-6 or abs(vega) < 1e-10:
            break
        
        # Update sigma
        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 5.0))  # Keep bounds
    
    return sigma if 0.001 < sigma < 5.0 else np.nan

@jit(nopython=True, fastmath=True, parallel=True)
def calculate_iv_vectorized(prices: np.ndarray, S: np.ndarray, K: np.ndarray, 
                           T: float, r: float, is_call: np.ndarray) -> np.ndarray:
    """Vectorized implied volatility calculation for arrays of option prices."""
    n = len(prices)
    ivs = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        ivs[i] = implied_vol_newton(prices[i], S[i], K[i], T, r, is_call[i])
    
    return ivs

@jit(nopython=True, fastmath=True, parallel=True)
def calculate_greeks_vectorized(S: np.ndarray, K: np.ndarray, T: float, 
                               r: float, sigma: np.ndarray, 
                               is_call: bool) -> Tuple[np.ndarray, np.ndarray, 
                                                              np.ndarray, np.ndarray]:
    """Vectorized Black-Scholes Greeks calculation"""
    n = len(S)
    deltas = np.empty(n, dtype=np.float64)
    gammas = np.empty(n, dtype=np.float64)
    thetas = np.empty(n, dtype=np.float64)
    vegas = np.empty(n, dtype=np.float64)
    
    # Use max to avoid issues with very small T
    T_safe = max(T, 1e-6)
    sqrt_T = math.sqrt(T_safe)
    inv_sqrt_T = 1.0 / sqrt_T 

    # Precompute exp(-rT) - same for all options
    exp_neg_rT = math.exp(-r * T_safe)
    
    for i in prange(n):
        # Skip if invalid inputs
        if np.isnan(sigma[i]) or sigma[i] <= 0 or S[i] <= 0 or K[i] <= 0:
            deltas[i] = np.nan
            gammas[i] = np.nan
            thetas[i] = np.nan
            vegas[i] = np.nan
            continue
        
        S_over_K = S[i] / K[i]
        log_S_over_K = math.log(S_over_K)
        half_sigma2 = 0.5 * sigma[i] * sigma[i]

        d1 = (log_S_over_K + (r + half_sigma2) * T_safe) / (sigma[i] * sqrt_T)
        d2 = d1 - sigma[i] * sqrt_T

        pdf_d1 = norm_pdf_numba(d1)
        cdf_d1 = norm_cdf_numba(d1)
        cdf_d2 = norm_cdf_numba(d2)
        cdf_neg_d2 = 1.0 - cdf_d2
        
        if is_call:
            deltas[i] = cdf_d1
        else:
            deltas[i] = cdf_d1 - 1.0
        
        gammas[i] = pdf_d1 * inv_sqrt_T / (S[i] * sigma[i])
        vegas[i] = S[i] * pdf_d1 * sqrt_T / 100.0
        
        first_term = -S[i] * sigma[i] * pdf_d1 * 0.5 * inv_sqrt_T 

        if is_call:
            thetas[i] = (first_term - r * K[i] * exp_neg_rT * cdf_d2) / 365.0
        else:
            thetas[i] = (first_term + r * K[i] * exp_neg_rT * cdf_neg_d2) / 365.0
    
    return deltas, gammas, thetas, vegas

def calculate_greeks_scalar(S: float, K: float, T: float, r: float, 
                           sigma: float, is_call: bool) -> Tuple[float, float, float, float]:
    """
    Calculate Greeks for a single option (scalar inputs, scalar outputs).
    Wraps your existing vectorized function.
    """
    # Convert to 1-element arrays
    S_arr = np.array([S], dtype=np.float64)
    K_arr = np.array([K], dtype=np.float64)
    sigma_arr = np.array([sigma], dtype=np.float64)
    
    # Use your vectorized function
    delta_arr, gamma_arr, theta_arr, vega_arr = calculate_greeks_vectorized(
        S_arr, K_arr, T, r, sigma_arr, is_call
    )
    
    # Return scalars
    return delta_arr[0], gamma_arr[0], theta_arr[0], vega_arr[0]

class MatrixGreeksCalculator:
   
    def __init__(self, risk_free_rate: float = 0.065, days_to_expiry: int = 7):
        self.risk_free_rate = risk_free_rate
        self.time_to_expiry = days_to_expiry / 365.0
    
    def get_greeks(self, is_call: bool, matrix: np.ndarray, 
                            days_to_expiry: Optional[int] = None) -> np.ndarray:

        if matrix.shape[0] != 13:
            raise ValueError(f"Expected 13 channels, got {matrix.shape[0]}")
        
        n_strikes = matrix.shape[1]
        T = (days_to_expiry or int(self.time_to_expiry * 365)) / 365.0
        
        # Extract channels
        if is_call:
            bid = matrix[0, :]
            ask = matrix[1, :]
        else:
            bid = matrix[2, :]
            ask = matrix[3, :]

        # Calculate mid prices
        mid = (bid + ask) / 2.0

        K = matrix[10, :] #strikes
        S = matrix[11, :]  # underlying_ltp, Same value across all strikes
        
        iv = calculate_iv_vectorized(mid, S, K, T, self.risk_free_rate, is_call)

        delta, gamma, theta, vega = calculate_greeks_vectorized(
            S, K, T, self.risk_free_rate, iv, is_call
        )
        
        return iv, delta, gamma, theta, vega
    
    def get_greeks_only(self, matrix: np.ndarray,
                       days_to_expiry: Optional[int] = None) -> np.ndarray:
        """
        Return ONLY Greeks matrix (10 channels) - doesn't append to input.
        
        Args:
            matrix: Input matrix (13 channels × n_strikes)
            days_to_expiry: Override days to expiry (optional)
        
        Returns:
            Greeks matrix (10 channels × n_strikes):
            [CE_IV, PE_IV, CE_DELTA, PE_DELTA, CE_GAMMA, PE_GAMMA,
             CE_THETA, PE_THETA, CE_VEGA, PE_VEGA]
        """
        if matrix.shape[0] != 13:
            raise ValueError(f"Expected 13 channels, got {matrix.shape[0]}")
        
        n_strikes = matrix.shape[1]
        T = (days_to_expiry or int(self.time_to_expiry * 365)) / 365.0
        
        # Extract channels
        ce_bid = matrix[0, :]
        ce_ask = matrix[1, :]
        pe_bid = matrix[2, :]
        pe_ask = matrix[3, :]
        strikes = matrix[10, :]
        underlying_ltp = matrix[11, :]
        
        # Calculate mid prices
        ce_mid = (ce_bid + ce_ask) / 2.0
        pe_mid = (pe_bid + pe_ask) / 2.0
        
        # Prepare arrays
        S = underlying_ltp
        K = strikes
        
        # Calculate IV
        is_call_ce = np.ones(n_strikes, dtype=np.bool_)
        is_call_pe = np.zeros(n_strikes, dtype=np.bool_)
        
        ce_iv = calculate_iv_vectorized(ce_mid, S, K, T, self.risk_free_rate, is_call_ce)
        pe_iv = calculate_iv_vectorized(pe_mid, S, K, T, self.risk_free_rate, is_call_pe)
        
        # Calculate Greeks
        ce_delta, ce_gamma, ce_theta, ce_vega = calculate_greeks_vectorized(
            S, K, T, self.risk_free_rate, ce_iv, is_call_ce
        )
        
        pe_delta, pe_gamma, pe_theta, pe_vega = calculate_greeks_vectorized(
            S, K, T, self.risk_free_rate, pe_iv, is_call_pe
        )
        
        # Stack into 10-channel Greeks matrix
        greeks_matrix = np.stack([
            ce_iv, pe_iv,
            ce_delta, pe_delta,
            ce_gamma, pe_gamma,
            ce_theta, pe_theta,
            ce_vega, pe_vega
        ], axis=0)
        
        return greeks_matrix  # Shape: (10, n_strikes)
    
    def calculate_portfolio_greeks(self, matrix: np.ndarray, 
                                   ce_positions: Optional[np.ndarray] = None,
                                   pe_positions: Optional[np.ndarray] = None) -> dict:
        """
        Calculate aggregated portfolio Greeks
        
        Args:
            matrix: Matrix with Greeks (23 channels × n_strikes)
            ce_positions: Call positions by strike (positive = long, negative = short)
            pe_positions: Put positions by strike
        
        Returns:
            Dictionary with aggregated Greeks
        """
        if matrix.shape[0] < 23:
            raise ValueError("Matrix must have Greeks channels (23 channels)")
        
        n_strikes = matrix.shape[1]
        
        # Default to neutral portfolio (no positions)
        if ce_positions is None:
            ce_positions = np.zeros(n_strikes)
        if pe_positions is None:
            pe_positions = np.zeros(n_strikes)
        
        # Extract Greeks
        ce_delta = matrix[15, :]
        pe_delta = matrix[16, :]
        ce_gamma = matrix[17, :]
        pe_gamma = matrix[18, :]
        ce_theta = matrix[19, :]
        pe_theta = matrix[20, :]
        ce_vega = matrix[21, :]
        pe_vega = matrix[22, :]
        
        # Aggregate (ignoring NaN values)
        portfolio_delta = np.nansum(ce_delta * ce_positions) + np.nansum(pe_delta * pe_positions)
        portfolio_gamma = np.nansum(ce_gamma * ce_positions) + np.nansum(pe_gamma * pe_positions)
        portfolio_theta = np.nansum(ce_theta * ce_positions) + np.nansum(pe_theta * pe_positions)
        portfolio_vega = np.nansum(ce_vega * ce_positions) + np.nansum(pe_vega * pe_positions)
        
        underlying_ltp = matrix[11, 0]
        net_exposure = portfolio_delta * underlying_ltp
        
        return {
            'delta': portfolio_delta,
            'gamma': portfolio_gamma,
            'theta': portfolio_theta,
            'vega': portfolio_vega,
            'net_exposure': net_exposure,
            'underlying_price': underlying_ltp
        }
    
    def get_atm_greeks(self, matrix: np.ndarray) -> dict:
        """
        Get Greeks for ATM options (closest to underlying price)
        
        Args:
            matrix: Matrix with Greeks (23 channels × n_strikes)
        
        Returns:
            Dictionary with ATM call and put Greeks
        """
        if matrix.shape[0] < 23:
            raise ValueError("Matrix must have Greeks channels")
        
        strikes = matrix[10, :]
        underlying_ltp = matrix[11, 0]
        
        # Find ATM index
        atm_idx = np.nanargmin(np.abs(strikes - underlying_ltp))
        
        return {
            'atm_strike': strikes[atm_idx],
            'underlying_ltp': underlying_ltp,
            'ce': {
                'iv': matrix[13, atm_idx],
                'delta': matrix[15, atm_idx],
                'gamma': matrix[17, atm_idx],
                'theta': matrix[19, atm_idx],
                'vega': matrix[21, atm_idx]
            },
            'pe': {
                'iv': matrix[14, atm_idx],
                'delta': matrix[16, atm_idx],
                'gamma': matrix[18, atm_idx],
                'theta': matrix[20, atm_idx],
                'vega': matrix[22, atm_idx]
            }
        }


if __name__ == "__main__":
    n_strikes = 7
    sample_matrix = np.full((13, n_strikes), np.nan, dtype=np.float64)
    
    # Simulate option chain data
    underlying_price = 1000.0
    strikes = np.array([950, 975, 1000, 1025, 1050, 1075, 1100], dtype=np.float64)
    
    # Calculate Greeks
    calculator = MatrixGreeksCalculator(risk_free_rate=0.065, days_to_expiry=7)

    extended_matrix = calculator.add_greeks_to_matrix(sample_matrix)