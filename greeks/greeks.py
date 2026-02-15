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
                           T: float, r: float, is_call: bool) -> np.ndarray:
    """Vectorized implied volatility calculation for arrays of option prices."""
    n = len(prices)
    ivs = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        ivs[i] = implied_vol_newton(prices[i], S[i], K[i], T, r, is_call)
    
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
    vannas = np.empty(n, dtype=np.float64)
    volgas = np.empty(n, dtype=np.float64)
    
    # Use max to avoid issues with very small T
    T_safe = max(T, 1e-6)
    sqrt_T = math.sqrt(T_safe)
    inv_sqrt_T = 1.0 / sqrt_T
    exp_neg_rT = math.exp(-r * T_safe)

    for i in prange(n):
        # Skip if invalid inputs
        if np.isnan(sigma[i]) or sigma[i] <= 0 or S[i] <= 0 or K[i] <= 0:
            deltas[i] = gammas[i] = thetas[i] = vegas[i] = vannas[i] = volgas[i] = np.nan
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
        
        #first order greeks
        deltas[i] = cdf_d1 if is_call else cdf_d1 - 1.0
        gammas[i] = pdf_d1 * inv_sqrt_T / (S[i] * sigma[i])
        vegas[i] = S[i] * pdf_d1 * sqrt_T / 100.0
        
        first_term = -S[i] * sigma[i] * pdf_d1 * 0.5 * inv_sqrt_T 

        if is_call:
            thetas[i] = (first_term - r * K[i] * exp_neg_rT * cdf_d2) / 365.0
        else:
            thetas[i] = (first_term + r * K[i] * exp_neg_rT * cdf_neg_d2) / 365.0

        vega_raw = S[i] * pdf_d1 * sqrt_T
        vanna_raw = -pdf_d1 * d2 / sigma[i]            # ∂Vega_raw/∂S
        volga_raw = vega_raw * (d1 * d2 / sigma[i])    # ∂Vega_raw/∂σ

        vannas[i] = vanna_raw / 100.0                  # match Vega scaling
        volgas[i] = volga_raw / 100.0                  # per 1% vol change
    
    return deltas, gammas, thetas, vegas, vannas, volgas

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
    res = calculate_greeks_vectorized(S_arr, K_arr, T, r, sigma_arr, is_call)
    return tuple(r[0] for r in res)   # delta, gamma, theta, vega, vanna, volga


