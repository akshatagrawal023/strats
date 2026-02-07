import time
import numpy as np
from numba import jit, prange
import math

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

# Test data
n_options = 10000  # Simulating 10,000 option strikes
prices = np.random.uniform(5, 50, n_options)
S = np.full(n_options, 1000.0)  # Same underlying
K = np.linspace(900, 1100, n_options)
T = 7/365
r = 0.065
is_call = np.random.choice([True, False], n_options)

# Benchmark
start = time.time()
result = calculate_iv_vectorized(prices, S, K, T, r, is_call)
elapsed = time.time() - start

print(f"Calculated {n_options} IVs in {elapsed:.3f}s")
print(f"Rate: {n_options/elapsed:.0f} IVs/sec")