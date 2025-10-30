"""
Matrix-based Greeks Calculator - Optimized for Real-Time Processing

Designed to work with OptionDataProcessor's matrix structure (channels × strikes).
Uses Numba JIT compilation for maximum performance on vectorized operations.

Input: NumPy matrix (13 channels, n_strikes)
Output: Greeks as additional matrix channels (10 new channels)
"""
import numpy as np
from numba import jit, vectorize, float64, prange
import math
from typing import Tuple, Optional

# ============================================================================
# NUMBA-OPTIMIZED MATH FUNCTIONS
# ============================================================================

@jit(nopython=True, fastmath=True, inline='always')
def norm_pdf_numba(x: float) -> float:
    """Fast standard normal PDF"""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

@jit(nopython=True, fastmath=True, inline='always')
def norm_cdf_numba(x: float) -> float:
    """Fast standard normal CDF using error function"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ============================================================================
# IMPLIED VOLATILITY CALCULATION
# ============================================================================

@jit(nopython=True, fastmath=True)
def implied_vol_newton(price: float, S: float, K: float, T: float, r: float, 
                       is_call: bool, max_iter: int = 50) -> float:
    """
    Newton-Raphson IV calculation - optimized for speed
    
    Args:
        price: Market price of option
        S: Underlying price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        is_call: True for call, False for put
        max_iter: Maximum iterations
    
    Returns:
        Implied volatility (annual)
    """
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
    """
    Vectorized IV calculation for multiple strikes
    
    Args:
        prices: Option prices array
        S: Underlying prices array (can be same value repeated)
        K: Strike prices array
        T: Time to expiry (years)
        r: Risk-free rate
        is_call: Boolean array (True for calls, False for puts)
    
    Returns:
        Array of implied volatilities
    """
    n = len(prices)
    ivs = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        ivs[i] = implied_vol_newton(prices[i], S[i], K[i], T, r, is_call[i])
    
    return ivs

# ============================================================================
# GREEKS CALCULATION
# ============================================================================

@jit(nopython=True, fastmath=True, parallel=True)
def calculate_greeks_vectorized(S: np.ndarray, K: np.ndarray, T: float, 
                               r: float, sigma: np.ndarray, 
                               is_call: np.ndarray) -> Tuple[np.ndarray, np.ndarray, 
                                                              np.ndarray, np.ndarray]:
    """
    Vectorized Black-Scholes Greeks calculation
    
    Args:
        S: Underlying prices (can be same value)
        K: Strike prices
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Implied volatilities
        is_call: Boolean array for option type
    
    Returns:
        Tuple of (delta, gamma, theta, vega) arrays
    """
    n = len(S)
    deltas = np.empty(n, dtype=np.float64)
    gammas = np.empty(n, dtype=np.float64)
    thetas = np.empty(n, dtype=np.float64)
    vegas = np.empty(n, dtype=np.float64)
    
    # Use max to avoid issues with very small T
    T_safe = max(T, 1e-6)
    sqrt_T = math.sqrt(T_safe)
    
    for i in prange(n):
        # Skip if invalid inputs
        if np.isnan(sigma[i]) or sigma[i] <= 0 or S[i] <= 0 or K[i] <= 0:
            deltas[i] = np.nan
            gammas[i] = np.nan
            thetas[i] = np.nan
            vegas[i] = np.nan
            continue
        
        # Calculate d1 and d2
        d1 = (math.log(S[i] / K[i]) + (r + 0.5 * sigma[i] * sigma[i]) * T_safe) / (sigma[i] * sqrt_T)
        d2 = d1 - sigma[i] * sqrt_T
        
        pdf_d1 = norm_pdf_numba(d1)
        cdf_d1 = norm_cdf_numba(d1)
        cdf_d2 = norm_cdf_numba(d2)
        
        # Delta
        if is_call[i]:
            deltas[i] = cdf_d1
        else:
            deltas[i] = cdf_d1 - 1.0
        
        # Gamma (same for call and put)
        gammas[i] = pdf_d1 / (S[i] * sigma[i] * sqrt_T)
        
        # Vega (same for call and put, divided by 100 for 1% change)
        vegas[i] = S[i] * pdf_d1 * sqrt_T / 100.0
        
        # Theta (per day)
        first_term = -(S[i] * sigma[i] * pdf_d1) / (2.0 * sqrt_T)
        if is_call[i]:
            thetas[i] = (first_term - r * K[i] * math.exp(-r * T_safe) * cdf_d2) / 365.0
        else:
            thetas[i] = (first_term + r * K[i] * math.exp(-r * T_safe) * norm_cdf_numba(-d2)) / 365.0
    
    return deltas, gammas, thetas, vegas

# ============================================================================
# MATRIX PROCESSOR CLASS
# ============================================================================

class MatrixGreeksCalculator:
    """
    Calculate Greeks for matrix-structured option chain data
    
    Matrix Input Format (13 channels × n_strikes):
        0: CE_BID, 1: CE_ASK, 2: PE_BID, 3: PE_ASK
        4: CE_VOL, 5: PE_VOL, 6: CE_OI, 7: PE_OI
        8: CE_OICH, 9: PE_OICH, 10: STRIKE
        11: UNDERLYING_LTP, 12: FUTURE_PRICE
    
    Output: Matrix with 10 additional channels (23 channels × n_strikes):
        13: CE_IV, 14: PE_IV
        15: CE_DELTA, 16: PE_DELTA
        17: CE_GAMMA, 18: PE_GAMMA
        19: CE_THETA, 20: PE_THETA
        21: CE_VEGA, 22: PE_VEGA
    """
    
    def __init__(self, risk_free_rate: float = 0.065, days_to_expiry: int = 7):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 6.5%)
            days_to_expiry: Days until option expiry
        """
        self.risk_free_rate = risk_free_rate
        self.time_to_expiry = days_to_expiry / 365.0
    
    def add_greeks_to_matrix(self, matrix: np.ndarray, 
                            days_to_expiry: Optional[int] = None) -> np.ndarray:
        """
        Add Greeks as new channels to existing matrix
        
        Args:
            matrix: Input matrix (13 channels × n_strikes)
            days_to_expiry: Override days to expiry (optional)
        
        Returns:
            Extended matrix (23 channels × n_strikes) with Greeks
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
        underlying_ltp = matrix[11, :]  # Same value across all strikes
        
        # Calculate mid prices
        ce_mid = (ce_bid + ce_ask) / 2.0
        pe_mid = (pe_bid + pe_ask) / 2.0
        
        # Prepare arrays for vectorized calculation
        S = underlying_ltp  # Already repeated across strikes
        K = strikes
        
        # Calculate IV for calls and puts
        is_call_ce = np.ones(n_strikes, dtype=np.bool_)
        is_call_pe = np.zeros(n_strikes, dtype=np.bool_)
        
        ce_iv = calculate_iv_vectorized(ce_mid, S, K, T, self.risk_free_rate, is_call_ce)
        pe_iv = calculate_iv_vectorized(pe_mid, S, K, T, self.risk_free_rate, is_call_pe)
        
        # Calculate Greeks for calls
        ce_delta, ce_gamma, ce_theta, ce_vega = calculate_greeks_vectorized(
            S, K, T, self.risk_free_rate, ce_iv, is_call_ce
        )
        
        # Calculate Greeks for puts
        pe_delta, pe_gamma, pe_theta, pe_vega = calculate_greeks_vectorized(
            S, K, T, self.risk_free_rate, pe_iv, is_call_pe
        )
        
        # Create extended matrix (23 channels)
        extended_matrix = np.full((23, n_strikes), np.nan, dtype=np.float64)
        extended_matrix[:13, :] = matrix  # Copy original channels
        
        # Add IV channels
        extended_matrix[13, :] = ce_iv
        extended_matrix[14, :] = pe_iv
        
        # Add Greeks channels
        extended_matrix[15, :] = ce_delta
        extended_matrix[16, :] = pe_delta
        extended_matrix[17, :] = ce_gamma
        extended_matrix[18, :] = pe_gamma
        extended_matrix[19, :] = ce_theta
        extended_matrix[20, :] = pe_theta
        extended_matrix[21, :] = ce_vega
        extended_matrix[22, :] = pe_vega
        
        return extended_matrix
    
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

# ============================================================================
# CHANNEL MAPPING REFERENCE
# ============================================================================

CHANNEL_NAMES = [
    'CE_BID', 'CE_ASK', 'PE_BID', 'PE_ASK',           # 0-3
    'CE_VOL', 'PE_VOL', 'CE_OI', 'PE_OI',              # 4-7
    'CE_OICH', 'PE_OICH', 'STRIKE',                    # 8-10
    'UNDERLYING_LTP', 'FUTURE_PRICE',                  # 11-12
    'CE_IV', 'PE_IV',                                  # 13-14
    'CE_DELTA', 'PE_DELTA',                            # 15-16
    'CE_GAMMA', 'PE_GAMMA',                            # 17-18
    'CE_THETA', 'PE_THETA',                            # 19-20
    'CE_VEGA', 'PE_VEGA'                               # 21-22
]

def get_channel_index(name: str) -> int:
    """Get channel index by name"""
    return CHANNEL_NAMES.index(name)

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example: Process matrix from OptionDataProcessor
    print("Matrix Greeks Calculator - Example Usage\n")
    
    # Create sample matrix (13 channels × 7 strikes)
    n_strikes = 7
    sample_matrix = np.full((13, n_strikes), np.nan, dtype=np.float64)
    
    # Simulate option chain data
    underlying_price = 1000.0
    strikes = np.array([950, 975, 1000, 1025, 1050, 1075, 1100], dtype=np.float64)
    
    # Fill sample data
    sample_matrix[10, :] = strikes  # Strikes
    sample_matrix[11, :] = underlying_price  # Underlying LTP (repeated)
    sample_matrix[12, :] = underlying_price + 5  # Future price
    
    # Simulate bid-ask for CE and PE
    for i, strike in enumerate(strikes):
        moneyness = (strike - underlying_price) / underlying_price
        
        # CE prices (decrease with strike)
        ce_mid = max(5, 100 - abs(moneyness * 500))
        sample_matrix[0, i] = ce_mid - 1  # CE_BID
        sample_matrix[1, i] = ce_mid + 1  # CE_ASK
        
        # PE prices (increase with strike)
        pe_mid = max(5, 50 + abs(moneyness * 500))
        sample_matrix[2, i] = pe_mid - 1  # PE_BID
        sample_matrix[3, i] = pe_mid + 1  # PE_ASK
        
        # Volume and OI
        sample_matrix[4, i] = 1000 * (1 + abs(moneyness))  # CE_VOL
        sample_matrix[5, i] = 1200 * (1 + abs(moneyness))  # PE_VOL
        sample_matrix[6, i] = 5000  # CE_OI
        sample_matrix[7, i] = 6000  # PE_OI
    
    # Calculate Greeks
    calculator = MatrixGreeksCalculator(risk_free_rate=0.065, days_to_expiry=7)
    
    print("Calculating IV and Greeks...")
    extended_matrix = calculator.add_greeks_to_matrix(sample_matrix)
    
    print(f"Extended matrix shape: {extended_matrix.shape}")
    print(f"Channels: {extended_matrix.shape[0]} (13 original + 10 Greeks)")
    print(f"Strikes: {extended_matrix.shape[1]}\n")
    
    # Get ATM Greeks
    atm_greeks = calculator.get_atm_greeks(extended_matrix)
    print("ATM Greeks:")
    print(f"  Strike: {atm_greeks['atm_strike']:.2f}")
    print(f"  Underlying: {atm_greeks['underlying_ltp']:.2f}")
    print(f"\n  Call:")
    print(f"    IV: {atm_greeks['ce']['iv']:.2%}")
    print(f"    Delta: {atm_greeks['ce']['delta']:.3f}")
    print(f"    Gamma: {atm_greeks['ce']['gamma']:.6f}")
    print(f"    Theta: {atm_greeks['ce']['theta']:.2f}")
    print(f"    Vega: {atm_greeks['ce']['vega']:.2f}")
    print(f"\n  Put:")
    print(f"    IV: {atm_greeks['pe']['iv']:.2%}")
    print(f"    Delta: {atm_greeks['pe']['delta']:.3f}")
    print(f"    Gamma: {atm_greeks['pe']['gamma']:.6f}")
    print(f"    Theta: {atm_greeks['pe']['theta']:.2f}")
    print(f"    Vega: {atm_greeks['pe']['vega']:.2f}")

