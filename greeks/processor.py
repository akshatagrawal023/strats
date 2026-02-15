import numpy as np
from typing import  Optional
from greeks import calculate_iv_vectorized, calculate_greeks_vectorized, calculate_greeks_scalar, implied_vol_newton
from models import OptionGreeks, SingleOptionGreeks
from datetime import date, datetime, timezone
import pytz
IST = pytz.timezone("Asia/Kolkata")

def days_to_expiry(expiry_date: date, reference_date: Optional[date] = None) -> float:
    """Return fractional days until expiry (can be negative if expired)."""
    if reference_date is None:
        reference_date = datetime.now(IST).date()
    delta = (expiry_date - reference_date).days
    return max(delta, 1e-6) 

def time_to_expiry_years(expiry_date: date, reference_date: Optional[date] = None) -> float:
    """Time to expiry in years (365-day convention)."""
    return days_to_expiry(expiry_date, reference_date) / 365.0

class GreeksProcessor:
   
    def __init__(self, risk_free_rate: float = 0.065):
        self.risk_free_rate = risk_free_rate

    def get_matrix_greeks(self, is_call: bool, matrix: np.ndarray, expiry_date: str,
                            ) -> np.ndarray:
        
        greeks = self.get_greeks_from_matrix(is_call, matrix, expiry_date)

        greeks_matrix = np.full((7, n_strikes), np.nan, dtype=np.float64)
        
        greeks_matrix[0, :] = greeks.iv
        greeks_matrix[1, :] = greeks.delta
        greeks_matrix[2, :] = greeks.gamma
        greeks_matrix[3, :] = greeks.theta
        greeks_matrix[4, :] = greeks.vega
        greeks_matrix[5, :] = greeks.vanna
        greeks_matrix[6, :] = greeks.volga
        
        return greeks_matrix
    
    def get_greeks_from_matrix(self, is_call: bool, matrix: np.ndarray, expiry_date: str,
                            ) -> np.ndarray:

        if matrix.shape[0] != 13:
            raise ValueError(f"Expected 13 channels, got {matrix.shape[0]}")
        
        n_strikes = matrix.shape[1]
        T = time_to_expiry_years(expiry_date)
        
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

        return self.get_greeks_vectorized(is_call, S, K, mid, T, self.risk_free_rate)
    
    def get_greeks_vectorized(self, is_call: bool, S_arr: np.ndarray, K_arr: np.ndarray, mid_arr: np.ndarray, T: float, r: float) -> OptionGreeks:

        iv = calculate_iv_vectorized(mid_arr, S_arr, K_arr, T, r, is_call)
        
        delta, gamma, theta, vega, vanna, volga = calculate_greeks_vectorized(
            S_arr, K_arr, T, r, iv, is_call
        )
        
        return OptionGreeks(iv =iv, delta = delta, gamma = gamma, theta = theta, vega = vega, vanna = vanna, volga = volga)

    def get_greeks_scalar(self, is_call: bool, S: float, K: float, T: float, r: float, mid: float) -> SingleOptionGreeks:

        iv = implied_vol_newton(mid, S, K, T, r, is_call)
        
        delta, gamma, theta, vega, vanna, volga = calculate_greeks_scalar(
            S, K, T, r, iv, is_call
        )
        
        return SingleOptionGreeks(iv, delta, gamma, theta, vega, vanna, volga)

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
    calculator = GreeksProcessor(risk_free_rate=0.065, days_to_expiry=7)

    extended_matrix = calculator.get_greeks_vectorized(sample_matrix)