import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import time
from greeks import calculate_iv_vectorized, calculate_greeks_vectorized

class GreeksProcessor:
   
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