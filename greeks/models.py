from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class OptionGreeks:
    """Greeks for a single option type (calls or puts)"""
    iv: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    vega: np.ndarray
    
    def __post_init__(self):
        # Ensure all arrays have same length
        lengths = [len(arr) for arr in [self.iv, self.delta, self.gamma, self.theta, self.vega]]
        if len(set(lengths)) != 1:
            raise ValueError("All Greek arrays must have same length")

@dataclass
class OptionChainGreeks:
    """Complete Greeks for an option chain (both calls and puts)"""
    strikes: np.ndarray
    underlying_price: float
    calls: OptionGreeks
    puts: OptionGreeks
    timestamp: Optional[float] = None
    
    @property
    def n_strikes(self) -> int:
        return len(self.strikes)