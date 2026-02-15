from dataclasses import dataclass, field
from datetime import datetime
from typing import Union, Optional
import numpy as np

@dataclass
class OptionGreeks:
    """Vectorized Greeks for multiple options of same type."""
    iv: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    vega: np.ndarray
    vanna: np.ndarray
    volga: np.ndarray

    def __post_init__(self):
        lengths = [len(arr) for arr in [self.iv, self.delta, self.gamma, self.theta, self.vega, self.vanna, self.volga]]
        if len(set(lengths)) != 1:
            raise ValueError("All Greek arrays must have same length")

@dataclass
class SingleOptionGreeks:
    """Scalar Greeks for a single option."""
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    vanna: float
    volga: float
    
@dataclass
class OptionChainGreeks:
    """Complete Greeks for an option chain."""
    strikes: np.ndarray
    underlying_price: float
    expiry_date: datetime
    calls: OptionGreeks
    puts: OptionGreeks
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def n_strikes(self) -> int:
        return len(self.strikes)