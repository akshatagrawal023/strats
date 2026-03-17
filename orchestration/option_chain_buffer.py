import numpy as np
from collections import deque
from typing import Optional, Tuple
import threading
import time

class OptionChainBuffer:
    def __init__(self, buffer_seconds: int = 300,  # 5 minutes
                 interval_seconds: int = 3,        # Every 3 seconds
                 n_channels: int = 23,              
                 max_strikes: int = 50):           
        
        self.buffer_size = buffer_seconds // interval_seconds
        self.n_channels = n_channels
        self.max_strikes = max_strikes
        
        # Main 3D buffer: (time, channel, strike)
        self.buffer = np.full((self.buffer_size, n_channels, self.max_strikes), 
                              np.nan, dtype=np.float64)
        
        # Metadata for each time slot
        self.timestamps = deque(maxlen=self.buffer_size)
        self.underlying_prices = deque(maxlen=self.buffer_size)
        self.expiry_dates = deque(maxlen=self.buffer_size)
        
        # Current write position (circular buffer)
        self.current_idx = 0
        self.is_filled = False
        
        # Thread safety for concurrent access
        self.lock = threading.RLock()
        
    def add_matrix(self, raw_matrix: np.ndarray, 
                   timestamp: float, 
                   underlying: float,
                   expiry: str) -> None:
        """
        Add raw API matrix mapped to the buffer.
        """
        with self.lock:
            # Ensure matrix has correct dimensions
            if raw_matrix.shape != (self.n_channels, self.max_strikes):
                matrix = self._normalize_matrix(raw_matrix)
            else:
                matrix = raw_matrix
                
            # Add to circular buffer
            self.buffer[self.current_idx] = matrix
            self.timestamps.append(timestamp)
            self.underlying_prices.append(underlying)
            self.expiry_dates.append(expiry)
            
            # Update write position
            self.current_idx = (self.current_idx + 1) % self.buffer_size
            
            # Mark as filled after first full cycle
            if not self.is_filled and len(self.timestamps) == self.buffer_size:
                self.is_filled = True
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Handle variable strike counts by padding/truncating."""
        current_strikes = matrix.shape[1]
        normalized = np.full((self.n_channels, self.max_strikes), np.nan)
        
        if current_strikes >= self.max_strikes:
            # Take ATM strikes (middle of the chain)
            start = (current_strikes - self.max_strikes) // 2
            normalized[:, :] = matrix[:, start:start + self.max_strikes]
        else:
            # Pad with NaN on both sides, keeping strikes centered
            pad = (self.max_strikes - current_strikes) // 2
            normalized[:, pad:pad + current_strikes] = matrix
            
        return normalized
    
    def get_time_slice(self, lookback: int = None) -> np.ndarray:
        """
        Get most recent time slices in chronological order.
        Returns: Array of shape (lookback, n_channels, n_strikes)
        """
        with self.lock:
            n_avail = len(self.timestamps)
            if n_avail == 0:
                return None
                
            if lookback is None or lookback > n_avail:
                lookback = n_avail
            
            # Get indices in chronological order
            if self.is_filled:
                indices = [(self.current_idx - lookback + i) % self.buffer_size 
                          for i in range(lookback)]
            else:
                indices = list(range(n_avail - lookback, n_avail))
            
            return self.buffer[indices]
    
    def get_feature_slice(self, feature_channels: list, 
                          lookback: int = None) -> np.ndarray:
        """
        Extract specific channels for feature calculation.
        """
        full_slice = self.get_time_slice(lookback)
        if full_slice is None:
            return None
            
        return full_slice[:, feature_channels, :]
    
    def get_strike_slice(self, moneyness_targets: list,
                         lookback: int = None) -> np.ndarray:
        """
        Extract specific moneyness points across time exactly when required for ML.
        This allows keeping the original raw density for Greeks.
        """
        full_slice = self.get_time_slice(lookback)
        if full_slice is None:
            return None
            
        n_time = full_slice.shape[0]
        n_targets = len(moneyness_targets)
        
        # Get strikes from channel 10
        strikes = full_slice[:, 10, :]  # (time, strikes)
        
        # Get underlying prices from metadata
        with self.lock:
            n_avail = len(self.underlying_prices)
            lookback_idx = min(lookback or n_avail, n_avail)
            # Retrieve chronological underlying prices matching the slice
            if self.is_filled:
                indices = [(self.current_idx - lookback_idx + i) % self.buffer_size for i in range(lookback_idx)]
                underlyings = [self.underlying_prices[i] for i in indices]
            else:
                underlyings = list(self.underlying_prices)[-lookback_idx:]
                
        underlying = np.array(underlyings).reshape(-1, 1)  # (time, 1)
        
        # Calculate moneyness for each time
        moneyness = strikes / underlying
        
        result = np.full((n_time, n_targets, self.n_channels), np.nan)
        
        for t in range(n_time):
            for i, target in enumerate(moneyness_targets):
                # Find closest strike
                idx = np.argmin(np.abs(moneyness[t] - target))
                result[t, i, :] = full_slice[t, :, idx]
        
        return result