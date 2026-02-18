# features/utils.py
from collections import defaultdict, deque
import numpy as np

class HistoricalIVStore:
    """Stores recent ATM IVs per symbol to compute percentiles."""
    
    def __init__(self, maxlen=100):
        self.store = defaultdict(lambda: deque(maxlen=maxlen))
    
    def update(self, symbol, atm_iv):
        self.store[symbol].append(atm_iv)
    
    def get_percentile(self, symbol, current_iv):
        hist = list(self.store[symbol])
        if len(hist) < 20:
            return np.nan
        return np.sum(np.array(hist) < current_iv) / len(hist)