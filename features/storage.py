import numpy as np
from collections import deque
import h5py  # or your DB of choice

class FeatureTimeSeries:
    """Stores feature matrices in rolling window + persistent storage."""
    
    def __init__(self, window_size: int = 300, db_path: str = "features.h5"):
        self.window = deque(maxlen=window_size)
        self.db_path = db_path
        
    def add(self, timestamp: float, matrix: np.ndarray):
        """Add new feature matrix to rolling window."""
        self.window.append((timestamp, matrix))
        
    def get_window(self) -> np.ndarray:
        """Get stacked window for ML input."""
        if len(self.window) < self.window.maxlen:
            return None
        return np.stack([m for _, m in self.window])
    
    def save_to_disk(self):
        """Persist to HDF5 for later training."""
        with h5py.File(self.db_path, 'a') as f:
            # Store with timestamp as key
            for timestamp, matrix in self.window:
                f.create_dataset(str(timestamp), data=matrix)