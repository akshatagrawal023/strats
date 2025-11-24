"""
Minimal code to load training data from HDF5 files.
"""
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
# from data_validation import validate_matrix


def load_training_data(symbol: str, data_dir: str = None):
    """
    Load training data from HDF5 file.
    
    Args:
        symbol: Stock symbol (e.g., "HDFCBANK")
        data_dir: Directory containing the HDF5 file (default: script directory)
    
    Returns:
        dict with keys: 'features', 'greeks', 'timestamps'
    """
    if data_dir is None:
        # Use directory where this script is located
        data_dir = Path(__file__).parent
    
    filepath = Path(data_dir) / f"{symbol}_training.h5"
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        data = {
            'features': f['features'][:],      # (n_snapshots, 11, strikes)
            'greeks': f['greeks'][:],          # (n_snapshots, 10, strikes)
            'timestamps': f['timestamps'][:]   # (n_snapshots,)
        }
    
    return data

def print_data_summary(symbol: str, data_dir: str = None, show_first_n: int = 5):
    """Print summary of loaded data."""
    data = load_training_data(symbol, data_dir)
    
    n_snapshots = len(data['timestamps'])
    n_to_show = min(show_first_n, n_snapshots)
    
    print(f"\n=== {symbol} Training Data ===")
    print(f"Total snapshots: {n_snapshots}")
    print(f"Features shape: {data['features'].shape}")  # (n, 11, strikes)
    print(f"Greeks shape: {data['greeks'].shape}")      # (n, 10, strikes)
    print(f"\nTime range:")
    print(f"  First: {datetime.fromtimestamp(data['timestamps'][0])}")
    print(f"  Last:  {datetime.fromtimestamp(data['timestamps'][-1])}")
    
    print(f"\n=== First {n_to_show} Snapshots ===")
    for i in range(n_to_show):
        print(f"\n--- Snapshot {i+1} ({datetime.fromtimestamp(data['timestamps'][i])}) ---")
        print(f"Features ({data['features'][i].shape[0]} channels × {data['features'][i].shape[1]} strikes):")
        print(data['features'][i])
        print(f"\nGreeks ({data['greeks'][i].shape[0]} channels × {data['greeks'][i].shape[1]} strikes):")
        print(data['greeks'][i])


if __name__ == "__main__":
    # Example usage
    symbol = "HDFCBANK"
    
    # Validate matrices
    # validation = validate_matrix(symbol)
    
    # Print summary
    print_data_summary(symbol, show_first_n=3)

