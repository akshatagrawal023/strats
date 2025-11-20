"""
Utility to load collected training data from NPZ files.

Example usage:
    from expiry_analysis.load_training_data import load_underlying_data
    
    data = load_underlying_data("training_data/RELIANCE")
    print(f"Loaded {len(data)} snapshots")
    print(f"Matrix shape: {data[0]['combined_matrix'].shape}")
"""
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional


def load_underlying_data(underlying_dir: str) -> List[Dict]:
    """
    Load all snapshots for an underlying from NPZ files.
    
    Args:
        underlying_dir: Path to underlying directory (e.g., "training_data/RELIANCE")
    
    Returns:
        List of dictionaries, each containing:
            - 'timestamp': Unix timestamp
            - 'datetime': ISO datetime string
            - 'base_matrix': Base matrix (13, strikes)
            - 'greeks_matrix': Greeks matrix (10, strikes)
            - 'combined_matrix': Combined matrix (23, strikes)
            - 'underlying': Underlying symbol
    """
    underlying_dir = Path(underlying_dir)
    metadata_file = underlying_dir / "metadata.json"
    
    if not metadata_file.exists():
        return []
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    data = []
    underlying = underlying_dir.name
    
    # Load each snapshot
    for entry in metadata:
        snapshot_id = entry['snapshot_id']
        npz_file = underlying_dir / f"{snapshot_id}.npz"
        
        if not npz_file.exists():
            continue
        
        # Load NPZ file
        npz_data = np.load(npz_file)
        
        snapshot = {
            'timestamp': float(npz_data['timestamp']),
            'datetime': entry['datetime'],
            'underlying': underlying,
            'base_matrix': npz_data['base_matrix'],
            'greeks_matrix': npz_data['greeks_matrix'],
            'combined_matrix': npz_data['combined_matrix']
        }
        
        data.append(snapshot)
    
    # Sort by timestamp
    data.sort(key=lambda x: x['timestamp'])
    
    return data


def load_all_underlyings(data_dir: str = "training_data") -> Dict[str, List[Dict]]:
    """
    Load data for all underlyings in a directory.
    
    Args:
        data_dir: Directory containing underlying subdirectories
    
    Returns:
        Dictionary mapping underlying -> list of snapshots
    """
    data_dir = Path(data_dir)
    all_data = {}
    
    # Find all underlying directories
    for underlying_dir in data_dir.iterdir():
        if underlying_dir.is_dir() and (underlying_dir / "metadata.json").exists():
            underlying = underlying_dir.name
            all_data[underlying] = load_underlying_data(str(underlying_dir))
    
    return all_data


def create_training_windows(
    data: List[Dict],
    window_size: int = 300,
    step_size: int = 1
) -> List[np.ndarray]:
    """
    Create rolling windows from collected snapshots for training.
    
    Args:
        data: List of snapshots (must be sorted by timestamp)
        window_size: Number of timesteps per window
        step_size: Step size between windows
    
    Returns:
        List of 3D arrays (window_size, channels, strikes)
    """
    if len(data) < window_size:
        return []
    
    windows = []
    
    for i in range(0, len(data) - window_size + 1, step_size):
        window_data = [snapshot['combined_matrix'] for snapshot in data[i:i+window_size]]
        window_array = np.array(window_data)  # (window_size, channels, strikes)
        windows.append(window_array)
    
    return windows


# Example usage
if __name__ == "__main__":
    import sys
    import os
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load data for one underlying
    data_dir = "training_data/RELIANCE"
    
    if Path(data_dir).exists():
        data = load_underlying_data(data_dir)
        print(f"Loaded {len(data)} snapshots from {data_dir}")
        
        if len(data) > 0:
            print(f"\nFirst snapshot:")
            print(f"  Timestamp: {data[0]['timestamp']}")
            print(f"  Datetime: {data[0]['datetime']}")
            print(f"  Base matrix shape: {data[0]['base_matrix'].shape}")
            print(f"  Greeks matrix shape: {data[0]['greeks_matrix'].shape}")
            print(f"  Combined matrix shape: {data[0]['combined_matrix'].shape}")
            
            # Create windows
            windows = create_training_windows(data, window_size=300)
            print(f"\nCreated {len(windows)} training windows (window_size=300)")
            if len(windows) > 0:
                print(f"  Window shape: {windows[0].shape}")
    else:
        print(f"Directory not found: {data_dir}")
        print("Run collect_training_data.py first to collect data")

