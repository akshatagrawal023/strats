"""
Option Chain Data Processor - Optimized for ML/DL Models

Stores option chain data in a matrix format (channels x strikes) for efficient feature engineering.
Uses strike-count based indexing (moneyness) instead of absolute strikes for better generalization.

Matrix Structure:
- Shape: (13 channels, 2*strike_count+1 strikes)
- Channels: CE_BID, CE_ASK, PE_BID, PE_ASK, CE_VOL, PE_VOL, CE_OI, PE_OI,
            CE_OICH, PE_OICH, STRIKE, UNDERLYING_LTP, FUTURE_PRICE
- Stored as 3D array: (time_steps, channels, strikes)

Key Features:
- Fixed window size for memory efficiency
- No dependency on absolute strike values
- NumPy arrays for fast vectorized operations
- HDF5 support for persistent storage
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_utils import get_option_chain

class OptionDataProcessor:
    def __init__(self, window_size=300, feature_window=20, strike_count=3):
        self.window_size = window_size
        self.feature_window = feature_window
        self.strike_count = strike_count  # n in get_option_chain(sym, n) -> 2n+1 strikes
        self.num_strikes = 2 * strike_count + 1
        self.data = {}  # underlying -> structured data
    
    def process_option_chain(self, underlying, resp):
        """Process option chain response and store data"""
        if not resp or resp.get('s') != 'ok':
            print(f"Invalid response for {underlying}")
            return
            
        data = resp['data']
        options = data.get('optionsChain', [])
        ts = time.time()
        
        # Initialize if not exists
        if underlying not in self.data:
            self.data[underlying] = {
                'timestamps': [],
                'matrix_data': []
            }
        
        store = self.data[underlying]
        
        # Create matrix for this timestamp
        # Channels: 0 CE_BID, 1 CE_ASK, 2 PE_BID, 3 PE_ASK, 
        #           4 CE_VOL, 5 PE_VOL, 6 CE_OI, 7 PE_OI,
        #           8 CE_OICH, 9 PE_OICH, 10 STRIKE, 11 UNDERLYING_LTP, 12 FUTURE_PRICE
        mat = np.full((13, self.num_strikes), np.nan, dtype=float)
        
        # First row contains underlying and future data
        if len(options) > 0:
            first_row = options[0]
            underlying_ltp = first_row.get('ltp', 0)
            future_price = first_row.get('fp', 0)  # Future price
            
            # Store underlying_ltp and future_price in all columns (same value across)
            mat[11, :] = underlying_ltp  # Underlying LTP channel
            mat[12, :] = future_price    # Future price channel
        
        # Process options in pairs (CE, PE) - skip first row
        si = 0  # Strike index (no dict needed!)
        for i in range(1, len(options), 2):
            if i + 1 >= len(options) or si >= self.num_strikes:
                break
            
            ce_row = options[i]      # Call
            pe_row = options[i + 1]  # Put
            
            strike = ce_row.get('strike_price')
            if strike is None:
                continue
            
            # Store strike value
            mat[10, si] = strike
            
            # Store CE data
            mat[0, si] = ce_row.get('bid', np.nan)
            mat[1, si] = ce_row.get('ask', np.nan)
            mat[4, si] = ce_row.get('volume', np.nan)
            mat[6, si] = ce_row.get('oi', np.nan)
            mat[8, si] = ce_row.get('oich', np.nan)
            
            # Store PE data
            mat[2, si] = pe_row.get('bid', np.nan)
            mat[3, si] = pe_row.get('ask', np.nan)
            mat[5, si] = pe_row.get('volume', np.nan)
            mat[7, si] = pe_row.get('oi', np.nan)
            mat[9, si] = pe_row.get('oich', np.nan)
            
            si += 1  # Next strike
        
        # Store matrix with timestamp
        store['timestamps'].append(ts)
        store['matrix_data'].append(mat)
        
        # Keep only last window_size snapshots
        if len(store['timestamps']) > self.window_size:
            store['timestamps'] = store['timestamps'][-self.window_size:]
            store['matrix_data'] = store['matrix_data'][-self.window_size:]
    
    def get_matrix(self, underlying, window=None):
        """Get matrix data for underlying with timestamps
        Returns: (timestamps, matrix_array)
        matrix_array shape: (time_steps, channels, strikes)
        """
        if underlying not in self.data:
            return None, None
            
        store = self.data[underlying]
        if not store.get('timestamps') or not store.get('matrix_data'):
            return None, None
        
        timestamps = store['timestamps']
        matrices = store['matrix_data']
        
        if window:
            timestamps = timestamps[-window:]
            matrices = matrices[-window:]
        
        # Stack matrices into 3D array
        matrix_array = np.stack(matrices, axis=0) if matrices else None
        return timestamps, matrix_array
    
    def print_matrix(self, underlying, show_full=False, channel_names=None):
        """Print matrix data in readable format"""
        if channel_names is None:
            channel_names = ['CE_BID', 'CE_ASK', 'PE_BID', 'PE_ASK', 
                           'CE_VOL', 'PE_VOL', 'CE_OI', 'PE_OI',
                           'CE_OICH', 'PE_OICH', 'STRIKE', 'UNDERLYING_LTP', 'FUTURE_PRICE']
        
        timestamps, matrix_array = self.get_matrix(underlying)
        if matrix_array is None:
            print(f"No matrix data for {underlying}")
            return
        
        store = self.data[underlying]
        num_timestamps = len(timestamps)
        
        # Get latest matrix (last time step)
        latest_mat = matrix_array[-1]  # shape: (channels, strikes)
        
        print(f"\n=== Matrix Data for {underlying} ===")
        print(f"Total snapshots: {num_timestamps}/{self.window_size}")
        print(f"Latest timestamp: {pd.to_datetime(timestamps[-1], unit='s')}")
        print(f"Matrix shape per snapshot: (channels={len(channel_names)}, strikes={latest_mat.shape[1]})")
        
        if show_full:
            # Show full matrix for latest snapshot
            print(f"\n--- Latest Snapshot ---")
            # Create DataFrame for better display
            df_mat = pd.DataFrame(latest_mat, 
                                 index=channel_names,
                                 columns=[f"S{i}" for i in range(latest_mat.shape[1])])
            print(df_mat)
        else:
            # Show compact view
            print(f"\n--- Sample (first 3 channels, first 5 strikes) ---")
            for ch_idx in range(min(3, len(channel_names))):
                print(f"{channel_names[ch_idx]}: {latest_mat[ch_idx, :5]}")
            print(f"\nSTRIKE: {latest_mat[10, :5]}")  # Show strikes
            print(f"UNDERLYING_LTP: {latest_mat[11, :5]}")  # Show underlying LTP
            print(f"FUTURE_PRICE: {latest_mat[12, :5]}")  # Show future price
    
    def save_to_hdf5(self, underlying, filepath):
        """Save matrix data to HDF5 for efficient storage and retrieval
        Usage: processor.save_to_hdf5('RELIANCE', 'data/reliance.h5')
        """
        timestamps, matrix_array = self.get_matrix(underlying)
        if matrix_array is None:
            print(f"No data to save for {underlying}")
            return
        
        with pd.HDFStore(filepath, mode='a') as hdf:
            # Save matrix as dataset
            hdf.put(f'{underlying}/matrix', 
                   pd.DataFrame(matrix_array.reshape(len(timestamps), -1),
                               index=pd.to_datetime(timestamps, unit='s')))
            
            # Save metadata
            hdf.get_storer(f'{underlying}/matrix').attrs.shape = matrix_array.shape
            hdf.get_storer(f'{underlying}/matrix').attrs.strike_count = self.strike_count
        
        print(f"Saved {underlying} data to {filepath}")

# Usage example
processor = OptionDataProcessor(window_size=300, feature_window=20, strike_count=3)


# Process option chain data
def process_underlying_data(underlying, resp):
    processor.process_option_chain(underlying, resp)
    processor.print_matrix(underlying)

# Example usage in your loop
iteration = 0
while True:
    iteration += 1
    print(f"\n{'='*50}")
    print(f"ITERATION {iteration}")
    print(f"{'='*50}")
    
    for underlying in ["RELIANCE", "HDFCBANK"]:
        sym = f"NSE:{underlying}-EQ"
        resp = get_option_chain(sym, 3)
        process_underlying_data(underlying, resp)
    
    # Show overall status with matrix info
    print(f"\n=== OVERALL STATUS ===")
    for underlying in ["RELIANCE", "HDFCBANK"]:
        if underlying in processor.data:
            store = processor.data[underlying]
            num_snapshots = len(store.get('timestamps', []))
            print(f"{underlying}: {num_snapshots}/{processor.window_size} snapshots")
    
    time.sleep(2)  # Adjust frequency as needed