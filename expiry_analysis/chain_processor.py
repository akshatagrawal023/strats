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
    def __init__(self, window_size=300, feature_window=20):
        self.window_size = window_size
        self.feature_window = feature_window
        self.data = {}  # underlying -> structured data
        self.df_data = {}  # underlying -> list of DataFrame rows
        
    def init_underlying(self, underlying, strikes):
        """Initialize data structures for one underlying"""
        self.data[underlying] = {
            'underlying': np.full((self.window_size, 4), np.nan),  # [ts, ltp, bid, ask]
            'future': np.full((self.window_size, 4), np.nan),
            'options': np.full((self.window_size, len(strikes) * 2, 6), np.nan),  # [ts, ltp, bid, ask, oi, oich]
            'strikes': strikes,
            'index': 0
        }
    
    def process_option_chain(self, underlying, resp):
        """Process option chain response and store data"""
        if not resp or resp.get('s') != 'ok':
            print(f"Invalid response for {underlying}")
            return
            
        data = resp['data']
        options = data.get('optionsChain', [])
        ts = time.time()
        
        # Initialize if not exists - extract strikes directly from CE/PE pairs (no pre-iteration!)
        if underlying not in self.data:
            # Fast strike extraction: options[1], options[3], options[5]... are CE rows
            strikes = sorted([options[i].get('strike_price') for i in range(1, len(options), 2) 
                            if i < len(options) and options[i].get('strike_price')])
            self.init_underlying(underlying, strikes)
            
            # Initialize matrix and mappings
            self.data[underlying]['strike_to_idx'] = {s: i for i, s in enumerate(strikes)}
            self.data[underlying]['timestamps'] = []  # Store timestamps for indexing
            self.data[underlying]['matrix_data'] = []  # List of (timestamp, matrix_slice)
        
        store = self.data[underlying]
        strike_to_idx = store['strike_to_idx']
        idx = store['index'] % self.window_size
        
        # Create matrix for this timestamp (channels x strikes)
        num_strikes = len(store['strikes'])
        mat = np.full((8, num_strikes), np.nan, dtype=float)
        
        # Create minimal row data (keep DF small; matrices hold rich data)
        row_data = {'timestamp': ts}
        
        # Fast processing leveraging fixed pattern: first row = underlying/future, rest = CE/PE pairs
        if len(options) > 0:
            # First row is always underlying or future
            first_row = options[0]
            if 'FUT' not in first_row.get('symbol', ''):
                store['underlying'][idx] = [ts, first_row.get('ltp', 0), first_row.get('bid', 0), first_row.get('ask', 0)]
                row_data['underlying_ltp'] = first_row.get('ltp', 0)
        
        # Process options in pairs (CE, PE, CE, PE...) - skip first row (underlying)
        for i in range(1, len(options), 2):
            if i + 1 >= len(options):
                break
            
            ce_row = options[i]      # Call
            pe_row = options[i + 1]  # Put
            
            strike = ce_row.get('strike_price')
            if strike is None or strike not in strike_to_idx:
                continue
            
            si = strike_to_idx[strike]
            
            # Process CE (Call) - mat[channel, strike_index]
            ce_bid = ce_row.get('bid')
            ce_ask = ce_row.get('ask')
            ce_vol = ce_row.get('volume')
            ce_oi = ce_row.get('oi')
            
            if ce_bid is not None: mat[0, si] = ce_bid
            if ce_ask is not None: mat[1, si] = ce_ask
            if ce_vol is not None: mat[4, si] = ce_vol
            if ce_oi is not None:  mat[6, si] = ce_oi
            
            # Process PE (Put)
            pe_bid = pe_row.get('bid')
            pe_ask = pe_row.get('ask')
            pe_vol = pe_row.get('volume')
            pe_oich = pe_row.get('oich')
            
            if pe_bid is not None: mat[2, si] = pe_bid
            if pe_ask is not None: mat[3, si] = pe_ask
            if pe_vol is not None: mat[5, si] = pe_vol
            if pe_oich is not None: mat[7, si] = pe_oich
        
        # Store matrix with timestamp as index
        store['timestamps'].append(ts)
        store['matrix_data'].append(mat)
        
        # Keep only last window_size rows
        if len(store['timestamps']) > self.window_size:
            store['timestamps'] = store['timestamps'][-self.window_size:]
            store['matrix_data'] = store['matrix_data'][-self.window_size:]
        
        # Add to DataFrame data for this specific underlying
        if underlying not in self.df_data:
            self.df_data[underlying] = []
        
        self.df_data[underlying].append(row_data)
        
        # Keep only last 300 rows for this underlying
        if len(self.df_data[underlying]) > self.window_size:
            self.df_data[underlying] = self.df_data[underlying][-self.window_size:]
        
        store['index'] += 1
    
    def get_rolling_features(self, underlying, window=None):
        """Calculate rolling features for all symbols"""
        if underlying not in self.data:
            return None
            
        store = self.data[underlying]
        window = window or self.feature_window
        
        if store['index'] < window:
            return None
            
        start_idx = max(0, store['index'] - window)
        end_idx = store['index']
        
        # Get data slices
        underlying_data = store['underlying'][start_idx:end_idx, 1]  # LTP column
        future_data = store['future'][start_idx:end_idx, 1]
        options_data = store['options'][start_idx:end_idx, :, 1]  # All options LTP
        
        # Calculate features
        features = {
            'underlying_ma': np.nanmean(underlying_data),
            'underlying_std': np.nanstd(underlying_data),
            'underlying_vol': np.nanstd(underlying_data) / np.nanmean(underlying_data) if np.nanmean(underlying_data) > 0 else 0,
            
            'future_ma': np.nanmean(future_data),
            'future_std': np.nanstd(future_data),
            
            'options_ma': np.nanmean(options_data, axis=0),  # MA for each option
            'options_std': np.nanstd(options_data, axis=0),   # Std for each option
            'options_vol': np.nanstd(options_data, axis=0) / np.nanmean(options_data, axis=0),
            
            'strikes': store['strikes'],
            'timestamp': time.time()
        }
        
        return features
    
    def get_latest_data(self, underlying):
        """Get latest data point for all symbols"""
        if underlying not in self.data:
            return None
            
        store = self.data[underlying]
        latest_idx = (store['index'] - 1) % self.window_size
        
        return {
            'underlying': store['underlying'][latest_idx],
            'future': store['future'][latest_idx],
            'options': store['options'][latest_idx],
            'strikes': store['strikes']
        }
    
    def get_dataframe(self, underlying=None):
        """Get the accumulated DataFrame for a specific underlying or all underlyings"""
        if underlying:
            # Return DataFrame for specific underlying with timestamp as index
            if underlying not in self.df_data or not self.df_data[underlying]:
                return pd.DataFrame()
            df = pd.DataFrame(self.df_data[underlying])
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
            return df
        else:
            # Return combined DataFrame for all underlyings
            if not self.df_data:
                return pd.DataFrame()
            
            all_data = []
            for u, data in self.df_data.items():
                for row in data:
                    row_copy = row.copy()
                    row_copy['underlying'] = u
                    all_data.append(row_copy)
            
            df = pd.DataFrame(all_data) if all_data else pd.DataFrame()
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
            return df
    
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
    
    def print_matrix(self, underlying, channel_names=None):
        """Print latest matrix data in readable format"""
        if channel_names is None:
            channel_names = ['CE_BID', 'CE_ASK', 'PE_BID', 'PE_ASK', 
                           'CE_VOL', 'PE_VOL', 'CE_OI', 'PE_OICH']
        
        timestamps, matrix_array = self.get_matrix(underlying, window=1)
        if matrix_array is None:
            print(f"No matrix data for {underlying}")
            return
        
        store = self.data[underlying]
        strikes = store['strikes']
        
        # Get latest matrix (last time step)
        latest_mat = matrix_array[-1]  # shape: (channels, strikes)
        
        print(f"\n=== Matrix Data for {underlying} ===")
        print(f"Timestamp: {pd.to_datetime(timestamps[-1], unit='s')}")
        print(f"Strikes: {strikes}")
        print(f"\nMatrix shape: (channels={len(channel_names)}, strikes={len(strikes)})")
        
        for ch_idx, ch_name in enumerate(channel_names):
            print(f"\n{ch_name}:")
            print(latest_mat[ch_idx])
    
    def save_to_hdf5(self, underlying, filepath):
        """Save matrix data to HDF5 for efficient storage and retrieval
        Usage: processor.save_to_hdf5('RELIANCE', 'data/reliance.h5')
        """
        timestamps, matrix_array = self.get_matrix(underlying)
        if matrix_array is None:
            print(f"No data to save for {underlying}")
            return
        
        store = self.data[underlying]
        
        with pd.HDFStore(filepath, mode='a') as hdf:
            # Save matrix as dataset
            hdf.put(f'{underlying}/matrix', 
                   pd.DataFrame(matrix_array.reshape(len(timestamps), -1),
                               index=pd.to_datetime(timestamps, unit='s')))
            
            # Save metadata
            hdf.put(f'{underlying}/strikes', pd.Series(store['strikes']))
            hdf.get_storer(f'{underlying}/matrix').attrs.shape = matrix_array.shape
        
        print(f"Saved {underlying} data to {filepath}")

# Usage example
processor = OptionDataProcessor(window_size=300, feature_window=20)


# Process option chain data
def process_underlying_data(underlying, resp):
    processor.process_option_chain(underlying, resp)
    
    # Get and print DataFrame for this specific underlying
    df = processor.get_dataframe(underlying)
    if not df.empty:
        print(f"\n=== DataFrame for {underlying} (Rows: {len(df)}) ===")
        print(df.tail(3))  # Show last 3 rows
    else:
        print(f"DataFrame is empty for {underlying}!")
    
    # Print matrix data
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
        resp = get_option_chain(sym)
        process_underlying_data(underlying, resp)
    
    # Show overall DataFrame status
    print(f"\n=== OVERALL STATUS ===")
    total_rows = 0
    for underlying in ["RELIANCE", "HDFCBANK"]:
        df = processor.get_dataframe(underlying)
        rows = len(df)
        total_rows += rows
        print(f"{underlying}: {rows} rows")
    
    print(f"Total rows across all underlyings: {total_rows}")
    
    time.sleep(2)  # Adjust frequency as needed