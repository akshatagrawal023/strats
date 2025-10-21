import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import time
from option_chain import get_option_chain
import sys
import os


class OptionDataProcessor:
    def __init__(self, window_size=300, feature_window=20):
        self.window_size = window_size
        self.feature_window = feature_window
        self.data = {}  # underlying -> structured data
        
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
            return
            
        data = resp['data']
        options = data.get('optionsChain', [])
        ts = time.time()
        
        # Initialize if not exists
        if underlying not in self.data:
            # Extract strikes from response
            strikes = sorted(set(row.get('strike_price') for row in options 
                              if row.get('strike_price') and row.get('strike_price') > 0))
            self.init_underlying(underlying, strikes)
        
        store = self.data[underlying]
        idx = store['index'] % self.window_size
        
        # Process each row in optionsChain
        for row in options:
            symbol = row.get('symbol', '')
            option_type = row.get('option_type', '')
            strike = row.get('strike_price')
            
            if option_type == '':  # Underlying or future
                if 'FUT' in symbol:
                    store['future'][idx] = [ts, row.get('ltp', 0), row.get('bid', 0), row.get('ask', 0)]
                else:  # Underlying EQ
                    store['underlying'][idx] = [ts, row.get('ltp', 0), row.get('bid', 0), row.get('ask', 0)]
            elif option_type in ['CE', 'PE'] and strike:
                if strike in store['strikes']:
                    strike_idx = store['strikes'].index(strike)
                    option_idx = strike_idx * 2 + (0 if option_type == 'CE' else 1)
                    store['options'][idx, option_idx] = [
                        ts, row.get('ltp', 0), row.get('bid', 0), row.get('ask', 0),
                        row.get('oi', 0), row.get('oich', 0)
                    ]
        
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

# Usage example
processor = OptionDataProcessor(window_size=300, feature_window=20)

# Process option chain data
def process_underlying_data(underlying, resp):
    processor.process_option_chain(underlying, resp)
    
    # Get rolling features
    features = processor.get_rolling_features(underlying)
    if features:
        print(f"Features for {underlying}:")
        print(f"  Underlying MA: {features['underlying_ma']:.2f}")
        print(f"  Future MA: {features['future_ma']:.2f}")
        print(f"  Options MA: {features['options_ma']}")
    
    # Get latest data
    latest = processor.get_latest_data(underlying)
    if latest:
        print(f"Latest underlying LTP: {latest['underlying'][1]}")
        print(f"Latest future LTP: {latest['future'][1]}")

# Example usage in your loop
while True:
    for underlying in ["RELIANCE", "HDFCBANK"]:
        sym = f"NSE:{underlying}-EQ"
        resp = get_option_chain(sym)
        process_underlying_data(underlying, resp)
    
    time.sleep(1)  # Adjust frequency as needed