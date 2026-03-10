# pipeline/feature_pipeline.py
"""
Orchestrates the entire data → Greeks → features pipeline.
Runs every 3 seconds, manages state, handles errors.
"""
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from collections import deque

from data.fetcher import FyersDataFetcher
from greeks.processor import GreeksProcessor
from features.matrix_features import FastMatrixFeatures
from utils.historical_store import HistoricalSkewStore  # We'll create this

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SymbolPipelineState:
    """State maintained for each symbol across iterations."""
    symbol: str
    last_matrix: Optional[np.ndarray] = None
    last_greeks: Optional[np.ndarray] = None
    feature_history: deque = field(default_factory=lambda: deque(maxlen=300))
    price_history: deque = field(default_factory=lambda: deque(maxlen=300))
    
    def update(self, matrix: np.ndarray, greeks: np.ndarray, features: Dict):
        """Update state with new data."""
        self.last_matrix = matrix
        self.last_greeks = greeks
        self.feature_history.append(features)
        self.price_history.append(matrix[11, 0])  # underlying price

class FeaturePipeline:
    """
    Main orchestration class for the feature generation pipeline.
    Runs continuously, manages state per symbol, handles errors.
    """
    
    def __init__(self, symbols: List[str], 
                 risk_free_rate: float = 0.065,
                 poll_interval: float = 3.0):
        """
        Args:
            symbols: List of symbols to track (e.g., ['NIFTY', 'BANKNIFTY'])
            risk_free_rate: For Greeks calculation
            poll_interval: Seconds between data fetches
        """
        self.symbols = symbols
        self.poll_interval = poll_interval
        self.running = False
        
        # Initialize components
        self.fetcher = FyersDataFetcher(max_concurrent=len(symbols))
        self.greeks_processor = GreeksProcessor(risk_free_rate=risk_free_rate)
        self.historical_store = HistoricalSkewStore()  # For skew percentiles
        
        # State per symbol
        self.state: Dict[str, SymbolPipelineState] = {
            sym: SymbolPipelineState(symbol=sym) for sym in symbols
        }
        
        # Feature cache for ML model (last 5 minutes = 100 iterations at 3s)
        self.feature_cache = deque(maxlen=100)
        
    async def run(self):
        """Main pipeline loop."""
        logger.info(f"Starting feature pipeline for symbols: {self.symbols}")
        self.running = True
        
        async with self.fetcher:  # Ensures proper cleanup
            while self.running:
                try:
                    start_time = datetime.now()
                    
                    # Step 1: Fetch all symbols concurrently
                    matrices = await self._fetch_all_symbols()
                    
                    # Step 2: Process each symbol
                    all_features = {}
                    for symbol, matrix in matrices.items():
                        if matrix is not None:
                            features = await self._process_symbol(symbol, matrix)
                            all_features[symbol] = features
                    
                    # Step 3: Log/Store features
                    self._store_features(all_features, start_time)
                    
                    # Step 4: Log progress
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Pipeline iteration completed in {elapsed:.2f}s")
                    
                    # Wait for next iteration
                    await asyncio.sleep(max(0, self.poll_interval - elapsed))
                    
                except Exception as e:
                    logger.exception(f"Pipeline error: {e}")
                    await asyncio.sleep(1)  # Brief pause before retry
    
    async def _fetch_all_symbols(self) -> Dict[str, Optional[np.ndarray]]:
        """Fetch matrices for all symbols concurrently."""
        try:
            # Your fetcher returns dict symbol -> matrix
            matrices = await self.fetcher.fetch_multiple_chains(
                self.symbols, strike_count=1  # Adjust as needed
            )
            return matrices
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return {sym: None for sym in self.symbols}
    
    async def _process_symbol(self, symbol: str, matrix: np.ndarray) -> Dict[str, float]:
        """
        Process one symbol through the pipeline.
        matrix: 13-channel raw data matrix
        Returns feature dictionary.
        """
        try:
            state = self.state[symbol]
            
            # Step 1: Calculate Greeks (extends matrix to 23 channels)
            # Using your existing method
            expiry_date = self._get_expiry_for_symbol(symbol)  # You need this
            greeks_matrix = self.greeks_processor.get_matrix_greeks(
                is_call=False,  # You might need both call/put separately
                matrix=matrix,
                days_to_expiry=7  # Compute from expiry_date
            )
            
            # Step 2: Extract underlying price for momentum
            underlying = matrix[11, 0]
            state.price_history.append(underlying)
            
            # Step 3: Calculate features using FastMatrixFeatures
            T = self.greeks_processor._get_T(None)  # Get time to expiry
            feature_extractor = FastMatrixFeatures(
                matrix=greeks_matrix,  # Use the extended matrix with Greeks
                T=T,
                r=self.greeks_processor.risk_free_rate
            )
            
            # Update price history in feature extractor for momentum
            feature_extractor.price_history = state.price_history
            
            # Extract all features
            features = {}
            
            # Core features from earlier discussion
            features.update(feature_extractor.skew_dislocation(self.historical_store))
            features.update(feature_extractor.momentum_features())
            
            # Add additional features you want
            features['atm_iv'] = (
                greeks_matrix[13, feature_extractor.atm_idx] + 
                greeks_matrix[14, feature_extractor.atm_idx]
            ) / 2
            
            features['bid_ask_spread_call'] = np.nanmean(
                (matrix[1, :] - matrix[0, :]) / ((matrix[1, :] + matrix[0, :])/2)
            )
            
            # Update historical store for future skew calculations
            self.historical_store.update(symbol, features)
            
            # Update symbol state
            state.update(matrix, greeks_matrix, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return {}
    
    def _store_features(self, all_features: Dict[str, Dict], timestamp: datetime):
        """
        Store features for ML model consumption.
        You can save to:
        - In-memory cache for real-time model
        - Database for training data
        - File for backtesting
        """
        # Add to rolling cache for model inference
        self.feature_cache.append({
            'timestamp': timestamp,
            'features': all_features
        })
        
        # Optionally save to database
        # self.db.insert(all_features, timestamp)
        
    def get_ml_input(self, symbol: str) -> Optional[np.ndarray]:
        """
        Get features for ML model in the right format.
        Returns array of shape (n_samples, n_features) for last N iterations.
        """
        if len(self.feature_cache) < 10:
            return None
            
        # Extract features for specific symbol
        symbol_features = []
        for entry in self.feature_cache:
            if symbol in entry['features']:
                feat_dict = entry['features'][symbol]
                # Convert to fixed-order array
                feature_vector = [
                    feat_dict.get('skew_z_-0.050', 0),
                    feat_dict.get('skew_z_0.050', 0),
                    feat_dict.get('skew_dislocation_score', 0),
                    feat_dict.get('momentum_1min', 0),
                    feat_dict.get('momentum_5min', 0),
                    feat_dict.get('acceleration', 0),
                    feat_dict.get('move_efficiency', 0),
                    feat_dict.get('vol_regime', 1.0),
                    feat_dict.get('atm_iv', 0),
                    feat_dict.get('bid_ask_spread_call', 0)
                ]
                symbol_features.append(feature_vector)
        
        return np.array(symbol_features) if symbol_features else None
    
    def stop(self):
        """Stop the pipeline gracefully."""
        self.running = False
        logger.info("Pipeline stopping...")

# Helper class for historical skew storage
class HistoricalSkewStore:
    """Simple in-memory store for historical skew values."""
    
    def __init__(self, maxlen: int = 1000):
        self.store = {}
        self.maxlen = maxlen
        
    def update(self, symbol: str, features: Dict):
        """Store skew values for this symbol."""
        if symbol not in self.store:
            self.store[symbol] = {}
            
        for key, value in features.items():
            if 'skew_' in key and not np.isnan(value):
                if key not in self.store[symbol]:
                    self.store[symbol][key] = deque(maxlen=self.maxlen)
                self.store[symbol][key].append(value)
    
    def get(self, key: str, default=None):
        """Get historical values for a specific skew metric."""
        # This is simplified - in practice you'd need symbol-specific
        for symbol_data in self.store.values():
            if key in symbol_data:
                return list(symbol_data[key])
        return default