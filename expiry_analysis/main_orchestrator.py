"""
Main Trading System Orchestrator

Wires together all components:
- chain_processor: Ingests API data → Matrix storage
- runtime_pipeline: Async DB + Feature computation
- feature_pipeline: DL tensor + compact features generation
- Your model: Inference on features → Trading signals

Usage:
    system = TradingSystem(underlyings=["RELIANCE", "HDFCBANK"])
    system.on_signal = your_signal_handler  # Optional: handle signals
    system.start()  # Starts polling/websocket loop
"""
import sys
import os
import time
import threading
from typing import List, Optional, Callable
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expiry_analysis.chain_processor import OptionDataProcessor
from expiry_analysis.runtime_pipeline import AsyncSinks
from expiry_analysis.feature_pipeline import FeatureGenerator
from utils.api_utils import get_option_chain


class TradingSystem:
    """
    Complete trading system orchestrator.
    
    Data Flow:
    1. API/WebSocket → process_option_chain()
    2. chain_processor → Matrix (ring buffer)
    3. runtime_pipeline → Async DB write + Greeks computation
    4. feature_pipeline → DL tensor + compact features
    5. Your model → Trading signal
    """
    
    def __init__(self, 
                 underlyings: List[str],
                 strike_count: int = 3,
                 window_size: int = 300,
                 poll_interval: float = 2.0,
                 db_path: str = "expiry_analysis/stream_data.db",
                 risk_free_rate: float = 0.065,
                 days_to_expiry: int = 7):
        """
        Args:
            underlyings: List of underlying symbols (e.g., ["RELIANCE", "HDFCBANK"])
            strike_count: Number of strikes to fetch (n → 2n+1 strikes)
            window_size: Rolling window size for matrices
            poll_interval: Seconds between API polls (if using polling mode)
            db_path: SQLite database path
            risk_free_rate: Risk-free rate for Greeks calculation
            days_to_expiry: Days to expiry for Greeks calculation
        """
        self.underlyings = underlyings
        self.poll_interval = poll_interval
        self._running = False
        self._stop_event = threading.Event()
        
        # Step 1: Initialize processor (ingestion)
        self.processor = OptionDataProcessor(
            window_size=window_size,
            strike_count=strike_count
        )
        
        # Step 2: Setup async sinks (DB + Feature worker)
        self.sinks = AsyncSinks(
            db_path=db_path,
            risk_free_rate=risk_free_rate,
            days_to_expiry=days_to_expiry,
            feature_callback=self._on_features_ready  # Callback when features computed
        )
        
        # Step 3: Wire processor → sinks
        self.processor.on_snapshot = self.sinks.handle_snapshot
        
        # Step 4: Initialize feature generator (for on-demand feature extraction)
        self.feature_gen = FeatureGenerator(
            processor=self.processor,
            risk_free_rate=risk_free_rate,
            days_to_expiry=days_to_expiry
        )
        
        # User callbacks (optional)
        self.on_signal: Optional[Callable] = None  # Called with (underlying, signal_dict)
        self.on_features: Optional[Callable] = None  # Called with (underlying, tensor, features_df)
        
    def _on_features_ready(self, underlying: str, ts: float, extended_matrix: np.ndarray, metrics: dict):
        """
        Internal callback when features are ready from async worker.
        Extended matrix shape: (23 channels, strikes)
        """
        # Optionally notify user
        if callable(self.on_features):
            try:
                # Get compact features for this underlying
                features_df = self.feature_gen.build_compact_features(underlying, window=20)
                self.on_features(underlying, extended_matrix, features_df)
            except Exception:
                pass
    
    def start(self):
        """Start the system (polling mode)"""
        if self._running:
            return
        
        print(f"[System] Starting trading system for {self.underlyings}")
        
        # Start async workers
        self.sinks.start()
        self._running = True
        
        # Start polling loop
        threading.Thread(target=self._poll_loop, daemon=True, name="PollLoop").start()
        
        print("[System] System started. Press Ctrl+C to stop.")
        
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def _poll_loop(self):
        """Main polling loop - fetches data from API"""
        while not self._stop_event.is_set():
            for underlying in self.underlyings:
                try:
                    sym = f"NSE:{underlying}-EQ"
                    resp = get_option_chain(sym, self.processor.strike_count)
                    self.processor.process_option_chain(underlying, resp)
                except Exception as e:
                    print(f"[Error] Failed to fetch {underlying}: {e}")
            
            self._stop_event.wait(self.poll_interval)
    
    def process_snapshot(self, underlying: str, resp: dict):
        """
        Process a single snapshot (for WebSocket mode).
        Call this from your WebSocket handler.
        """
        self.processor.process_option_chain(underlying, resp)
    
    def get_latest_features(self, underlying: str, window: int = 64):
        """
        Get latest features on-demand (for model inference).
        
        Returns:
            (tensor, features_df) or (None, None) if not enough data
        """
        ts, tensor = self.feature_gen.build_dl_tensor(underlying, window=window)
        features_df = self.feature_gen.build_compact_features(underlying, window=20)
        
        return tensor, features_df
    
    def get_signal(self, underlying: str, model=None):
        """
        Generate trading signal from latest features.
        
        Args:
            underlying: Underlying symbol
            model: Your model (callable that takes tensor → signal)
        
        Returns:
            dict with signal, confidence, features
        """
        tensor, features_df = self.get_latest_features(underlying, window=64)
        
        if tensor is None or features_df is None or features_df.empty:
            return None
        
        signal = None
        confidence = 0.0
        
        # Use your model if provided
        if callable(model):
            try:
                # Prepare tensor for model (add batch dimension if needed)
                if tensor.ndim == 3:
                    tensor_batch = tensor[np.newaxis, ...]  # (1, time, channels, strikes)
                else:
                    tensor_batch = tensor
                
                # Model inference
                prediction = model(tensor_batch)
                
                # Convert to signal (adjust based on your model output)
                if isinstance(prediction, np.ndarray):
                    signal = 'BUY' if prediction[0] > 0.5 else 'SELL'
                    confidence = float(abs(prediction[0] - 0.5) * 2)  # 0-1 scale
            except Exception as e:
                print(f"[Error] Model inference failed: {e}")
        
        # Fallback: Use compact features for simple rule-based signal
        if signal is None:
            pcr_atm = features_df['pcr_atm'].iloc[0]
            pe_oi_mom = features_df['pe_oi_momentum'].iloc[0]
            
            if pcr_atm > 1.2 and pe_oi_mom > 0:
                signal = 'BUY'
                confidence = 0.6
            elif pcr_atm < 0.8:
                signal = 'SELL'
                confidence = 0.6
            else:
                signal = 'HOLD'
                confidence = 0.3
        
        result = {
            'underlying': underlying,
            'signal': signal,
            'confidence': confidence,
            'features': features_df.iloc[0].to_dict(),
            'timestamp': time.time()
        }
        
        # Notify user callback
        if callable(self.on_signal):
            try:
                self.on_signal(underlying, result)
            except Exception:
                pass
        
        return result
    
    def stop(self):
        """Stop the system gracefully"""
        print("[System] Stopping...")
        self._running = False
        self._stop_event.set()
        self.sinks.stop()
        print("[System] Stopped.")
    
    def status(self):
        """Print system status"""
        print(f"\n=== System Status ===")
        print(f"Underlyings: {self.underlyings}")
        print(f"Running: {self._running}")
        
        for underlying in self.underlyings:
            if underlying in self.processor.data:
                store = self.processor.data[underlying]
                count = store['count']
                print(f"{underlying}: {count}/{self.processor.window_size} snapshots")


# ============================================================================
# Example Usage
# ============================================================================

def example_signal_handler(underlying: str, signal_dict: dict):
    """Example signal handler - replace with your trading logic"""
    print(f"[Signal] {underlying}: {signal_dict['signal']} (confidence: {signal_dict['confidence']:.2f})")
    print(f"  Features: PCR={signal_dict['features']['pcr_atm']:.2f}, "
          f"PE_OI_mom={signal_dict['features']['pe_oi_momentum']:.2f}")

def example_feature_handler(underlying: str, tensor: np.ndarray, features_df):
    """Example feature handler - for logging/monitoring"""
    if features_df is not None and not features_df.empty:
        print(f"[Features] {underlying}: PCR={features_df['pcr_atm'].iloc[0]:.2f}")

if __name__ == "__main__":
    # Create system
    system = TradingSystem(
        underlyings=["RELIANCE", "HDFCBANK"],
        strike_count=3,
        poll_interval=2.0
    )
    
    # Optional: Add signal handler
    system.on_signal = example_signal_handler
    system.on_features = example_feature_handler
    
    # Start system (blocking)
    system.start()

