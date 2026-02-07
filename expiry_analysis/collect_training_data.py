"""
Minimal script to collect live option chain data and Greek matrices for training.

Saves each snapshot separately (no windowing) to files per underlying.
Runs in parallel threads for multiple underlyings.
"""
import sys
import os
import time
import numpy as np
import threading
from datetime import datetime
from pathlib import Path
from typing import List
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expiry_analysis.chain_processor import OptionDataProcessor
from greeks.matrix_greeks import MatrixGreeksCalculator
from utils.api_utils import get_option_chain
from expiry_analysis.config import (
    UNDERLYINGS, STRIKE_COUNT, POLL_INTERVAL,
    RISK_FREE_RATE, DAYS_TO_EXPIRY
)

class DataCollector:
    """Collects and saves option chain data + Greeks for training."""
    
    def __init__(self, underlying: str, output_dir: str = "training_data"):
        self.underlying = underlying
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create directory for this underlying
        self.underlying_dir = self.output_dir / underlying
        self.underlying_dir.mkdir(exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.underlying_dir / "metadata.json"
        self.file_lock = threading.Lock()
        
        # Initialize counters
        self.count = 0
        self.last_save_time = time.time()
        
        # Load existing metadata
        self.metadata = []
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
                self.count = len(self.metadata)
    
    def save_snapshot(self, timestamp: float, base_matrix: np.ndarray, greeks_matrix: np.ndarray):
        """
        Save a single snapshot to NPZ file.
        
        Args:
            timestamp: Unix timestamp
            base_matrix: Base matrix (13 channels, strikes)
            greeks_matrix: Greeks matrix (10 channels, strikes)
        """
        combined_matrix = np.vstack([base_matrix, greeks_matrix])  # (23, strikes)
        
        # Create filename with timestamp
        snapshot_id = f"snapshot_{int(timestamp * 1000)}"
        npz_file = self.underlying_dir / f"{snapshot_id}.npz"
        
        with self.file_lock:
            # Save matrices to NPZ
            np.savez_compressed(
                npz_file,
                base_matrix=base_matrix,
                greeks_matrix=greeks_matrix,
                combined_matrix=combined_matrix,
                timestamp=timestamp
            )
            
            # Update metadata
            metadata_entry = {
                'snapshot_id': snapshot_id,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                'num_channels': int(combined_matrix.shape[0]),
                'num_strikes': int(combined_matrix.shape[1])
            }
            self.metadata.append(metadata_entry)
            
            # Save metadata (every 10 snapshots to reduce I/O)
            if len(self.metadata) % 10 == 0:
                with open(self.metadata_file, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
        
        self.count += 1
        
        # Print progress every 10 saves
        if self.count % 10 == 0:
            elapsed = time.time() - self.last_save_time
            rate = 10 / elapsed if elapsed > 0 else 0
            print(f"[{self.underlying}] Saved {self.count} snapshots | Rate: {rate:.1f}/s")
            self.last_save_time = time.time()
    
    def finalize(self):
        """Save final metadata."""
        with self.file_lock:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)


def collect_underlying(
    underlying: str,
    strike_count: int,
    interval: float,
    risk_free_rate: float,
    days_to_expiry: int,
    output_dir: str,
    stop_event: threading.Event
):
    """
    Collect data for a single underlying in a separate thread.
    
    Args:
        underlying: Underlying symbol
        strike_count: Number of strikes
        interval: Polling interval
        risk_free_rate: Risk-free rate
        days_to_expiry: Days to expiry
        output_dir: Output directory
        stop_event: Threading event to signal stop
    """
    # Initialize components
    processor = OptionDataProcessor(window_size=1, strike_count=strike_count)  # No windowing needed
    greeks_calc = MatrixGreeksCalculator(risk_free_rate=risk_free_rate, days_to_expiry=days_to_expiry)
    collector = DataCollector(underlying, output_dir)
    
    print(f"[{underlying}] Started data collection")
    
    try:
        while not stop_event.is_set():
            try:
                # Fetch from API
                sym = f"NSE:{underlying}-EQ"
                resp = get_option_chain(sym, strike_count)
                
                # Process and get latest matrix
                latest_matrix = processor.process_and_get_latest(underlying, resp)
                if latest_matrix is None:
                    continue
                
                # Calculate Greeks
                greeks_matrix = greeks_calc.get_greeks_only(latest_matrix)
                
                # Save snapshot
                timestamp = time.time()
                collector.save_snapshot(timestamp, latest_matrix, greeks_matrix)
                
            except Exception as e:
                print(f"[Error] {underlying}: {e}")
            
            stop_event.wait(interval)
    finally:
        collector.finalize()
        print(f"[{underlying}] Stopped data collection. Total: {collector.count} snapshots")


def main():
    """Main entry point - runs parallel collection for all underlyings."""
    from expiry_analysis.config import UNDERLYINGS, STRIKE_COUNT, POLL_INTERVAL, RISK_FREE_RATE, DAYS_TO_EXPIRY
    
    output_dir = "training_data"
    stop_event = threading.Event()
    threads = []
    
    print(f"[Start] Collecting data for {UNDERLYINGS}")
    print(f"Output directory: {output_dir}")
    print(f"Interval: {POLL_INTERVAL}s")
    print("Press Ctrl+C to stop\n")
    
    # Start thread for each underlying
    for underlying in UNDERLYINGS:
        thread = threading.Thread(
            target=collect_underlying,
            args=(underlying, STRIKE_COUNT, POLL_INTERVAL, RISK_FREE_RATE, DAYS_TO_EXPIRY, output_dir, stop_event),
            name=f"Collect-{underlying}",
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    try:
        # Wait for all threads
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\n[Stop] Shutting down...")
        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)
        
        # Print summary
        print("\n[Summary]")
        for underlying in UNDERLYINGS:
            underlying_dir = Path(output_dir) / underlying
            metadata_file = underlying_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    count = len(metadata)
                    print(f"  {underlying}: {count} snapshots in {underlying_dir}")


if __name__ == "__main__":
    main()

