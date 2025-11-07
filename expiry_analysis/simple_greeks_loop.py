"""
Simple continuous loop: API → Matrix → Greeks

Refactored for parallel execution and config-based setup.
"""
import sys
import os
import time
import numpy as np
import threading
from datetime import datetime
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expiry_analysis.chain_processor import OptionDataProcessor
from vol.matrix_greeks import MatrixGreeksCalculator
from utils.api_utils import get_option_chain
from expiry_analysis.config import (
    UNDERLYINGS, STRIKE_COUNT, POLL_INTERVAL,
    RISK_FREE_RATE, DAYS_TO_EXPIRY, WINDOW_SIZE
)


def process_underlying(underlying: str, processor: OptionDataProcessor, 
                      greeks_calc: MatrixGreeksCalculator, 
                      strike_count: int, interval: float, 
                      stop_event: threading.Event):
    """
    Process a single underlying continuously.
    Can be run in parallel threads for multiple underlyings.
    
    Args:
        underlying: Underlying symbol
        processor: OptionDataProcessor instance
        greeks_calc: MatrixGreeksCalculator instance
        strike_count: Number of strikes to fetch
        interval: Polling interval in seconds
        stop_event: Threading event to signal stop
    """
    while not stop_event.is_set():
        try:
            # Step 1: Fetch from API
            sym = f"NSE:{underlying}-EQ"
            resp = get_option_chain(sym, strike_count)
            
            # Step 2: Process and get latest matrix in one call
            latest_matrix = processor.process_and_get_latest(underlying, resp)
            if latest_matrix is None:
                continue
            
            # Step 3: Get Greeks with timing
            start_time = time.perf_counter()
            greeks_matrix = greeks_calc.get_greeks_only(latest_matrix)
            calc_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Step 4: Combine matrices
            combined_matrix = np.vstack([latest_matrix, greeks_matrix])  # (23, strikes)
            
            # Step 5: Display complete matrix
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] {underlying} | ⏱ {calc_time:.2f}ms")
            print_matrix(combined_matrix)
            
        except Exception as e:
            print(f"[Error] {underlying}: {e}")
        
        stop_event.wait(interval)


def print_matrix(matrix: np.ndarray):
    """
    Print complete matrix in readable format.
    
    Args:
        matrix: Combined matrix (23 channels × strikes)
    """
    from vol.matrix_greeks import CHANNEL_NAMES
    
    n_strikes = matrix.shape[1]
    strikes = matrix[10, :]  # STRIKE channel
    
    # Print header
    print(f"{'Channel':<20} " + " ".join([f"S{i:2d}({strikes[i]:.0f})" for i in range(n_strikes)]))
    print("-" * 80)
    
    # Print each channel
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        if ch_idx < matrix.shape[0]:
            values = matrix[ch_idx, :]
            if ch_idx in [0, 1, 2, 3]:  # Prices
                print(f"{ch_name:<20} " + " ".join([f"{v:8.2f}" if not np.isnan(v) else "     nan" for v in values]))
            elif ch_idx in [4, 5, 6, 7]:  # Volume/OI
                print(f"{ch_name:<20} " + " ".join([f"{v:8.0f}" if not np.isnan(v) else "     nan" for v in values]))
            elif ch_idx in [8, 9]:  # OICH
                print(f"{ch_name:<20} " + " ".join([f"{v:8.0f}" if not np.isnan(v) else "     nan" for v in values]))
            elif ch_idx == 10:  # STRIKE
                print(f"{ch_name:<20} " + " ".join([f"{v:8.0f}" if not np.isnan(v) else "     nan" for v in values]))
            elif ch_idx in [11, 12]:  # Spot/Future
                print(f"{ch_name:<20} " + " ".join([f"{v:8.2f}" if not np.isnan(v) else "     nan" for v in values]))
            elif ch_idx in [13, 14]:  # IV
                print(f"{ch_name:<20} " + " ".join([f"{v:8.2%}" if not np.isnan(v) else "     nan" for v in values]))
            elif ch_idx in [15, 16]:  # Delta
                print(f"{ch_name:<20} " + " ".join([f"{v:8.3f}" if not np.isnan(v) else "     nan" for v in values]))
            elif ch_idx in [17, 18]:  # Gamma
                print(f"{ch_name:<20} " + " ".join([f"{v:8.6f}" if not np.isnan(v) else "     nan" for v in values]))
            elif ch_idx in [19, 20]:  # Theta
                print(f"{ch_name:<20} " + " ".join([f"{v:8.2f}" if not np.isnan(v) else "     nan" for v in values]))
            elif ch_idx in [21, 22]:  # Vega
                print(f"{ch_name:<20} " + " ".join([f"{v:8.2f}" if not np.isnan(v) else "     nan" for v in values]))


def run_parallel(underlyings: List[str], strike_count: int, interval: float,
                 risk_free_rate: float, days_to_expiry: int, window_size: int):
    """
    Run processing in parallel threads for multiple underlyings.
    
    Args:
        underlyings: List of underlying symbols
        strike_count: Number of strikes
        interval: Polling interval
        risk_free_rate: Risk-free rate for Greeks
        days_to_expiry: Days to expiry
        window_size: Window size for processor
    """
    # Shared instances (thread-safe for read operations)
    processor = OptionDataProcessor(window_size=window_size, strike_count=strike_count)
    greeks_calc = MatrixGreeksCalculator(risk_free_rate=risk_free_rate, days_to_expiry=days_to_expiry)
    
    stop_event = threading.Event()
    threads = []
    
    print(f"[Start] Processing {underlyings} every {interval}s")
    print("Press Ctrl+C to stop\n")
    
    # Start thread for each underlying
    for underlying in underlyings:
        thread = threading.Thread(
            target=process_underlying,
            args=(underlying, processor, greeks_calc, strike_count, interval, stop_event),
            name=f"Process-{underlying}",
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
            thread.join(timeout=1.0)


def main():
    """Main entry point - uses config file"""
    run_parallel(
        underlyings=UNDERLYINGS,
        strike_count=STRIKE_COUNT,
        interval=POLL_INTERVAL,
        risk_free_rate=RISK_FREE_RATE,
        days_to_expiry=DAYS_TO_EXPIRY,
        window_size=WINDOW_SIZE
    )


if __name__ == "__main__":
    main()

