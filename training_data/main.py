
import sys
import os
import time
import numpy as np
import threading
from datetime import datetime
from typing import List
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expiry_analysis.chain_processor import OptionDataProcessor
from vol.matrix_greeks import MatrixGreeksCalculator
from utils.api_utils import get_option_chain
from expiry_analysis.config import (
    UNDERLYINGS, STRIKE_COUNT, POLL_INTERVAL,
    RISK_FREE_RATE, DAYS_TO_EXPIRY, WINDOW_SIZE
)
from training_data.data_base import save_matrices

# Rate limiter: max 200 calls per minute
_rate_limiter = {'lock': threading.Lock(), 'calls': deque()}

def _wait_for_rate_limit():
    """Ensure we don't exceed 200 calls per minute"""
    with _rate_limiter['lock']:
        now = time.time()
        # Remove calls older than 60 seconds
        while _rate_limiter['calls'] and now - _rate_limiter['calls'][0] > 60:
            _rate_limiter['calls'].popleft()
        
        # If at limit, wait
        if len(_rate_limiter['calls']) >= 200:
            wait_time = 60 - (now - _rate_limiter['calls'][0]) + 0.1
            if wait_time > 0:
                time.sleep(wait_time)
            # Clean again after wait
            now = time.time()
            while _rate_limiter['calls'] and now - _rate_limiter['calls'][0] > 60:
                _rate_limiter['calls'].popleft()
        
        _rate_limiter['calls'].append(time.time())


def process_underlying(underlying: str, processor: OptionDataProcessor, 
                      greeks_calc: MatrixGreeksCalculator, 
                      strike_count: int, interval: float, 
                      stop_event: threading.Event):
    
    while not stop_event.is_set():
        try:
            # Rate limiting
            _wait_for_rate_limit()
            
            # Step 1: Fetch from API
            sym = f"NSE:{underlying}-EQ"
            resp = get_option_chain(sym, strike_count)
            
            # Step 2: Get matrix only (no windowing)
            latest_matrix, spot, future = processor.create_matrix_from_response(resp)
            if latest_matrix is None:
                continue
            
            # Step 3: Get Greeks with timing
            start_time = time.perf_counter()
            greeks_matrix = greeks_calc.get_greeks_only(latest_matrix)
            calc_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Step 4: Save matrices for training
            save_matrices(underlying, latest_matrix, greeks_matrix, time.time())
            
            # Step 5: Display status
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {underlying} | ⏱ {calc_time:.2f}ms | 💾 Saved")
            
        except Exception as e:
            print(f"[Error] {underlying}: {e}")
        
        stop_event.wait(interval)


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

