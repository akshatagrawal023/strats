"""
Minimal script to collect live option chain data and Greek matrices for training.
10 stocks, every 3 seconds = 200 calls/minute (API limit).
"""
import sys
import os
import time
import threading
import numpy as np
from datetime import datetime
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expiry_analysis.chain_processor import OptionDataProcessor
from vol.matrix_greeks import MatrixGreeksCalculator
from utils.api_utils import get_option_chain
from expiry_analysis.config import (
    UNDERLYINGS, STRIKE_COUNT, POLL_INTERVAL,
    RISK_FREE_RATE, DAYS_TO_EXPIRY
)
from training_data.data_base import save_matrices


def process_underlying(underlying: str, processor: OptionDataProcessor, 
                      greeks_calc: MatrixGreeksCalculator, 
                      strike_count: int, interval: float, 
                      stop_event: threading.Event, fyers, start_delay: float = 0.0):
    """Collect data for one underlying with staggered start time."""
    if start_delay > 0:
        time.sleep(start_delay)
    
    while not stop_event.is_set():
        try:
            resp = get_option_chain(f"NSE:{underlying}-EQ", strike_count, fyers=fyers)
            if resp is None:
                print(f"[Warning] {underlying}: No response from API")
                stop_event.wait(interval)
                continue
                
            matrix, spot, future = processor.create_matrix_from_response(resp)
            
            if matrix is not None and spot is not None:
                # Augment matrix with spot and future (11 -> 13 channels)
                n_strikes = matrix.shape[1]
                spot_row = np.full((1, n_strikes), spot, dtype=float)
                future_row = np.full((1, n_strikes), future if not np.isnan(future) else spot, dtype=float)
                augmented_matrix = np.vstack([matrix, spot_row, future_row])  # (13, strikes)
                
                greeks = greeks_calc.get_greeks_only(augmented_matrix)
                save_matrices(underlying, matrix, greeks, time.time())
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {underlying} | Saved")
            else:
                print(f"[Warning] {underlying}: Invalid matrix or spot")
        except Exception as e:
            print(f"[Error] {underlying}: {e}")
            import traceback
            traceback.print_exc()
        
        stop_event.wait(interval)


def main():
    """Main entry point."""
    print("[Init] Initializing components...")
    processor = OptionDataProcessor(window_size=1, strike_count=STRIKE_COUNT)
    greeks_calc = MatrixGreeksCalculator(risk_free_rate=RISK_FREE_RATE, days_to_expiry=DAYS_TO_EXPIRY)
    
    # Authenticate BEFORE starting threads to avoid blocking
    print("[Auth] Authenticating with API...")
    try:
        from utils.fyers_instance import FyersInstance
        fyers = FyersInstance.get_instance()
        print("[Auth] Authentication successful")
        
        # Test API call to ensure it works
        print("[Test] Testing API connection...")
        test_resp = get_option_chain("NSE:RELIANCE-EQ", 1)
        if test_resp and test_resp.get('s') == 'ok':
            print("[Test] API connection successful")
        else:
            print(f"[Test] API test failed: {test_resp}")
            return
    except Exception as e:
        print(f"[Auth] Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    stop_event = threading.Event()
    threads = []
    num_stocks = len(UNDERLYINGS)
    
    # Stagger start times to evenly distribute calls
    stagger_step = POLL_INTERVAL / num_stocks
    
    print(f"[Start] Collecting {num_stocks} stocks every {POLL_INTERVAL}s")
    print(f"Rate: {num_stocks * (60/POLL_INTERVAL):.0f} calls/min (evenly spaced)")
    print("Press Ctrl+C to stop\n")
    
    for idx, underlying in enumerate(UNDERLYINGS):
        start_delay = idx * stagger_step
        thread = threading.Thread(
            target=process_underlying,
            args=(underlying, processor, greeks_calc, STRIKE_COUNT, POLL_INTERVAL, stop_event, fyers, start_delay),
            daemon=True
        )
        thread.start()
        threads.append(thread)
        print(f"[Thread] Started {underlying} (delay: {start_delay:.2f}s)")
    
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\n[Stop] Shutting down...")
        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)


if __name__ == "__main__":
    main()

