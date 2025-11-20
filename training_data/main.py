"""
Minimal script to collect live option chain data and Greek matrices for training.
10 stocks, every 3 seconds = 200 calls/minute (API limit).
"""
import sys
import os
import time
import threading
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
                      stop_event: threading.Event):
    """Collect data for one underlying."""
    while not stop_event.is_set():
        try:
            resp = get_option_chain(f"NSE:{underlying}-EQ", strike_count)
            matrix, spot, future = processor.create_matrix_from_response(resp)
            
            if matrix is not None:
                greeks = greeks_calc.get_greeks_only(matrix)
                save_matrices(underlying, matrix, greeks, time.time())
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {underlying} | Saved")
        except Exception as e:
            print(f"[Error] {underlying}: {e}")
        
        stop_event.wait(interval)


def main():
    """Main entry point."""
    processor = OptionDataProcessor(window_size=1, strike_count=STRIKE_COUNT)
    greeks_calc = MatrixGreeksCalculator(risk_free_rate=RISK_FREE_RATE, days_to_expiry=DAYS_TO_EXPIRY)
    
    stop_event = threading.Event()
    threads = []
    
    print(f"[Start] Collecting {len(UNDERLYINGS)} stocks every {POLL_INTERVAL}s")
    print(f"Rate: {len(UNDERLYINGS) * (60/POLL_INTERVAL):.0f} calls/min")
    print("Press Ctrl+C to stop\n")
    
    for underlying in UNDERLYINGS:
        thread = threading.Thread(
            target=process_underlying,
            args=(underlying, processor, greeks_calc, STRIKE_COUNT, POLL_INTERVAL, stop_event),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
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

