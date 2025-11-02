"""
Simple continuous loop: API → Matrix → Greeks

Just what you need right now - no extra complexity.
"""
import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expiry_analysis.chain_processor import OptionDataProcessor
from vol.matrix_greeks import MatrixGreeksCalculator
from utils.api_utils import get_option_chain


def main():
    # Initialize
    processor = OptionDataProcessor(window_size=300, strike_count=3)
    greeks_calc = MatrixGreeksCalculator(risk_free_rate=0.065, days_to_expiry=7)
    
    underlyings = ["RELIANCE", "HDFCBANK"]
    interval = 2.0  # 2 seconds
    
    print(f"[Start] Fetching data every {interval}s for {underlyings}")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            for underlying in underlyings:
                try:
                    # Step 1: Fetch from API
                    sym = f"NSE:{underlying}-EQ"
                    resp = get_option_chain(sym, processor.strike_count)
                    
                    # Step 2: Process into matrix
                    processor.process_option_chain(underlying, resp)
                    
                    # Step 3: Get latest matrix
                    ts, matrices = processor.get_matrix(underlying, window=1)
                    if matrices is None:
                        continue
                    
                    latest_matrix = matrices[-1]  # Shape: (13, strikes)
                    
                    # Step 4: Get Greeks (returns only Greeks matrix, 10 channels)
                    greeks_matrix = greeks_calc.get_greeks_only(latest_matrix)
                    
                    # Find ATM index (closest strike to spot)
                    spot = latest_matrix[11, 0]
                    strikes = latest_matrix[10, :]
                    atm_idx = int(np.nanargmin(np.abs(strikes - spot)))
                    
                    # Display
                    print(f"[{underlying}] Matrix: {latest_matrix.shape} → Greeks: {greeks_matrix.shape} | "
                          f"Spot: {spot:.2f} | "
                          f"CE_IV_ATM: {greeks_matrix[0, atm_idx]:.2%} | "
                          f"PE_IV_ATM: {greeks_matrix[1, atm_idx]:.2%} | "
                          f"CE_Delta: {greeks_matrix[2, atm_idx]:.3f} | "
                          f"PE_Delta: {greeks_matrix[3, atm_idx]:.3f}")
                    
                except Exception as e:
                    print(f"[Error] {underlying}: {e}")
            
            print()  # Blank line between iterations
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n[Stop] Shutting down...")


if __name__ == "__main__":
    main()

