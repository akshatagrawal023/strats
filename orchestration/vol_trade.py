import asyncio
import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestration.prod_pipeline import ProductionHFTPipeline
from greeks.greeks import calculate_iv_vectorized, calculate_greeks_vectorized

async def monitor_and_calculate_greeks(pipeline: ProductionHFTPipeline):
    """Periodically pull the dense raw strike buffer and calculate Greeks."""
    
    while pipeline.running:
        for symbol, buffer in pipeline.buffers.items():
            
            # Get the latest raw snapshot
            latest_slice = buffer.get_time_slice(lookback=1)
            
            if latest_slice is not None:
                mat = latest_slice[0]  # Shape: (11, max_strikes)
                
                with buffer.lock:
                    spot = buffer.underlying_prices[-1]
                    expiry_ts = buffer.expiry_dates[-1]
                
                # Channels: 0 CE_BID, 1 CE_ASK, 2 PE_BID, 3 PE_ASK, 10 STRIKE
                ce_bid, ce_ask = mat[0], mat[1]
                pe_bid, pe_ask = mat[2], mat[3]
                strikes = mat[10]
                
                # Calculate mid prices
                ce_mid = np.nanmean([ce_bid, ce_ask], axis=0)
                pe_mid = np.nanmean([pe_bid, pe_ask], axis=0)
                
                # Filter valid strikes for CE
                valid_ce = ~np.isnan(ce_mid) & ~np.isnan(strikes) & (strikes > 0)
                if np.any(valid_ce):
                    v_ce_mid = ce_mid[valid_ce]
                    v_strikes = strikes[valid_ce]
                    v_spot = np.full(len(v_ce_mid), spot)
                    
                    # Calculate exact time to expiration (T) in Years
                    # We use max(..., 60) to ensure a minimum of 1 minute to avoid division by zero for 0DTE options at expiration
                    current_time = time.time()
                    time_to_expiry_sec = max(expiry_ts - current_time, 60.0)
                    T = time_to_expiry_sec / (365.25 * 24 * 3600)
                    r = 0.065  # Standardized RBI repo rate (Apr 2026) — matches market_features.RISK_FREE_RATE
                    
                    # Calculate CE IVs
                    ce_ivs = calculate_iv_vectorized(
                        prices=v_ce_mid, S=v_spot, K=v_strikes, 
                        T=T, r=r, is_call=True
                    )
                    
                    # Calculate CE Greeks
                    deltas, gammas, thetas, vegas, vannas, volgas = calculate_greeks_vectorized(
                        S=v_spot, K=v_strikes, T=T, r=r, sigma=ce_ivs, is_call=True
                    )
                    
                    print(f"[{symbol}] Latest Spot: {spot:.2f}")
                    print(f"  -> Calculated {len(v_ce_mid)} CE Greeks.")
                    print(f"  -> ATM Delta approx: {deltas[np.argmin(np.abs(v_strikes - spot))]:.3f}")
                    
        await asyncio.sleep(3)

async def main():
    symbols = ["NSE:NIFTY50-INDEX"]
    
    # Initialize pipeline
    pipeline = ProductionHFTPipeline(
        symbols=symbols,
        strike_count=25,
        buffer_seconds=60,  # 1 minute buffer for testing
        interval_seconds=3
    )
    
    # Start pipeline pulling data in background
    pipeline_task = asyncio.create_task(pipeline.run())
    
    print("Warming up buffers...")
    await asyncio.sleep(5)
    
    # Start consumer
    greeks_task = asyncio.create_task(monitor_and_calculate_greeks(pipeline))
    
    try:
        await asyncio.gather(pipeline_task, greeks_task)
    except KeyboardInterrupt:
        pipeline.stop()
        print("Pipeline stopped.")

if __name__ == "__main__":
    asyncio.run(main())