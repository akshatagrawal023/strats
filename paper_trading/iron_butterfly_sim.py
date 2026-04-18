import asyncio
import numpy as np
import sys
import os
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestration.prod_pipeline import ProductionHFTPipeline
from paper_trading.virtual_broker import VirtualBroker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def iron_butterfly_scanner(pipeline: ProductionHFTPipeline, broker: VirtualBroker, symbol: str):
    qty = 250 # 10 lots of Nifty
    
    # We will trigger entries once we have valid data
    positions_entered = False
    
    while pipeline.running:
        buffer = pipeline.buffers.get(symbol)
        if not buffer:
            await asyncio.sleep(1)
            continue
            
        latest_slice = buffer.get_time_slice(lookback=1)
        if latest_slice is None:
            await asyncio.sleep(1)
            continue
            
        mat = latest_slice[0]
        
        t0 = time.perf_counter()
        
        with buffer.lock:
            spot = buffer.underlying_prices[-1]
            
        ce_bid, ce_ask = mat[0], mat[1]
        pe_bid, pe_ask = mat[2], mat[3]
        strikes = mat[10]
        
        t1 = time.perf_counter()
        
        # Build dictionary mapping unique symbol strikes for the virtual broker PnL tracker
        current_prices = {}
        for i, k in enumerate(strikes):
            if not np.isnan(k):
                current_prices[f"CE_{int(k)}"] = {'bid': ce_bid[i], 'ask': ce_ask[i]}
                current_prices[f"PE_{int(k)}"] = {'bid': pe_bid[i], 'ask': pe_ask[i]}
                
        t2 = time.perf_counter()
        
        # Phase 1: Entry Trigger
        if not positions_entered:
            # 1. Structural Symmetry Optimization
            # The matrix is structurally symmetric and validated by the strict OptionChainBuffer.
            # ATM is logically guaranteed to be the exact geometric center of the array.
            num_strikes = mat.shape[1]
            mid_idx = num_strikes // 2
            atm_strike = int(strikes[mid_idx])
            
            logging.info(f"Targeting ATM Strike: {atm_strike} (Spot: {spot:.2f})")
            
            # 2. Adjacent Array Offsets
            # Instead of nominal values, we use strictly adjacent indices.
            # Index offsets [1, 2, 3] cleanly map to 50/100/150 for Nifty, or 100/200/300 for Sensex.
            strike_offsets = [1, 2, 3]
            
            for offset in strike_offsets:
                call_idx = mid_idx + offset
                put_idx = mid_idx - offset
                
                otm_call_strike = int(strikes[call_idx])
                otm_put_strike = int(strikes[put_idx])
                nominal_width = int(otm_call_strike - atm_strike)
                    
                # Build the 4-legged Iron Butterfly using O(1) direct geometric indexing
                legs = [
                    {'symbol': f'CE_{atm_strike}', 'qty': qty, 'side': -1, 'price': ce_bid[mid_idx]}, # Short CE
                    {'symbol': f'PE_{atm_strike}', 'qty': qty, 'side': -1, 'price': pe_bid[mid_idx]}, # Short PE
                    {'symbol': f'CE_{otm_call_strike}', 'qty': qty, 'side': 1, 'price': ce_ask[call_idx]}, # Long CE
                    {'symbol': f'PE_{otm_put_strike}', 'qty': qty, 'side': 1, 'price': pe_ask[put_idx]}  # Long PE
                ]
                
                sim_id = f"IB_W{nominal_width}"
                
                # Margin calculation estimation for 1 lot perfectly hedged iron fly on Nifty
                margin_locked = float(nominal_width * qty) * 1.5 
                
                broker.virtual_place_basket(sim_id, margin_locked, legs)
            
            positions_entered = True
            
        # Phase 2: MTM Execution & Tracking
        else:
            mtm_log = []
            for sim_id in list(broker.positions.keys()):
                pnl = broker.update_pnl(sim_id, current_prices)
                if pnl is not None:
                    mtm_log.append(f"{sim_id}: \u20b9{pnl:.2f}")
                    
            t3 = time.perf_counter()
            
            if mtm_log:
                logging.info(f"LIVE MTM -> " + " | ".join(mtm_log) + f" [Extract: {(t1-t0)*1000:.3f}ms | Dict Map: {(t2-t1)*1000:.3f}ms | MTM Calc: {(t3-t2)*1000:.3f}ms]")
                
        await asyncio.sleep(3)

async def main():
    symbols = ["NSE:NIFTY50-INDEX"]
    # Initialize main quantitative pipeline
    pipeline = ProductionHFTPipeline(
        symbols=symbols,
        strike_count=25,
        buffer_seconds=60,
        interval_seconds=3
    )
    
    # Initialize the Virtual Paper Broker Engine
    broker = VirtualBroker(start_capital=5000000.0) # 50 Lakhs Starting Capital
    
    pipeline_task = asyncio.create_task(pipeline.run())
    
    logging.info("Warming up buffers for 5 seconds...")
    await asyncio.sleep(5)
    
    scanner_task = asyncio.create_task(iron_butterfly_scanner(pipeline, broker, symbols[0]))
    
    try:
        await asyncio.gather(pipeline_task, scanner_task)
    except KeyboardInterrupt:
        pipeline.stop()
        logging.info("Paper trading session legally exited. CSV logged in paper_trading_logs.")

if __name__ == "__main__":
    asyncio.run(main())
