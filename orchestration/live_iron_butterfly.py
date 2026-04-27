import asyncio
import numpy as np
import sys
import os
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestration.prod_pipeline import ProductionHFTPipeline
from orchestration.order_manager import OrderManager
from utils.api_utils import place_order, cancel_order

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def live_iron_butterfly_execution(pipeline: ProductionHFTPipeline, order_manager: OrderManager, symbol: str):
    # Configuration
    qty = 325 # Jan 2026 NIFTY 50 Lot Size (65 * 5 lots)
    nominal_width = 100 
    
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
        
        with buffer.lock:
            spot = buffer.underlying_prices[-1]
            
        ce_bid, ce_ask = mat[0], mat[1]
        pe_bid, pe_ask = mat[2], mat[3]
        strikes = mat[10]
        
        if not positions_entered:
            num_strikes = mat.shape[1]
            mid_idx = num_strikes // 2
            atm_strike = int(strikes[mid_idx])
            
            # Use Index Offset to find width
            # For width 100 on NIFTY, offset = 2
            offset = 2 
            call_idx = mid_idx + offset
            put_idx = mid_idx - offset
            
            otm_call_strike = int(strikes[call_idx])
            otm_put_strike = int(strikes[put_idx])
            
            if np.isnan(ce_bid[mid_idx]) or np.isnan(ce_ask[call_idx]):
                await asyncio.sleep(1)
                continue
                
            logging.info(f"[LIVE] Identified Valid Structure. Spot: {spot:.2f} | ATM: {atm_strike}")
            
            # ----------------------------------------------------
            # STEP 1: WING-FIRST EXECUTION
            # ----------------------------------------------------
            
            # Calculate strict Mathematical Mid-Prices, exactly 1 decimal point format.
            long_ce_mid = round((ce_bid[call_idx] + ce_ask[call_idx]) / 2.0, 1)
            long_pe_mid = round((pe_bid[put_idx] + pe_ask[put_idx]) / 2.0, 1)
            
            # Build Limit Order payloads using Fyers SDK format
            wing_orders = [
                {
                    "symbol": f"NSE:NIFTY{atm_strike+nominal_width}CE", # Need exact expiration standardizer here in prod
                    "qty": qty,
                    "type": 1, # 1 = Limit Order
                    "side": 1, # Buy
                    "productType": "MARGIN",
                    "limitPrice": long_ce_mid,
                    "validity": "DAY",
                    "offlineOrder": False
                },
                {
                    "symbol": f"NSE:NIFTY{atm_strike-nominal_width}PE",
                    "qty": qty,
                    "type": 1, 
                    "side": 1, 
                    "productType": "MARGIN",
                    "limitPrice": long_pe_mid,
                    "validity": "DAY",
                    "offlineOrder": False
                }
            ]
            
            # Fire Async Multi-Order
            logging.info(f"[WING ENTRY] Emitting Limit Orders @ {long_ce_mid:.1f} and {long_pe_mid:.1f}...")
            # Note: For testing safety, this is strictly commented out.
            # response = place_order(wing_orders) 
            
            # Mocking Order IDs for demonstration as we do not fire to NSE in this version
            mock_order_ids = ["W_123456", "W_123457"]
            
            # ----------------------------------------------------
            # STEP 2: THE CHASER ALGORITHM
            # ----------------------------------------------------
            try:
                # Wait for Websocket to flag Status = 2 for both wings, max 5 seconds wait
                # await asyncio.wait_for(order_manager.wait_for_fills(mock_order_ids), timeout=5.0)
                logging.info("[CHASER] (Mock) Wait sequence passed. Both wings strictly filled via Event Socket.")
            except asyncio.TimeoutError:
                logging.warning("[CHASER TIMEOUT] 5 Seconds elapsed. Killing limit orders and repricing...")
                # cancel_order([{"id": oid} for oid in mock_order_ids])
                # continue (which restarts the while loop, recalculates the matrix mid, and places again automatically!)
                pass
                
            # ----------------------------------------------------
            # STEP 3: ATM SPREAD
            # ----------------------------------------------------
            # Only reached if Wings filled
            short_ce_mid = round((ce_bid[mid_idx] + ce_ask[mid_idx]) / 2.0, 1)
            short_pe_mid = round((pe_bid[mid_idx] + pe_ask[mid_idx]) / 2.0, 1)
            
            logging.info(f"[STRADDLE ENTRY] Wings locked. Emitting Short Limit Orders @ {short_ce_mid:.1f} and {short_pe_mid:.1f}...")
            # Fire Short limits ...
            
            positions_entered = True
            
        else:
            # Trailing stop loss logic / Exit conditions would live here...
            pass
            
        await asyncio.sleep(2)

async def main():
    symbols = ["NSE:NIFTY50-INDEX"]
    
    # Init Pipeline
    pipeline = ProductionHFTPipeline(
        symbols=symbols, 
        strike_count=25, 
        buffer_seconds=60, 
        interval_seconds=3
    )
    
    # Init Order Manager
    order_manager = OrderManager()
    
    pipeline_task = asyncio.create_task(pipeline.run())
    
    logging.info("Warming up buffers for 5 seconds...")
    await asyncio.sleep(5)
    
    scanner_task = asyncio.create_task(live_iron_butterfly_execution(pipeline, order_manager, symbols[0]))
    
    try:
         await asyncio.gather(pipeline_task, scanner_task)
    except KeyboardInterrupt:
         pipeline.stop()
         order_manager.ws_client.stop_sockets()
         logging.info("Live engine legally stopped.")

if __name__ == "__main__":
    asyncio.run(main())
