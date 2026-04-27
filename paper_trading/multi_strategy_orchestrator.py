import asyncio
import numpy as np
import os
import time
import logging
from datetime import datetime
from collections import deque

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from orchestration.prod_pipeline import ProductionHFTPipeline
from paper_trading.virtual_broker import VirtualBroker
from paper_trading.hdf5_archiver import HDF5Archiver

# Import reactive evaluators
from paper_trading.strategies.iron_butterfly.scanner import evaluate_market_tick as eval_ib
from paper_trading.strategies.directional_spread.scanner import evaluate_market_tick as eval_ds
from paper_trading.strategies.ratio_spread.scanner import evaluate_market_tick as eval_rs
from paper_trading.strategies.calendar_spread.scanner import evaluate_market_tick as eval_cs
from paper_trading.strategies.backspread.scanner import evaluate_market_tick as eval_bs

# Import shared central features
from paper_trading.market_features import (
    compute_25delta_skew,
    compute_iv_zscore,
    compute_spot_ewm_volatility,
    detect_panic,
)

SHARED_LOGS_DIR = os.path.join(CURRENT_DIR, "shared_logs")
os.makedirs(SHARED_LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(SHARED_LOGS_DIR, "orchestrator.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Orchestrator")

def is_past_equity_market_close():
    now = datetime.now()
    if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
        return True
    return False

# Manage per-symbol history centralized in Orchestrator
class MarketState:
    def __init__(self):
        self.spot_history = deque(maxlen=1200)
        self.atm_iv_history = deque(maxlen=1200)
        self.skew_history = deque(maxlen=1200)

async def main():
    symbols = ["NSE:NIFTY50-INDEX", "BSE:SENSEX-INDEX"]
    
    pipeline = ProductionHFTPipeline(
        symbols=symbols,
        strike_count=12,
        interval_seconds=1
    )
    
    broker = VirtualBroker(start_capital=1_000_000.0)
    archiver = HDF5Archiver(output_dir=SHARED_LOGS_DIR, flush_interval=60)
    
    pipeline_task = asyncio.create_task(pipeline.run())
    
    logger.info("Warming up shared pipeline for 5 seconds...")
    await asyncio.sleep(5)

    symbol_states = {sym: MarketState() for sym in symbols}
    strategy_states = {
        "IB": {sym: {} for sym in symbols},
        "DS": {sym: {} for sym in symbols},
        "RS": {sym: {} for sym in symbols},
        "BS": {sym: {} for sym in symbols},
        "CS": {sym: {} for sym in symbols},
    }

    warmup_period = 20
    last_global_mtm_print = time.time()

    logger.info("Starting Centralized Reactive Strategies...")
    
    try:
        while pipeline.running:
            if is_past_equity_market_close():
                logger.info("Market close reached (15:30 IST). Shutting down all strategies.")
                pipeline.stop()
                break
                
            for symbol in symbols:
                buffer = pipeline.buffers.get(symbol)
                if not buffer:
                    continue
                
                latest_slice = buffer.get_time_slice(lookback=1)
                greeks_slice = buffer.get_greeks_slice(lookback=1)
                
                if latest_slice is None or greeks_slice is None:
                    continue
                    
                mat = latest_slice[0]
                greeks_mat = greeks_slice[0]
                
                with buffer.lock:
                    spot = float(buffer.underlying_prices[-1])
                    
                ce_bid, ce_ask = mat[0], mat[1]
                pe_bid, pe_ask = mat[2], mat[3]
                strikes        = mat[10]

                # 1. CENTRALIZED FEATURE CALCULATION
                atm_idx = int(np.nanargmin(np.abs(greeks_mat[8] - 1.0)))
                atm_iv = float(greeks_mat[6, atm_idx])
                skew = compute_25delta_skew(greeks_mat)
                
                ms = symbol_states[symbol]
                ms.spot_history.append(spot)
                if not np.isnan(atm_iv): ms.atm_iv_history.append(atm_iv)
                if not np.isnan(skew): ms.skew_history.append(skew)

                # Initialize defaults if warmup not complete
                features = {
                    'atm_idx': atm_idx,
                    'iv_z_score': None,
                    'skew_z': None,
                    'panic': False,
                    'momentum': None,
                    'vol': None,
                    'ce_iv_vel': greeks_mat[12, atm_idx],
                    'pe_iv_vel': greeks_mat[13, atm_idx],
                    'charm': greeks_mat[11, atm_idx]
                }
                
                if len(ms.spot_history) >= warmup_period:
                    spot_stats = compute_spot_ewm_volatility(ms.spot_history)
                    _, _, iv_z = compute_iv_zscore(ms.atm_iv_history)
                    panic, skew_z = detect_panic(ms.skew_history, skew, spot_stats)
                    
                    features.update({
                        'iv_z_score': iv_z,
                        'skew_z': skew_z,
                        'panic': panic,
                        'momentum': spot_stats['momentum'],
                        'vol': spot_stats['ewm_vol']
                    })

                # Prepare common prices structure for PnL
                current_prices = {}
                for i, k in enumerate(strikes):
                    if not np.isnan(k):
                        current_prices[f"CE_{int(k)}"] = {'bid': ce_bid[i], 'ask': ce_ask[i]}
                        current_prices[f"PE_{int(k)}"] = {'bid': pe_bid[i], 'ask': pe_ask[i]}

                # Apply Quantity logic requested by user
                qty = 650 if "NIFTY" in symbol else 200

                # 2. TRIGGER REACTIVE STRATEGIES
                eval_args = (broker, symbol, qty, strikes, current_prices, greeks_mat, features)
                
                eval_ib(*eval_args, strategy_states["IB"][symbol])
                eval_ds(*eval_args, strategy_states["DS"][symbol])
                eval_rs(*eval_args, strategy_states["RS"][symbol])
                eval_bs(*eval_args, strategy_states["BS"][symbol])
                eval_cs(*eval_args, strategy_states["CS"][symbol])

            # Global Print every 15s instead of per-strategy
            now = time.time()
            if now - last_global_mtm_print >= 15:
                if broker.positions:
                    total_pnl = sum(p.get('unrealized_pnl', 0.0) for p in broker.positions.values())
                    logger.info(f"[GLOBAL MTM] Open Trades: {len(broker.positions)} | Total PnL: \u20b9{total_pnl:.2f}")
                last_global_mtm_print = now

            # Sleep briefly to yield loop
            await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        pipeline.stop()
        await pipeline_task
        broker.flush_pnl_csv()
        archiver.shutdown()
        logger.info("All resources flushed. Orchestrator shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())
