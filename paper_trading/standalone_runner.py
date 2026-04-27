import asyncio
import numpy as np
import os
import time
import logging
from collections import deque
from datetime import datetime

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from orchestration.prod_pipeline import ProductionHFTPipeline
from paper_trading.virtual_broker import VirtualBroker

from paper_trading.market_features import (
    compute_25delta_skew,
    compute_iv_zscore,
    compute_spot_ewm_volatility,
    detect_panic,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("StandaloneRunner")

class MarketState:
    def __init__(self):
        self.spot_history = deque(maxlen=1200)
        self.atm_iv_history = deque(maxlen=1200)
        self.skew_history = deque(maxlen=1200)

async def run_standalone(eval_func, symbol="NSE:NIFTY50-INDEX", qty=650):
    logger.info(f"Starting standalone test for {eval_func.__name__} on {symbol} (Qty: {qty})")
    
    pipeline = ProductionHFTPipeline(
        symbols=[symbol],
        strike_count=12,
        interval_seconds=1
    )
    
    broker = VirtualBroker(start_capital=1_000_000.0)
    pipeline_task = asyncio.create_task(pipeline.run())
    
    logger.info("Warming up pipeline for 5 seconds...")
    await asyncio.sleep(5)

    ms = MarketState()
    strategy_state = {}
    warmup_period = 20

    try:
        while pipeline.running:
            buffer = pipeline.buffers.get(symbol)
            if not buffer:
                await asyncio.sleep(1)
                continue
            
            latest_slice = buffer.get_time_slice(lookback=1)
            greeks_slice = buffer.get_greeks_slice(lookback=1)
            
            if latest_slice is None or greeks_slice is None:
                await asyncio.sleep(1)
                continue
                
            mat = latest_slice[0]
            greeks_mat = greeks_slice[0]
            
            with buffer.lock:
                spot = float(buffer.underlying_prices[-1])
                
            ce_bid, ce_ask = mat[0], mat[1]
            pe_bid, pe_ask = mat[2], mat[3]
            strikes        = mat[10]

            atm_idx = int(np.nanargmin(np.abs(greeks_mat[8] - 1.0)))
            atm_iv = float(greeks_mat[6, atm_idx])
            skew = compute_25delta_skew(greeks_mat)
            
            ms.spot_history.append(spot)
            if not np.isnan(atm_iv): ms.atm_iv_history.append(atm_iv)
            if not np.isnan(skew): ms.skew_history.append(skew)

            features = {
                'atm_idx': atm_idx,
                'iv_z_score': None,
                'skew_z': None,
                'panic': False,
                'momentum': None,
                'vol': None,
                'ce_iv_vel': greeks_mat[11, atm_idx],
                'pe_iv_vel': greeks_mat[12, atm_idx]
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

            current_prices = {}
            for i, k in enumerate(strikes):
                if not np.isnan(k):
                    current_prices[f"CE_{int(k)}"] = {'bid': ce_bid[i], 'ask': ce_ask[i]}
                    current_prices[f"PE_{int(k)}"] = {'bid': pe_bid[i], 'ask': pe_ask[i]}

            # Trigger the specific strategy
            eval_func(broker, symbol, qty, strikes, current_prices, greeks_mat, features, strategy_state)

            await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down standalone test.")
    finally:
        pipeline.stop()
        await pipeline_task
