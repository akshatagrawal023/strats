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
from paper_trading.strategies.VolTrade.scanner import evaluate_market_tick as eval_vt
from paper_trading.strategies.Vol2.scanner import evaluate_market_tick as eval_v2
from paper_trading.strategies.GammaScalp.scanner import evaluate_market_tick as eval_gs

# Import shared central features
from paper_trading.market_features import (
    compute_25delta_skew,
    compute_smile_spread,
    compute_rolling_zscore,
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
    ],
    force=True
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
        self.spread_history = deque(maxlen=1200)

async def main():
    symbols = ["NSE:NIFTY50-INDEX", "BSE:SENSEX-INDEX"]
    
    pipeline = ProductionHFTPipeline(
        symbols=symbols,
        strike_count=24,
        interval_seconds=1
    )
    
    broker = VirtualBroker(start_capital=1_000_000.0)
    
    # Use the new dedicated HDF5 archives directory inside paper_trading
    archiver_dir = os.path.join(CURRENT_DIR, "hdf5_data_archives")
    archivers = {sym: HDF5Archiver(symbol=sym, output_dir=archiver_dir, flush_interval=60) for sym in symbols}
    
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
        "VT": {sym: {} for sym in symbols},
        "VOL2": {sym: {} for sym in symbols},
        "GS": {sym: {} for sym in symbols},
    }

    warmup_period = 300
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
                    expiry_ts = float(buffer.expiry_dates[-1])
                    T_val = float(buffer.T_values[-1])
                    ts_now = float(buffer.timestamps[-1])
                    
                ce_bid, ce_ask = mat[0], mat[1]
                pe_bid, pe_ask = mat[2], mat[3]
                strikes        = mat[10]

                # 1. CENTRALIZED FEATURE CALCULATION
                # Find ATM index safely
                valid_strikes = ~np.isnan(greeks_mat[8])
                if not np.any(valid_strikes): continue
                
                atm_idx = int(np.nanargmin(np.abs(greeks_mat[8] - 1.0)))
                atm_iv = float(greeks_mat[6, atm_idx])
                skew = compute_25delta_skew(greeks_mat)
                spread = compute_smile_spread(greeks_mat, atm_idx, wing_offset=2)
                
                ms = symbol_states[symbol]
                ms.spot_history.append(spot)
                if not np.isnan(atm_iv): ms.atm_iv_history.append(atm_iv)
                if not np.isnan(skew): ms.skew_history.append(skew)
                if not np.isnan(spread): ms.spread_history.append(spread)

                # Initialize defaults if warmup not complete
                features = {
                    'atm_idx': atm_idx,
                    'iv_z_score': 0.0,
                    'smile_z': 0.0,
                    'skew_z': 0.0,
                    'panic': False,
                    'momentum': 0.0,
                    'vol': 0.0
                }
                
                if len(ms.spot_history) >= warmup_period:
                    spot_stats = compute_spot_ewm_volatility(ms.spot_history)
                    iv_z = compute_rolling_zscore(ms.atm_iv_history)
                    smile_z = compute_rolling_zscore(ms.spread_history)
                    panic, skew_z = detect_panic(ms.skew_history, skew, spot_stats)
                    
                    # Detect if today is expiry (T < 0.003 is roughly < 12 hours)
                    is_expiry_day = T_val < 0.005 

                    # Gamma Surface (Gamma/Theta Ratio)
                    ce_gamma = greeks_mat[1, atm_idx]
                    pe_gamma = greeks_mat[1, atm_idx] # PE_Gamma is essentially same as CE_Gamma for ATM
                    total_gamma = ce_gamma + pe_gamma

                    features.update({
                        'iv_z_score': iv_z,
                        'smile_z': smile_z,
                        'skew_z': skew_z,
                        'panic': panic,
                        'momentum': spot_stats['momentum'],
                        'acceleration': spot_stats['acceleration'],
                        'vol': spot_stats['ewm_vol'],
                        'is_expiry_day': is_expiry_day,
                        'atm_gamma': total_gamma
                    })

                # 2. HDF5 RECORDING (Optional but requested)
                archiver = archivers.get(symbol)
                if archiver:
                    mtm_snapshot = broker.get_latest_pnl_snapshot()
                    archiver.record_tick(
                        timestamp=ts_now,
                        spot=spot,
                        expiry_ts=expiry_ts,
                        T=T_val,
                        atm_iv=atm_iv,
                        skew=skew,
                        iv_z_score=features['iv_z_score'],
                        panic_flag=features['panic'],
                        spot_ewm_vol=features['vol'],
                        raw_matrix=mat,
                        greeks_matrix=greeks_mat,
                        mtm_pnl=mtm_snapshot
                    )

                # 3. Prepare common prices structure for PnL
                current_prices = {'spot': spot}
                for i, k in enumerate(strikes):
                    if not np.isnan(k):
                        current_prices[f"CE_{int(k)}"] = {'bid': ce_bid[i], 'ask': ce_ask[i]}
                        current_prices[f"PE_{int(k)}"] = {'bid': pe_bid[i], 'ask': pe_ask[i]}

                # Apply Quantity logic requested by user: 20 lots 
                # (Nifty 65 per lot -> 1300, Sensex 20 per lot -> 400)
                qty = 1300 if "NIFTY" in symbol else 400

                # 4. TRIGGER REACTIVE STRATEGIES
                eval_args = (broker, symbol, qty, strikes, current_prices, greeks_mat, features)
                
                # eval_ib(*eval_args, strategy_states["IB"][symbol])
                # eval_ds(*eval_args, strategy_states["DS"][symbol])
                eval_rs(*eval_args, strategy_states["RS"][symbol])
                eval_bs(*eval_args, strategy_states["BS"][symbol])
                eval_cs(*eval_args, strategy_states["CS"][symbol])
                eval_vt(*eval_args, strategy_states["VT"][symbol])
                eval_v2(*eval_args, strategy_states["VOL2"][symbol])
                eval_gs(*eval_args, strategy_states["GS"][symbol])

            # Global Print every 15s with strategy breakup
            now = time.time()
            if now - last_global_mtm_print >= 15:
                if broker.positions:
                    total_pnl = 0.0
                    strat_pnl = {}
                    
                    for sim_id, pos in broker.positions.items():
                        pnl = pos.get('unrealized_pnl', 0.0)
                        total_pnl += pnl
                        
                        s_name = "Other"
                        if "VT_IC" in sim_id: s_name = "VolTrade"
                        elif "SS_" in sim_id: s_name = "ShortStraddle"
                        elif "GS_LONG" in sim_id: s_name = "GammaScalp"
                        elif "VOL2" in sim_id: s_name = "Vol2"
                        elif "BACKSPREAD" in sim_id: s_name = "Backspread"
                        
                        strat_pnl[s_name] = strat_pnl.get(s_name, 0.0) + pnl
                    
                    breakup_str = " | ".join([f"{k}: {v:+.0f}" for k, v in strat_pnl.items()])
                    logger.info(f"[GLOBAL MTM] Open: {len(broker.positions)} | Total: ₹{total_pnl:+.2f} ({breakup_str})")
                last_global_mtm_print = now

            # Sleep briefly to yield loop
            await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        pipeline.stop()
        await pipeline_task
        broker.flush_pnl_csv()
        for arch in archivers.values():
            arch.shutdown()
        logger.info("All resources flushed. Orchestrator shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())
