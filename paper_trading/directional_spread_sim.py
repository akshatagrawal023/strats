import asyncio
import numpy as np
import sys
import os
import time
import logging
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestration.prod_pipeline import ProductionHFTPipeline
from paper_trading.virtual_broker import VirtualBroker
from paper_trading.market_features import (
    compute_25delta_skew,
    compute_smile_polynomial,
    compute_iv_zscore,
    compute_spot_ewm_volatility,
    detect_panic,
)
from paper_trading.hdf5_archiver import HDF5Archiver

import datetime
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def is_past_equity_market_close():
    """Returns True if current IST time is past 15:30."""
    now = datetime.datetime.now()
    # Assuming local time is IST as per metadata (+05:30)
    if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
        return True
    return False

async def directional_spread_scanner(
    pipeline: ProductionHFTPipeline,
    broker: VirtualBroker,
    archiver: HDF5Archiver,
    symbol: str,
):
    """
    Directional Vertical Spreads Scanner.
    Targets ~1% swings by exploiting short-term momentum pulses.
    
    Logic:
      - Buy Side: Buy ATM, Sell OTM (1-strike away)
      - Entry: abs(momentum) > 0.5 * ewm_vol
      - Exit: Profit Target or Stop Loss (MTM based)
    """
    qty = 200  # 4 lots of Nifty (50 qty/lot) or adjusts for Crude
    if "CRUDE" in symbol:
        qty = 100 # 1 lot Crude
        
    warmup_period = 20
    atm_iv_history = deque(maxlen=1200)
    skew_history   = deque(maxlen=1200)
    spot_history   = deque(maxlen=1200)

    # Momentum thresholds
    MOMENTUM_CONFIRMATION_MULT = 1.2  # momentum must be > 1.2 * ewm_vol
    
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
            spot      = buffer.underlying_prices[-1]
            expiry_ts = buffer.expiry_dates[-1]
            T         = buffer.T_values[-1]
            timestamp = buffer.timestamps[-1]

        ce_bid, ce_ask = mat[0], mat[1]
        pe_bid, pe_ask = mat[2], mat[3]
        strikes        = mat[10]
        t1 = time.perf_counter()

        # Get Greeks (pre-computed in pipeline)
        greeks_slice = buffer.get_greeks_slice(lookback=1)
        if greeks_slice is None:
            await asyncio.sleep(1)
            continue
        greeks_mat = greeks_slice[0] 
        t2 = time.perf_counter()

        # ATM Detection
        atm_idx = int(np.nanargmin(np.abs(greeks_mat[8] - 1.0)))
        atm_iv = float(greeks_mat[6, atm_idx])
        skew = compute_25delta_skew(greeks_mat)
        
        # Tracking
        spot_history.append(float(spot))
        if not np.isnan(atm_iv): atm_iv_history.append(atm_iv)
        if not np.isnan(skew): skew_history.append(skew)

        # Signal Generation
        spot_stats = compute_spot_ewm_volatility(spot_history)
        momentum = spot_stats['momentum']
        vol = spot_stats['ewm_vol']
        
        iv_mean, iv_std, iv_z_score = compute_iv_zscore(atm_iv_history)
        panic, skew_z = detect_panic(skew_history, skew, spot_stats)
        
        t3 = time.perf_counter()

        # Build prices for MTM
        current_prices = {}
        for i, k in enumerate(strikes):
            if not np.isnan(k):
                current_prices[f"CE_{int(k)}"] = {'bid': ce_bid[i], 'ask': ce_ask[i]}
                current_prices[f"PE_{int(k)}"] = {'bid': pe_bid[i], 'ask': pe_ask[i]}

        # ----------------------------------------------------
        # POSITION MANAGEMENT (EXIT)
        # ----------------------------------------------------
        mtm_log = []
        mtm_snapshot = {}
        for sim_id in list(broker.positions.keys()):
            pnl = broker.update_pnl(sim_id, current_prices)
            if pnl is not None:
                mtm_log.append(f"{sim_id}: \u20b9{pnl:.2f}")
                mtm_snapshot[sim_id] = pnl
                
                # Dynamic Exit Example:
                # 1. Take Profit at 1.5% of margin
                # 2. Stop Loss at 0.8% of margin
                margin = broker.positions[sim_id]['margin_locked']
                if pnl > margin * 0.015:
                    logging.info(f"[TAKE PROFIT] {sim_id} reached ₹{pnl:.2f} (1.5% margin). Closing.")
                    broker.virtual_close_all(sim_id, current_prices)
                elif pnl < -margin * 0.008:
                    logging.info(f"[STOP LOSS] {sim_id} at ₹{pnl:.2f} (0.8% loss). Closing.")
                    broker.virtual_close_all(sim_id, current_prices)

        # ----------------------------------------------------
        # ENTRY LOGIC (DIRECTIONAL)
        # ----------------------------------------------------
        if len(spot_history) >= warmup_period:
            # Entry condition: Momentum burst relative to recent volatility
            # Also filter for 'panic' (don't enter bullish if skew is spiking)
            is_bullish_burst = (momentum > vol * MOMENTUM_CONFIRMATION_MULT) and (not panic)
            is_bearish_burst = (momentum < -vol * MOMENTUM_CONFIRMATION_MULT)
            
            # Check if we already have an open trade for this bias
            has_bull_trade = any("BULL" in k for k in broker.positions.keys())
            has_bear_trade = any("BEAR" in k for k in broker.positions.keys())

            if is_bullish_burst and not has_bull_trade:
                # Enter Bull Call Spread: Buy ATM, Sell OTM (+1 strike)
                up_idx = atm_idx + 1
                if up_idx < len(strikes):
                    s_atm = int(strikes[atm_idx])
                    s_otm = int(strikes[up_idx])
                    legs = [
                        {'symbol': f'CE_{s_atm}', 'qty': qty, 'side':  1, 'price': ce_ask[atm_idx]},
                        {'symbol': f'CE_{s_otm}', 'qty': qty, 'side': -1, 'price': ce_bid[up_idx]},
                    ]
                    sim_id = f"BULL_SPREAD_{s_atm}_{s_otm}"
                    margin = (ce_ask[atm_idx] - ce_bid[up_idx]) * qty # Debit cost is the max risk
                    broker.virtual_place_basket(sim_id, margin, legs)
                    logging.info(f"[ENTRY] Bullish Momentum Burst: {momentum:.2f} > {vol:.2f}*1.2")

            elif is_bearish_burst and not has_bear_trade:
                # Enter Bear Put Spread: Buy ATM, Sell OTM (-1 strike)
                dn_idx = atm_idx - 1
                if dn_idx >= 0:
                    s_atm = int(strikes[atm_idx])
                    s_otm = int(strikes[dn_idx])
                    legs = [
                        {'symbol': f'PE_{s_atm}', 'qty': qty, 'side':  1, 'price': pe_ask[atm_idx]},
                        {'symbol': f'PE_{s_otm}', 'qty': qty, 'side': -1, 'price': pe_bid[dn_idx]},
                    ]
                    sim_id = f"BEAR_SPREAD_{s_atm}_{s_otm}"
                    margin = (pe_ask[atm_idx] - pe_bid[dn_idx]) * qty
                    broker.virtual_place_basket(sim_id, margin, legs)
                    logging.info(f"[ENTRY] Bearish Momentum Burst: {momentum:.2f} < -{vol:.2f}*1.2")

        t4 = time.perf_counter()
        if mtm_log:
            logging.info(f"LIVE MTM -> " + " | ".join(mtm_log))

        # Archive tick
        archiver.record_tick(
            timestamp=float(timestamp), spot=float(spot), expiry_ts=float(expiry_ts), T=float(T),
            atm_iv=atm_iv, skew=skew, iv_z_score=iv_z_score, panic_flag=panic,
            spot_ewm_vol=vol, raw_matrix=mat, greeks_matrix=greeks_mat,
            mtm_pnl=mtm_snapshot or None
        )
        
        await asyncio.sleep(3)

async def main():
    symbols = ["NSE:NIFTY50-INDEX"]
    # symbols = ["MCX:CRUDEOIL25APR-FUT"]

    pipeline = ProductionHFTPipeline(symbols=symbols, strike_count=10, interval_seconds=3)
    broker = VirtualBroker(start_capital=1_000_000.0)
    archiver = HDF5Archiver(output_dir="directional_logs")

    pipeline_task = asyncio.create_task(pipeline.run())
    await asyncio.sleep(5)
    scanner_task = asyncio.create_task(directional_spread_scanner(pipeline, broker, archiver, symbols[0]))

    while pipeline.running:
        if "NSE" in symbols[0] and is_past_equity_market_close():
            logging.info("Current time is past 15:30 IST. Equity market is closed. Stopping simulation.")
            pipeline.stop()
            break
        await asyncio.sleep(5)

    try:
        if not pipeline.running:
            await asyncio.gather(pipeline_task, scanner_task, return_exceptions=True)
        else:
            await asyncio.gather(pipeline_task, scanner_task)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        broker.flush_pnl_csv()
        archiver.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
