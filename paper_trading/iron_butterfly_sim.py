import asyncio
import numpy as np
import sys
import os
import time
import logging

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
from collections import deque

import datetime
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def is_past_equity_market_close():
    """Returns True if current IST time is past 15:30."""
    now = datetime.datetime.now()
    # Assuming local time is IST as per metadata (+05:30)
    if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
        return True
    return False


async def iron_butterfly_scanner(
    pipeline: ProductionHFTPipeline,
    broker: VirtualBroker,
    archiver: HDF5Archiver,
    symbol: str,
):
    qty = 200  # 10 lots of Nifty (25 qty/lot)

    # ----------------------------------------------------
    # QUANT TRACKING STATE — 1 hour of memory at 3s ticks
    # ----------------------------------------------------
    warmup_period = 20
    atm_iv_history = deque(maxlen=1200)
    skew_history   = deque(maxlen=1200)
    spot_history   = deque(maxlen=1200)

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

        # --- Pull buffer metadata under lock ---
        with buffer.lock:
            spot      = buffer.underlying_prices[-1]
            expiry_ts = buffer.expiry_dates[-1]
            T         = buffer.T_values[-1]
            timestamp = buffer.timestamps[-1]

        ce_bid, ce_ask = mat[0], mat[1]
        pe_bid, pe_ask = mat[2], mat[3]
        strikes        = mat[10]
        t1 = time.perf_counter()

        # -------------------------------------------------------
        # READ PRE-COMPUTED GREEKS FROM PIPELINE BUFFER
        # prod_pipeline computes the full 8-channel greeks matrix
        # for ALL strikes once per tick and stores it centrally.
        # All strategies share this — zero redundant computation.
        # Channels: [Delta, Gamma, Theta, Vega, Vanna, Volga, CE_IV, PE_IV,  moneyness, theta/vega ratio]
        # -------------------------------------------------------
        greeks_slice = buffer.get_greeks_slice(lookback=1)
        if greeks_slice is None:
            await asyncio.sleep(1)
            continue
        greeks_mat = greeks_slice[0]   # shape: (13, n_strikes)
        t2 = time.perf_counter()

        # True ATM = strike where moneyness (Ch 8) is closest to 1.0
        atm_idx = int(np.nanargmin(np.abs(greeks_mat[8] - 1.0)))

        # ATM IV from channel 6 (CE_IV)
        atm_iv = float(greeks_mat[6, atm_idx]) if atm_idx < greeks_mat.shape[1] else float('nan')

        # 25-Delta Skew adapts to spot movement
        skew = compute_25delta_skew(greeks_mat)

        # Catch full smile curve for tracking (a0=ATM, a1=Skew, a2=Curvature)
        smile_a0, smile_a1, smile_a2 = compute_smile_polynomial(greeks_mat)

        # Append to rolling histories
        if not np.isnan(atm_iv):
            atm_iv_history.append(atm_iv)
        if not np.isnan(skew):
            skew_history.append(float(skew))
        spot_history.append(float(spot))

        # ----------------------------------------------------
        # EWM-BASED SPOT VOLATILITY  (replaces first-vs-last)
        # ----------------------------------------------------
        spot_stats = compute_spot_ewm_volatility(spot_history)

        # ----------------------------------------------------
        # STATISTICAL EVALUATOR
        # ----------------------------------------------------
        panic_detected = False
        iv_z_score     = 0.0
        skew_z         = 0.0

        if len(atm_iv_history) >= warmup_period:
            panic_detected, skew_z = detect_panic(skew_history, skew, spot_stats)
            _, _, iv_z_score = compute_iv_zscore(atm_iv_history)

        t3 = time.perf_counter()

        # Build current_prices dict for MTM — only in Phase 2 to avoid wasted work
        current_prices = {}
        mtm_snapshot   = {}

        # ----------------------------------------------------
        # PHASE 1: ENTRY TRIGGER
        # ----------------------------------------------------
        if not positions_entered:
            if len(atm_iv_history) < warmup_period:
                logging.info(
                    f"[WARMUP] ({len(atm_iv_history)}/{warmup_period}) "
                    f"ATM_IV: {atm_iv:.4f} | Skew: {skew*100:.2f}% "
                    f"(reading from pipeline greeks buffer)"
                )
                await asyncio.sleep(2)
                continue

            logging.info(
                f"[SCAN] ATM: {int(strikes[atm_idx])} | IV_Z: {iv_z_score:.2f} | "
                f"Skew_Z: {skew_z:.2f} | EWM_Vol: {spot_stats['ewm_vol']:.2f} | "
                f"Panic: {panic_detected} | Smile(a1,a2): {smile_a1:.3f}, {smile_a2:.3f}"
            )

            if panic_detected or iv_z_score < 1.5:
                await asyncio.sleep(3)
                continue

            logging.info(
                f"[ALPHA TRIGGERED] IV Z-Score {iv_z_score:.2f} >= 1.5. "
                f"Panic clear. Executing Sniper Entry."
            )

            atm_strike = int(strikes[atm_idx])
            # Offsets [1, 2, 3] → widths 50/100/150 for Nifty (50-pt strike spacing)
            for offset in [1, 2, 3]:
                call_idx = atm_idx + offset
                put_idx  = atm_idx - offset

                otm_call_strike = int(strikes[call_idx])
                otm_put_strike  = int(strikes[put_idx])
                nominal_width   = otm_call_strike - atm_strike

                legs = [
                    {'symbol': f'CE_{atm_strike}',      'qty': qty, 'side': -1, 'price': ce_bid[atm_idx]},
                    {'symbol': f'PE_{atm_strike}',      'qty': qty, 'side': -1, 'price': pe_bid[atm_idx]},
                    {'symbol': f'CE_{otm_call_strike}', 'qty': qty, 'side':  1, 'price': ce_ask[call_idx]},
                    {'symbol': f'PE_{otm_put_strike}',  'qty': qty, 'side':  1, 'price': pe_ask[put_idx]},
                ]

                sim_id        = f"IB_W{nominal_width}"
                margin_locked = float(nominal_width * qty) * 1.5
                broker.virtual_place_basket(sim_id, margin_locked, legs)

            positions_entered = True

        # ----------------------------------------------------
        # PHASE 2: MTM TRACKING
        # ----------------------------------------------------
        else:
            # Build price dict only when needed (Phase 2 only)
            for i, k in enumerate(strikes):
                if not np.isnan(k):
                    current_prices[f"CE_{int(k)}"] = {'bid': ce_bid[i], 'ask': ce_ask[i]}
                    current_prices[f"PE_{int(k)}"] = {'bid': pe_bid[i], 'ask': pe_ask[i]}

            mtm_log = []
            for sim_id in list(broker.positions.keys()):
                pnl = broker.update_pnl(sim_id, current_prices)
                if pnl is not None:
                    mtm_log.append(f"{sim_id}: \u20b9{pnl:.2f}")
                    mtm_snapshot[sim_id] = pnl

            t4 = time.perf_counter()

            if mtm_log:
                logging.info(
                    f"LIVE MTM -> " + " | ".join(mtm_log) +
                    f" [Extract: {(t1-t0)*1000:.3f}ms | Greeks(buf): {(t2-t1)*1000:.3f}ms | "
                    f"Quant: {(t3-t2)*1000:.3f}ms | MTM: {(t4-t3)*1000:.3f}ms]"
                )

            # Flush broker PnL buffer periodically (~1 min)
            if len(broker._pnl_buffer) >= 20:
                broker.flush_pnl_csv()

        # ----------------------------------------------------
        # ARCHIVE TICK — non-blocking, queued to background thread
        # ----------------------------------------------------
        archiver.record_tick(
            timestamp    = float(timestamp),
            spot         = float(spot),
            expiry_ts    = float(expiry_ts),
            T            = float(T),
            atm_iv       = atm_iv if not np.isnan(atm_iv) else float('nan'),
            skew         = float(skew) if not np.isnan(skew) else float('nan'),
            iv_z_score   = float(iv_z_score),
            panic_flag   = panic_detected,
            spot_ewm_vol = float(spot_stats['ewm_vol']),
            raw_matrix   = mat,
            greeks_matrix= greeks_mat,
            mtm_pnl      = mtm_snapshot or None,
        )

        await asyncio.sleep(3)


async def main():
    symbols = ["NSE:NIFTY50-INDEX"]
    # symbols = ["MCX:CRUDEOIL25APR-FUT"]

    pipeline = ProductionHFTPipeline(
        symbols=symbols,
        strike_count=10,
        buffer_seconds=60,
        interval_seconds=3
    )

    broker   = VirtualBroker(start_capital=5_000_000.0)
    archiver = HDF5Archiver(output_dir="paper_trading_logs", flush_interval=60)

    pipeline_task = asyncio.create_task(pipeline.run())

    logging.info("Warming up buffers for 5 seconds...")
    await asyncio.sleep(5)

    scanner_task = asyncio.create_task(
        iron_butterfly_scanner(pipeline, broker, archiver, symbols[0])
    )

    while pipeline.running:
        if "NSE" in symbols[0] and is_past_equity_market_close():
            logging.info("Current time is past 15:30 IST. Equity market is closed. Stopping simulation.")
            pipeline.stop()
            break
        await asyncio.sleep(5)

    try:
        if not pipeline.running:
            # If we exited due to time check, wait for tasks to finish
            await asyncio.gather(pipeline_task, scanner_task, return_exceptions=True)
        else:
            await asyncio.gather(pipeline_task, scanner_task)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        broker.flush_pnl_csv()
        archiver.shutdown()
        logging.info("Paper trading session cleanly exited. HDF5 + CSV flushed.")


if __name__ == "__main__":
    asyncio.run(main())
