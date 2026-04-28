import logging
import os
import csv
import time
import numpy as np
from datetime import datetime

# Local logging setup for VolTrade
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
MTM_LOG = os.path.join(LOG_DIR, "mtm_live.csv")
TRADE_LOG = os.path.join(LOG_DIR, "trades_placed.csv")

# Ensure headers for CSVs
for log_file, headers in [(MTM_LOG, ["timestamp", "symbol", "sim_id", "pnl", "smile_z"]), 
                          (TRADE_LOG, ["timestamp", "symbol", "sim_id", "type", "price", "credit", "trigger"])]:
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(headers)

# Symbol-specific thresholds to handle different volatility and cost regimes
SYMBOL_CONFIG = {
    "NIFTY50": {"entry_z": 2.5, "exit_z": -0.4}, # 95th Percentile Entry | 10th Percentile Exit
    "SENSEX":   {"entry_z": 4.5, "exit_z": -1.3}, # Extreme Tail Entry | 10th Percentile Exit to cover costs
    "DEFAULT":  {"entry_z": 3.0, "exit_z": 0.0}
}

def evaluate_market_tick(
    broker, symbol, qty, strikes, current_prices, 
    greeks_mat, features, state_dict
):
    """
    VolTrade — Smile Spread Mean Reversion Strategy.
    Signal: Smile Spread Z-Score (ATM_IV - Wing_IV).
    """
    logger = logging.getLogger("VolTrade")
    smile_z = features.get('smile_z', 0.0)
    atm_idx = features.get('atm_idx')
    
    if atm_idx is None:
        return

    # Clean symbol for sim_id matching (e.g. NSE:NIFTY50-INDEX -> NIFTY50)
    sym_prefix = symbol.split(":")[1].split("-")[0]
    config = SYMBOL_CONFIG.get(sym_prefix, SYMBOL_CONFIG["DEFAULT"])

    # --- 0. SAFETY CIRCUIT BREAKERS ---
    consecutive_losses = state_dict.get('consecutive_losses', 0)
    if consecutive_losses >= 3:
        return # Locked (Circuit Breaker)

    # HARD EXPIRY EXIT: 2:50 PM Wall (Wait for next tick to flush)
    now_dt = datetime.now()
    is_expiry_day = features.get('is_expiry_day', False)
    if is_expiry_day and (now_dt.hour > 14 or (now_dt.hour == 14 and now_dt.minute >= 50)):
        for sim_id in list(broker.positions.keys()):
            if "VT_IC" in sim_id:
                logger.warning(f"[PANIC EXIT] {sim_id} 2:50 PM Wall reached. Closing all legs.")
                broker.virtual_close_all(sim_id, current_prices, features=features, trigger="EXP_PANIC")
        return # Stop all activity

    # No Expiry Afternoon entries (After 12:30 IST)
    can_enter = not (is_expiry_day and (now_dt.hour > 12 or (now_dt.hour == 12 and now_dt.minute >= 30)))

    # --- MTM TRACKING & LOGGING ---
    has_trade = False
    for sim_id in list(broker.positions.keys()):
        if sym_prefix in sim_id and "VT_IC" in sim_id:
            has_trade = True
            pnl = broker.update_pnl(sim_id, current_prices, features=features)
            if pnl is not None:
                # Log MTM to local strategy CSV
                try:
                    with open(MTM_LOG, "a", newline="") as f:
                        csv.writer(f).writerow([datetime.now(), symbol, sim_id, round(pnl, 2), round(smile_z, 2)])
                except Exception as e:
                    logger.error(f"Failed to write MTM log: {e}")
                
                # Check Stop Loss (1.5x Credit)
                pos = broker.positions[sim_id]
                entry_credit = pos.get('entry_credit', 0.0)
                if pnl < -entry_credit * 1.5:
                    logger.warning(f"[STOP LOSS] {sim_id} at 150%. Closing.")
                    final_pnl = broker.virtual_close_all(sim_id, current_prices, features=features, trigger="SL_150%")
                    state_dict['last_exit_time'] = time.time()
                    if final_pnl is not None and final_pnl < 0:
                        state_dict['consecutive_losses'] = state_dict.get('consecutive_losses', 0) + 1
                        logger.warning(f"[CIRCUIT BREAKER] Loss recorded. Counter: {state_dict['consecutive_losses']}/3")
                
                # Check Profit Taking (Custom exit per symbol)
                elif smile_z <= config["exit_z"]:
                    logger.info(f"[TAKE PROFIT] {sim_id} Reverted (Z={smile_z:.2f}).")
                    final_pnl = broker.virtual_close_all(sim_id, current_prices, features=features, trigger="REVERSION")
                    state_dict['last_exit_time'] = time.time()
                    if final_pnl is not None:
                        if final_pnl < 0:
                            state_dict['consecutive_losses'] = state_dict.get('consecutive_losses', 0) + 1
                            logger.warning(f"[CIRCUIT BREAKER] Loss recorded. Counter: {state_dict['consecutive_losses']}/3")
                        else:
                            state_dict['consecutive_losses'] = 0 # Reset on win
                continue

    # --- ENTRY TRIGGER (Symbol Specific) ---
    # COOLDOWN: 60-second lockout after any trade exit to prevent churn
    last_exit = state_dict.get('last_exit_time', 0)
    cooldown_active = (time.time() - last_exit) < 60
    
    if not has_trade and not cooldown_active and can_enter and smile_z >= config["entry_z"]:
        logger.info(f"[{symbol} ENTRY] Extreme Spike Z: {smile_z:.2f} >= {config['entry_z']}. Entering Delta-Neutral Iron Condor.")
        
        # --- DELTA-NEUTRAL LEG SELECTION ---
        # greeks_mat rows: 0:CE_Delta, 1:PE_Delta, 2:CE_Gamma, 3:PE_Gamma, 4:CE_Theta, 5:PE_Theta, 6:CE_IV, 7:PE_IV, 8:Strike
        ce_deltas = greeks_mat[0]
        pe_deltas = greeks_mat[1]
        
        try:
            # 1. Find the 0.5 Delta (ATM) strikes for both sides independently
            ce_atm_idx = np.nanargmin(np.abs(ce_deltas - 0.5))
            pe_atm_idx = np.nanargmin(np.abs(pe_deltas + 0.5))
            
            s_ce_atm = int(strikes[ce_atm_idx])
            s_pe_atm = int(strikes[pe_atm_idx])
            
            # 2. Find the 0.15 Delta (Wing) strikes independently
            ce_wing_idx = np.nanargmin(np.abs(ce_deltas - 0.15))
            pe_wing_idx = np.nanargmin(np.abs(pe_deltas + 0.15))
            
            s_ce_wing = int(strikes[ce_wing_idx])
            s_pe_wing = int(strikes[pe_wing_idx])

            # Fetch Prices
            ce_atm_bid = current_prices[f"CE_{s_ce_atm}"]['bid']
            pe_atm_bid = current_prices[f"PE_{s_pe_atm}"]['bid']
            ce_wing_ask = current_prices[f"CE_{s_ce_wing}"]['ask']
            pe_wing_ask = current_prices[f"PE_{s_pe_wing}"]['ask']
            
            net_credit_unit = (ce_atm_bid + pe_atm_bid) - (ce_wing_ask + pe_wing_ask)
            total_credit = net_credit_unit * qty
            
            legs = [
                {'symbol': f'CE_{s_ce_atm}', 'qty': qty, 'side': -1, 'price': ce_atm_bid},
                {'symbol': f'PE_{s_pe_atm}', 'qty': qty, 'side': -1, 'price': pe_atm_bid},
                {'symbol': f'CE_{s_ce_wing}', 'qty': qty, 'side':  1, 'price': ce_wing_ask},
                {'symbol': f'PE_{s_pe_wing}', 'qty': qty, 'side':  1, 'price': pe_wing_ask},
            ]
            
            sim_id = f"VT_IC_{sym_prefix}_{s_ce_atm}_{s_pe_atm}"
            # Estimate Margin: (CE Strike - PE Strike) * qty
            margin_req = (s_ce_atm - s_pe_atm + 200) * qty 
            
            broker.virtual_place_basket(sim_id, margin_req, legs, features=features, trigger=f"Z:{smile_z:.2f}")
            broker.positions[sim_id]['entry_credit'] = total_credit
            
            # Local Strategy Log
            with open(TRADE_LOG, "a", newline="") as f:
                csv.writer(f).writerow([datetime.now(), symbol, sim_id, "ENTRY", "NA", round(total_credit, 2), f"Z:{smile_z:.2f}"])
                
            logger.info(f"[STRATEGY ENTRY] {sim_id} Delta Neutral Iron Condor Placed. Net Credit: {total_credit:.2f}")
            
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Failed to select Delta-Neutral legs for {symbol}: {e}")

if __name__ == "__main__":
    import asyncio
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from paper_trading.standalone_runner import run_standalone
    asyncio.run(run_standalone(evaluate_market_tick, symbol="NSE:NIFTY50-INDEX", qty=1500)) # 20 lots
