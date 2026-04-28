import logging
import os
import csv
import time
import numpy as np
from datetime import datetime

# Local logging setup for Vol2 (OTM Vol Scalp)
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
MTM_LOG = os.path.join(LOG_DIR, "mtm_live.csv")
TRADE_LOG = os.path.join(LOG_DIR, "trades_placed.csv")

# Initialize logs
for log_file, headers in [
    (MTM_LOG, ["timestamp", "symbol", "sim_id", "pnl", "smile_z"]),
    (TRADE_LOG, ["timestamp", "symbol", "sim_id", "type", "pnl", "net_credit", "trigger"])
]:
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(headers)

# Symbol-specific thresholds for OTM Vol
# Since we are trading OTM wings, we can afford sharper entries
SYMBOL_CONFIG = {
    "NIFTY50": {"entry_z": 2.2, "exit_z": 0.2, "target_delta": 0.25},
    "SENSEX":   {"entry_z": 3.5, "exit_z": -0.5, "target_delta": 0.20}, # Low target delta to beat taxes
    "DEFAULT":  {"entry_z": 2.5, "exit_z": 0.0, "target_delta": 0.25}
}

def evaluate_market_tick(
    broker, symbol, qty, strikes, current_prices, 
    greeks_mat, features, state_dict
):
    """
    Vol2 — Tax-Efficient OTM Volatility Arbitrage.
    Targets 0.25 Delta wings to maximize Net Alpha vs Transaction Costs.
    """
    logger = logging.getLogger("Vol2")
    smile_z = features.get('smile_z', 0.0)
    atm_idx = features.get('atm_idx')
    
    if atm_idx is None:
        return

    sym_prefix = symbol.split(":")[1].split("-")[0]
    config = SYMBOL_CONFIG.get(sym_prefix, SYMBOL_CONFIG["DEFAULT"])

    # --- 0. SAFETY CIRCUIT BREAKERS ---
    consecutive_losses = state_dict.get('consecutive_losses', 0)
    if consecutive_losses >= 3:
        return # Locked (Circuit Breaker)
    
    # HARD EXPIRY EXIT: 2:50 PM Wall
    now_dt = datetime.now()
    is_expiry_day = features.get('is_expiry_day', False)
    if is_expiry_day and (now_dt.hour > 14 or (now_dt.hour == 14 and now_dt.minute >= 50)):
        for sim_id in list(broker.positions.keys()):
            if "VOL2" in sim_id:
                logger.warning(f"[PANIC EXIT] {sim_id} 2:50 PM Wall reached. Closing OTM wings.")
                broker.virtual_close_all(sim_id, current_prices, features=features, trigger="EXP_PANIC")
        return # Stop all activity

    # No Expiry Afternoon entries (After 12:30 IST)
    can_enter = not (is_expiry_day and (now_dt.hour > 12 or (now_dt.hour == 12 and now_dt.minute >= 30)))

    # --- 1. MTM TRACKING & LOGGING ---
    has_trade = False
    for sim_id in list(broker.positions.keys()):
        if sym_prefix in sim_id and "VOL2" in sim_id:
            has_trade = True
            pnl = broker.update_pnl(sim_id, current_prices, features=features)
            if pnl is not None:
                # Log MTM
                try:
                    with open(MTM_LOG, "a", newline="") as f:
                        csv.writer(f).writerow([datetime.now(), symbol, sim_id, round(pnl, 2), round(smile_z, 2)])
                except Exception as e:
                    logger.error(f"Failed to write Vol2 MTM log: {e}")
                
                # Check Profit Taking (Custom exit)
                if smile_z <= config["exit_z"]:
                    logger.info(f"[Vol2 PROFIT] {sim_id} Reverted (Z={smile_z:.2f}). Closing.")
                    final_pnl = broker.virtual_close_all(sim_id, current_prices, features=features, trigger="REVERSION")
                    state_dict['last_exit_time'] = time.time()
                    
                    if final_pnl is not None:
                        if final_pnl < 0:
                            state_dict['consecutive_losses'] = state_dict.get('consecutive_losses', 0) + 1
                            logger.warning(f"[VOL2 CIRCUIT BREAKER] Loss recorded. Counter: {state_dict['consecutive_losses']}/3")
                        else:
                            state_dict['consecutive_losses'] = 0 # Reset on win

    # --- ENTRY TRIGGER ---
    last_exit = state_dict.get('last_exit_time', 0)
    cooldown_active = (time.time() - last_exit) < 120 
    
    if not has_trade and not cooldown_active and can_enter and smile_z >= config["entry_z"]:
        logger.info(f"[{symbol} Vol2 ENTRY] Skew Spike Z: {smile_z:.2f}. Targeting {config['target_delta']} Delta.")
        
        ce_deltas = greeks_mat[0]
        pe_deltas = greeks_mat[1]
        
        try:
            # 1. Selection based on TARGET DELTA (Tax Efficient)
            target = config['target_delta']
            ce_short_idx = np.nanargmin(np.abs(ce_deltas - target))
            pe_short_idx = np.nanargmin(np.abs(pe_deltas + target))
            
            s_ce_short = int(strikes[ce_short_idx])
            s_pe_short = int(strikes[pe_short_idx])
            
            # 2. Wings for margin protection (further away)
            ce_wing_idx = np.nanargmin(np.abs(ce_deltas - (target - 0.10)))
            pe_wing_idx = np.nanargmin(np.abs(pe_deltas + (target - 0.10)))
            
            s_ce_wing = int(strikes[ce_wing_idx])
            s_pe_wing = int(strikes[pe_wing_idx])

            # Fetch Prices
            ce_bid = current_prices[f"CE_{s_ce_short}"]['bid']
            pe_bid = current_prices[f"PE_{s_pe_short}"]['bid']
            ce_w_ask = current_prices[f"CE_{s_ce_wing}"]['ask']
            pe_w_ask = current_prices[f"PE_{s_pe_wing}"]['ask']
            
            net_credit_unit = (ce_bid + pe_bid) - (ce_w_ask + pe_w_ask)
            total_credit = net_credit_unit * qty
            
            legs = [
                {'symbol': f'CE_{s_ce_short}', 'qty': qty, 'side': -1, 'price': ce_bid},
                {'symbol': f'PE_{s_pe_short}', 'qty': qty, 'side': -1, 'price': pe_bid},
                {'symbol': f'CE_{s_ce_wing}', 'qty': qty, 'side':  1, 'price': ce_w_ask},
                {'symbol': f'PE_{s_pe_wing}', 'qty': qty, 'side':  1, 'price': pe_w_ask},
            ]
            
            sim_id = f"VOL2_{sym_prefix}_{s_ce_short}_{s_pe_short}"
            margin_req = (s_ce_short - s_pe_short + 100) * qty 
            
            broker.virtual_place_basket(sim_id, margin_req, legs, features=features, trigger=f"Z:{smile_z:.2f}")
            broker.positions[sim_id]['entry_credit'] = total_credit
            
            with open(TRADE_LOG, "a", newline="") as f:
                csv.writer(f).writerow([datetime.now(), symbol, sim_id, "ENTRY", "NA", round(total_credit, 2), f"Z:{smile_z:.2f}"])
                
            logger.info(f"[Vol2 ENTRY] {sim_id} Net Credit: {total_credit:.2f} (Delta: {target})")
            
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Failed to select OTM legs for Vol2 {symbol}: {e}")
