import logging
import os
import csv
import time
import numpy as np
from datetime import datetime

# Local logging setup
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
MTM_LOG = os.path.join(LOG_DIR, "mtm_live.csv")
TRADE_LOG = os.path.join(LOG_DIR, "trades_placed.csv")

for log_file, headers in [(MTM_LOG, ["timestamp", "symbol", "sim_id", "pnl", "iv_z"]), 
                          (TRADE_LOG, ["timestamp", "symbol", "sim_id", "type", "price", "premium", "trigger"])]:
    if not os.path.exists(log_file):
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(headers)

def evaluate_market_tick(
    broker, symbol, qty, strikes, current_prices, 
    greeks_mat, features, state_dict
):
    """
    GammaScalp — Bet on Chaos.
    Buys ATM Straddles when Vol is crushed and scalps quick Gamma bursts.
    """
    logger = logging.getLogger("GammaScalp")
    
    iv_z = features.get('iv_z_score', 0.0)
    momentum = features.get('momentum', 0.0)
    acceleration = features.get('acceleration', 0.0)
    total_gamma = features.get('atm_gamma', 0.0)
    is_expiry_day = features.get('is_expiry_day', False)
    atm_idx = features.get('atm_idx')
    
    if atm_idx is None: return
    sym_prefix = symbol.split(":")[1].split("-")[0]

    # --- 0. SAFETY GATES ---
    now_time = datetime.now().time()
    panic_cutoff = datetime.strptime("14:50", "%H:%M").time()
    is_panic = now_time >= panic_cutoff
    
    # HARD EXPIRY EXIT: 2:50 PM onwards, we are done.
    if is_expiry_day and is_panic:
        for sim_id in list(broker.positions.keys()):
            if sym_prefix in sim_id and "GS_LONG" in sim_id:
                logger.warning(f"[PANIC EXIT] {sim_id} 2:50 PM Expiry Wall reached. Force closing.")
                broker.virtual_close_all(sim_id, current_prices, features=features, trigger="EXP_PANIC")
        return # Block all further evaluation/entries

    # --- 1. MTM & EXIT ---
    has_trade = False
    for sim_id in list(broker.positions.keys()):
        if sym_prefix in sim_id and "GS_LONG" in sim_id:
            has_trade = True
            pnl = broker.update_pnl(sim_id, current_prices, features=features)
            
            if pnl is not None:
                pos = broker.positions[sim_id]
                entry_premium = pos.get('entry_premium', 1.0)
                
                # TAKE PROFIT: Quick 12% Scalp
                if pnl > entry_premium * 0.12:
                    logger.info(f"[GS PROFIT] 12% Scalped.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="TP_12%")
                    state_dict['last_exit_time'] = time.time()
                    continue
                
                # OPTIMIZED ACCELERATION EXHAUSTION
                if pnl > entry_premium * 0.03 and acceleration < -1.0:
                    logger.info(f"[GS EXIT] Burst stall (Accel:{acceleration:.2f}).")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="ACCEL_EXHAUST")
                    state_dict['last_exit_time'] = time.time()
                    continue

                # HARD STOP: 30% of Premium
                if pnl < -entry_premium * 0.30:
                    logger.error(f"[GS STOP] 30% Premium lost.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="SL_30%")
                    state_dict['last_exit_time'] = time.time()
                    continue

    # --- 2. ENTRY TRIGGER ---
    last_exit = state_dict.get('last_exit_time', 0)
    cooldown_active = (time.time() - last_exit) < 60 # 1 min cooldown for HFT scalping
    
    if not has_trade and not cooldown_active:
        # GOLDEN RATIO TRIGGER (Optimized via 12PM-2:45PM Sweep)
        if (iv_z < -2.0) or (iv_z < -1.3 and acceleration > 1.5):
            logger.info(f"[GS ENTRY] Target Acquired (Accel:{acceleration:.2f}, Z:{iv_z:.2f})")
            
            s_atm = int(strikes[atm_idx])
            try:
                ce_price = current_prices[f"CE_{s_atm}"]['ask']
                pe_price = current_prices[f"PE_{s_atm}"]['ask']
                
                total_premium = (ce_price + pe_price) * qty
                
                legs = [
                    {'symbol': f'CE_{s_atm}', 'qty': qty, 'side': 1, 'price': ce_price},
                    {'symbol': f'PE_{s_atm}', 'qty': qty, 'side': 1, 'price': pe_price},
                ]
                sim_id = f"GS_LONG_{sym_prefix}_{s_atm}"
                # Margin for long is just the premium
                broker.virtual_place_basket(sim_id, total_premium, legs, features=features)
                broker.positions[sim_id]['entry_premium'] = total_premium
                
                with open(TRADE_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([datetime.now(), symbol, sim_id, "ENTRY", ce_price+pe_price, total_premium, f"Z:{iv_z:.2f}"])
                
                logger.info(f"[ENTRY] {sim_id} Gamma Scalp Placed. Premium: {total_premium:.2f}")
            except KeyError: pass
