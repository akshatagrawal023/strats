import logging
import os
import csv
import time
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

    # --- MTM TRACKING & LOGGING ---
    has_trade = False
    for sim_id in list(broker.positions.keys()):
        if symbol in sim_id and "VT_IC" in sim_id:
            has_trade = True
            pnl = broker.update_pnl(sim_id, current_prices)
            if pnl is not None:
                # Log MTM
                with open(MTM_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([datetime.now(), symbol, sim_id, round(pnl, 2), round(smile_z, 2)])
                
                # Check Stop Loss (1.5x Credit)
                pos = broker.positions[sim_id]
                entry_credit = pos.get('entry_credit', 0.0)
                if pnl < -entry_credit * 1.5:
                    logger.warning(f"[STOP LOSS] {sim_id} at 150%. Closing.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="SL_150%")
                
                # Check Profit Taking (Smile Reversion)
                elif smile_z <= -0.5:
                    logger.info(f"[TAKE PROFIT] {sim_id} Smile Reverted (Z={smile_z:.2f}).")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="REVERSION")

    # --- ENTRY TRIGGER ---
    if not has_trade and smile_z >= 1.0:
        logger.info(f"[{symbol} ENTRY] Smile Z: {smile_z:.2f} >= 1.0. Entering Iron Condor.")
        
        atm_strike = int(strikes[atm_idx])
        wing_offset = 2
        call_wing_idx = min(atm_idx + wing_offset, len(strikes)-1)
        put_wing_idx  = max(atm_idx - wing_offset, 0)
        
        cw_strike = int(strikes[call_wing_idx])
        pw_strike = int(strikes[put_wing_idx])
        
        # Prices
        ce_atm_bid = current_prices[f"CE_{atm_strike}"]['bid']
        pe_atm_bid = current_prices[f"PE_{atm_strike}"]['bid']
        ce_otm_ask = current_prices[f"CE_{cw_strike}"]['ask']
        pe_otm_ask = current_prices[f"PE_{pw_strike}"]['ask']
        
        net_credit_unit = (ce_atm_bid + pe_atm_bid) - (ce_otm_ask + pe_otm_ask)
        total_credit = net_credit_unit * qty
        
        legs = [
            {'symbol': f'CE_{atm_strike}', 'qty': qty, 'side': -1, 'price': ce_atm_bid},
            {'symbol': f'PE_{atm_strike}', 'qty': qty, 'side': -1, 'price': pe_atm_bid},
            {'symbol': f'CE_{cw_strike}',  'qty': qty, 'side':  1, 'price': ce_otm_ask},
            {'symbol': f'PE_{pw_strike}',  'qty': qty, 'side':  1, 'price': pe_otm_ask},
        ]
        
        sim_id = f"VT_IC_{symbol.split(':')[1]}_{atm_strike}"
        margin_req = (cw_strike - pw_strike) * qty # Rough margin
        
        broker.virtual_place_basket(sim_id, margin_req, legs, features=features, trigger=f"Z:{smile_z:.2f}")
        # Store metadata for PnL logic
        broker.positions[sim_id]['entry_credit'] = total_credit
        
        # Log Trade
        with open(TRADE_LOG, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now(), symbol, sim_id, "ENTRY", "NA", round(total_credit, 2), f"Z:{smile_z:.2f}"])

if __name__ == "__main__":
    import asyncio
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from paper_trading.standalone_runner import run_standalone
    asyncio.run(run_standalone(evaluate_market_tick, symbol="NSE:NIFTY50-INDEX", qty=1500)) # 20 lots
