import logging
import os
import csv
from datetime import datetime

# Local logging setup for ShortStraddle
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
    ShortStraddle — 2-Leg Smile Mean Reversion.
    Higher efficiency than Iron Condor due to lower friction.
    """
    logger = logging.getLogger("ShortStraddle")
    smile_z = features.get('smile_z', 0.0)
    atm_idx = features.get('atm_idx')
    if atm_idx is None: return

    # --- MTM TRACKING ---
    has_trade = False
    for sim_id in list(broker.positions.keys()):
        if symbol in sim_id and "SS_SMILE" in sim_id:
            has_trade = True
            pnl = broker.update_pnl(sim_id, current_prices)
            if pnl is not None:
                with open(MTM_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([datetime.now(), symbol, sim_id, round(pnl, 2), round(smile_z, 2)])
                
                pos = broker.positions[sim_id]
                entry_credit = pos.get('entry_credit', 0.0)
                
                # Stop Loss 1.5x credit
                if pnl < -entry_credit * 1.5:
                    logger.warning(f"[STOP LOSS] {sim_id} SL 150%.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="SL_150%")
                elif smile_z <= -0.5:
                    logger.info(f"[TAKE PROFIT] {sim_id} Reverted (Z={smile_z:.2f}).")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="REVERSION")

    # --- ENTRY ---
    if not has_trade and smile_z >= 1.0:
        logger.info(f"[{symbol}] ENTER Straddle. Smile Z: {smile_z:.2f}")
        atm_strike = int(strikes[atm_idx])
        ce_bid = current_prices[f"CE_{atm_strike}"]['bid']
        pe_bid = current_prices[f"PE_{atm_strike}"]['bid']
        total_credit = (ce_bid + pe_bid) * qty
        
        legs = [
            {'symbol': f'CE_{atm_strike}', 'qty': qty, 'side': -1, 'price': ce_bid},
            {'symbol': f'PE_{atm_strike}', 'qty': qty, 'side': -1, 'price': pe_bid},
        ]
        sim_id = f"SS_SMILE_{symbol.split(':')[1]}_{atm_strike}"
        broker.virtual_place_basket(sim_id, margin_locked=qty*500, legs=legs, features=features, trigger=f"Z:{smile_z:.2f}")
        broker.positions[sim_id]['entry_credit'] = total_credit
        
        with open(TRADE_LOG, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now(), symbol, sim_id, "ENTRY", "NA", round(total_credit, 2), f"Z:{smile_z:.2f}"])

if __name__ == "__main__":
    import asyncio, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from paper_trading.standalone_runner import run_standalone
    asyncio.run(run_standalone(evaluate_market_tick, symbol="NSE:NIFTY50-INDEX", qty=1300))
