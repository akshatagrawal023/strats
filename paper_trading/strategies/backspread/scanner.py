import logging
import time
from datetime import datetime

def evaluate_market_tick(
    broker, symbol, qty, strikes, current_prices, 
    greeks_mat, features, state_dict
):
    strat_logger = logging.getLogger("Backspread")

    iv_z_score = features.get('iv_z_score')
    atm_idx = features.get('atm_idx')
    momentum = features.get('momentum', 0.0)
    spot = current_prices.get('spot', 0.0) # Provided by orchestrator in current_prices
    
    if iv_z_score is None or atm_idx is None:
        return

    sym_prefix = symbol.split(":")[1].split("-")[0]

    # --- 0. SAFETY CIRCUIT BREAKERS ---
    consecutive_losses = state_dict.get('consecutive_losses', 0)
    if consecutive_losses >= 3:
        return # Locked (Circuit Breaker)
    
    # HARD EXPIRY EXIT: 2:50 PM Wall
    now_dt = datetime.now()
    is_expiry_day = features.get('is_expiry_day', False)
    if is_expiry_day and (now_dt.hour > 14 or (now_dt.hour == 14 and now_dt.minute >= 50)):
        for sim_id in list(broker.positions.keys()):
            if "BACKSPREAD" in sim_id:
                strat_logger.warning(f"[PANIC EXIT] {sim_id} 2:50 PM Wall reached.")
                broker.virtual_close_all(sim_id, current_prices, features=features, trigger="EXP_PANIC")
        return # Block all activity

    # No Expiry Afternoon entries (After 12:30 IST)
    can_enter = not (is_expiry_day and (now_dt.hour > 12 or (now_dt.hour == 12 and now_dt.minute >= 30)))

    # --- 1. MTM TRACKING & SMART EXIT ---
    for sim_id in list(broker.positions.keys()):
        if f"{sym_prefix}_BACKSPREAD" in sim_id:
            pnl = broker.update_pnl(sim_id, current_prices, features=features)
            if pnl is not None:
                pos = broker.positions[sim_id]
                margin = pos['margin_locked']
                
                # Parse long strike from sim_id for defensive exit (format: PREFIX_STRAT_ATM_OTM)
                parts = sim_id.split("_")
                long_strike = float(parts[-1]) if parts[-1].isdigit() else 0.0
                
                # Update Circuit Breaker Logic on Exits
                def close_and_track(trig):
                    final_pnl = broker.virtual_close_all(sim_id, current_prices, features=features, trigger=trig)
                    state_dict['last_exit_time'] = time.time()
                    if final_pnl is not None:
                        if final_pnl < 0:
                            state_dict['consecutive_losses'] = state_dict.get('consecutive_losses', 0) + 1
                        else:
                            state_dict['consecutive_losses'] = 0 # Reset on win

                # 1. VEGA PROFIT (Vol Mean Reversion)
                if iv_z_score >= 0.2:
                    strat_logger.info(f"[VEGA PROFIT] {sim_id} Vol Reverted to {iv_z_score:.2f}. Closing.")
                    close_and_track("VEGA_REVERSION")
                
                # 2. GAMMA PROFIT (Breakout)
                elif pnl > margin * 0.15: 
                    strat_logger.info(f"[GAMMA PROFIT] {sim_id} at 15% breakout. Closing.")
                    close_and_track("TP_15%")
                
                # 3. DEFENSIVE BAILOUT (Avoid the Gamma Trap)
                # If spot is near long_strike and losing momentum, bail.
                elif long_strike > 0 and (abs(spot - long_strike) / long_strike < 0.005) and momentum < 0:
                    strat_logger.warning(f"[DEFENSIVE EXIT] {sim_id} Stalled at Long Strike. Bailing out.")
                    close_and_track("DEFENSIVE_BAILOUT")

                # 4. STRICT STOP LOSS
                elif pnl < -margin * 0.15:
                    strat_logger.error(f"[STOP LOSS] {sim_id} at 15%. Protecting capital.")
                    close_and_track("SL_15%")

    has_trade = any(f"{sym_prefix}_BACKSPREAD" in k for k in broker.positions.keys())
    # --- SMART ENTRY (High Conviction Only) ---
    if not has_trade and can_enter:
        # Require Deep Low Vol AND Positive Momentum pulse
        if iv_z_score < -2.0 and momentum > 5.0:
            up_idx = atm_idx + 2
            if up_idx < len(strikes):
                s_atm = int(strikes[atm_idx])
                s_otm = int(strikes[up_idx])
                
                try:
                    ce_atm_bid = current_prices[f"CE_{s_atm}"]['bid']
                    ce_otm_ask = current_prices[f"CE_{s_otm}"]['ask']
                    
                    legs = [
                        {'symbol': f'CE_{s_atm}', 'qty': qty,   'side': -1, 'price': ce_atm_bid, 'spot': spot},
                        {'symbol': f'CE_{s_otm}', 'qty': qty*2, 'side':  1, 'price': ce_otm_ask, 'spot': spot},
                    ]
                    sim_id = f"{sym_prefix}_BACKSPREAD_{s_atm}_{s_otm}"
                    margin = (s_otm - s_atm) * qty 
                    trigger_msg = f"IVZ:{iv_z_score:.2f}|Mom:{momentum:.1f}"
                    broker.virtual_place_basket(sim_id, margin, legs, features=features, trigger=trigger_msg)
                    strat_logger.info(f"[SMART ENTRY] {symbol} Vol Spike Imminent. Entered Backspread at {iv_z_score:.2f}")
                except KeyError as e:
                    strat_logger.error(f"Price data missing for backspread legs: {e}")

if __name__ == "__main__":
    import asyncio
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from paper_trading.standalone_runner import run_standalone
    
    asyncio.run(run_standalone(evaluate_market_tick, symbol="NSE:NIFTY50-INDEX", qty=650))
