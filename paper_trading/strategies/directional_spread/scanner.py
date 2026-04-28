import logging
import time

def evaluate_market_tick(
    broker, symbol, qty, strikes, current_prices, 
    greeks_mat, features, state_dict
):
    """
    Directional Spread — Trend-Hunter Mode.
    Only enters during high-momentum trending phases.
    """
    logger = logging.getLogger("DirSpread")
    
    momentum = features.get('momentum', 0.0)
    panic = features.get('panic', False)
    atm_idx = features.get('atm_idx')
    
    if atm_idx is None: return

    sym_prefix = symbol.split(":")[1].split("-")[0]

    # --- POSITION TRACKING & DYNAMIC EXIT ---
    has_trade = False
    for sim_id in list(broker.positions.keys()):
        if sym_prefix in sim_id and ("BULL" in sim_id or "BEAR" in sim_id):
            has_trade = True
            broker.update_pnl(sim_id, current_prices, features=features)
            
            # --- MOMENTUM EXHAUSTION EXIT ---
            # If the trend weakens significantly, exit to preserve alpha
            if "BULL" in sim_id and momentum < 3.0:
                logger.info(f"[TREND EXHAUST] {sim_id} Momentum dropped to {momentum:.1f}. Closing Bull Spread.")
                broker.virtual_close_all(sim_id, current_prices, features=features, trigger="TREND_WEAK")
                state_dict['last_exit_time'] = time.time()
                continue
            elif "BEAR" in sim_id and momentum > -3.0:
                logger.info(f"[TREND EXHAUST] {sim_id} Momentum rose to {momentum:.1f}. Closing Bear Spread.")
                broker.virtual_close_all(sim_id, current_prices, features=features, trigger="TREND_WEAK")
                state_dict['last_exit_time'] = time.time()
                continue

    # --- ENTRY TRIGGER (Trend-Hunter Mode) ---
    # COOLDOWN: 120s lockout after any trade exit
    last_exit = state_dict.get('last_exit_time', 0)
    cooldown_active = (time.time() - last_exit) < 120
    
    # THRESHOLDS:
    # Requires STRONG momentum (abs > 10.0) and NO panic
    if not has_trade and not cooldown_active and not panic:
        
        if momentum > 10.0:  # BULL CASE
            logger.info(f"[{symbol} BULL ENTRY] Strong Up-Trend (Mom:{momentum:.1f}).")
            s_base = int(strikes[atm_idx])
            s_long = s_base
            s_short = s_base + (50 if "NIFTY" in sym_prefix else 200)
            
            try:
                l_price = current_prices[f"CE_{s_long}"]['ask']
                s_price = current_prices[f"CE_{s_short}"]['bid']
                
                legs = [
                    {'symbol': f'CE_{s_long}',  'qty': qty, 'side': 1,  'price': l_price},
                    {'symbol': f'CE_{s_short}', 'qty': qty, 'side': -1, 'price': s_price},
                ]
                sim_id = f"{sym_prefix}_BULL_ST_{s_long}"
                margin = (s_short - s_long) * qty
                broker.virtual_place_basket(sim_id, margin, legs, features=features)
                logger.info(f"[ENTRY] {sim_id} Trend-Hunter Bull Spread Placed.")
            except KeyError: pass

        elif momentum < -10.0: # BEAR CASE
            logger.info(f"[{symbol} BEAR ENTRY] Strong Down-Trend (Mom:{momentum:.1f}).")
            s_base = int(strikes[atm_idx])
            s_long = s_base
            s_short = s_base - (50 if "NIFTY" in sym_prefix else 200)
            
            try:
                l_price = current_prices[f"PE_{s_long}"]['ask']
                s_price = current_prices[f"PE_{s_short}"]['bid']
                
                legs = [
                    {'symbol': f'PE_{s_long}',  'qty': qty, 'side': 1,  'price': l_price},
                    {'symbol': f'PE_{s_short}', 'qty': qty, 'side': -1, 'price': s_price},
                ]
                sim_id = f"{sym_prefix}_BEAR_ST_{s_long}"
                margin = (s_long - s_short) * qty
                broker.virtual_place_basket(sim_id, margin, legs, features=features)
                logger.info(f"[ENTRY] {sim_id} Trend-Hunter Bear Spread Placed.")
            except KeyError: pass

if __name__ == "__main__":
    import asyncio
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from paper_trading.standalone_runner import run_standalone
    
    asyncio.run(run_standalone(evaluate_market_tick, symbol="NSE:NIFTY50-INDEX", qty=650))
