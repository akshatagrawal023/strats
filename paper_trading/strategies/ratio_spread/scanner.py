import logging

def evaluate_market_tick(
    broker, symbol, qty, strikes, current_prices, 
    greeks_mat, features, state_dict
):
    strat_logger = logging.getLogger("RatioSpread")
    
    skew_z = features.get('skew_z')
    panic = features.get('panic')
    atm_idx = features.get('atm_idx')
    
    if skew_z is None:
        return

    sym_prefix = symbol.split(":")[1].split("-")[0]
    
    # --- MTM TRACKING ---
    has_trade = any(f"{sym_prefix}_RATIO_SPREAD" in k for k in broker.positions.keys())
    for sim_id in list(broker.positions.keys()):
        if f"{sym_prefix}_RATIO_SPREAD" in sim_id:
            pnl = broker.update_pnl(sim_id, current_prices, features=features)
            if pnl is not None:
                margin = broker.positions[sim_id]['margin_locked']
                if pnl > margin * 0.05:
                    strat_logger.info(f"[TAKE PROFIT] {sim_id} at 5%. Closing.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="TP_5%")
                elif pnl < -margin * 0.20:
                    strat_logger.info(f"[STOP LOSS] {sim_id} at 20%. Closing.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="SL_20%")

    # --- ENTRY TRIGGER ---
    if not has_trade:
        if skew_z > 2.0 and not panic:
            dn_idx = atm_idx - 2
            if dn_idx >= 0:
                s_atm = int(strikes[atm_idx])
                s_otm = int(strikes[dn_idx])
                
                pe_atm_bid = current_prices[f"PE_{s_atm}"]['bid']
                pe_otm_ask = current_prices[f"PE_{s_otm}"]['ask']
                
                legs = [
                    {'symbol': f'PE_{s_atm}', 'qty': qty,   'side': -1, 'price': pe_atm_bid},
                    {'symbol': f'PE_{s_otm}', 'qty': qty*2, 'side':  1, 'price': pe_otm_ask},
                ]
                sim_id = f"{sym_prefix}_RATIO_SPREAD_{s_atm}_{s_otm}"
                margin = (s_atm - s_otm) * qty 
                trigger_msg = f"SkewZ:{skew_z:.2f}>2.0"
                broker.virtual_place_basket(sim_id, margin, legs, features=features, trigger=trigger_msg)
                strat_logger.info(f"[ENTRY] {symbol} High Skew: {skew_z:.2f}. Entered Put Ratio Spread.")

if __name__ == "__main__":
    import asyncio
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from paper_trading.standalone_runner import run_standalone
    
    asyncio.run(run_standalone(evaluate_market_tick, symbol="NSE:NIFTY50-INDEX", qty=650))
