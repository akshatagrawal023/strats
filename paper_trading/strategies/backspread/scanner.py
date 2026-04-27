import logging

def evaluate_market_tick(
    broker, symbol, qty, strikes, current_prices, 
    greeks_mat, features, state_dict
):
    strat_logger = logging.getLogger("Backspread")

    iv_z_score = features.get('iv_z_score')
    atm_idx = features.get('atm_idx')
    
    if iv_z_score is None:
        return

    sym_prefix = symbol.split(":")[1].split("-")[0]

    # --- MTM TRACKING ---
    has_trade = any(f"{sym_prefix}_BACKSPREAD" in k for k in broker.positions.keys())
    for sim_id in list(broker.positions.keys()):
        if f"{sym_prefix}_BACKSPREAD" in sim_id:
            pnl = broker.update_pnl(sim_id, current_prices, features=features)
            if pnl is not None:
                margin = broker.positions[sim_id]['margin_locked']
                if pnl > margin * 0.10: 
                    strat_logger.info(f"[TAKE PROFIT] {sim_id} at 10% (Gamma Explosion). Closing.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="TP_10%")
                elif pnl < -margin * 0.20:
                    strat_logger.info(f"[STOP LOSS] {sim_id} at 20%. Closing.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="SL_20%")

    # --- ENTRY TRIGGER ---
    if not has_trade:
        if iv_z_score < -1.5:
            up_idx = atm_idx + 2
            if up_idx < len(strikes):
                s_atm = int(strikes[atm_idx])
                s_otm = int(strikes[up_idx])
                
                ce_atm_bid = current_prices[f"CE_{s_atm}"]['bid']
                ce_otm_ask = current_prices[f"CE_{s_otm}"]['ask']
                
                legs = [
                    {'symbol': f'CE_{s_atm}', 'qty': qty,   'side': -1, 'price': ce_atm_bid},
                    {'symbol': f'CE_{s_otm}', 'qty': qty*2, 'side':  1, 'price': ce_otm_ask},
                ]
                sim_id = f"{sym_prefix}_BACKSPREAD_{s_atm}_{s_otm}"
                margin = (s_otm - s_atm) * qty 
                trigger_msg = f"IVZ:{iv_z_score:.2f}<-1.5"
                broker.virtual_place_basket(sim_id, margin, legs, features=features, trigger=trigger_msg)
                strat_logger.info(f"[ENTRY] {symbol} Low Volatility {iv_z_score:.2f}. Entered Call Backspread.")

if __name__ == "__main__":
    import asyncio
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from paper_trading.standalone_runner import run_standalone
    
    asyncio.run(run_standalone(evaluate_market_tick, symbol="NSE:NIFTY50-INDEX", qty=650))
