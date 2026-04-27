import logging

def evaluate_market_tick(
    broker, symbol, qty, strikes, current_prices, 
    greeks_mat, features, state_dict
):
    """
    Reactive evaluate function for Iron Butterfly.
    Triggered instantly by the orchestrator upon every tick.
    """
    strat_logger = logging.getLogger("IronButterfly")
    
    if "positions_entered" not in state_dict:
        state_dict["positions_entered"] = False

    atm_idx = features['atm_idx']
    iv_z_score = features['iv_z_score']
    panic_detected = features['panic']
    
    # Only compute if warmup passes (signaled by iv_z_score != None)
    if iv_z_score is None:
        return

    # --- MTM TRACKING ---
    has_trade = False
    for sim_id in list(broker.positions.keys()):
        if symbol in sim_id and "IB_W" in sim_id:
            has_trade = True
            pnl = broker.update_pnl(sim_id, current_prices)
            if pnl is not None:
                margin = broker.positions[sim_id]['margin_locked']
                if pnl > margin * 0.10:
                    strat_logger.info(f"[TAKE PROFIT] {sim_id} at 10%. Closing.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="TP_10%")
                elif pnl < -margin * 0.20:
                    strat_logger.info(f"[STOP LOSS] {sim_id} at 20%. Closing.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="SL_20%")

    # --- ENTRY TRIGGER ---
    if not state_dict["positions_entered"] and not has_trade:
        if not (panic_detected or iv_z_score < 1.5):
            strat_logger.info(f"[{symbol} TRIGGER] IV Z-Score {iv_z_score:.2f} >= 1.5. Entering IB.")
            atm_strike = int(strikes[atm_idx])
            
            # Using current_prices dict populated by orchestrator
            ce_atm_bid = current_prices[f"CE_{atm_strike}"]['bid']
            pe_atm_bid = current_prices[f"PE_{atm_strike}"]['bid']
            
            for offset in [1, 2, 3]:
                call_idx = atm_idx + offset
                put_idx  = atm_idx - offset
                if call_idx >= len(strikes) or put_idx < 0:
                    continue
                    
                s_call_otm = int(strikes[call_idx])
                s_put_otm = int(strikes[put_idx])
                
                ce_otm_ask = current_prices[f"CE_{s_call_otm}"]['ask']
                pe_otm_ask = current_prices[f"PE_{s_put_otm}"]['ask']
                
                nominal_width = s_call_otm - atm_strike

                legs = [
                    {'symbol': f'CE_{atm_strike}', 'qty': qty, 'side': -1, 'price': ce_atm_bid},
                    {'symbol': f'PE_{atm_strike}', 'qty': qty, 'side': -1, 'price': pe_atm_bid},
                    {'symbol': f'CE_{s_call_otm}', 'qty': qty, 'side':  1, 'price': ce_otm_ask},
                    {'symbol': f'PE_{s_put_otm}',  'qty': qty, 'side':  1, 'price': pe_otm_ask},
                ]
                # Include symbol in sim_id to track multi-symbol separately
                sym_prefix = symbol.split(":")[1].split("-")[0]
                sim_id = f"{sym_prefix}_IB_W{nominal_width}"
                margin_locked = float(nominal_width * qty) * 1.5
                trigger_msg = f"IVZ:{iv_z_score:.2f}>=1.5"
                broker.virtual_place_basket(sim_id, margin_locked, legs, features=features, trigger=trigger_msg)

            state_dict["positions_entered"] = True

if __name__ == "__main__":
    import asyncio
    import sys
    import os
    # Add root to path so imports work when running directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from paper_trading.standalone_runner import run_standalone
    
    asyncio.run(run_standalone(evaluate_market_tick, symbol="NSE:NIFTY50-INDEX", qty=650))
