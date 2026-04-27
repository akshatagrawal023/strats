import logging
import time

def evaluate_market_tick(
    broker, symbol, qty, strikes, current_prices, 
    greeks_mat, features, state_dict
):
    """
    Reactive evaluate function for Directional Spread.
    """
    strat_logger = logging.getLogger("DirectionalSpread")
    
    if "last_exit_time" not in state_dict:
        state_dict["last_exit_time"] = 0

    EXIT_COOLDOWN_SEC = 300
    MOMENTUM_CONFIRMATION_MULT = 1.8

    atm_idx = features['atm_idx']
    momentum = features['momentum']
    vol = features['vol']
    panic = features['panic']
    ce_iv_vel = features.get('ce_iv_vel', 0.0)
    pe_iv_vel = features.get('pe_iv_vel', 0.0)
    
    if momentum is None:
        return # Warmup phase

    sym_prefix = symbol.split(":")[1].split("-")[0]

    # --- EXIT MANAGEMENT ---
    for sim_id in list(broker.positions.keys()):
        if f"{sym_prefix}_BULL_SPREAD" in sim_id or f"{sym_prefix}_BEAR_SPREAD" in sim_id:
            pnl = broker.update_pnl(sim_id, current_prices, features=features)
            if pnl is not None:
                margin = broker.positions[sim_id]['margin_locked']
                if pnl > margin * 0.05:
                    strat_logger.info(f"[TAKE PROFIT] {sim_id} at 5%. Closing.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="TP_5%")
                    state_dict["last_exit_time"] = time.time()
                elif pnl < -margin * 0.20:
                    strat_logger.info(f"[STOP LOSS] {sim_id} at 20%. Closing.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="SL_20%")
                    state_dict["last_exit_time"] = time.time()

    # --- ENTRY TRIGGER ---
    if (time.time() - state_dict["last_exit_time"]) > EXIT_COOLDOWN_SEC:
        is_bullish_burst = (momentum > vol * MOMENTUM_CONFIRMATION_MULT) and (not panic)
        is_bearish_burst = (momentum < -vol * MOMENTUM_CONFIRMATION_MULT)
        
        has_bull_trade = any(f"{sym_prefix}_BULL" in k for k in broker.positions.keys())
        has_bear_trade = any(f"{sym_prefix}_BEAR" in k for k in broker.positions.keys())

        if is_bullish_burst and not has_bull_trade:
            if ce_iv_vel > -0.001:
                up_idx = atm_idx + 2
                if up_idx < len(strikes):
                    s_atm = int(strikes[atm_idx])
                    s_otm = int(strikes[up_idx])
                    legs = [
                        {'symbol': f'CE_{s_atm}', 'qty': qty, 'side':  1, 'price': current_prices[f"CE_{s_atm}"]['ask']},
                        {'symbol': f'CE_{s_otm}', 'qty': qty, 'side': -1, 'price': current_prices[f"CE_{s_otm}"]['bid']},
                    ]
                    sim_id = f"{sym_prefix}_BULL_SPREAD_{s_atm}_{s_otm}"
                    margin = (current_prices[f"CE_{s_atm}"]['ask'] - current_prices[f"CE_{s_otm}"]['bid']) * qty
                    trigger_msg = f"Mom:{momentum:.2f} (IV:{ce_iv_vel:.4f})"
                    broker.virtual_place_basket(sim_id, margin, legs, features=features, trigger=trigger_msg)
                    strat_logger.info(f"[ENTRY] {symbol} Bullish Momentum: {momentum:.2f} (IV Vel: {ce_iv_vel:.4f})")

        elif is_bearish_burst and not has_bear_trade:
            if pe_iv_vel > -0.001:
                dn_idx = atm_idx - 2
                if dn_idx >= 0:
                    s_atm = int(strikes[atm_idx])
                    s_otm = int(strikes[dn_idx])
                    legs = [
                        {'symbol': f'PE_{s_atm}', 'qty': qty, 'side':  1, 'price': current_prices[f"PE_{s_atm}"]['ask']},
                        {'symbol': f'PE_{s_otm}', 'qty': qty, 'side': -1, 'price': current_prices[f"PE_{s_otm}"]['bid']},
                    ]
                    sim_id = f"{sym_prefix}_BEAR_SPREAD_{s_atm}_{s_otm}"
                    margin = (current_prices[f"PE_{s_atm}"]['ask'] - current_prices[f"PE_{s_otm}"]['bid']) * qty
                    trigger_msg = f"Mom:{momentum:.2f} (IV:{pe_iv_vel:.4f})"
                    broker.virtual_place_basket(sim_id, margin, legs, features=features, trigger=trigger_msg)
                    strat_logger.info(f"[ENTRY] {symbol} Bearish Momentum: {momentum:.2f} (IV Vel: {pe_iv_vel:.4f})")

if __name__ == "__main__":
    import asyncio
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from paper_trading.standalone_runner import run_standalone
    
    asyncio.run(run_standalone(evaluate_market_tick, symbol="NSE:NIFTY50-INDEX", qty=650))
