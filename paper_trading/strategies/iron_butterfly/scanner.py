import logging
import time
import numpy as np

def evaluate_market_tick(
    broker, symbol, qty, strikes, current_prices, 
    greeks_mat, features, state_dict
):
    """
    Reactive evaluate function for Iron Butterfly.
    Triggered instantly by the orchestrator upon every tick.
    """
    strat_logger = logging.getLogger("IronButterfly")
    
def evaluate_market_tick(
    broker, symbol, qty, strikes, current_prices, 
    greeks_mat, features, state_dict
):
    """
    Iron Butterfly — Range-Bound Theta Harvester.
    Sentinel Mode: Only enters in quiet, non-trending markets.
    """
    logger = logging.getLogger("IronFly")
    
    iv_z = features.get('iv_z_score', 0.0)
    momentum = features.get('momentum', 0.0)
    panic = features.get('panic', False)
    atm_idx = features.get('atm_idx')
    
    if atm_idx is None: return

    sym_prefix = symbol.split(":")[1].split("-")[0]

    # --- POSITION TRACKING ---
    has_trade = False
    for sim_id in list(broker.positions.keys()):
        if sym_prefix in sim_id and "IB_W" in sim_id:
            has_trade = True
            pnl = broker.update_pnl(sim_id, current_prices, features=features)
            
            # --- STORM BAILOUT ---
            # If market panics or trends hard (>15), close the Butterfly instantly
            if panic or abs(momentum) > 15.0:
                logger.warning(f"[STORM BAILOUT] {sim_id} Closing. Momentum:{momentum:.1f} | Panic:{panic}")
                broker.virtual_close_all(sim_id, current_prices, features=features, trigger="MOMENTUM_BREAKOUT")
                state_dict['last_exit_time'] = time.time()
                continue

            # Standard Take Profit at 10% of margin
            if pnl is not None:
                margin = broker.positions[sim_id].get('margin_locked', 100000)
                if pnl > margin * 0.10:
                    logger.info(f"[TAKE PROFIT] {sim_id} at 10%. Closing.")
                    broker.virtual_close_all(sim_id, current_prices, features=features, trigger="TP_10%")
                    state_dict['last_exit_time'] = time.time()

    # --- ENTRY TRIGGER (Sentinel Mode) ---
    # COOLDOWN: 120s lockout after any trade exit
    last_exit = state_dict.get('last_exit_time', 0)
    cooldown_active = (time.time() - last_exit) < 120
    
    # THRESHOLDS:
    # 1. Market must be STATIC (abs momentum < 5.0)
    # 2. No panic spike in progress
    # 3. IV should be high (>1.0) and ready to crush OR very quiet (<-1.5)
    entry_allowed = abs(momentum) < 5.0 and not panic and (iv_z > 1.0 or iv_z < -1.5)

    if not has_trade and not cooldown_active and entry_allowed:
        logger.info(f"[{symbol} IB ENTRY] Market is Static (Mom:{momentum:.1f}, IVZ:{iv_z:.2f}). Entering Sentinel Fly.")
        
        # Delta-Neutral Leg selection
        ce_deltas = greeks_mat[0]
        pe_deltas = greeks_mat[1]
        
        try:
            # 1. Center the Butterfly at 0.5 Delta
            ce_idx = np.nanargmin(np.abs(ce_deltas - 0.5))
            pe_idx = np.nanargmin(np.abs(pe_deltas + 0.5))
            
            s_ce = int(strikes[ce_idx])
            s_pe = int(strikes[pe_idx])
            
            # 2. Wings (Fixed width based on symbol)
            wing_dist = 100 if "NIFTY" in sym_prefix else 300
            s_ce_w = s_ce + wing_dist
            s_pe_w = s_pe - wing_dist
            
            # 3. Prices
            ce_bid = current_prices[f"CE_{s_ce}"]['bid']
            pe_bid = current_prices[f"PE_{s_pe}"]['bid']
            ce_w_ask = current_prices[f"CE_{s_ce_w}"]['ask']
            pe_w_ask = current_prices[f"PE_{s_pe_w}"]['ask']
            
            legs = [
                {'symbol': f'CE_{s_ce}',   'qty': qty, 'side': -1, 'price': ce_bid},
                {'symbol': f'PE_{s_pe}',   'qty': qty, 'side': -1, 'price': pe_bid},
                {'symbol': f'CE_{s_ce_w}', 'qty': qty, 'side':  1, 'price': ce_w_ask},
                {'symbol': f'PE_{s_pe_w}', 'qty': qty, 'side':  1, 'price': pe_w_ask},
            ]
            
            sim_id = f"IB_W{wing_dist}_{sym_prefix}_{s_ce}"
            margin = wing_dist * qty
            
            broker.virtual_place_basket(sim_id, margin, legs, features=features, trigger=f"MOM:{momentum:.1f}")
            broker.positions[sim_id]['margin_locked'] = margin
            logger.info(f"[STRATEGY ENTRY] {sim_id} Defensive Butterfly Placed.")

        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Price data missing for IB selection: {e}")

if __name__ == "__main__":
    import asyncio
    import sys
    import os
    # Add root to path so imports work when running directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from paper_trading.standalone_runner import run_standalone
    
    asyncio.run(run_standalone(evaluate_market_tick, symbol="NSE:NIFTY50-INDEX", qty=650))
