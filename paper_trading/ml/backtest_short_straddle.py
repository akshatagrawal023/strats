"""
backtest_short_straddle.py
--------------------------
2-Leg version of the Smile Mean Reversion strategy.
Captures the same relative-value edge with 50% less friction.
"""

import os
import sys
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from collections import deque
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.trade_costs import estimate_options_costs

NIFTY_LOT = 65

def bs_price(S, K, T, iv, option_type='CE', r=0.065):
    if T <= 0 or iv <= 0 or np.isnan(iv) or np.isnan(T):
        return max(0.0, (S - K) if option_type == 'CE' else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) if option_type == 'CE' else \
           (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

def compute_smile_spread(mat, atm_idx, wing_offset=2):
    ci, pi = atm_idx + wing_offset, atm_idx - wing_offset
    if ci >= mat.shape[1] or pi < 0: return np.nan
    a_iv = (mat[6, atm_idx] + mat[7, atm_idx]) / 2.0
    w_iv = (mat[6, ci] + mat[7, pi]) / 2.0
    return a_iv - w_iv

@dataclass
class Position:
    entry_tick: int; entry_ce: float; entry_pe: float; credit: float; cost: float; current_pnl: float = 0.0

def run_backtest(filepath: str, qty: int = 1300):
    print(f"Backtesting Short Straddle on {os.path.basename(filepath)} with Qty {qty}...")
    with h5py.File(filepath, 'r') as f:
        df = pd.DataFrame(f['ticks'][:]); g_3d = f['greeks_matrices'][:]
    
    n = len(df); spot = df['spot'].values
    spread_hist = deque(maxlen=1200); last_exit = -300
    positions = []; active: Optional[Position] = None; equity = np.zeros(n)

    for t in range(1, n):
        mat = g_3d[t]; atm_idx = int(np.nanargmin(np.abs(mat[8] - 1.0)))
        spread = compute_smile_spread(mat, atm_idx)
        if not np.isnan(spread): spread_hist.append(spread)
        smile_z = 0.0
        if len(spread_hist) >= 60:
            arr = np.array(spread_hist); mu, sig = arr.mean(), arr.std()
            if sig > 1e-8: smile_z = (spread - mu) / sig
            
        if active:
            T_now = float(df['T'].iloc[t]) if 'T' in df.columns else 0.001
            moneyness = mat[8]; K_atm = spot[t] / moneyness[atm_idx]
            c_ce = bs_price(spot[t], K_atm, T_now, float(mat[6, atm_idx]), 'CE')
            c_pe = bs_price(spot[t], K_atm, T_now, float(mat[7, atm_idx]), 'PE')
            active.current_pnl = (active.entry_ce - c_ce + active.entry_pe - c_pe) * qty
            
            hold = t - active.entry_tick
            if (smile_z <= -0.5 and hold > 60) or (active.current_pnl < -active.credit * 1.5) or (t == n-1):
                exit_cost = estimate_options_costs((c_ce+c_pe)/2, qty, is_sell=False) * 2
                slip = (0.05 * 4) * qty # 2 entry + 2 exit = 4 legs total slippage
                net = active.current_pnl - exit_cost - active.cost - slip
                positions.append({'net_pnl': net, 'reason': 'SIGNAL' if smile_z <= -0.5 else 'EOD/SL'})
                active, last_exit = None, t
                
        if not active and (t - last_exit) >= 300 and smile_z >= 1.0 and len(spread_hist) >= 60:
            T_ent = float(df['T'].iloc[t]) if 'T' in df.columns else 0.01
            moneyness = mat[8]; K_atm = spot[t] / moneyness[atm_idx]
            e_ce = bs_price(spot[t], K_atm, T_ent, float(mat[6, atm_idx]), 'CE')
            e_pe = bs_price(spot[t], K_atm, T_ent, float(mat[7, atm_idx]), 'PE')
            e_cost = estimate_options_costs((e_ce+e_pe)/2, qty, is_sell=True) * 2
            active = Position(t, e_ce, e_pe, (e_ce+e_pe)*qty, e_cost)
        equity[t] = sum(p['net_pnl'] for p in positions)

    df_res = pd.DataFrame(positions)
    print(f"Short Straddle Results:\nTrades: {len(df_res)} | Net PnL: \u20b9{equity[-1]:,.2f}")
    plt.figure(figsize=(10,6)); plt.plot(equity); plt.title("Short Straddle Equity"); plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "backtest_straddle.png"))
    return df_res

if __name__ == "__main__":
    h5 = glob.glob(os.path.join(os.path.dirname(__file__), "../hdf5_data_archives/*.h5"))[0]
    run_backtest(h5, qty=65*20)
