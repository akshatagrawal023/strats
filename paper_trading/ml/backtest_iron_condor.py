"""
backtest_iron_condor.py
-----------------------
v5.2 — PRECISE FRICTION & 65-UNIT LOTS:
  - Exact tax/brokerage calculation per trade using utils.trade_costs.
  - Lot size = 65 (User defined).
  - 20-lot default simulation (1,300 qty).
  - Slippage = 0.05 per leg.
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

NIFTY_LOT = 65  # User requested lot size

def bs_price(S, K, T, iv, option_type='CE', r=0.065):
    if T <= 0 or iv <= 0 or np.isnan(iv) or np.isnan(T):
        return max(0.0, (S - K) if option_type == 'CE' else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) if option_type == 'CE' else \
           (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

def price_ic_legs(mat, atm_idx, wing_offset, spot, T):
    n = mat.shape[1]
    ci = min(atm_idx + wing_offset, n - 1); pi = max(atm_idx - wing_offset, 0)
    def s(v): return float(v) if not np.isnan(v) else 0.15
    K_atm = spot / s(mat[8, atm_idx]) if s(mat[8, atm_idx]) > 0 else spot
    K_ci = spot / s(mat[8, ci]) if s(mat[8, ci]) > 0 else spot * 1.01
    K_pi = spot / s(mat[8, pi]) if s(mat[8, pi]) > 0 else spot * 0.99
    
    a_ce = bs_price(spot, K_atm, T, s(mat[6, atm_idx]), 'CE')
    a_pe = bs_price(spot, K_atm, T, s(mat[7, atm_idx]), 'PE')
    w_ce = bs_price(spot, K_ci,  T, s(mat[6, ci]), 'CE')
    w_pe = bs_price(spot, K_pi,  T, s(mat[7, pi]), 'PE')
    return a_ce, a_pe, w_ce, w_pe, K_atm, K_ci, K_pi

def compute_smile_spread(mat, atm_idx, wing_offset):
    ci, pi = atm_idx + wing_offset, atm_idx - wing_offset
    if ci >= mat.shape[1] or pi < 0: return np.nan
    a_iv = (mat[6, atm_idx] + mat[7, atm_idx]) / 2.0
    w_iv = (mat[6, ci] + mat[7, pi]) / 2.0
    return a_iv - w_iv

@dataclass
class Position:
    entry_tick: int; entry_atm_ce: float; entry_atm_pe: float; entry_otm_ce: float; entry_otm_pe: float
    net_credit_unit: float; entry_cost: float; current_pnl: float = 0.0

@dataclass
class Config:
    entry_spread_z: float = 1.0; exit_spread_z: float = -0.5
    qty: int = NIFTY_LOT * 20
    slippage: float = 0.05

def run_backtest(filepath: str, config: Config = None) -> pd.DataFrame:
    if config is None: config = Config()
    with h5py.File(filepath, 'r') as f:
        df = pd.DataFrame(f['ticks'][:]); g_3d = f['greeks_matrices'][:]
    
    n = len(df); spot = df['spot'].values
    spread_hist = deque(maxlen=1200); last_exit = -300
    positions = []; active: Optional[Position] = None; equity = np.zeros(n)
    
    def calc_transaction_costs(atm_p, otm_p, qty):
        # Entry/Exit for 4 legs (2 short ATM, 2 long OTM)
        short_cost = estimate_options_costs(atm_p, qty, is_sell=True) * 2
        long_cost  = estimate_options_costs(otm_p, qty, is_buy=True) * 2
        return short_cost + long_cost

    for t in range(1, n):
        mat = g_3d[t]; atm_idx = int(np.nanargmin(np.abs(mat[8] - 1.0)))
        spread = compute_smile_spread(mat, atm_idx, 2)
        if not np.isnan(spread): spread_hist.append(spread)
        
        smile_z = 0.0
        if len(spread_hist) >= 60:
            arr = np.array(spread_hist); mu, sig = arr.mean(), arr.std()
            if sig > 1e-8: smile_z = (spread - mu) / sig
            
        if active:
            T_now = float(df['T'].iloc[t]) if 'T' in df.columns else 0.001
            c_ace, c_ape, c_wce, c_wpe, _, _, _ = price_ic_legs(mat, atm_idx, 2, spot[t], T_now)
            active.current_pnl = ((active.entry_atm_ce - c_ace + active.entry_atm_pe - c_ape) + (c_wce - active.entry_otm_ce + c_wpe - active.entry_otm_pe)) * config.qty
            
            hold = t - active.entry_tick
            if (smile_z <= config.exit_spread_z and hold > 60) or (active.current_pnl < -active.net_credit_unit * config.qty * 1.5) or (hold > 7200):
                exit_cost = calc_transaction_costs((c_ace+c_ape)/2, (c_wce+c_wpe)/2, config.qty)
                slip = (config.slippage * 8) * config.qty
                net = active.current_pnl - exit_cost - active.entry_cost - slip
                positions.append({'net_pnl': net, 'exit_reason': 'SIGNAL' if smile_z <= config.exit_spread_z else 'HARD'})
                active, last_exit = None, t
                
        if not active and (t - last_exit) >= 300 and smile_z >= config.entry_spread_z and len(spread_hist) >= 60:
            T_ent = float(df['T'].iloc[t]) if 'T' in df.columns else 0.01
            e_ace, e_ape, e_wce, e_wpe, _, _, _ = price_ic_legs(mat, atm_idx, 2, spot[t], T_ent)
            e_cost = calc_transaction_costs((e_ace+e_ape)/2, (e_wce+e_wpe)/2, config.qty)
            active = Position(t, e_ace, e_ape, e_wce, e_wpe, (e_ace+e_ape-e_wce-e_wpe), e_cost)
        equity[t] = sum(p['net_pnl'] for p in positions)

    print(f"\nIRON CONDOR (Qty: {config.qty}) | Net PnL: \u20b9{equity[-1]:,.2f}")
    plt.plot(equity); plt.savefig(os.path.join(os.path.dirname(__file__), "backtest_results_v5.png"))
    return pd.DataFrame(positions)

if __name__ == "__main__":
    h5 = glob.glob(os.path.join(os.path.dirname(__file__), "../hdf5_data_archives/*.h5"))[0]
    run_backtest(h5)
