"""
backtest_iron_condor.py
-----------------------
Walk-forward backtester for Iron Condor/Butterfly — Smile Mean Reversion.

v5.1 — SCALE, SLIPPAGE & PREMIUM PnL:
  - Actual BS Premium PnL (Sell ATM, Buy Wings)
  - Scaling support (Lot size vs fixed costs)
  - Realistic Slippage penalty (0.05 per leg)
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
from paper_trading.ml.vol_data_prep import compute_realized_vol
from utils.trade_costs import estimate_options_costs

NIFTY_LOT = 75

CH_GAMMA = 1
CH_THETA = 2
CH_VEGA = 3
CH_CE_IV = 6
CH_PE_IV = 7
CH_MONEYNESS = 8
CH_SKEW = 10


def bs_price(S, K, T, iv, option_type='CE', r=0.065):
    if T <= 0 or iv <= 0 or np.isnan(iv) or np.isnan(T):
        return max(0.0, (S - K) if option_type == 'CE' else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    if option_type == 'CE':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def price_ic_legs(mat, atm_idx, wing_offset, spot, T):
    n = mat.shape[1]
    ci = min(atm_idx + wing_offset, n - 1); pi = max(atm_idx - wing_offset, 0)
    def safe(v): return float(v) if not np.isnan(v) else 0.15
    moneyness = mat[CH_MONEYNESS]
    K_atm = spot / safe(moneyness[atm_idx]) if safe(moneyness[atm_idx]) > 0 else spot
    K_ci = spot / safe(moneyness[ci]) if safe(moneyness[ci]) > 0 else spot * (1 + wing_offset * 0.005)
    K_pi = spot / safe(moneyness[pi]) if safe(moneyness[pi]) > 0 else spot * (1 - wing_offset * 0.005)
    atm_ce = bs_price(spot, K_atm, T, safe(mat[CH_CE_IV, atm_idx]), 'CE')
    atm_pe = bs_price(spot, K_atm, T, safe(mat[CH_PE_IV, atm_idx]), 'PE')
    otm_ce = bs_price(spot, K_ci,  T, safe(mat[CH_CE_IV, ci]), 'CE')
    otm_pe = bs_price(spot, K_pi,  T, safe(mat[CH_PE_IV, pi]), 'PE')
    return atm_ce, atm_pe, otm_ce, otm_pe, K_atm, K_ci, K_pi


def compute_smile_spread(mat, atm_idx, wing_offset):
    n = mat.shape[1]
    ci, pi = atm_idx + wing_offset, atm_idx - wing_offset
    if ci >= n or pi < 0: return np.nan
    atm_iv = (mat[CH_CE_IV, atm_idx] + mat[CH_PE_IV, atm_idx]) / 2.0
    wing_avg = (mat[CH_CE_IV, ci] + mat[CH_PE_IV, pi]) / 2.0
    return atm_iv - wing_avg if not (np.isnan(atm_iv) or np.isnan(wing_avg)) else np.nan


@dataclass
class Position:
    entry_tick: int
    entry_spot: float
    entry_spread: float
    entry_spread_z: float
    entry_atm_iv: float
    entry_atm_ce: float; entry_atm_pe: float; entry_otm_ce: float; entry_otm_pe: float
    net_credit: float
    current_pnl: float = 0.0
    exit_tick: Optional[int] = None; exit_reason: str = ""


@dataclass
class Config:
    entry_spread_z: float = 1.0; exit_spread_z: float = -0.5
    wing_offset: int = 2; qty: int = NIFTY_LOT
    stop_loss_pct: float = 1.5; cooldown_ticks: int = 300
    spread_history_len: int = 1200; slippage_per_leg: float = 0.05


def run_backtest(filepath: str, config: Config = None) -> pd.DataFrame:
    if config is None: config = Config()
    with h5py.File(filepath, 'r') as f:
        ticks = f['ticks'][:]; df = pd.DataFrame(ticks); greeks_3d = f['greeks_matrices'][:]
    
    n = len(df); spot = df['spot'].values
    round_trip = estimate_options_costs(premium=100, lot_size=1, is_sell=True) * 8
    spread_hist = deque(maxlen=config.spread_history_len)
    positions = []; active: Optional[Position] = None; last_exit = -config.cooldown_ticks
    equity = np.zeros(n); cum_pnl = 0.0
    spread_series, spread_z_series = np.full(n, np.nan), np.full(n, np.nan)

    for t in range(1, n):
        mat = greeks_3d[t]; moneyness = mat[CH_MONEYNESS]
        atm_idx = int(np.nanargmin(np.abs(moneyness - 1.0)))
        spread = compute_smile_spread(mat, atm_idx, config.wing_offset); spread_series[t] = spread
        if not np.isnan(spread): spread_hist.append(spread)
        
        spread_z = 0.0
        if len(spread_hist) >= 60:
            arr = np.array(spread_hist); mu, sigma = arr.mean(), arr.std()
            if sigma > 1e-8: spread_z = (spread - mu) / sigma
        spread_z_series[t] = spread_z
        
        if active is not None:
            T_now = float(df['T'].iloc[t]) if 'T' in df.columns else 0.001
            c_ace, c_ape, c_oce, c_ope, _, _, _ = price_ic_legs(mat, atm_idx, config.wing_offset, spot[t], T_now)
            active.current_pnl = ((active.entry_atm_ce - c_ace + active.entry_atm_pe - c_ape) + (c_oce - active.entry_otm_ce + c_ope - active.entry_otm_pe)) * config.qty
            
            hold = t - active.entry_tick
            exited = False
            if spread_z <= config.exit_spread_z and hold > 60: active.exit_tick, active.exit_reason = t, "SPREAD_REVERTED"; exited = True
            elif active.net_credit > 0 and active.current_pnl < -active.net_credit * config.stop_loss_pct: active.exit_tick, active.exit_reason = t, "STOP_LOSS"; exited = True
            elif hold >= 7200: active.exit_tick, active.exit_reason = t, "TIME_EXIT"; exited = True
            
            if exited:
                slp = (config.slippage_per_leg * 8) * config.qty
                net_pnl = active.current_pnl - slp - round_trip
                cum_pnl += net_pnl
                positions.append({
                    'entry_tick': active.entry_tick, 'exit_tick': t, 'net_pnl': net_pnl, 'gross_pnl': active.current_pnl,
                    'slippage': slp, 'exit_reason': active.exit_reason
                })
                last_exit, active = t, None

        if active is None and (t - last_exit) >= config.cooldown_ticks:
            if len(spread_hist) >= 60 and spread_z >= config.entry_spread_z:
                T_ent = float(df['T'].iloc[t]) if 'T' in df.columns else 0.01
                e_ace, e_ape, e_oce, e_ope, K_atm, K_ci, K_pi = price_ic_legs(mat, atm_idx, config.wing_offset, spot[t], T_ent)
                active = Position(entry_tick=t, entry_spot=spot[t], entry_spread=spread, entry_spread_z=spread_z,
                                  entry_atm_iv=float(df['atm_iv'].iloc[t]) if 'atm_iv' in df.columns else 0,
                                  entry_atm_ce=e_ace, entry_atm_pe=e_ape, entry_otm_ce=e_oce, entry_otm_pe=e_ope,
                                  net_credit=((e_ace + e_ape) - (e_oce + e_ope)) * config.qty)
        equity[t] = cum_pnl

    trades_df = pd.DataFrame(positions)
    print(f"\n{'='*60}\n  IRON CONDOR v5.1 (Scale & Slippage)\n{'='*60}")
    print(f"  Trades: {len(trades_df)} | Net PnL: ₹{trades_df['net_pnl'].sum() if not trades_df.empty else 0:,.2f}")
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    axes[0].plot(equity); axes[0].set_title("Equity Curve"); axes[1].plot(spot); axes[1].set_title("Spot")
    axes[2].plot(spread_series * 100); axes[2].set_title("Smile Spread (%)"); axes[3].plot(spread_z_series); axes[3].set_title("Z-Score")
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "backtest_results_v5.png"))
    trades_df.to_csv(os.path.join(os.path.dirname(__file__), "backtest_trades_v5.csv"), index=False)
    return trades_df

if __name__ == "__main__":
    h5_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hdf5_data_archives")
    h5_files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
    if h5_files: run_backtest(h5_files[0], Config(qty=NIFTY_LOT * 20))
