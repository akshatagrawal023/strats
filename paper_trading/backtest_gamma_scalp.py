import h5py
import numpy as np
import pandas as pd
import os
import sys
import time
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_trading.market_features import compute_spot_ewm_volatility

FILES = [
    "/Users/akshatagrawal/Desktop/strats/strats/paper_trading/hdf5_data_archives/NIFTY50_20260428_.h5",
    "/Users/akshatagrawal/Desktop/strats/strats/paper_trading/hdf5_data_archives/NIFTY50_20260428.h5",
    "/Users/akshatagrawal/Desktop/strats/strats/paper_trading/hdf5_data_archives/NIFTY50_20260428_s49.h5"
]

START_TS = time.mktime((2026, 4, 28, 12, 0, 0, 0, 0, 0))

def run_simulation(df, target_z, target_accel, exit_accel):
    spot_history = deque(maxlen=200)
    in_position = False
    entry_price = 0
    entry_ts = 0
    trades = []

    for i, row in df.iterrows():
        spot = row['spot']
        spot_history.append(spot)
        if len(spot_history) < 30: continue
        
        stats = compute_spot_ewm_volatility(spot_history)
        accel = stats['acceleration']
        iv_z = row['iv_z']
        
        if not in_position:
            if iv_z < target_z or (iv_z < -1.0 and accel > target_accel):
                in_position = True
                entry_price = spot
                entry_ts = row['ts']
        else:
            pnl_points = abs(spot - entry_price)
            time_passed_mins = (row['ts'] - entry_ts) / 60
            decay = time_passed_mins * 1.5 
            net_pnl = pnl_points - decay
            
            # Take Profit (20%)
            if net_pnl > 12:
                trades.append(net_pnl - 1.0) # Subtract 1 pt for Slippage + Costs
                in_position = False
            elif accel < exit_accel and net_pnl > 2:
                trades.append(net_pnl - 1.0) # Subtract 1 pt for Slippage + Costs
                in_position = False
            elif net_pnl < -10:
                trades.append(net_pnl - 1.0) # Subtract 1 pt for Slippage + Costs
                in_position = False
    
    return trades

def sweep_parameters():
    all_data = []
    for path in FILES:
        if not os.path.exists(path): continue
        with h5py.File(path, 'r') as f:
            raw = f['ticks'][:]
            df_chunk = pd.DataFrame({'ts': raw['timestamp'], 'spot': raw['spot'], 'iv_z': raw['iv_z_score']})
            all_data.append(df_chunk)
    
    df = pd.concat(all_data).sort_values('ts').drop_duplicates('ts')
    df = df[df['ts'] >= START_TS].reset_index(drop=True)

    results = []
    print(f"Running Parameter Sweep...")
    
    for z in [-1.5, -2.0, -2.5]:
        for accel in [1.5, 2.0, 3.0]:
            for ex in [-0.5, -1.0, -2.0]:
                trades = run_simulation(df, z, accel, ex)
                if trades:
                    results.append({
                        'z': z, 'accel': accel, 'exit_accel': ex,
                        'n': len(trades), 'pnl': sum(trades), 'win_rate': len([t for t in trades if t>0])/len(trades)
                    })

    res_df = pd.DataFrame(results).sort_values('pnl', ascending=False)
    print("\n--- PARAMETER SWEEP RESULTS ---")
    print(res_df.head(10).to_string(index=False))
    
    best = res_df.iloc[0]
    print(f"\nWINNER: Z={best['z']}, Accel={best['accel']}, ExitAccel={best['exit_accel']}")
    print(f"PnL: {best['pnl']:.2f} pts | Trades: {best['n']}")

if __name__ == "__main__":
    sweep_parameters()
