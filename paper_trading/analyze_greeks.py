import h5py
import pandas as pd
import numpy as np
import os
import glob
import logging

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Greek Channel Mapping
GREEK_CHANNELS = {
    0: 'delta',
    1: 'gamma',
    2: 'theta',
    3: 'vega',
    4: 'vanna',
    5: 'volga',
    6: 'ce_iv',
    7: 'pe_iv',
    8: 'moneyness',
    9: 'theta_vega_ratio',
    10: 'raw_skew',
    11: 'charm',
    12: 'ce_iv_vel',
    13: 'pe_iv_vel'
}

def load_latest_h5(folder="paper_trading/hdf5_data_archives"):
    """Find and return the path to the most recent H5 file."""
    files = glob.glob(os.path.join(folder, "*.h5"))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def h5_to_dataframe(filepath):
    """
    Parses HDF5 trading logs into a clean, combined Pandas DataFrame.
    Automatically aligns scalar features with ATM-extracted Greeks.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    logging.info(f"Loading data from {filepath}...")
    with h5py.File(filepath, 'r') as f:
        # 1. Load the flat 'ticks' record array (scalars like spot, skew_z, iv_z)
        ticks = f['ticks'][:]
        df = pd.DataFrame(ticks)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            # Adjust to IST if needed, but let's keep it raw for now
        
        # 2. Extract ATM Greeks from the 3D matrix
        # Matrix shape: (n_ticks, n_channels, n_strikes)
        greeks_mat = f['greeks_matrices'][:]
        
        logging.info(f"Greeks Matrix Shape: {greeks_mat.shape} (Ticks, Channels, Strikes)")

        # We need to find the ATM strike for each tick. 
        # Moneyness (Ch 8) =~ 1.0 is ATM.
        moneyness_ch = greeks_mat[:, 8, :]
        atm_indices = np.nanargmin(np.abs(moneyness_ch - 1.0), axis=1)
        
        # Extract each Greek channel at the ATM index for every tick
        for ch_idx, ch_name in GREEK_CHANNELS.items():
            if ch_idx < greeks_mat.shape[1]:
                # Slice: for each tick i, pick channel ch_idx at strike atm_indices[i]
                df[f'atm_{ch_name}'] = greeks_mat[np.arange(len(df)), ch_idx, atm_indices]

        # 3. Feature Engineering: Spot Returns & Lead/Lag
        df['spot_ret'] = df['spot'].pct_change()
        # Look ahead 10 ticks (30s) to see if features "predict" movement
        df['spot_ret_future_10t'] = df['spot_ret'].shift(-10) 
        
        return df

def run_basic_analysis(df):
    """
    Performs critical statistical checks to see if features correlate with market movement.
    """
    if df is None or df.empty:
        print("DataFrame is empty.")
        return

    # Calculate correlations
    features = ['atm_ce_iv', 'atm_pe_iv', 'skew', 'iv_z_score', 'atm_ce_iv_vel', 'atm_pe_iv_vel']
    target = 'spot_ret_future_10t'
    
    # Filter for valid rows
    clean_df = df.dropna(subset=features + [target])
    
    if clean_df.empty:
        print("Not enough data points for correlation analysis.")
        return

    corr_matrix = clean_df[features + [target]].corr()
    
    print("\n" + "="*50)
    print(" FEATURE CORRELATION WITH FUTURE SPOT RETURNS (10 Ticks Ahead) ")
    print("="*50)
    print(corr_matrix[target].sort_values(ascending=False))
    print("="*50)

    if PLOT_AVAILABLE:
        # Plot Spot vs Skew & Panic
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(df['datetime'], df['spot'], color='blue', label='Spot Price')
        ax1.set_title("Spot Price vs Volatility Markers")
        ax1.legend(loc='upper left')
        
        ax2.plot(df['datetime'], df['iv_z_score'], color='orange', label='IV Z-Score', alpha=0.7)
        ax2.plot(df['datetime'], df['atm_ce_iv_vel'] * 100, color='red', label='IV Velocity (x100)', alpha=0.5)
        ax2.axhline(1.5, color='black', linestyle='--', label='Alpha Threshold')
        
        ax2.set_ylabel("Z-Score / Velocity")
        ax2.legend(loc='upper left')
        
        plt.tight_layout()
        plot_path = "paper_trading/hdf5_data_archives/analysis_plot.png"
        plt.savefig(plot_path)
        print(f"\nAnalysis plot saved to: {plot_path}")
    else:
        print("\nMatplotlib not installed. Skipping plot generation.")

if __name__ == "__main__":
    latest_file = load_latest_h5()
    if latest_file:
        df = h5_to_dataframe(latest_file)
        if df is not None:
            print("\n--- DATAFRAME HEAD ---")
            print(df[['datetime', 'spot', 'atm_ce_iv', 'skew', 'iv_z_score']].head())
            run_basic_analysis(df)
    else:
        print("No HDF5 log files found in paper_trading/hdf5_data_archives/")
