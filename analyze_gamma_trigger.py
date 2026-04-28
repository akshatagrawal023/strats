import h5py
import numpy as np

H5_PATH = "/Users/akshatagrawal/Desktop/strats/strats/paper_trading/hdf5_data_archives/NIFTY50_20260428_s49.h5"

try:
    with h5py.File(H5_PATH, 'r') as f:
        data = f['ticks']
        print(f"Dataset Shape: {data.shape}")
        if len(data.shape) == 2:
            iv_z = data[:, 6]
            print(f"Current IV_Z: {iv_z[-1]:.4f}")
            print(f"Today Range: [{np.min(iv_z):.2f}, {np.max(iv_z):.2f}]")
        else:
            print("Dataset is 1D. Checking first row...")
            print(data[0])

except Exception as e:
    print(f"Error: {e}")
