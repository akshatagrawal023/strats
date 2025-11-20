import h5py
import numpy as np
import os

def save_matrices(symbol, feature_matrix, greeks_matrix, timestamp):
    """Save feature and greeks matrices to HDF5"""
    os.makedirs('training_data', exist_ok=True)
    filepath = f'training_data/{symbol}_training.h5'
    
    with h5py.File(filepath, 'a') as f:
        # Append feature matrix
        if 'features' not in f:
            f.create_dataset('features', data=feature_matrix[np.newaxis, :, :],
                           maxshape=(None, feature_matrix.shape[0], feature_matrix.shape[1]),
                           compression='gzip')
            f.create_dataset('greeks', data=greeks_matrix[np.newaxis, :, :],
                           maxshape=(None, greeks_matrix.shape[0], greeks_matrix.shape[1]),
                           compression='gzip')
            f.create_dataset('timestamps', data=[timestamp], maxshape=(None,))
        else:
            f['features'].resize(f['features'].shape[0] + 1, axis=0)
            f['features'][-1] = feature_matrix
            f['greeks'].resize(f['greeks'].shape[0] + 1, axis=0)
            f['greeks'][-1] = greeks_matrix
            f['timestamps'].resize(f['timestamps'].shape[0] + 1, axis=0)
            f['timestamps'][-1] = timestamp

