"""
Data Loader for Option Chain Model Training

Prepares data from OptionDataProcessor into PyTorch-ready format.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from expiry_analysis.chain_processor import OptionDataProcessor
from greeks.matrix_greeks import MatrixGreeksCalculator
from expiry_analysis.models.feature_engineering import add_derived_features, normalize_matrix


class OptionChainDataset(Dataset):
    """
    PyTorch Dataset for option chain data.
    
    Loads rolling windows of matrices and generates labels based on future price movement.
    """
    
    def __init__(
        self,
        processor: OptionDataProcessor,
        greeks_calc: MatrixGreeksCalculator,
        underlying: str,
        window_size: int = 300,
        prediction_horizon: int = 10,  # timesteps ahead
        move_threshold: float = 0.002,  # 0.2% move threshold
        add_derived: bool = True,
        normalize: bool = True
    ):
        """
        Args:
            processor: OptionDataProcessor with historical data
            greeks_calc: MatrixGreeksCalculator
            underlying: Underlying symbol
            window_size: Number of timesteps to use
            prediction_horizon: How many timesteps ahead to predict
            move_threshold: Minimum move % for label generation
            add_derived: Whether to add derived features
            normalize: Whether to normalize data
        """
        self.processor = processor
        self.greeks_calc = greeks_calc
        self.underlying = underlying
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.move_threshold = move_threshold
        self.add_derived = add_derived
        self.normalize = normalize
        
        # Load all available data
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess all available data."""
        timestamps, matrices = self.processor.get_matrix(self.underlying, window=None)
        
        if matrices is None or len(matrices) < self.window_size + self.prediction_horizon:
            self.samples = []
            return
        
        # Add Greeks to each matrix
        matrices_with_greeks = []
        for mat in matrices:
            greeks = self.greeks_calc.get_greeks_only(mat)
            combined = np.vstack([mat, greeks])  # (23, strikes)
            matrices_with_greeks.append(combined)
        
        matrices_with_greeks = np.array(matrices_with_greeks)  # (time, 23, strikes)
        
        # Add derived features if requested
        if self.add_derived:
            extended_matrices = []
            for i in range(len(matrices_with_greeks)):
                # Use last 20 timesteps for momentum
                hist_start = max(0, i - 20)
                hist = matrices_with_greeks[hist_start:i+1]
                extended = add_derived_features(matrices_with_greeks[i], hist)
                extended_matrices.append(extended)
            matrices_with_greeks = np.array(extended_matrices)
        
        # Generate labels
        self.samples = []
        spot_prices = matrices_with_greeks[:, 11, 0]  # UNDERLYING_LTP channel
        
        for i in range(len(matrices_with_greeks) - self.window_size - self.prediction_horizon):
            # Input window
            X = matrices_with_greeks[i:i+self.window_size]  # (window_size, channels, strikes)
            
            # Label: direction of future price movement
            current_price = spot_prices[i + self.window_size - 1]
            future_price = spot_prices[i + self.window_size + self.prediction_horizon - 1]
            
            if np.isnan(current_price) or np.isnan(future_price):
                continue
            
            price_change = (future_price - current_price) / current_price
            
            if price_change > self.move_threshold:
                label = 1  # Up
            elif price_change < -self.move_threshold:
                label = 0  # Down
            else:
                label = -1  # Neutral (skip for binary classification)
            
            if label == -1:
                continue
            
            # Normalize if requested
            if self.normalize:
                X, _, _ = normalize_matrix(X)
            
            self.samples.append({
                'X': X,
                'y': label,
                'price_change': price_change
            })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]
        X = torch.FloatTensor(sample['X'])
        y = torch.LongTensor([sample['y']])
        return X, y.squeeze()


def create_dataloader(
    processor: OptionDataProcessor,
    greeks_calc: MatrixGreeksCalculator,
    underlying: str,
    batch_size: int = 32,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Create PyTorch DataLoader.
    
    Args:
        processor: OptionDataProcessor
        greeks_calc: MatrixGreeksCalculator
        underlying: Underlying symbol
        batch_size: Batch size
        shuffle: Whether to shuffle
        **dataset_kwargs: Additional arguments for OptionChainDataset
    
    Returns:
        DataLoader
    """
    dataset = OptionChainDataset(
        processor=processor,
        greeks_calc=greeks_calc,
        underlying=underlying,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )


# Example usage
if __name__ == "__main__":
    from expiry_analysis.config import WINDOW_SIZE, STRIKE_COUNT, RISK_FREE_RATE, DAYS_TO_EXPIRY
    
    # Initialize
    processor = OptionDataProcessor(window_size=WINDOW_SIZE, strike_count=STRIKE_COUNT)
    greeks_calc = MatrixGreeksCalculator(risk_free_rate=RISK_FREE_RATE, days_to_expiry=DAYS_TO_EXPIRY)
    
    # Note: You need to populate processor with data first
    # This is just a structure example
    
    # Create dataloader
    dataloader = create_dataloader(
        processor=processor,
        greeks_calc=greeks_calc,
        underlying="RELIANCE",
        batch_size=16,
        window_size=300,
        prediction_horizon=10
    )
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    # Iterate
    for batch_X, batch_y in dataloader:
        print(f"Batch X shape: {batch_X.shape}")  # (batch, 300, channels, strikes)
        print(f"Batch y shape: {batch_y.shape}")  # (batch,)
        break

