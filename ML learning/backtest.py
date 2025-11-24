"""
Backtest Script for Option Chain Models

Uses rolling window of 300 seconds (100 steps at 3s intervals)
Predicts 30 seconds ahead (10 steps)
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd

from load_data import load_training_data
from features import compute_all_features
from models import create_model


class BacktestEngine:
    """
    Backtest engine for option chain models.
    """
    
    def __init__(
        self,
        symbol: str,
        model_path: Optional[str] = None,
        model_type: str = 'hybrid',
        window_size: int = 100,  # 300 seconds = 100 steps at 3s intervals
        prediction_horizon: int = 10,  # 30 seconds = 10 steps
        move_threshold: float = 0.001,  # 0.1% minimum move
        data_dir: Optional[str] = None
    ):
        self.symbol = symbol
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.move_threshold = move_threshold
        
        # Load data
        print(f"Loading data for {symbol}...")
        data = load_training_data(symbol, data_dir)
        self.features = data['features']  # (n, 11, strikes)
        self.greeks = data['greeks']      # (n, 10, strikes)
        self.timestamps = data['timestamps']
        
        # Combine features and greeks
        self.combined = np.concatenate([self.features, self.greeks], axis=1)  # (n, 21, strikes)
        
        # Extract spot prices for labeling
        # Note: UNDERLYING_LTP is not stored in saved HDF5 data
        # We approximate spot from strikes and option prices
        from features import get_spot_price
        self.spot_prices = np.array([get_spot_price(f) for f in self.features])
        
        # Load or create model
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}...")
            self.model = self._load_model(model_path, model_type)
        else:
            print(f"Creating new {model_type} model...")
            self.model = create_model(
                model_type=model_type,
                input_channels=21,
                num_strikes=self.features.shape[2],
                time_steps=window_size
            )
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Data shape: {self.combined.shape}")
        print(f"Window size: {window_size} steps ({window_size * 3} seconds)")
        print(f"Prediction horizon: {prediction_horizon} steps ({prediction_horizon * 3} seconds)")
    
    def _load_model(self, model_path: str, model_type: str) -> nn.Module:
        """Load trained model."""
        model = create_model(
            model_type=model_type,
            input_channels=21,
            num_strikes=self.features.shape[2],
            time_steps=self.window_size
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def generate_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate labels for backtesting.
        
        Returns:
            (labels, price_changes) where labels: 1=up, 0=down, -1=neutral
        """
        n_samples = len(self.combined)
        labels = np.full(n_samples, -1, dtype=int)
        price_changes = np.full(n_samples, np.nan)
        
        for i in range(n_samples - self.window_size - self.prediction_horizon):
            current_idx = i + self.window_size - 1
            future_idx = current_idx + self.prediction_horizon
            
            current_price = self.spot_prices[current_idx]
            future_price = self.spot_prices[future_idx]
            
            if np.isnan(current_price) or np.isnan(future_price):
                continue
            
            price_change = (future_price - current_price) / current_price
            
            if price_change > self.move_threshold:
                labels[current_idx] = 1  # Up
            elif price_change < -self.move_threshold:
                labels[current_idx] = 0  # Down
            # else: -1 (neutral, skip)
            
            price_changes[current_idx] = price_change
        
        return labels, price_changes
    
    def create_windows(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create rolling windows for backtesting.
        
        Returns:
            (X, y, indices) where:
            - X: (n_samples, window_size, 21, strikes)
            - y: (n_samples,) labels
            - indices: (n_samples,) original indices
        """
        n_samples = len(self.combined)
        windows = []
        labels = []
        indices = []
        
        label_arr, price_changes = self.generate_labels()
        
        for i in range(n_samples - self.window_size - self.prediction_horizon):
            # Input window
            X = self.combined[i:i+self.window_size]  # (window_size, 21, strikes)
            
            # Label
            label_idx = i + self.window_size - 1
            y = label_arr[label_idx]
            
            # Skip neutral labels
            if y == -1:
                continue
            
            windows.append(X)
            labels.append(y)
            indices.append(label_idx)
        
        if len(windows) == 0:
            return np.array([]), np.array([]), np.array([])
        
        return np.array(windows), np.array(labels), np.array(indices)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input windows.
        
        Args:
            X: Input windows (n_samples, window_size, 21, strikes)
        
        Returns:
            Predictions (n_samples, 2) - logits for [down, up]
        """
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy()
    
    def run_backtest(
        self,
        step_size: int = 10,  # Step between predictions (30 seconds)
        confidence_threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Run backtest on the model.
        
        Args:
            step_size: Step between predictions
            confidence_threshold: Minimum probability for signal
        
        Returns:
            DataFrame with backtest results
        """
        print(f"\nRunning backtest for {self.symbol}...")
        print(f"Step size: {step_size} steps ({step_size * 3} seconds)")
        print(f"Confidence threshold: {confidence_threshold}")
        
        n_samples = len(self.combined)
        results = []
        
        label_arr, price_changes = self.generate_labels()
        
        for i in range(0, n_samples - self.window_size - self.prediction_horizon, step_size):
            # Input window
            X = self.combined[i:i+self.window_size]  # (window_size, 21, strikes)
            X = X[np.newaxis, :]  # (1, window_size, 21, strikes)
            
            # Label
            label_idx = i + self.window_size - 1
            future_idx = label_idx + self.prediction_horizon
            
            if label_idx >= len(label_arr) or future_idx >= len(self.spot_prices):
                continue
            
            true_label = label_arr[label_idx]
            if true_label == -1:  # Skip neutral
                continue
            
            current_price = self.spot_prices[label_idx]
            future_price = self.spot_prices[future_idx]
            actual_change = price_changes[label_idx]
            
            # Predict
            pred_logits = self.predict(X)[0]  # (2,)
            pred_probs = torch.softmax(torch.FloatTensor(pred_logits), dim=0).numpy()
            pred_label = int(np.argmax(pred_probs))
            pred_confidence = float(pred_probs[pred_label])
            
            # Compute features for analysis
            features_window = self.features[i:i+self.window_size]
            greeks_window = self.greeks[i:i+self.window_size]
            feature_dict = compute_all_features(features_window, greeks_window)
            
            # Record result
            result = {
                'timestamp': datetime.fromtimestamp(self.timestamps[label_idx]),
                'index': label_idx,
                'current_price': float(current_price),
                'future_price': float(future_price),
                'actual_change': float(actual_change),
                'actual_label': int(true_label),
                'pred_label': pred_label,
                'pred_confidence': pred_confidence,
                'pred_prob_up': float(pred_probs[1]),
                'pred_prob_down': float(pred_probs[0]),
                'correct': int(pred_label == true_label),
                **feature_dict
            }
            
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # Calculate metrics
        if len(df) > 0:
            accuracy = df['correct'].mean()
            up_accuracy = df[df['actual_label'] == 1]['correct'].mean() if len(df[df['actual_label'] == 1]) > 0 else 0
            down_accuracy = df[df['actual_label'] == 0]['correct'].mean() if len(df[df['actual_label'] == 0]) > 0 else 0
            
            print(f"\n=== Backtest Results ===")
            print(f"Total predictions: {len(df)}")
            print(f"Overall accuracy: {accuracy:.2%}")
            print(f"Up accuracy: {up_accuracy:.2%}")
            print(f"Down accuracy: {down_accuracy:.2%}")
            print(f"Up signals: {len(df[df['pred_label'] == 1])}")
            print(f"Down signals: {len(df[df['pred_label'] == 0])}")
        
        return df
    
    def run_backtest_with_features(
        self,
        step_size: int = 10,
        confidence_threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Run backtest and include feature analysis.
        """
        return self.run_backtest(step_size, confidence_threshold)


def main():
    """Example backtest usage."""
    import sys
    
    # Configuration
    symbol = "HDFCBANK"
    model_type = 'hybrid'  # 'hybrid', '3dcnn', 'lstm', 'transformer'
    model_path = None  # Path to trained model, or None for untrained
    
    # Create backtest engine
    engine = BacktestEngine(
        symbol=symbol,
        model_path=model_path,
        model_type=model_type,
        window_size=100,  # 300 seconds
        prediction_horizon=10,  # 30 seconds
        move_threshold=0.001  # 0.1%
    )
    
    # Run backtest
    results_df = engine.run_backtest(step_size=10, confidence_threshold=0.6)
    
    # Save results
    output_path = Path(__file__).parent / f"{symbol}_backtest_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print sample results
    print("\n=== Sample Results ===")
    print(results_df[['timestamp', 'actual_label', 'pred_label', 'pred_confidence', 'correct']].head(10))


if __name__ == "__main__":
    main()

