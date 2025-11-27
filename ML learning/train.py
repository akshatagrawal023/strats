"""
Training Script for Option Chain Models

Trains all model architectures:
- Hybrid CNN-RNN
- 3D CNN
- Simple LSTM
- Transformer
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import json

from load_data import load_training_data
from features import get_spot_price
from models import create_model


class OptionChainDataset(Dataset):
    """PyTorch Dataset for option chain training data."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalize: bool = True
    ):
        """
        Args:
            X: Input windows (n_samples, window_size, channels, strikes)
            y: Labels (n_samples,)
            normalize: Whether to normalize inputs
        """
        self.X = X
        self.y = y
        self.normalize = normalize
        
        if normalize:
            self._normalize_data()
    
    def _normalize_data(self):
        """Normalize input data by channel."""
        # X shape: (n_samples, window_size, channels, strikes)
        n_samples, window_size, channels, strikes = self.X.shape
        
        # Normalize each channel independently
        for ch in range(channels):
            channel_data = self.X[:, :, ch, :]
            
            # Compute stats (ignoring NaN)
            mean = np.nanmean(channel_data)
            std = np.nanstd(channel_data)
            
            if std > 1e-6:
                self.X[:, :, ch, :] = (channel_data - mean) / std
            else:
                self.X[:, :, ch, :] = channel_data - mean
            
            # Replace NaN with 0
            self.X[:, :, ch, :] = np.nan_to_num(self.X[:, :, ch, :], nan=0.0)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])
        return X, y.squeeze()


class ModelTrainer:
    """Trainer for option chain models."""
    
    def __init__(
        self,
        symbol: str,
        window_size: int = 100,  # 300 seconds
        prediction_horizon: int = 10,  # 30 seconds
        move_threshold: float = 0.001,  # 0.1%
        train_split: float = 0.8,
        data_dir: Optional[str] = None,
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ):
        self.symbol = symbol
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.move_threshold = move_threshold
        self.train_split = train_split
        self.batch_size = batch_size
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load data
        print(f"\n{'='*60}")
        print(f"Loading data for {symbol}...")
        print(f"{'='*60}")
        data = load_training_data(symbol, data_dir)
        self.features = data['features']  # (n, 11, strikes)
        self.greeks = data['greeks']      # (n, 10, strikes)
        self.timestamps = data['timestamps']
        
        # Combine features and greeks
        self.combined = np.concatenate([self.features, self.greeks], axis=1)  # (n, 21, strikes)
        
        # Extract spot prices for labeling
        print("Extracting spot prices...")
        self.spot_prices = np.array([get_spot_price(f) for f in self.features])
        
        print(f"Data loaded: {self.combined.shape}")
        print(f"Window size: {window_size} steps ({window_size * 3} seconds)")
        print(f"Prediction horizon: {prediction_horizon} steps ({prediction_horizon * 3} seconds)")
        print(f"Device: {self.device}")
    
    def create_windows_and_labels(
        self,
        step_size: int = 1  # Step between windows (1 = every timestep)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create rolling windows and labels.
        
        Args:
            step_size: Step between windows
        
        Returns:
            (X, y) where:
            - X: (n_samples, window_size, channels, strikes)
            - y: (n_samples,) labels (0=down, 1=up)
        """
        print(f"\nCreating windows and labels (step_size={step_size})...")
        
        n_samples = len(self.combined)
        windows = []
        labels = []
        
        valid_count = 0
        neutral_count = 0
        
        for i in range(0, n_samples - self.window_size - self.prediction_horizon, step_size):
            # Input window
            X = self.combined[i:i+self.window_size]  # (window_size, 21, strikes)
            
            # Label
            current_idx = i + self.window_size - 1
            future_idx = current_idx + self.prediction_horizon
            
            if future_idx >= len(self.spot_prices):
                continue
            
            current_price = self.spot_prices[current_idx]
            future_price = self.spot_prices[future_idx]
            
            if np.isnan(current_price) or np.isnan(future_price):
                continue
            
            price_change = (future_price - current_price) / current_price
            
            # Generate label
            if price_change > self.move_threshold:
                y = 1  # Up
                valid_count += 1
            elif price_change < -self.move_threshold:
                y = 0  # Down
                valid_count += 1
            else:
                neutral_count += 1
                continue  # Skip neutral
            
            windows.append(X)
            labels.append(y)
        
        print(f"Created {len(windows)} samples")
        print(f"  Up labels: {sum(labels)}")
        print(f"  Down labels: {len(labels) - sum(labels)}")
        print(f"  Neutral (skipped): {neutral_count}")
        
        if len(windows) == 0:
            raise ValueError("No valid samples created!")
        
        return np.array(windows), np.array(labels)
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and validation sets (time-based)."""
        split_idx = int(len(X) * self.train_split)
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        
        return X_train, y_train, X_val, y_val
    
    def train_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Train a single model.
        
        Args:
            model_type: 'hybrid', '3dcnn', 'lstm', or 'transformer'
            X_train, y_train: Training data
            X_val, y_val: Validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization
            save_dir: Directory to save model
        
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*60}")
        
        # Create model
        model = create_model(
            model_type=model_type,
            input_channels=21,
            num_strikes=self.features.shape[2],
            time_steps=self.window_size
        ).to(self.device)
        
        # Create datasets
        train_dataset = OptionChainDataset(X_train, y_train, normalize=True)
        val_dataset = OptionChainDataset(X_val, y_val, normalize=True)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        
        # Handle class imbalance with weights
        class_counts = np.bincount(y_train)
        class_weights = torch.FloatTensor([
            len(y_train) / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0,
            len(y_train) / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
        ]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'best_epoch': 0
        }
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(num_epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for X_batch, y_batch in train_pbar:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += y_batch.size(0)
                train_correct += (predicted == y_batch).sum().item()
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * train_correct / train_total:.2f}%'
                })
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                history['best_val_acc'] = best_val_acc
                history['best_epoch'] = epoch + 1
                patience_counter = 0
                
                if save_dir:
                    save_path_dir = Path(save_dir)
                    save_path_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_path_dir / f"{self.symbol}_{model_type}_best.pth"
                    torch.save(model.state_dict(), save_path)
                    print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"\nBest validation accuracy: {best_val_acc:.2f}% (epoch {history['best_epoch']})")
        
        return history
    
    def train_all_models(
        self,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        save_dir: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Train all model types.
        
        Returns:
            Dictionary with training history for each model
        """
        # Create windows and labels
        X, y = self.create_windows_and_labels(step_size=1)
        
        # Split data
        X_train, y_train, X_val, y_val = self.split_data(X, y)
        
        # Train each model
        model_types = ['hybrid', '3dcnn', 'lstm', 'transformer']
        all_histories = {}
        
        for model_type in model_types:
            try:
                history = self.train_model(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    save_dir=save_dir
                )
                all_histories[model_type] = history
            except Exception as e:
                print(f"\nError training {model_type}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save training summary
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            summary_path = save_path / f"{self.symbol}_training_summary.json"
            summary = {
                'symbol': self.symbol,
                'window_size': self.window_size,
                'prediction_horizon': self.prediction_horizon,
                'move_threshold': self.move_threshold,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'histories': {
                    k: {
                        'best_val_acc': v['best_val_acc'],
                        'best_epoch': v['best_epoch'],
                        'final_train_acc': v['train_acc'][-1] if v['train_acc'] else 0,
                        'final_val_acc': v['val_acc'][-1] if v['val_acc'] else 0
                    }
                    for k, v in all_histories.items()
                }
            }
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nTraining summary saved to {summary_path}")
        
        return all_histories


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train option chain models')
    parser.add_argument('--symbol', type=str, default='HDFCBANK', help='Stock symbol')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--window_size', type=int, default=100, help='Window size in steps')
    parser.add_argument('--horizon', type=int, default=10, help='Prediction horizon in steps')
    parser.add_argument('--threshold', type=float, default=0.001, help='Move threshold for labels')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ModelTrainer(
        symbol=args.symbol,
        window_size=args.window_size,
        prediction_horizon=args.horizon,
        move_threshold=args.threshold,
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    
    # Train all models
    histories = trainer.train_all_models(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    for model_type, history in histories.items():
        print(f"{model_type.upper()}:")
        print(f"  Best Val Acc: {history['best_val_acc']:.2f}% (epoch {history['best_epoch']})")
        print(f"  Final Train Acc: {history['train_acc'][-1]:.2f}%")
        print(f"  Final Val Acc: {history['val_acc'][-1]:.2f}%")


if __name__ == "__main__":
    main()

