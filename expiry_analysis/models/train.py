"""
Training Script for Option Chain Direction Prediction Model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from expiry_analysis.models.cnn_rnn_model import create_model
from expiry_analysis.models.data_loader import create_dataloader
from expiry_analysis.chain_processor import OptionDataProcessor
from vol.matrix_greeks import MatrixGreeksCalculator
from expiry_analysis.config import (
    WINDOW_SIZE, STRIKE_COUNT, RISK_FREE_RATE, DAYS_TO_EXPIRY
)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X, y in tqdm(dataloader, desc="Training"):
        X = X.to(device)
        y = y.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Validating"):
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def train_model(
    underlying: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_type: str = "cnn_rnn",
    device: Optional[torch.device] = None
):
    """
    Train model for a given underlying.
    
    Args:
        underlying: Underlying symbol
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        model_type: "cnn_rnn" or "3d_cnn"
        device: PyTorch device (auto-detect if None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Initialize components
    processor = OptionDataProcessor(window_size=WINDOW_SIZE, strike_count=STRIKE_COUNT)
    greeks_calc = MatrixGreeksCalculator(risk_free_rate=RISK_FREE_RATE, days_to_expiry=DAYS_TO_EXPIRY)
    
    # TODO: Load historical data into processor
    # For now, this is a structure - you need to populate processor with data first
    
    # Create dataloaders
    train_loader = create_dataloader(
        processor=processor,
        greeks_calc=greeks_calc,
        underlying=underlying,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Split validation (80/20)
    dataset_size = len(train_loader.dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_loader.dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    # Determine input channels (23 base + derived features if added)
    sample_X, _ = next(iter(train_loader))
    input_channels = sample_X.shape[2]
    num_strikes = sample_X.shape[3]
    
    model = create_model(
        model_type=model_type,
        input_channels=input_channels,
        num_strikes=num_strikes,
        time_steps=WINDOW_SIZE
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"models/best_{underlying}_{model_type}.pth")
            print(f"Saved best model (val_acc: {val_acc:.4f})")
    
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    # Example usage
    train_model(
        underlying="RELIANCE",
        num_epochs=10,
        batch_size=16,
        learning_rate=0.001,
        model_type="cnn_rnn"
    )

