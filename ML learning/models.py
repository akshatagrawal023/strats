"""
Model Architectures for Option Chain Direction Prediction

Inspired by model_architecture.md, implements multiple architectures:
1. Hybrid CNN-RNN (Recommended)
2. 3D CNN
3. Transformer-based
4. Simple LSTM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HybridCNNRNN(nn.Module):
    """
    Hybrid CNN-RNN Architecture (Recommended)
    
    Input: (batch, time=100, channels=21, strikes=7)
    - CNN extracts spatial patterns (strike relationships) per timestep
    - RNN captures temporal evolution
    """
    
    def __init__(
        self,
        input_channels: int = 21,
        num_strikes: int = 7,
        time_steps: int = 100,
        cnn_filters: int = 64,
        lstm_hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.time_steps = time_steps
        self.num_strikes = num_strikes
        
        # 2D CNN for spatial patterns (strike dimension)
        self.conv1 = nn.Conv2d(input_channels, cnn_filters, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(cnn_filters, cnn_filters, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters)
        self.bn2 = nn.BatchNorm2d(cnn_filters)
        
        # Time-distributed pooling
        self.pool = nn.AdaptiveAvgPool2d((1, num_strikes))
        
        # Bidirectional LSTM for temporal patterns
        self.lstm = nn.LSTM(
            cnn_filters * num_strikes,
            lstm_hidden,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x: (batch, time, channels, strikes)
        batch_size = x.size(0)
        
        # Process each timestep through CNN
        cnn_outputs = []
        for t in range(self.time_steps):
            timestep = x[:, t, :, :]  # (batch, channels, strikes)
            timestep = timestep.unsqueeze(1)  # (batch, 1, channels, strikes)
            timestep = timestep.transpose(1, 2)  # (batch, channels, 1, strikes)
            
            # CNN
            out = F.relu(self.bn1(self.conv1(timestep)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.pool(out)  # (batch, filters, 1, strikes)
            out = out.squeeze(2)  # (batch, filters, strikes)
            out = out.reshape(batch_size, -1)  # (batch, filters * strikes)
            
            cnn_outputs.append(out)
        
        # Stack for LSTM: (batch, time, features)
        cnn_sequence = torch.stack(cnn_outputs, dim=1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_sequence)
        # Use last hidden state
        lstm_final = lstm_out[:, -1, :]  # (batch, hidden * 2)
        
        # Classification
        out = F.relu(self.fc1(lstm_final))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class CNN3D(nn.Module):
    """
    3D CNN Architecture
    
    Input: (batch, time=100, channels=21, strikes=7)
    - 3D convolutions capture spatio-temporal patterns
    """
    
    def __init__(
        self,
        input_channels: int = 21,
        num_strikes: int = 7,
        time_steps: int = 100,
        cnn_filters: int = 64,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        # 3D CNN layers
        self.conv3d1 = nn.Conv3d(1, cnn_filters, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(cnn_filters)
        self.conv3d2 = nn.Conv3d(cnn_filters, cnn_filters * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(cnn_filters * 2)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Output layers
        self.fc1 = nn.Linear(cnn_filters * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: (batch, time, channels, strikes)
        # Add channel dimension for 3D conv
        x = x.unsqueeze(1)  # (batch, 1, time, channels, strikes)
        
        # 3D CNN
        out = F.relu(self.bn1(self.conv3d1(x)))
        out = F.relu(self.bn2(self.conv3d2(out)))
        out = self.pool(out)  # (batch, filters, 1, 1, 1)
        out = out.view(out.size(0), -1)  # (batch, filters)
        
        # Classification
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class SimpleLSTM(nn.Module):
    """
    Simple LSTM Architecture
    
    Input: (batch, time=100, channels=21, strikes=7)
    - Flattens spatial dimension, uses LSTM for temporal patterns
    """
    
    def __init__(
        self,
        input_channels: int = 21,
        num_strikes: int = 7,
        time_steps: int = 100,
        lstm_hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Flatten spatial dimensions
        input_size = input_channels * num_strikes
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size,
            lstm_hidden,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x: (batch, time, channels, strikes)
        batch_size = x.size(0)
        time_steps = x.size(1)
        
        # Flatten spatial dimensions
        x = x.view(batch_size, time_steps, -1)  # (batch, time, channels * strikes)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_final = lstm_out[:, -1, :]  # (batch, hidden * 2)
        
        # Classification
        out = F.relu(self.fc1(lstm_final))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class TransformerModel(nn.Module):
    """
    Transformer-based Architecture
    
    Input: (batch, time=100, channels=21, strikes=7)
    - Multi-head attention for strike relationships and temporal dependencies
    """
    
    def __init__(
        self,
        input_channels: int = 21,
        num_strikes: int = 7,
        time_steps: int = 100,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Flatten spatial dimensions
        input_size = input_channels * num_strikes
        
        # Embedding
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(time_steps, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x: (batch, time, channels, strikes)
        batch_size = x.size(0)
        time_steps = x.size(1)
        
        # Flatten spatial dimensions
        x = x.view(batch_size, time_steps, -1)  # (batch, time, channels * strikes)
        
        # Embedding
        x = self.embedding(x)  # (batch, time, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0)
        
        # Transformer
        transformer_out = self.transformer(x)  # (batch, time, d_model)
        
        # Use last timestep
        final = transformer_out[:, -1, :]  # (batch, d_model)
        
        # Classification
        out = F.relu(self.fc1(final))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def create_model(model_type: str = 'hybrid', **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'hybrid', '3dcnn', 'lstm', or 'transformer'
        **kwargs: Model-specific parameters
    
    Returns:
        Model instance
    """
    models = {
        'hybrid': HybridCNNRNN,
        '3dcnn': CNN3D,
        'lstm': SimpleLSTM,
        'transformer': TransformerModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](**kwargs)

