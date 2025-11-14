"""
CNN-RNN Hybrid Model for Option Chain Direction Prediction

Architecture:
- 3D CNN layers extract spatial patterns (strike relationships) at each timestep
- RNN layers capture temporal evolution
- Dense layers for final prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class OptionChainCNNRNN(nn.Module):
    """
    Hybrid CNN-RNN model for predicting underlying direction from option chain data.
    
    Input: (batch, time_steps=300, channels=23, strikes=7)
    Output: (batch, 2) - [P(down), P(up)] probabilities
    """
    
    def __init__(
        self,
        input_channels: int = 23,
        num_strikes: int = 7,
        time_steps: int = 300,
        cnn_filters: int = 64,
        lstm_hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        external_features: int = 0,
        num_classes: int = 2
    ):
        """
        Args:
            input_channels: Number of channels (Greeks + raw data)
            num_strikes: Number of strikes in option chain
            time_steps: Number of historical timesteps
            cnn_filters: Number of CNN feature maps
            lstm_hidden: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            external_features: Number of external features (rates, commodities)
            num_classes: Output classes (2 for binary, 5 for multi-class)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_strikes = num_strikes
        self.time_steps = time_steps
        self.external_features = external_features
        
        # 3D CNN for spatial pattern extraction (strike relationships)
        # Input: (batch, channels, time, strikes)
        self.conv1 = nn.Conv3d(
            in_channels=1,  # Treat channels as depth
            out_channels=32,
            kernel_size=(3, 3, 3),  # (time, channel, strike)
            padding=(1, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=cnn_filters,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(cnn_filters)
        
        # Adaptive pooling to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool3d((time_steps, 1, 1))
        
        # Reshape for RNN: (batch, time_steps, cnn_filters)
        # Bidirectional LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Final layers
        lstm_output_size = lstm_hidden * 2  # Bidirectional
        combined_size = lstm_output_size + external_features
        
        self.fc1 = nn.Linear(combined_size, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, num_classes)
        
    def forward(
        self, 
        x: torch.Tensor, 
        external: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, time_steps, channels, strikes)
            external: External features (batch, external_features)
        
        Returns:
            Logits (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape for 3D CNN: (batch, 1, time_steps, channels, strikes)
        # We'll process channels as depth dimension
        x = x.permute(0, 2, 1, 3)  # (batch, channels, time_steps, strikes)
        x = x.unsqueeze(1)  # (batch, 1, channels, time_steps, strikes)
        
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Pool to reduce dimensions
        x = self.pool(x)  # (batch, cnn_filters, time_steps, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch, cnn_filters, time_steps)
        x = x.permute(0, 2, 1)  # (batch, time_steps, cnn_filters)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last timestep output
        lstm_features = lstm_out[:, -1, :]  # (batch, lstm_hidden * 2)
        
        # Concatenate external features if provided
        if external is not None:
            combined = torch.cat([lstm_features, external], dim=1)
        else:
            combined = lstm_features
        
        # Dense layers
        x = F.relu(self.fc1(combined))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc_out(x)
        
        return logits


class OptionChain3DCNN(nn.Module):
    """
    Pure 3D CNN alternative (simpler, faster, but may miss long-term dependencies).
    
    Input: (batch, time_steps=300, channels=23, strikes=7)
    Output: (batch, 2) - [P(down), P(up)]
    """
    
    def __init__(
        self,
        input_channels: int = 23,
        num_strikes: int = 7,
        time_steps: int = 300,
        num_filters: int = 64,
        dropout: float = 0.3,
        external_features: int = 0,
        num_classes: int = 2
    ):
        super().__init__()
        
        # 3D CNN layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, num_filters, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(num_filters)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        # Dense layers
        combined_size = num_filters + external_features
        self.fc1 = nn.Linear(combined_size, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, num_classes)
        
    def forward(
        self, 
        x: torch.Tensor, 
        external: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Reshape: (batch, 1, time_steps, channels, strikes)
        x = x.permute(0, 2, 1, 3).unsqueeze(1)
        
        # CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Pool
        x = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        
        # External features
        if external is not None:
            x = torch.cat([x, external], dim=1)
        
        # Dense
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc_out(x)
        
        return logits


def create_model(
    model_type: str = "cnn_rnn",
    input_channels: int = 23,
    num_strikes: int = 7,
    time_steps: int = 300,
    external_features: int = 0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create model.
    
    Args:
        model_type: "cnn_rnn" or "3d_cnn"
        input_channels: Number of input channels
        num_strikes: Number of strikes
        time_steps: Historical timesteps
        external_features: External feature count
        **kwargs: Additional model parameters
    
    Returns:
        Initialized model
    """
    if model_type == "cnn_rnn":
        return OptionChainCNNRNN(
            input_channels=input_channels,
            num_strikes=num_strikes,
            time_steps=time_steps,
            external_features=external_features,
            **kwargs
        )
    elif model_type == "3d_cnn":
        return OptionChain3DCNN(
            input_channels=input_channels,
            num_strikes=num_strikes,
            time_steps=time_steps,
            external_features=external_features,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_model(
        model_type="cnn_rnn",
        input_channels=23,
        num_strikes=7,
        time_steps=300,
        external_features=5  # e.g., interest rates, commodities
    )
    
    # Dummy input
    batch_size = 4
    x = torch.randn(batch_size, 300, 23, 7)  # (batch, time, channels, strikes)
    external = torch.randn(batch_size, 5)  # External features
    
    # Forward pass
    logits = model(x, external)
    probs = F.softmax(logits, dim=1)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output probabilities: {probs}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

