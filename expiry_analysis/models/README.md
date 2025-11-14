# Option Chain Direction Prediction Models

## Overview

This module implements deep learning models to predict underlying price direction using option chain data. The approach leverages:

1. **Spatial patterns** in option chains (skew, smile across strikes)
2. **Temporal evolution** (how patterns change over 300 timesteps)
3. **Greeks** (IV, Delta, Gamma, Theta, Vega) as market sentiment indicators
4. **Derived features** (PCR, skew, max pain, momentum)

## Your Approach - Assessment

✅ **Excellent foundation:**
- CNN for spatial patterns (strike relationships) - **Correct**
- Time series component (300 rolling matrices) - **Essential**
- Greeks as features - **Industry standard**
- Manipulation detection premise - **Valid** (option flow often precedes moves)

✅ **Architecture choice validated:**
- Your data structure `(300, 23, 7)` is perfect for CNN-RNN hybrid
- Spatial dimension (strikes) → CNN
- Temporal dimension (time) → RNN
- This is the right approach for this problem

## Model Architecture

### CNN-RNN Hybrid (Recommended)
```
Input: (batch, 300, 23, 7)
  ↓
3D CNN → Extract spatial patterns per timestep
  ↓
Bidirectional LSTM → Capture temporal evolution
  ↓
Dense layers → Probability output
```

**Why this works:**
- Captures **option chain patterns** (skew, smile) at each moment
- Learns **how patterns evolve** over time
- Handles both short-term (recent activity) and long-term (trends)

### Alternative: Pure 3D CNN
- Simpler, faster
- May miss long-term dependencies
- Good starting point for experimentation

## Files Structure

```
models/
├── __init__.py
├── cnn_rnn_model.py      # Model architectures (CNN-RNN, 3D CNN)
├── feature_engineering.py # Derived features (PCR, skew, momentum)
├── data_loader.py         # PyTorch Dataset/DataLoader
├── train.py              # Training script
└── README.md             # This file
```

## Usage

### 1. Prepare Data
```python
from expiry_analysis.chain_processor import OptionDataProcessor
from vol.matrix_greeks import MatrixGreeksCalculator

processor = OptionDataProcessor(window_size=300, strike_count=3)
greeks_calc = MatrixGreeksCalculator(risk_free_rate=0.065, days_to_expiry=7)

# Populate processor with historical data
# (use your data fetching pipeline)
```

### 2. Create DataLoader
```python
from expiry_analysis.models.data_loader import create_dataloader

dataloader = create_dataloader(
    processor=processor,
    greeks_calc=greeks_calc,
    underlying="RELIANCE",
    batch_size=32,
    window_size=300,
    prediction_horizon=10,  # Predict 10 timesteps ahead
    move_threshold=0.002     # 0.2% move threshold
)
```

### 3. Train Model
```python
from expiry_analysis.models.train import train_model

train_model(
    underlying="RELIANCE",
    num_epochs=20,
    batch_size=32,
    learning_rate=0.001,
    model_type="cnn_rnn"
)
```

### 4. Inference
```python
from expiry_analysis.models.cnn_rnn_model import create_model
import torch

model = create_model(model_type="cnn_rnn", input_channels=23, num_strikes=7, time_steps=300)
model.load_state_dict(torch.load("models/best_RELIANCE_cnn_rnn.pth"))
model.eval()

# Get latest 300 timesteps
timestamps, matrices = processor.get_matrix("RELIANCE", window=300)
X = torch.FloatTensor(matrices).unsqueeze(0)  # (1, 300, 23, 7)

with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=1)
    print(f"P(Down): {probs[0, 0]:.3f}, P(Up): {probs[0, 1]:.3f}")
```

## Feature Engineering

### Derived Features (Auto-added)
- **PCR** (Put-Call Ratio): `PE_OI / CE_OI`
- **IV Skew**: `PE_IV - CE_IV` (negative = bearish)
- **OI Skew**: `PE_OI - CE_OI`
- **Max Pain Gap**: Distance from spot to max pain strike
- **OI Momentum**: Rate of change in OI over time

### External Features (Future)
- Interest rates
- Commodity prices
- Market indices
- VIX

## Training Strategy

### Label Generation
- **Forward-looking**: Compare price at `t+10` timesteps vs `t`
- **Threshold**: 0.2% move = signal (configurable)
- **Binary**: Up (1) vs Down (0)

### Data Split
- **Time-based**: Train on past, validate on recent
- **Walk-forward**: For backtesting

### Loss Function
- **Cross-entropy** for classification
- **Focal loss** if class imbalance (optional)

## Next Steps

1. **Collect Data**: Populate processor with historical option chain data
2. **Baseline Model**: Train CNN-RNN with Greeks only
3. **Add Features**: Incorporate derived features (PCR, skew)
4. **External Data**: Add interest rates, commodities
5. **Ensemble**: Combine multiple models
6. **Backtesting**: Validate on historical data
7. **Live Trading**: Deploy with risk management

## Performance Expectations

- **Baseline**: 50-55% accuracy (random)
- **Good Model**: 58-65% accuracy
- **Excellent**: 65%+ accuracy (very difficult in markets)

**Note**: Even 55-60% accuracy can be profitable with proper risk management and position sizing.

## Considerations

1. **Market Regimes**: Model may perform differently in bull/bear/volatile markets
2. **Overfitting**: Use regularization (dropout, L2)
3. **Data Quality**: Ensure clean, consistent data
4. **Latency**: Model inference should be < 10ms for live trading
5. **Risk Management**: Never rely solely on model - use stops, position sizing

## References

- Option chain patterns (skew, smile)
- Greeks interpretation
- CNN for spatial data
- RNN/LSTM for time series
- Market microstructure

