# Training Guide for Option Chain Models

## Quick Start

### Basic Usage

Train all models with default settings:

```bash
python "ML learning/train.py" --symbol HDFCBANK
```

### Advanced Usage

```bash
python "ML learning/train.py" \
    --symbol HDFCBANK \
    --epochs 100 \
    --lr 0.0001 \
    --batch_size 64 \
    --save_dir models \
    --window_size 100 \
    --horizon 10 \
    --threshold 0.001
```

## Parameters

- `--symbol`: Stock symbol (default: HDFCBANK)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 32)
- `--save_dir`: Directory to save models (default: models)
- `--data_dir`: Data directory (default: script directory)
- `--window_size`: Input window size in steps (default: 100 = 300 seconds)
- `--horizon`: Prediction horizon in steps (default: 10 = 30 seconds)
- `--threshold`: Minimum price move for label generation (default: 0.001 = 0.1%)

## What Gets Trained

The script trains **4 model architectures**:

1. **Hybrid CNN-RNN** (Recommended)
   - 2D CNN for spatial patterns + LSTM for temporal patterns
   - Best for capturing both strike relationships and time evolution

2. **3D CNN**
   - 3D convolutions for spatio-temporal patterns
   - Simpler but may miss long-term dependencies

3. **Simple LSTM**
   - Flattened spatial + LSTM
   - Fastest to train, good baseline

4. **Transformer**
   - Multi-head attention
   - Most complex, needs more data

## Output Files

After training, you'll get:

```
models/
├── HDFCBANK_hybrid_best.pth      # Best hybrid model
├── HDFCBANK_3dcnn_best.pth       # Best 3D CNN model
├── HDFCBANK_lstm_best.pth        # Best LSTM model
├── HDFCBANK_transformer_best.pth # Best transformer model
└── HDFCBANK_training_summary.json # Training summary
```

## Training Summary

The `training_summary.json` contains:

```json
{
  "symbol": "HDFCBANK",
  "window_size": 100,
  "prediction_horizon": 10,
  "move_threshold": 0.001,
  "train_samples": 2000,
  "val_samples": 500,
  "histories": {
    "hybrid": {
      "best_val_acc": 65.2,
      "best_epoch": 42,
      "final_train_acc": 68.5,
      "final_val_acc": 64.8
    },
    ...
  }
}
```

## Using Trained Models

After training, use the models in backtest:

```python
from backtest import BacktestEngine

# Use trained model
engine = BacktestEngine(
    symbol="HDFCBANK",
    model_path="models/HDFCBANK_hybrid_best.pth",
    model_type='hybrid'
)

results = engine.run_backtest()
```

## Training Features

- **Automatic normalization**: Data is normalized by channel
- **Class balancing**: Handles imbalanced datasets with weighted loss
- **Early stopping**: Stops if validation accuracy doesn't improve for 10 epochs
- **Learning rate scheduling**: Reduces LR on plateau
- **Time-based split**: Train on past, validate on recent data
- **Progress bars**: Shows training progress with tqdm

## Tips

1. **Start with fewer epochs**: Use `--epochs 20` for quick testing
2. **Adjust batch size**: Larger batch size (64-128) for more stable training
3. **Monitor overfitting**: Check if train_acc >> val_acc
4. **Try different thresholds**: Lower threshold (0.0005) for more samples
5. **Use GPU**: Training is much faster on GPU (automatically detected)

## Expected Training Time

- **CPU**: ~5-10 minutes per model per epoch (50 epochs = 4-8 hours)
- **GPU**: ~30-60 seconds per model per epoch (50 epochs = 25-50 minutes)

## Troubleshooting

**Out of memory error:**
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--window_size` (try 50)

**Poor accuracy:**
- Check data quality (use `data_validation.py`)
- Increase `--epochs`
- Try different `--threshold` values
- Ensure enough training samples (>1000)

**Model not improving:**
- Check learning rate (try `--lr 0.0001`)
- Verify labels are balanced
- Check for data leakage

