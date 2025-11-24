# Training Strategy for 3-Second Interval Option Chain Data

## Data Characteristics
- **Sampling Rate:** 3 seconds per snapshot
- **Available History:** 300 snapshots = 15 minutes (current window)
- **Data Collection:** Continuous during market hours
- **Stocks:** 10 underlyings simultaneously

---

## Part 1: Rolling Feature Window Sizes

### Time Conversions (3-second intervals)
```
10 steps  = 30 seconds  = 0.5 minutes
20 steps  = 60 seconds  = 1 minute
40 steps  = 120 seconds = 2 minutes
60 steps  = 180 seconds = 3 minutes
100 steps = 300 seconds = 5 minutes
200 steps = 600 seconds = 10 minutes
300 steps = 900 seconds = 15 minutes
```

### Recommended Window Sizes by Feature Type

#### **Short-Term Features (10-40 steps = 30s - 2 min)**
**Use for:** Real-time momentum, immediate signals
- **OI Change Momentum:** 10-20 steps (30s - 1 min)
  - Captures recent accumulation patterns
  - Fast enough to catch manipulation signals
  
- **Volume Momentum:** 10-20 steps (30s - 1 min)
  - Detects sudden volume bursts
  - Event-driven activity

- **Price Momentum (MA Cross):** 20-40 steps (1-2 min)
  - Short-term trend direction
  - Avoids noise from single snapshots

- **IV Changes:** 10-20 steps (30s - 1 min)
  - Rapid IV expansion/contraction
  - Fear/greed indicators

**Rationale:** Option manipulation signals are often short-lived. Institutional accumulation happens over minutes, not hours.

#### **Medium-Term Features (40-100 steps = 2-5 min)**
**Use for:** Trend confirmation, sustained patterns
- **OI Momentum (Sustained):** 40-60 steps (2-3 min)
  - Confirms accumulation is not just noise
  - Stronger signal than short-term

- **OI Persistence:** 40-60 steps (2-3 min)
  - Count of positive OI changes
  - Consistent buildup = institutional activity

- **Volatility Regime:** 60-100 steps (3-5 min)
  - Market state detection
  - High volatility = uncertain market

- **Moving Averages:** 40-100 steps (2-5 min)
  - Trend direction
  - Smooth out noise

**Rationale:** Medium-term windows balance signal strength with responsiveness. Good for confirming short-term signals.

#### **Long-Term Features (100-300 steps = 5-15 min)**
**Use for:** Context, market structure, baseline
- **OI Accumulation Rate:** 100-200 steps (5-10 min)
  - Long-term institutional buildup
  - Strong manipulation signal

- **Price Volatility:** 100-200 steps (5-10 min)
  - Historical volatility context
  - Normalize features by volatility

- **Max Pain Stability:** 200-300 steps (10-15 min)
  - How stable is max pain strike?
  - Stable max pain = stronger pull

**Rationale:** Long-term features provide context but may be too slow for trading signals. Use for filtering/validation.

### **Multi-Scale Approach (Recommended)**
Use **multiple window sizes** for the same feature type:
- **Short (20 steps):** Fast signals
- **Medium (60 steps):** Confirmation
- **Long (100 steps):** Context

Example: `OI_MOMENTUM_20`, `OI_MOMENTUM_60`, `OI_MOMENTUM_100`

---

## Part 2: Target Prediction Horizon

### Key Considerations

1. **Signal Decay:** Option signals decay over time
   - Strong signals: Valid for 1-5 minutes
   - Weak signals: Decay in 30 seconds

2. **Trading Strategy:**
   - **Scalping:** 30s - 2 min ahead
   - **Intraday:** 2-10 min ahead
   - **Swing:** 10-30 min ahead

3. **Market Microstructure:**
   - Price moves in 3-second intervals
   - Need enough time for signal to manifest
   - Too short = noise, too long = signal decay

4. **Risk Management:**
   - Holding period should match prediction horizon
   - Exit strategy must be defined

### Recommended Prediction Horizons

#### **Option 1: Short-Term (30-60 seconds ahead)**
**Horizon:** 10-20 steps (30s - 1 min)

**Pros:**
- Captures immediate manipulation signals
- Fast execution, quick exits
- High signal-to-noise ratio (if signal is strong)

**Cons:**
- More noise from market microstructure
- Requires fast execution
- May miss slower accumulation patterns

**Use Case:** High-frequency scalping, catching immediate OI buildup

**Label Generation:**
```python
# 30 seconds ahead (10 steps)
future_price = spot_prices[t + 10]
current_price = spot_prices[t]
price_change = (future_price - current_price) / current_price

# Binary classification
if price_change > 0.001:  # 0.1% move
    label = 1  # Up
elif price_change < -0.001:
    label = 0  # Down
else:
    label = -1  # Neutral (skip)
```

#### **Option 2: Medium-Term (2-5 minutes ahead) - **RECOMMENDED**
**Horizon:** 40-100 steps (2-5 min)

**Pros:**
- Balances signal strength with execution time
- Captures sustained accumulation patterns
- Less noise than short-term
- Signal still fresh (not decayed)

**Cons:**
- Requires holding position for 2-5 minutes
- May miss very fast moves

**Use Case:** Intraday trading, catching institutional accumulation

**Label Generation:**
```python
# 3 minutes ahead (60 steps)
future_price = spot_prices[t + 60]
current_price = spot_prices[t]
price_change = (future_price - current_price) / current_price

# Binary classification with threshold
if price_change > 0.002:  # 0.2% move
    label = 1  # Up
elif price_change < -0.002:
    label = 0  # Down
else:
    label = -1  # Neutral (skip)

# Multi-class (optional)
if price_change > 0.005:  # 0.5%
    label = 2  # Strong Up
elif price_change > 0.002:
    label = 1  # Weak Up
elif price_change < -0.005:
    label = -2  # Strong Down
elif price_change < -0.002:
    label = -1  # Weak Down
else:
    label = 0  # Neutral
```

#### **Option 3: Long-Term (5-15 minutes ahead)**
**Horizon:** 100-300 steps (5-15 min)

**Pros:**
- Captures long-term trends
- Less noise
- Better for swing trading

**Cons:**
- Signal may decay
- Requires longer holding period
- May miss short-term opportunities

**Use Case:** Swing trading, trend following

### **Multi-Horizon Approach (Advanced)**
Train **multiple models** for different horizons:
- **Model 1:** 30s ahead (scalping)
- **Model 2:** 3 min ahead (intraday) - **Primary**
- **Model 3:** 10 min ahead (swing)

**Ensemble:** Combine predictions for stronger signal

---

## Part 3: Label Generation Strategies

### Strategy 1: Binary Classification (Direction)
```python
def generate_binary_label(current_price, future_price, threshold=0.002):
    """
    threshold: Minimum move % to generate signal (default 0.2%)
    """
    price_change = (future_price - current_price) / current_price
    
    if price_change > threshold:
        return 1  # Up
    elif price_change < -threshold:
        return 0  # Down
    else:
        return -1  # Neutral (skip in training)
```

**Threshold Selection:**
- **0.1% (0.001):** Very sensitive, many signals, more noise
- **0.2% (0.002):** Balanced (recommended for 3-min horizon)
- **0.5% (0.005):** Conservative, fewer but stronger signals

### Strategy 2: Multi-Class Classification (Direction + Strength)
```python
def generate_multiclass_label(current_price, future_price):
    price_change = (future_price - current_price) / current_price
    
    if price_change > 0.005:
        return 2  # Strong Up
    elif price_change > 0.002:
        return 1  # Weak Up
    elif price_change < -0.005:
        return -2  # Strong Down
    elif price_change < -0.002:
        return -1  # Weak Down
    else:
        return 0  # Neutral
```

### Strategy 3: Regression (Magnitude)
```python
def generate_regression_label(current_price, future_price):
    """
    Predict actual price change percentage
    """
    price_change = (future_price - current_price) / current_price
    return price_change  # Continuous value
```

**Then classify:**
- `price_change > 0.002` → Up
- `price_change < -0.002` → Down
- Else → Neutral

### Strategy 4: Time-Weighted Labels
```python
def generate_time_weighted_label(prices, t, horizon, weights=None):
    """
    Use multiple future prices with weights
    weights: [0.5, 0.3, 0.2] for [t+1, t+2, t+3]
    """
    if weights is None:
        weights = np.ones(horizon) / horizon
    
    future_prices = prices[t+1:t+1+horizon]
    weighted_price = np.average(future_prices, weights=weights)
    
    return (weighted_price - prices[t]) / prices[t]
```

**Rationale:** Smooths out noise by using multiple future points

---

## Part 4: Data Requirements for Training

### Minimum Data Requirements

#### **For Basic Model:**
- **Training samples:** 10,000+ (per underlying)
- **Time period:** ~8-10 hours of market data
  - 10,000 samples × 3 seconds = 30,000 seconds = 8.3 hours
- **Window size:** 300 steps (15 min history)
- **Prediction horizon:** 60 steps (3 min ahead)

#### **For Robust Model:**
- **Training samples:** 50,000+ (per underlying)
- **Time period:** 2-3 weeks of market data
  - Captures different market regimes
  - Includes volatile and calm periods
- **Multiple underlyings:** 10 stocks = 500,000+ total samples

### Data Quality Checks

1. **Temporal Continuity:**
   - Check for gaps > 10 seconds (missing data)
   - Flag periods with low liquidity (wide spreads)

2. **Label Distribution:**
   - Should be roughly balanced (40-60% up/down)
   - If imbalanced, use class weights or focal loss

3. **Feature Validity:**
   - Remove samples with NaN in critical features
   - Flag samples with extreme values (outliers)

---

## Part 5: Training Methodology

### Step 1: Data Preparation

```python
# Load data from HDF5
data = load_training_data("HDFCBANK")

# Extract matrices and timestamps
features = data['features']  # (n_snapshots, 11, strikes)
greeks = data['greeks']      # (n_snapshots, 10, strikes)
timestamps = data['timestamps']  # (n_snapshots,)

# Combine features + greeks
combined = np.concatenate([features, greeks], axis=1)  # (n, 21, strikes)

# Extract spot prices for labeling
spot_prices = features[:, 10, 0]  # UNDERLYING_LTP channel, first strike
```

### Step 2: Create Training Windows

```python
def create_training_samples(
    matrices,           # (n_snapshots, channels, strikes)
    spot_prices,        # (n_snapshots,)
    window_size=300,    # Input history (15 min)
    horizon=60,         # Prediction horizon (3 min)
    step_size=10,       # Step between samples (30s)
    threshold=0.002     # Minimum move for label
):
    """
    Create rolling windows with labels
    """
    samples = []
    
    for i in range(0, len(matrices) - window_size - horizon, step_size):
        # Input: last window_size snapshots
        X = matrices[i:i+window_size]  # (300, 21, strikes)
        
        # Label: price change horizon steps ahead
        current_price = spot_prices[i + window_size - 1]
        future_price = spot_prices[i + window_size + horizon - 1]
        
        if np.isnan(current_price) or np.isnan(future_price):
            continue
        
        price_change = (future_price - current_price) / current_price
        
        # Generate label
        if price_change > threshold:
            y = 1  # Up
        elif price_change < -threshold:
            y = 0  # Down
        else:
            continue  # Skip neutral
        
        samples.append({
            'X': X,
            'y': y,
            'price_change': price_change,
            'timestamp': timestamps[i + window_size - 1]
        })
    
    return samples
```

### Step 3: Train/Validation Split

```python
# Time-based split (CRITICAL - don't shuffle by time)
split_idx = int(0.8 * len(samples))

train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

# Or use walk-forward validation
# Train on first 80% of time, validate on last 20%
```

### Step 4: Feature Engineering

```python
def add_rolling_features(matrix_window):
    """
    matrix_window: (300, 21, strikes)
    Add rolling features as additional channels
    """
    # Short-term (20 steps = 1 min)
    oi_momentum_20 = compute_oi_momentum(matrix_window, window=20)
    
    # Medium-term (60 steps = 3 min)
    oi_momentum_60 = compute_oi_momentum(matrix_window, window=60)
    
    # Long-term (100 steps = 5 min)
    oi_momentum_100 = compute_oi_momentum(matrix_window, window=100)
    
    # Add as new channels
    extended = np.concatenate([
        matrix_window,
        oi_momentum_20,
        oi_momentum_60,
        oi_momentum_100
    ], axis=1)
    
    return extended
```

---

## Part 6: Recommended Configuration

### **Primary Configuration (Recommended)**

```python
# Input window
WINDOW_SIZE = 300        # 15 minutes of history

# Rolling feature windows
SHORT_WINDOW = 20        # 1 minute (momentum)
MEDIUM_WINDOW = 60       # 3 minutes (trend)
LONG_WINDOW = 100        # 5 minutes (context)

# Prediction horizon
PREDICTION_HORIZON = 60  # 3 minutes ahead

# Label threshold
MOVE_THRESHOLD = 0.002   # 0.2% minimum move

# Step size for sample generation
STEP_SIZE = 10           # 30 seconds between samples
```

### **Alternative: Scalping Configuration**

```python
WINDOW_SIZE = 100        # 5 minutes history
SHORT_WINDOW = 10        # 30 seconds
MEDIUM_WINDOW = 20       # 1 minute
PREDICTION_HORIZON = 10  # 30 seconds ahead
MOVE_THRESHOLD = 0.001   # 0.1% (more sensitive)
STEP_SIZE = 5            # 15 seconds
```

### **Alternative: Swing Trading Configuration**

```python
WINDOW_SIZE = 300        # 15 minutes history
SHORT_WINDOW = 60        # 3 minutes
MEDIUM_WINDOW = 100      # 5 minutes
LONG_WINDOW = 200        # 10 minutes
PREDICTION_HORIZON = 100 # 5 minutes ahead
MOVE_THRESHOLD = 0.005   # 0.5% (conservative)
STEP_SIZE = 20           # 1 minute
```

---

## Part 7: Implementation Checklist

### Data Collection
- [ ] Collect 2-3 weeks of continuous data (10 stocks)
- [ ] Ensure 3-second intervals (no gaps > 10 seconds)
- [ ] Validate data quality (no NaN, reasonable ranges)

### Feature Engineering
- [ ] Implement latest value features (Category 1)
- [ ] Implement change features (Category 2)
- [ ] Implement rolling features with multiple window sizes (Category 3)
- [ ] Normalize features appropriately

### Label Generation
- [ ] Choose prediction horizon (recommend 60 steps = 3 min)
- [ ] Choose move threshold (recommend 0.002 = 0.2%)
- [ ] Generate labels for all samples
- [ ] Check label distribution (should be balanced)

### Training Setup
- [ ] Create time-based train/validation split
- [ ] Implement PyTorch Dataset/DataLoader
- [ ] Set up model architecture (CNN-RNN hybrid)
- [ ] Configure loss function (binary cross-entropy + class weights if imbalanced)

### Validation
- [ ] Walk-forward validation on out-of-sample data
- [ ] Monitor precision/recall for each class
- [ ] Track prediction accuracy by time of day
- [ ] Analyze false positives/negatives

---

## Part 8: Expected Results

### Realistic Expectations

**With 3-minute prediction horizon:**
- **Accuracy:** 55-65% (better than 50% random)
- **Precision (Up):** 60-70% (if signal is strong)
- **Recall (Up):** 50-60% (may miss some moves)

**Signal Quality:**
- Strong signals (PCR > 1.5, high OI momentum): 65-75% accuracy
- Weak signals (PCR 0.9-1.1, low OI momentum): 50-55% accuracy

**Trading Performance:**
- Win rate: 55-60%
- Average win: 0.3-0.5%
- Average loss: 0.2-0.3%
- Risk-reward: ~1.5:1 (if managed well)

---

## Summary: Quick Reference

| Feature Type | Window Size | Time Period | Use Case |
|-------------|-------------|-------------|----------|
| Short-term momentum | 10-20 steps | 30s - 1 min | Fast signals |
| Medium-term trend | 40-60 steps | 2-3 min | Confirmation |
| Long-term context | 100-200 steps | 5-10 min | Baseline |

| Prediction Horizon | Steps | Time | Strategy |
|-------------------|-------|------|----------|
| Scalping | 10-20 | 30s - 1 min | High-frequency |
| **Intraday (Recommended)** | **40-100** | **2-5 min** | **Primary** |
| Swing | 100-300 | 5-15 min | Trend following |

**Recommended Starting Point:**
- Input window: 300 steps (15 min)
- Rolling features: 20, 60, 100 steps (multi-scale)
- Prediction horizon: 60 steps (3 min)
- Move threshold: 0.002 (0.2%)

