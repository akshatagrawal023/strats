# Model Architecture for Option Chain-Based Direction Prediction

## Your Approach - Analysis

**Strengths:**
1. ✅ **CNN for spatial patterns**: Option chains have spatial structure (skew, smile patterns across strikes)
2. ✅ **Time series component**: 300 rolling matrices capture temporal dynamics
3. ✅ **Greeks as features**: IV, Delta, Gamma capture market sentiment
4. ✅ **Manipulation detection**: Option flow often precedes price moves

**Considerations:**
1. **3D Tensor Structure**: Your data is `(time=300, channels=23, strikes=7)` - perfect for 3D CNN or CNN+RNN
2. **Spatial + Temporal**: Need to capture both strike patterns (spatial) and time evolution (temporal)
3. **Feature Engineering**: Beyond Greeks, consider:
   - **PCR (Put-Call Ratio)**: OI-based sentiment
   - **Skew**: IV difference between OTM puts and calls
   - **Max Pain**: Strike with max OI
   - **OI Momentum**: Rate of change in OI
   - **Volume-Price Divergence**: Unusual activity
4. **External Features**: Interest rates, commodities, indices - add as separate channels or concatenate

## Recommended Architecture

### Option 1: Hybrid CNN-RNN (Recommended)
```
Input: (300, 23, 7) tensor
  ↓
3D CNN Layers (extract spatial patterns per timestep)
  → Output: (300, 64, 7) feature maps
  ↓
Time-distributed pooling → (300, 64)
  ↓
Bidirectional LSTM/GRU (capture temporal patterns)
  → Output: (128,) hidden state
  ↓
External features concatenation (rates, commodities)
  → (128 + N_external)
  ↓
Dense layers → Probability output
```

**Why this works:**
- CNN captures **spatial patterns** (skew, smile) at each timestep
- RNN captures **temporal evolution** (how patterns change over time)
- Handles both short-term (recent patterns) and long-term (trends)

### Option 2: 3D CNN
```
Input: (300, 23, 7) tensor
  ↓
3D Convolutional layers (spatial + temporal)
  → Extract spatio-temporal features
  ↓
Global pooling
  ↓
Dense layers → Probability
```

**Trade-off:** Simpler but may miss long-term dependencies

### Option 3: Transformer-based
```
Input: (300, 23×7) flattened per timestep
  ↓
Multi-head self-attention (learns strike relationships)
  ↓
Temporal attention (learns time dependencies)
  ↓
Output head → Probability
```

**Trade-off:** More complex, needs more data, but very powerful

## Output Design

**Binary Classification:**
- `P(up)` = probability of upward movement
- `P(down)` = probability of downward movement
- Threshold: 0.6+ for signal

**Multi-class:**
- `P(strong_up)`, `P(weak_up)`, `P(neutral)`, `P(weak_down)`, `P(strong_down)`

**Regression:**
- Predict expected move magnitude (in %)
- Then classify direction separately

## Feature Engineering Beyond Greeks

### Derived Features (add as channels):
1. **PCR (Put-Call Ratio)**: `PE_OI / CE_OI` per strike
2. **IV Skew**: `PE_IV - CE_IV` (negative = bearish)
3. **OI Skew**: `PE_OI - CE_OI` (positive = bearish)
4. **Max Pain**: Strike with max `(CE_OI + PE_OI)`
5. **OI Momentum**: `d(OI)/dt` (rate of change)
6. **Volume-Price Divergence**: High volume, low price change
7. **Gap to Max Pain**: `|Spot - MaxPain| / Spot`

### External Features (concatenate after CNN-RNN):
- Interest rates (10Y, 3M)
- Commodity prices (Gold, Crude)
- Market indices (Nifty, VIX)
- Sector-specific indicators

## Training Strategy

1. **Label Generation:**
   - Forward-looking: `label = 1 if price(t+5min) > price(t) * 1.002` (0.2% move)
   - Or: `label = sign(price(t+10min) - price(t))`

2. **Data Augmentation:**
   - Time shifts (market open vs close patterns)
   - Strike window shifts (if using moneyness)
   - Noise injection (small random perturbations)

3. **Validation:**
   - Time-based split (train on past, validate on recent)
   - Walk-forward validation for backtesting

4. **Loss Function:**
   - Binary cross-entropy for classification
   - Focal loss if class imbalance
   - Add regularization (L2, dropout)

## Implementation Priority

1. **Phase 1**: Basic CNN-RNN hybrid with Greeks only
2. **Phase 2**: Add derived features (PCR, Skew, etc.)
3. **Phase 3**: Add external features
4. **Phase 4**: Ensemble models, attention mechanisms

