# Option Chain Features for Direction Prediction

## Why CE_OI and PE_OI (Fixed)
- **OI (Open Interest)**: Total outstanding contracts
- **OICH (OI Change)**: Intraday change in OI

## Matrix Structure (13 Channels × 2n+1 Strikes)
```
0:  CE_BID    - Call option bid price
1:  CE_ASK    - Call option ask price
2:  PE_BID    - Put option bid price
3:  PE_ASK    - Put option ask price
4:  CE_VOL    - Call volume
5:  PE_VOL    - Put volume
6:  CE_OI     - Call open interest
7:  PE_OI     - Put open interest
8:  CE_OICH   - Call OI change
9:  PE_OICH   - Put OI change
10: STRIKE    - Strike prices
11: UNDERLYING_LTP - Underlying last traded price
12: FUTURE_PRICE   - Near-month future price
```

## Key Features for Direction Prediction

### 1. **Put-Call Ratio (PCR)** - PRIMARY SIGNAL
- **PCR > 1.2**: Strong bearish sentiment (more puts = expecting fall)
- **PCR < 0.8**: Strong bullish sentiment (more calls = expecting rise)
- **PCR at ATM**: Most important - shows near-term direction
- **PCR OTM**: Shows extreme hedging

### 2. **OI Momentum** - TREND CONFIRMATION
- **PE OI increasing + CE OI decreasing**: BULLISH (sellers hedging downside)
- **CE OI increasing + PE OI decreasing**: BEARISH (sellers hedging upside)
- **OICH ratio > 1**: More put activity = bullish
- **OICH ratio < 1**: More call activity = bearish

### 3. **Volume Analysis** - CONVICTION INDICATOR
- **High PE volume + Low CE volume**: Bearish conviction
- **High CE volume + Low PE volume**: Bullish conviction
- **Volume momentum**: Increasing volume confirms direction

### 4. **Max Pain Theory** - PRICE MAGNET
- Strike with maximum total OI often acts as price target
- **Distance to max pain**: Price tends to move toward max pain
- **OI concentration**: Higher concentration = stronger pull

### 5. **Option Skew** - MARKET SENTIMENT
- **Positive skew_diff**: Bullish (OTM calls expensive = expecting upside)
- **Negative skew_diff**: Bearish (OTM puts expensive = expecting downside)

### 6. **Order Flow** - REAL-TIME SENTIMENT
- **Bid pressure > 0.5**: Strong buying pressure
- **Bid pressure < 0.5**: Strong selling pressure
- Compare CE vs PE bid pressure for relative strength

### 7. **Liquidity Indicators**
- **Tight spreads**: High liquidity, reliable signals
- **Wide spreads**: Low liquidity, ignore signals

### 8. **Future Premium**
- **Positive premium**: Contango (normal market)
- **Negative premium**: Backwardation (stressed market)

## Composite Signals

### Bullish Signals (Direction > 0)
1. PCR > 1.2 (high put OI)
2. PE OI momentum > 0 (increasing puts)
3. CE OI momentum < 0 (decreasing calls)
4. High PE volume (put sellers)
5. MA cross > 0 (uptrend)

### Bearish Signals (Direction < 0)
1. PCR < 0.8 (low put OI)
2. CE OI momentum > 0 (increasing calls)
3. PE OI momentum < 0 (decreasing puts)
4. High CE volume (call sellers)
5. MA cross < 0 (downtrend)

## Feature Importance for ML Models

### High Importance (Primary Features)
1. **pcr_atm**: Direct sentiment indicator
2. **oich_ratio_atm**: Real-time flow
3. **direction_signal**: Composite score
4. **pe_oi_momentum**: Trend strength
5. **max_pain_distance**: Price target

### Medium Importance
1. **vol_ratio_atm**: Volume confirmation
2. **future_premium**: Market structure
3. **skew_diff**: Volatility sentiment
4. **ma_cross**: Price momentum

### Low Importance (Context)
1. **liquidity_ratio**: Data quality
2. **vol_regime**: Volatility state
3. **ce/pe_spread**: Transaction costs

## Usage Example

```python
from chain_processor import OptionDataProcessor
from feature_engineering import OptionFeatureEngine

# Initialize
processor = OptionDataProcessor(window_size=300, strike_count=3)
feature_engine = OptionFeatureEngine(processor)

# Process data
resp = get_option_chain("NSE:RELIANCE-EQ", 3)
processor.process_option_chain("RELIANCE", resp)

# Extract features
features = feature_engine.compute_all_features("RELIANCE", window=20)
df_features = feature_engine.get_feature_dataframe("RELIANCE", window=20)

# Check direction signal
if features['direction_signal'] > 0.3:
    print("BULLISH")
elif features['direction_signal'] < -0.3:
    print("BEARISH")
else:
    print("NEUTRAL")
```

## Notes on Moneyness

Moneyness = (Strike - Underlying) / Underlying × 100

- **Moneyness = 0**: ATM (At-The-Money)
- **Moneyness > 0**: OTM calls / ITM puts
- **Moneyness < 0**: ITM calls / OTM puts

By using moneyness instead of absolute strikes:
- Features generalize across different underlyings
- Model learns relative patterns (±1%, ±2%, etc.)
- Works even if strike range shifts due to price movement

