# Comprehensive Feature Analysis for Option Chain Manipulation Detection

## Hypothesis
**Option chain data contains manipulation patterns that can predict underlying price movements.** Market makers, institutions, and large traders leave footprints in:
- Open Interest (OI) buildup patterns
- Volume-OI divergences
- Bid-Ask spread anomalies
- Greeks mispricing
- Cross-strike OI concentration
- Time-based accumulation patterns

---

## Available Data Channels (23 total)

### Base Channels (0-12)
```
0:  CE_BID         - Call bid price
1:  CE_ASK         - Call ask price
2:  PE_BID         - Put bid price
3:  PE_ASK         - Put ask price
4:  CE_VOL         - Call volume
5:  PE_VOL         - Put volume
6:  CE_OI          - Call open interest
7:  PE_OI          - Put open interest
8:  CE_OICH        - Call OI change (intraday)
9:  PE_OICH        - Put OI change (intraday)
10: STRIKE         - Strike prices
11: UNDERLYING_LTP - Spot price
12: FUTURE_PRICE   - Future price
```

### Greeks Channels (13-22)
```
13: CE_IV          - Call implied volatility
14: PE_IV          - Put implied volatility
15: CE_DELTA       - Call delta
16: PE_DELTA       - Put delta
17: CE_GAMMA        - Call gamma
18: PE_GAMMA        - Put gamma
19: CE_THETA        - Call theta
20: PE_THETA        - Put theta
21: CE_VEGA         - Call vega
22: PE_VEGA         - Put vega
```

---

## Feature Categories by Computation Type

### CATEGORY 1: Latest Value Features (No History Needed)
**Use:** Current snapshot only (t[-1])

#### A. Price-Based
1. **Mid Prices** (CE/PE)
   - `CE_MID = (CE_BID + CE_ASK) / 2`
   - `PE_MID = (PE_BID + PE_ASK) / 2`
   - **Why:** Cleaner signal than bid/ask individually

2. **Bid-Ask Spreads** (Liquidity)
   - `CE_SPREAD = CE_ASK - CE_BID`
   - `PE_SPREAD = PE_ASK - PE_BID`
   - `SPREAD_RATIO = CE_SPREAD / PE_SPREAD`
   - **Why:** Wide spreads = low liquidity = unreliable signals

3. **Moneyness** (Strike-Independent)
   - `MONEYNESS = (STRIKE - UNDERLYING_LTP) / UNDERLYING_LTP * 100`
   - **Why:** Normalizes across different price levels

4. **Future Premium**
   - `FUTURE_PREMIUM = (FUTURE_PRICE - UNDERLYING_LTP) / UNDERLYING_LTP`
   - **Why:** Contango/backwardation indicates market stress

#### B. OI-Based (Manipulation Signals)
5. **Put-Call Ratios** (PCR)
   - `PCR_ATM = PE_OI[ATM] / CE_OI[ATM]`
   - `PCR_TOTAL = sum(PE_OI) / sum(CE_OI)`
   - `PCR_OTM = sum(PE_OI[OTM]) / sum(CE_OI[OTM])`
   - **Why:** PCR > 1.2 = bearish, PCR < 0.8 = bullish (contrarian)

6. **OI Concentration**
   - `MAX_PAIN_IDX = argmax(CE_OI + PE_OI)`
   - `MAX_PAIN_DISTANCE = (STRIKE[MAX_PAIN] - UNDERLYING_LTP) / UNDERLYING_LTP`
   - `OI_CONCENTRATION = max(CE_OI + PE_OI) / sum(CE_OI + PE_OI)`
   - **Why:** Price tends to gravitate toward max pain

7. **OI Skew**
   - `CE_OI_SLOPE = mean(diff(CE_OI))` (across strikes)
   - `PE_OI_SLOPE = mean(diff(PE_OI))`
   - `OI_SKEW_DIFF = CE_OI_SLOPE - PE_OI_SLOPE`
   - **Why:** Asymmetric OI buildup indicates directional bias

#### C. Volume-Based
8. **Volume Ratios**
   - `VOL_RATIO_ATM = PE_VOL[ATM] / CE_VOL[ATM]`
   - `VOL_RATIO_TOTAL = sum(PE_VOL) / sum(CE_VOL)`
   - **Why:** High volume = conviction

9. **Volume-OI Divergence** (Manipulation Signal)
   - `CE_VOL_OI_RATIO = CE_VOL / (CE_OI + 1e-6)`
   - `PE_VOL_OI_RATIO = PE_VOL / (PE_OI + 1e-6)`
   - **Why:** High volume, low OI = day trading (weak signal)
   - **Why:** Low volume, high OI = accumulation (strong signal)

#### D. Greeks-Based
10. **IV Skew** (Volatility Sentiment)
    - `IV_SKEW = PE_IV[OTM] - CE_IV[OTM]`
    - `IV_SMILE = (CE_IV[OTM] + PE_IV[OTM]) / 2 - CE_IV[ATM]`
    - **Why:** Negative skew = bearish, positive = bullish

11. **Delta Concentration**
    - `CE_DELTA_WEIGHTED = sum(CE_OI * CE_DELTA) / sum(CE_OI)`
    - `PE_DELTA_WEIGHTED = sum(PE_OI * PE_DELTA) / sum(PE_OI)`
    - **Why:** Net delta exposure indicates directional bias

12. **Gamma Exposure**
    - `GAMMA_EXPOSURE = sum((CE_OI * CE_GAMMA) - (PE_OI * PE_GAMMA))`
    - **Why:** High gamma = price sensitivity = potential volatility

#### E. Order Flow
13. **Bid Pressure**
    - `CE_BID_PRESSURE = CE_BID / (CE_BID + CE_ASK)`
    - `PE_BID_PRESSURE = PE_BID / (PE_BID + PE_ASK)`
    - **Why:** > 0.5 = buying pressure, < 0.5 = selling pressure

14. **Ask Pressure**
    - `CE_ASK_PRESSURE = CE_ASK / (CE_BID + CE_ASK)`
    - `PE_ASK_PRESSURE = PE_ASK / (PE_BID + PE_ASK)`
    - **Why:** Inverse of bid pressure

---

### CATEGORY 2: Change Features (t[-1] vs t[-2] or t[-k])
**Use:** Compare current vs previous snapshot(s)

#### A. Price Changes
15. **Spot Returns**
    - `SPOT_RETURN_1 = (UNDERLYING_LTP[-1] - UNDERLYING_LTP[-2]) / UNDERLYING_LTP[-2]`
    - `SPOT_RETURN_5 = (UNDERLYING_LTP[-1] - UNDERLYING_LTP[-6]) / UNDERLYING_LTP[-6]`
    - **Why:** Momentum indicator

16. **Option Price Changes**
    - `CE_MID_CHANGE = CE_MID[-1] - CE_MID[-2]`
    - `PE_MID_CHANGE = PE_MID[-1] - PE_MID[-2]`
    - **Why:** Option price moves ahead of spot (leading indicator)

#### B. OI Changes (Already Available as OICH)
17. **OI Change Momentum**
    - `CE_OICH_ACCEL = CE_OICH[-1] - CE_OICH[-2]` (if available)
    - `PE_OICH_ACCEL = PE_OICH[-1] - PE_OICH[-2]`
    - **Why:** Acceleration in OI buildup = stronger signal

18. **OICH Ratios**
    - `OICH_RATIO_ATM = PE_OICH[ATM] / (CE_OICH[ATM] + 1e-6)`
    - `OICH_RATIO_TOTAL = sum(PE_OICH) / (sum(CE_OICH) + 1e-6)`
    - **Why:** Real-time flow direction

#### C. Volume Changes
19. **Volume Momentum**
    - `CE_VOL_CHANGE = CE_VOL[-1] - CE_VOL[-2]`
    - `PE_VOL_CHANGE = PE_VOL[-1] - PE_VOL[-2]`
    - **Why:** Increasing volume = increasing conviction

#### D. Greeks Changes
20. **IV Changes**
    - `CE_IV_CHANGE = CE_IV[-1] - CE_IV[-2]`
    - `PE_IV_CHANGE = PE_IV[-1] - PE_IV[-2]`
    - `IV_SKEW_CHANGE = IV_SKEW[-1] - IV_SKEW[-2]`
    - **Why:** IV expansion/contraction indicates fear/greed

21. **Delta Changes**
    - `CE_DELTA_CHANGE = CE_DELTA[-1] - CE_DELTA[-2]`
    - `PE_DELTA_CHANGE = PE_DELTA[-1] - PE_DELTA[-2]`
    - **Why:** Delta drift indicates moneyness shift

---

### CATEGORY 3: Rolling Window Features (t[-window:] to t[-1])
**Use:** Statistics over trailing window (e.g., 20 steps = 1 minute, 100 steps = 5 minutes)

#### A. Price Momentum
22. **Moving Averages**
    - `SPOT_MA_5 = mean(UNDERLYING_LTP[-5:])`
    - `SPOT_MA_10 = mean(UNDERLYING_LTP[-10:])`
    - `SPOT_MA_20 = mean(UNDERLYING_LTP[-20:])`
    - `MA_CROSS = SPOT_MA_5 - SPOT_MA_10`
    - **Why:** Trend direction

23. **Volatility**
    - `SPOT_VOLATILITY = std(diff(UNDERLYING_LTP[-20:])) / mean(UNDERLYING_LTP[-20:])`
    - `VOLATILITY_REGIME = std(UNDERLYING_LTP[-10:]) / std(UNDERLYING_LTP[-20:])`
    - **Why:** High volatility = uncertain market

#### B. OI Momentum (Manipulation Detection)
24. **OI Accumulation Rate**
    - `CE_OI_MOMENTUM = (CE_OI[-1] - CE_OI[-10]) / (CE_OI[-10] + 1e-6)`
    - `PE_OI_MOMENTUM = (PE_OI[-1] - PE_OI[-10]) / (PE_OI[-10] + 1e-6)`
    - **Why:** Sustained OI buildup = institutional accumulation

25. **OI Change Rate**
    - `CE_OICH_MEAN = mean(CE_OICH[-10:])`
    - `PE_OICH_MEAN = mean(PE_OICH[-10:])`
    - `OICH_MOMENTUM_RATIO = PE_OICH_MEAN / (CE_OICH_MEAN + 1e-6)`
    - **Why:** Consistent OI change = trend confirmation

26. **OI Persistence**
    - `CE_OI_PERSISTENCE = count(CE_OICH > 0) / window_size` (last 20 steps)
    - `PE_OI_PERSISTENCE = count(PE_OICH > 0) / window_size`
    - **Why:** Persistent OI increase = strong signal

#### C. Volume Momentum
27. **Volume Trends**
    - `CE_VOL_MA = mean(CE_VOL[-10:])`
    - `PE_VOL_MA = mean(PE_VOL[-10:])`
    - `VOL_MOMENTUM = (CE_VOL[-1] + PE_VOL[-1]) - (CE_VOL[-10] + PE_VOL[-10])`
    - **Why:** Volume surge = event-driven

#### D. Greeks Momentum
28. **IV Trends**
    - `CE_IV_MA = mean(CE_IV[-10:])`
    - `PE_IV_MA = mean(PE_IV[-10:])`
    - `IV_TREND = (CE_IV[-1] + PE_IV[-1]) - (CE_IV[-10] + PE_IV[-10])`
    - **Why:** IV expansion = fear, IV contraction = complacency

29. **Delta Drift**
    - `CE_DELTA_DRIFT = mean(CE_DELTA[-1]) - mean(CE_DELTA[-10])`
    - `PE_DELTA_DRIFT = mean(PE_DELTA[-1]) - mean(PE_DELTA[-10])`
    - **Why:** Delta shift indicates underlying movement

#### E. Cross-Strike Patterns (Spatial Features)
30. **OI Distribution**
    - `CE_OI_CENTROID = sum(CE_OI * STRIKE) / sum(CE_OI)`
    - `PE_OI_CENTROID = sum(PE_OI * STRIKE) / sum(PE_OI)`
    - `OI_CENTROID_DIFF = PE_OI_CENTROID - CE_OI_CENTROID`
    - **Why:** OI concentration location indicates expected price level

31. **OI Spread**
    - `CE_OI_STD = std(CE_OI)` (across strikes)
    - `PE_OI_STD = std(PE_OI)`
    - **Why:** Concentrated OI = stronger max pain effect

32. **Volume Distribution**
    - `CE_VOL_CENTROID = sum(CE_VOL * STRIKE) / sum(CE_VOL)`
    - `PE_VOL_CENTROID = sum(PE_VOL * STRIKE) / sum(PE_VOL)`
    - **Why:** Volume concentration indicates active trading level

---

### CATEGORY 4: Advanced Manipulation Detection Features

#### A. Divergence Signals
33. **Price-OI Divergence**
    - `SPOT_UP_OI_DOWN = (SPOT_RETURN > 0) & (CE_OI_MOMENTUM < 0) & (PE_OI_MOMENTUM < 0)`
    - `SPOT_DOWN_OI_UP = (SPOT_RETURN < 0) & (CE_OI_MOMENTUM > 0) & (PE_OI_MOMENTUM > 0)`
    - **Why:** Divergence = potential reversal

34. **Volume-OI Divergence**
    - `HIGH_VOL_LOW_OI = (VOL_MOMENTUM > threshold) & (OI_MOMENTUM < 0)`
    - `LOW_VOL_HIGH_OI = (VOL_MOMENTUM < threshold) & (OI_MOMENTUM > 0)`
    - **Why:** High volume, low OI = day trading (weak)
    - **Why:** Low volume, high OI = accumulation (strong)

35. **IV-Price Divergence**
    - `SPOT_UP_IV_DOWN = (SPOT_RETURN > 0) & (IV_TREND < 0)`
    - `SPOT_DOWN_IV_UP = (SPOT_RETURN < 0) & (IV_TREND > 0)`
    - **Why:** IV compression during moves = continuation signal

#### B. Accumulation Patterns
36. **OI Buildup Rate**
    - `CE_OI_BUILDUP = sum(CE_OICH[-10:]) / sum(CE_OI[-10:])`
    - `PE_OI_BUILDUP = sum(PE_OICH[-10:]) / sum(PE_OI[-10:])`
    - **Why:** Rapid OI buildup = institutional activity

37. **Strike-Level OI Concentration**
    - `CE_OI_MAX_STRIKE = STRIKE[argmax(CE_OI)]`
    - `PE_OI_MAX_STRIKE = STRIKE[argmax(PE_OI)]`
    - `OI_MAX_DISTANCE = abs(CE_OI_MAX_STRIKE - PE_OI_MAX_STRIKE) / UNDERLYING_LTP`
    - **Why:** Asymmetric OI concentration = directional bias

#### C. Time-Based Patterns
38. **OI Change Persistence**
    - `CE_OICH_PERSISTENCE = count(CE_OICH[-20:] > 0) / 20`
    - `PE_OICH_PERSISTENCE = count(PE_OICH[-20:] > 0) / 20`
    - **Why:** Consistent OI increase = strong accumulation

39. **Volume Burst Detection**
    - `CE_VOL_BURST = (CE_VOL[-1] > 2 * mean(CE_VOL[-20:-1]))`
    - `PE_VOL_BURST = (PE_VOL[-1] > 2 * mean(PE_VOL[-20:-1]))`
    - **Why:** Sudden volume surge = news/event

#### D. Cross-Option Relationships
40. **CE-PE Imbalance**
    - `CE_PE_OI_IMBALANCE = (CE_OI - PE_OI) / (CE_OI + PE_OI)`
    - `CE_PE_VOL_IMBALANCE = (CE_VOL - PE_VOL) / (CE_VOL + PE_VOL)`
    - **Why:** Extreme imbalance = directional signal

41. **Strike-Level PCR**
    - `PCR_BY_STRIKE = PE_OI / (CE_OI + 1e-6)` (per strike)
    - `PCR_SKEW = std(PCR_BY_STRIKE)`
    - **Why:** PCR variation across strikes = sentiment gradient

---

## Feature Implementation Strategy

### Step 1: Latest Value Features (Category 1)
**Computation:** Direct from `matrix[-1]` (latest snapshot)
- No history needed
- Fast computation
- Good for real-time signals

**Examples:**
- PCR, Max Pain, Moneyness, Spreads, IV Skew

### Step 2: Change Features (Category 2)
**Computation:** Compare `matrix[-1]` vs `matrix[-2]` (or `matrix[-k]`)
- Need 2+ snapshots
- Captures momentum
- Good for trend confirmation

**Examples:**
- Spot returns, OI change acceleration, IV changes

### Step 3: Rolling Window Features (Category 3)
**Computation:** Statistics over `matrix[-window:]`
- Need window_size snapshots (e.g., 20 = 1 minute)
- Captures trends and patterns
- Good for manipulation detection

**Examples:**
- OI momentum, volume trends, moving averages, volatility

### Step 4: Advanced Features (Category 4)
**Computation:** Combine multiple categories
- Divergence detection
- Pattern recognition
- Composite signals

**Examples:**
- Price-OI divergence, accumulation patterns, burst detection

---

## Recommended Feature Priority for Manipulation Detection

### Tier 1: Core Manipulation Signals (Must Have)
1. **PCR_ATM** - Primary sentiment
2. **OI_MOMENTUM** (CE/PE) - Accumulation detection
3. **OICH_RATIO** - Real-time flow
4. **MAX_PAIN_DISTANCE** - Price target
5. **VOL_OI_DIVERGENCE** - Weak vs strong signals

### Tier 2: Confirmation Signals (High Value)
6. **IV_SKEW** - Volatility sentiment
7. **OI_PERSISTENCE** - Sustained accumulation
8. **PRICE_OI_DIVERGENCE** - Reversal signals
9. **OI_CENTROID** - Expected price level
10. **GAMMA_EXPOSURE** - Volatility sensitivity

### Tier 3: Context Features (Nice to Have)
11. **SPREAD_RATIO** - Liquidity filter
12. **VOLATILITY_REGIME** - Market state
13. **FUTURE_PREMIUM** - Market structure
14. **DELTA_DRIFT** - Moneyness shift
15. **VOLUME_BURST** - Event detection

---

## Implementation Notes

### Window Sizes
- **Short-term (1-2 min):** 20-40 steps (3s intervals)
- **Medium-term (5 min):** 100 steps
- **Long-term (15 min):** 300 steps (full window)

### Normalization
- **Price-based:** Normalize by spot price
- **OI/Volume:** Normalize by max or sum
- **Greeks:** Already normalized (IV = 0-1, Delta = -1 to +1)

### Missing Data Handling
- Use `np.nanmean`, `np.nanstd` for statistics
- Forward-fill for gaps (if needed)
- Flag low-liquidity strikes (wide spreads)

### Real-Time Computation
- Pre-compute latest value features (Category 1)
- Cache rolling statistics (Category 3)
- Update incrementally (don't recompute full window)

---

## Next Steps

1. **Implement Category 1 features** (latest values) - fastest to implement
2. **Add Category 2 features** (changes) - need 2+ snapshots
3. **Build Category 3 features** (rolling) - need window_size data
4. **Create Category 4 features** (advanced) - combine above
5. **Feature selection** - Use correlation analysis, feature importance
6. **Feature engineering pipeline** - Batch computation for training data

