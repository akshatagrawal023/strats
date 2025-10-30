"""
Feature Engineering for Options Direction Prediction

Creates features from option chain matrix data that capture:
1. Order flow and momentum signals
2. Option Greeks and implied volatility patterns
3. Open interest dynamics
4. Put-Call ratios and skew
5. Moneyness-based patterns
"""
import numpy as np
import pandas as pd

class OptionFeatureEngine:
    def __init__(self, processor):
        self.processor = processor
    
    def compute_all_features(self, underlying, window=20):
        """Compute all features for the given underlying"""
        timestamps, matrix = self.processor.get_matrix(underlying, window=window)
        
        if matrix is None or len(timestamps) < window:
            return None
        
        features = {}
        
        # Extract channels
        ce_bid = matrix[:, 0, :]
        ce_ask = matrix[:, 1, :]
        pe_bid = matrix[:, 2, :]
        pe_ask = matrix[:, 3, :]
        ce_vol = matrix[:, 4, :]
        pe_vol = matrix[:, 5, :]
        ce_oi = matrix[:, 6, :]
        pe_oi = matrix[:, 7, :]
        ce_oich = matrix[:, 8, :]
        pe_oich = matrix[:, 9, :]
        strikes = matrix[:, 10, :]
        underlying_ltp = matrix[:, 11, :]
        future_price = matrix[:, 12, :]
        
        # === 1. MONEYNESS FEATURES ===
        # Calculate moneyness % for each strike: (strike - underlying) / underlying * 100
        moneyness = (strikes - underlying_ltp) / underlying_ltp * 100
        features['moneyness_atm_idx'] = self._find_atm_index(moneyness[-1])  # Latest ATM position
        
        # === 2. PRICE MOMENTUM ===
        features['underlying_return'] = (underlying_ltp[-1, 0] - underlying_ltp[0, 0]) / underlying_ltp[0, 0]
        features['underlying_volatility'] = np.nanstd(np.diff(underlying_ltp[:, 0])) / np.nanmean(underlying_ltp[:, 0])
        features['future_premium'] = (future_price[-1, 0] - underlying_ltp[-1, 0]) / underlying_ltp[-1, 0]
        
        # === 3. BID-ASK SPREADS (Liquidity indicator) ===
        ce_spread = (ce_ask - ce_bid) / ce_bid  # Normalized spread
        pe_spread = (pe_ask - pe_bid) / pe_bid
        features['ce_spread_atm'] = np.nanmean(ce_spread[-5:, features['moneyness_atm_idx']])
        features['pe_spread_atm'] = np.nanmean(pe_spread[-5:, features['moneyness_atm_idx']])
        features['liquidity_ratio'] = features['ce_spread_atm'] / (features['pe_spread_atm'] + 1e-6)
        
        # === 4. OPEN INTEREST DYNAMICS ===
        # PCR (Put-Call Ratio) at different moneyness levels
        atm_idx = features['moneyness_atm_idx']
        features['pcr_atm'] = pe_oi[-1, atm_idx] / (ce_oi[-1, atm_idx] + 1e-6)
        features['pcr_otm'] = np.nansum(pe_oi[-1, atm_idx+1:]) / (np.nansum(ce_oi[-1, :atm_idx]) + 1e-6)
        features['pcr_total'] = np.nansum(pe_oi[-1, :]) / (np.nansum(ce_oi[-1, :]) + 1e-6)
        
        # OI Change momentum
        features['ce_oich_atm'] = ce_oich[-1, atm_idx]
        features['pe_oich_atm'] = pe_oich[-1, atm_idx]
        features['oich_ratio_atm'] = pe_oich[-1, atm_idx] / (ce_oich[-1, atm_idx] + 1e-6)
        
        # Rolling OI change (bullish if CE_OI increasing or PE_OI decreasing)
        features['ce_oi_momentum'] = (ce_oi[-1, atm_idx] - ce_oi[-10, atm_idx]) / (ce_oi[-10, atm_idx] + 1e-6)
        features['pe_oi_momentum'] = (pe_oi[-1, atm_idx] - pe_oi[-10, atm_idx]) / (pe_oi[-10, atm_idx] + 1e-6)
        
        # === 5. VOLUME ANALYSIS ===
        # Volume at ATM and OTM strikes
        features['ce_vol_atm'] = np.nanmean(ce_vol[-5:, atm_idx])
        features['pe_vol_atm'] = np.nanmean(pe_vol[-5:, atm_idx])
        features['vol_ratio_atm'] = features['pe_vol_atm'] / (features['ce_vol_atm'] + 1e-6)
        
        # Volume momentum (increasing volume = stronger signal)
        features['ce_vol_momentum'] = (ce_vol[-1, atm_idx] - ce_vol[-10, atm_idx]) / (ce_vol[-10, atm_idx] + 1e-6)
        features['pe_vol_momentum'] = (pe_vol[-1, atm_idx] - pe_vol[-10, atm_idx]) / (pe_vol[-10, atm_idx] + 1e-6)
        
        # === 6. OPTION SKEW (ITM vs OTM) ===
        # Price skew indicates market sentiment
        if atm_idx > 0 and atm_idx < strikes.shape[1] - 1:
            itm_ce = np.nanmean(ce_bid[-1, :atm_idx])
            otm_ce = np.nanmean(ce_ask[-1, atm_idx+1:])
            itm_pe = np.nanmean(pe_bid[-1, atm_idx+1:])
            otm_pe = np.nanmean(pe_ask[-1, :atm_idx])
            
            features['ce_skew'] = (itm_ce - otm_ce) / (itm_ce + 1e-6)
            features['pe_skew'] = (itm_pe - otm_pe) / (itm_pe + 1e-6)
            features['skew_diff'] = features['ce_skew'] - features['pe_skew']
        
        # === 7. ORDER FLOW (Bid-Ask Pressure) ===
        # Higher bid pressure = bullish, higher ask pressure = bearish
        ce_bid_pressure = ce_bid / (ce_bid + ce_ask + 1e-6)
        pe_bid_pressure = pe_bid / (pe_bid + pe_ask + 1e-6)
        features['ce_bid_pressure_atm'] = np.nanmean(ce_bid_pressure[-5:, atm_idx])
        features['pe_bid_pressure_atm'] = np.nanmean(pe_bid_pressure[-5:, atm_idx])
        
        # === 8. MAX PAIN / OI CONCENTRATION ===
        # Strike with maximum total OI (often acts as magnet)
        total_oi = ce_oi[-1, :] + pe_oi[-1, :]
        max_oi_idx = np.nanargmax(total_oi)
        features['max_pain_distance'] = (strikes[-1, max_oi_idx] - underlying_ltp[-1, 0]) / underlying_ltp[-1, 0]
        features['max_pain_pull'] = total_oi[max_oi_idx] / np.nansum(total_oi)  # Concentration ratio
        
        # === 9. CROSS-STRIKE PATTERNS ===
        # OI buildup pattern across strikes
        ce_oi_slope = np.nanmean(np.diff(ce_oi[-1, :]))
        pe_oi_slope = np.nanmean(np.diff(pe_oi[-1, :]))
        features['oi_slope_diff'] = ce_oi_slope - pe_oi_slope
        
        # === 10. TIME-BASED FEATURES ===
        # Rolling statistics
        features['underlying_ma_5'] = np.nanmean(underlying_ltp[-5:, 0])
        features['underlying_ma_10'] = np.nanmean(underlying_ltp[-10:, 0])
        features['ma_cross'] = features['underlying_ma_5'] - features['underlying_ma_10']
        
        # Volatility regime
        features['vol_regime'] = np.nanstd(underlying_ltp[-10:, 0]) / np.nanstd(underlying_ltp[-20:, 0])
        
        # === 11. COMPOSITE SIGNALS ===
        # Bullish signal: PE OI increase + CE OI decrease + High PCR
        features['bullish_signal'] = (
            (features['pe_oi_momentum'] > 0) * 1.0 +
            (features['ce_oi_momentum'] < 0) * 1.0 +
            (features['pcr_atm'] > 1.2) * 1.0
        ) / 3.0
        
        # Bearish signal: CE OI increase + PE OI decrease + Low PCR
        features['bearish_signal'] = (
            (features['ce_oi_momentum'] > 0) * 1.0 +
            (features['pe_oi_momentum'] < 0) * 1.0 +
            (features['pcr_atm'] < 0.8) * 1.0
        ) / 3.0
        
        features['direction_signal'] = features['bullish_signal'] - features['bearish_signal']
        
        return features
    
    def _find_atm_index(self, moneyness_row):
        """Find the index closest to ATM (moneyness = 0)"""
        return np.nanargmin(np.abs(moneyness_row))
    
    def get_feature_dataframe(self, underlying, window=20):
        """Get features as a pandas DataFrame row"""
        features = self.compute_all_features(underlying, window)
        if features is None:
            return None
        return pd.DataFrame([features])
    
    def explain_features(self):
        """Print explanation of all features"""
        explanations = {
            'MOMENTUM': [
                'underlying_return: Price change over window (positive = bullish)',
                'underlying_volatility: Price volatility (higher = more uncertain)',
                'future_premium: Future vs spot premium (positive = contango)',
                'ma_cross: MA5 - MA10 crossover signal'
            ],
            'LIQUIDITY': [
                'ce_spread_atm: Call bid-ask spread at ATM (lower = more liquid)',
                'pe_spread_atm: Put bid-ask spread at ATM',
                'liquidity_ratio: Relative liquidity between CE and PE'
            ],
            'OPEN_INTEREST': [
                'pcr_atm: Put-Call ratio at ATM (>1 = bearish, <1 = bullish)',
                'pcr_otm: Put-Call ratio for OTM options',
                'pcr_total: Overall Put-Call ratio',
                'ce_oi_momentum: Call OI change (increasing = bearish)',
                'pe_oi_momentum: Put OI change (increasing = bullish)',
                'oich_ratio_atm: PE_OICH / CE_OICH (>1 = bullish)'
            ],
            'VOLUME': [
                'vol_ratio_atm: PE_VOL / CE_VOL (>1 = bearish)',
                'ce_vol_momentum: Call volume momentum',
                'pe_vol_momentum: Put volume momentum'
            ],
            'SKEW': [
                'ce_skew: Call ITM vs OTM price difference',
                'pe_skew: Put ITM vs OTM price difference',
                'skew_diff: Relative skew (positive = bullish skew)'
            ],
            'ORDER_FLOW': [
                'ce_bid_pressure_atm: Call bid strength (>0.5 = strong buying)',
                'pe_bid_pressure_atm: Put bid strength'
            ],
            'MAX_PAIN': [
                'max_pain_distance: Distance to max OI strike',
                'max_pain_pull: OI concentration at max pain'
            ],
            'COMPOSITE': [
                'bullish_signal: 0-1 score, higher = more bullish',
                'bearish_signal: 0-1 score, higher = more bearish',
                'direction_signal: -1 to +1, negative = bearish, positive = bullish'
            ]
        }
        
        for category, features in explanations.items():
            print(f"\n=== {category} ===")
            for feature in features:
                print(f"  • {feature}")

if __name__ == "__main__":
    # Example usage
    from chain_processor import processor
    
    feature_engine = OptionFeatureEngine(processor)
    feature_engine.explain_features()

