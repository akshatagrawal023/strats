
import numpy as np

from features.base import FeatureExtractor

class PriceMomentumFeatures(FeatureExtractor):
    def compute(self, price_df):
        # price_df is a DataFrame with columns ['close', 'volume']
        return {'rsi': rsi(price_df['close']), 'macd': macd(price_df['close'])}