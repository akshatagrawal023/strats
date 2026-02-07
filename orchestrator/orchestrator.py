# orchestrator.py
from typing import Dict, List, Optional
import asyncio
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradingSignal:
    """ML-generated trading signal."""
    symbol: str
    timestamp: datetime
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0-1
    features: Dict[str, float]
    model_version: str
    expected_return: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class MLTradingOrchestrator:
    """
    Complete orchestration for ML-based options trading.
    """
    
    def __init__(self, greeks_engine: AsyncGreeksEngine):
        self.greeks_engine = greeks_engine
        self.feature_calculators = {
            'iv': IVFeatureCalculator(),
            'greek': GreekFeatureCalculator(),
            'volume': VolumeFeatureCalculator()  # You'd implement this
        }
        self.model_predictor = ModelPredictor()  # Your ML model wrapper
    
    async def generate_signals(self, symbols: List[str], 
                              strategy_config: Dict) -> List[TradingSignal]:
        """Generate trading signals for multiple symbols."""
        
        # 1. Fetch market data for all symbols
        market_data = await self._fetch_market_data(symbols)
        
        # 2. Calculate Greeks for all symbols (concurrently)
        greeks_requests = []
        for symbol, data in market_data.items():
            request = GreeksRequest(
                symbol=symbol,
                underlying_price=data['underlying_price'],
                strikes=tuple(data['strikes']),
                ce_bids=tuple(data['ce_bids']),
                ce_asks=tuple(data['ce_asks']),
                pe_bids=tuple(data['pe_bids']),
                pe_asks=tuple(data['pe_asks']),
                days_to_expiry=data['days_to_expiry']
            )
            greeks_requests.append(request)
        
        greeks_results = await self.greeks_engine.calculate_batch(greeks_requests)
        
        # 3. Calculate features for ML
        all_signals = []
        for symbol, greeks_result in greeks_results.items():
            signal = await self._process_single_symbol(
                symbol, greeks_result, strategy_config
            )
            if signal:
                all_signals.append(signal)
        
        return all_signals
    
    async def _process_single_symbol(self, symbol: str, 
                                   greeks_result: GreeksResult,
                                   strategy_config: Dict) -> Optional[TradingSignal]:
        """Process a single symbol through the ML pipeline."""
        
        # Calculate features
        features = {}
        
        # IV features
        iv_features = self.feature_calculators['iv'].iv_percentile(
            np.concatenate([greeks_result.ce_iv, greeks_result.pe_iv])
        )
        features['iv_percentile'] = iv_features
        
        # Skew features
        atm_idx = np.argmin(np.abs(greeks_result.strikes - 
                                 greeks_result.ce_iv.mean()))  # Approximation
        skew_features = self.feature_calculators['iv'].skew_smile(
            greeks_result.ce_iv, greeks_result.pe_iv,
            greeks_result.strikes, atm_idx
        )
        features.update({f'skew_{k}': v for k, v in skew_features.items()})
        
        # Greek-based features
        # Assume we have positions data (you'd fetch this from your portfolio)
        positions = np.zeros_like(greeks_result.strikes)  # Placeholder
        
        delta_exp = self.feature_calculators['greek'].delta_exposure(
            greeks_result.ce_delta, positions
        )
        features['delta_exposure'] = delta_exp
        
        gamma_exp = self.feature_calculators['greek'].gamma_exposure(
            greeks_result.ce_gamma, positions, 
            greeks_result.strikes.mean()  # Approx underlying
        )
        features['gamma_exposure'] = gamma_exp
        
        # 4. Run ML model inference
        model_input = self._prepare_model_input(features)
        
        # This could be XGBoost, Neural Net, etc.
        prediction = await self.model_predictor.predict(model_input)
        
        # 5. Generate trading signal
        if prediction['confidence'] > strategy_config.get('min_confidence', 0.7):
            signal = TradingSignal(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                signal=prediction['direction'],
                confidence=prediction['confidence'],
                features=features,
                model_version=prediction['model_version'],
                expected_return=prediction.get('expected_return'),
                stop_loss=self._calculate_stop_loss(features, strategy_config),
                take_profit=self._calculate_take_profit(features, strategy_config)
            )
            return signal
        
        return None
    
    async def _fetch_market_data(self, symbols: List[str]) -> Dict:
        """Fetch market data for symbols."""
        # Implement based on your data source (Kite, Bloomberg, etc.)
        # This should be async
        market_data = {}
        
        for symbol in symbols:
            # Simulated data - replace with actual API calls
            market_data[symbol] = {
                'underlying_price': np.random.uniform(1000, 2000),
                'strikes': np.linspace(950, 1050, 21),
                'ce_bids': np.random.uniform(5, 50, 21),
                'ce_asks': np.random.uniform(5, 50, 21),
                'pe_bids': np.random.uniform(5, 50, 21),
                'pe_asks': np.random.uniform(5, 50, 21),
                'days_to_expiry': 7
            }
        
        return market_data
    
    def _prepare_model_input(self, features: Dict) -> np.ndarray:
        """Prepare features for model inference."""
        # Normalize, scale, etc.
        feature_vector = np.array(list(features.values()))
        return np.nan_to_num(feature_vector, nan=0.0)
    
    def _calculate_stop_loss(self, features: Dict, config: Dict) -> float:
        """Calculate stop loss based on volatility."""
        iv = features.get('atm_iv', 0.2)
        return config.get('stop_loss_multiplier', 1.5) * iv
    
    def _calculate_take_profit(self, features: Dict, config: Dict) -> float:
        """Calculate take profit based on volatility."""
        iv = features.get('atm_iv', 0.2)
        return config.get('take_profit_multiplier', 2.0) * iv