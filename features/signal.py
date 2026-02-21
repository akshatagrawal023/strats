def trading_signal(self, historical_skew_store) -> Dict[str, any]:
        """
        Generate combined trading signal from both features.
        """
        # Get features
        skew = self.skew_dislocation(historical_skew_store)
        momentum = self.momentum_features()
        
        # Combine into signal
        signal = {
            'timestamp': datetime.now().isoformat(),
            'underlying': self.matrix[self.UNDERLYING, 0],
            **skew,
            **momentum
        }
        
        # Simple rule-based signal for your scalping strategy
        signal['action'] = 'HOLD'
        signal['confidence'] = 0.0
        signal['side'] = None
        
        # Conditions for buying calls:
        # 1. Skew dislocation shows puts expensive (negative skew z-score)
        # 2. Momentum is down (price falling)
        # 3. Move efficiency is high (clean move, not choppy)
        if (signal.get('skew_z_-0.050', 0) < -1.5 and  # Puts expensive (negative z-score means puts IV > calls IV)
            signal.get('momentum_1min', 0) < -0.1 and   # Price falling
            signal.get('move_efficiency', 0) > 0.7):    # Clean move
            
            signal['action'] = 'BUY_CALLS'
            signal['confidence'] = min(1.0, abs(signal['skew_z_-0.050']) / 3.0)
            signal['side'] = 'CALL'
            
        # Conditions for buying puts:
        # 1. Skew dislocation shows calls expensive (positive skew z-score)
        # 2. Momentum is up (price rising)
        # 3. Move efficiency is high
        elif (signal.get('skew_z_0.050', 0) > 1.5 and   # Calls expensive
              signal.get('momentum_1min', 0) > 0.1 and   # Price rising
              signal.get('move_efficiency', 0) > 0.7):
            
            signal['action'] = 'BUY_PUTS'
            signal['confidence'] = min(1.0, signal['skew_z_0.050'] / 3.0)
            signal['side'] = 'PUT'
        
        # Check for volatility expansion (both sides might be good)
        if signal.get('vol_regime', 1.0) > 1.5 and signal.get('skew_dislocation_score', 0) > 1.0:
            signal['vol_regime_signal'] = 'HIGH_VOL_OPPORTUNITY'
        
        return signal