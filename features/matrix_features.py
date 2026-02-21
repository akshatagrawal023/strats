# features/matrix_features.py
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

class FastMatrixFeatures:
    """
    Ultra-fast feature extraction directly from matrix.
    Now with skew dislocation and momentum features.
    """
    
    # Channel indices (your format)
    STRIKE = 10
    UNDERLYING = 11
    CE_IV = 13
    PE_IV = 14
    CE_DELTA = 15
    PE_DELTA = 16
    CE_VOL = 4
    PE_VOL = 5
    
    def __init__(self, matrix: np.ndarray, T: float, r: float = 0.065):
        self.matrix = matrix
        self.T = T
        self.r = r
        self.n_strikes = matrix.shape[1]
        
        # Pre-compute moneyness (vectorized)
        strikes = matrix[self.STRIKE, :]
        S = matrix[self.UNDERLYING, 0]
        F = S * np.exp(r * T)
        self.log_moneyness = np.log(strikes / F)
        
        # Cache ATM index
        self.atm_idx = np.argmin(np.abs(self.log_moneyness))
        
        # For momentum features (need historical prices)
        self.price_history = deque(maxlen=100)  # Last 100 ticks
        self.vol_history = deque(maxlen=100)    # Last 100 realized vol estimates
        
    def update_price(self, price: float):
        """Update with new underlying price for momentum calculations."""
        self.price_history.append(price)
        
        # Update realized vol if we have enough data
        if len(self.price_history) >= 10:
            returns = np.diff(list(self.price_history)) / np.array(list(self.price_history)[:-1])
            self.vol_history.append(np.std(returns) * np.sqrt(252 * 390))  # Annualized
    
    # ==================== FEATURE 1: SKEW DISLOCATION ====================
    
    def get_skew_at_moneyness(self, log_moneyness_target: float) -> float:
        """
        Get skew (call_iv - put_iv) at specific log-moneyness point.
        
        Args:
            log_moneyness_target: e.g., 0.05 for 5% OTM call
        
        Returns:
            skew value (positive if calls more expensive)
        """
        # Find strike index closest to target moneyness
        idx = np.argmin(np.abs(self.log_moneyness - log_moneyness_target))
        
        call_iv = self.matrix[self.CE_IV, idx]
        put_iv = self.matrix[self.PE_IV, idx]
        
        return call_iv - put_iv
    
    def get_skew_surface(self) -> np.ndarray:
        """
        Get skew across all moneyness points.
        Returns array of skew values at each strike.
        """
        return self.matrix[self.CE_IV, :] - self.matrix[self.PE_IV, :]
    
    def skew_dislocation(self, historical_skew_store) -> Dict[str, float]:
        """
        Measure how current skew deviates from normal.
        
        Returns:
            Dictionary with dislocation metrics at key moneyness points
        """
        key_points = [-0.05, -0.025, 0, 0.025, 0.05]  # 5% OTM put to 5% OTM call
        results = {}
        
        for k in key_points:
            current_skew = self.get_skew_at_moneyness(k)
            
            # Get historical distribution for this moneyness point
            hist_skews = historical_skew_store.get(f'skew_{k:.3f}', [])
            
            if len(hist_skews) >= 20:
                mean_skew = np.mean(hist_skews)
                std_skew = np.std(hist_skews)
                
                # Z-score of current skew
                zscore = (current_skew - mean_skew) / (std_skew + 1e-10)
                
                # Percentile (0-100)
                percentile = np.sum(np.array(hist_skews) < current_skew) / len(hist_skews) * 100
                
                results[f'skew_z_{k:.3f}'] = zscore
                results[f'skew_pct_{k:.3f}'] = percentile
            else:
                results[f'skew_z_{k:.3f}'] = np.nan
                results[f'skew_pct_{k:.3f}'] = np.nan
            
            results[f'skew_abs_{k:.3f}'] = current_skew
        
        # Overall dislocation score (average of absolute z-scores)
        z_scores = [v for k, v in results.items() if 'skew_z' in k and not np.isnan(v)]
        results['skew_dislocation_score'] = np.mean(np.abs(z_scores)) if z_scores else np.nan
        
        # Skew term structure (difference between short-dated and long-dated skew)
        # This would require multiple expiries - placeholder for now
        results['skew_term'] = 0.0
        
        return results
    
    # ==================== FEATURE 2: MOMENTUM OF THE MOVE ====================
    
    def momentum_features(self) -> Dict[str, float]:
        """
        Calculate momentum and acceleration of underlying move.
        
        Returns:
            Dictionary with momentum metrics
        """
        results = {}
        
        if len(self.price_history) < 10:
            # Not enough data
            results['momentum_1min'] = np.nan
            results['momentum_5min'] = np.nan
            results['acceleration'] = np.nan
            results['move_efficiency'] = np.nan
            results['volume_pressure'] = np.nan
            return results
        
        prices = np.array(list(self.price_history))
        
        # 1. Price momentum over different windows
        current_price = prices[-1]
        
        # 1-minute momentum (assuming ~1 sec per update, so 60 ticks)
        if len(prices) >= 60:
            price_1min_ago = prices[-60]
            results['momentum_1min'] = (current_price - price_1min_ago) / price_1min_ago * 100  # percent
        else:
            results['momentum_1min'] = (current_price - prices[0]) / prices[0] * 100
        
        # 5-minute momentum (300 ticks)
        if len(prices) >= 300:
            price_5min_ago = prices[-300]
            results['momentum_5min'] = (current_price - price_5min_ago) / price_5min_ago * 100
        else:
            results['momentum_5min'] = results['momentum_1min']  # fallback
        
        # 2. Acceleration (change in momentum)
        # Calculate short-term momentum trend
        if len(prices) >= 30:
            recent_returns = np.diff(prices[-30:]) / prices[-30:-1]
            older_returns = np.diff(prices[-60:-30]) / prices[-60:-31] if len(prices) >= 60 else recent_returns
            
            recent_momentum = np.mean(recent_returns)
            older_momentum = np.mean(older_returns)
            
            results['acceleration'] = (recent_momentum - older_momentum) * 10000  # basis points
        else:
            results['acceleration'] = 0.0
        
        # 3. Move efficiency (how directional vs choppy)
        # High efficiency = price moved in straight line
        if len(prices) >= 20:
            total_move = abs(prices[-1] - prices[-20])
            total_path = np.sum(np.abs(np.diff(prices[-20:])))
            results['move_efficiency'] = total_move / total_path if total_path > 0 else 1.0
        else:
            results['move_efficiency'] = 1.0
        
        # 4. Volume pressure (from options volume)
        # Are people piling into calls or puts during the move?
        total_call_vol = np.nansum(self.matrix[self.CE_VOL, :])
        total_put_vol = np.nansum(self.matrix[self.PE_VOL, :])
        total_vol = total_call_vol + total_put_vol
        
        if total_vol > 0:
            results['volume_pressure'] = (total_call_vol - total_put_vol) / total_vol
        else:
            results['volume_pressure'] = 0.0
        
        # 5. Volatility regime (from realized vol history)
        if len(self.vol_history) >= 20:
            current_vol = self.vol_history[-1]
            avg_vol = np.mean(list(self.vol_history))
            results['vol_regime'] = current_vol / avg_vol
        else:
            results['vol_regime'] = 1.0
        
        return results
    