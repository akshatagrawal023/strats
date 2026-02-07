import numpy as np
import pandas as pd
from numba import jit, float64, int8, prange
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import threading
from collections import defaultdict

# Implement Numba-compatible normal distribution functions
@jit(nopython=True, fastmath=True)
def norm_pdf_numba(x):
    """Numba-compatible standard normal PDF"""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

@jit(nopython=True, fastmath=True)
def norm_cdf_numba(x):
    """Numba-compatible standard normal CDF using error function"""
    if x >= 0:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    else:
        return 0.5 * (1.0 - math.erf(-x / math.sqrt(2.0)))

@dataclass
class Greeks:
    """Vectorized Greeks storage"""
    delta: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    vega: np.ndarray
    iv: np.ndarray

class VolatilityDashboard:
    def __init__(self, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.underlying_data = {}
        self.option_chains_df = pd.DataFrame()
        self.expiries = {}
        self.greeks_cache = {}
        self.last_update = 0
        self.cache_validity = 0.1  # 100ms cache validity
        self.lock = threading.RLock()
        
    def update_data(self, raw_data: dict):
        """Update with new market data - optimized for speed"""
        with self.lock:
            current_time = time.time()
            
            # Process underlying data
            underlying = raw_data['data']['optionsChain'][0]
            self.underlying_data = {
                'symbol': underlying['ex_symbol'],
                'price': underlying['ltp'],
                'bid': underlying['bid'],
                'ask': underlying['ask'],
                'change': underlying['ltpch']
            }
            
            # Process expiry data
            self.expiries = {exp['expiry']: exp['date'] for exp in raw_data['data']['expiryData']}
            
            # Process options chain
            options_data = []
            for opt in raw_data['data']['optionsChain'][1:]:  # Skip underlying
                if opt['option_type'] in ['CE', 'PE']:
                    options_data.append({
                        'symbol': opt['symbol'],
                        'strike': float(opt['strike_price']),
                        'expiry': self._extract_expiry_from_symbol(opt['symbol']),
                        'option_type': opt['option_type'],
                        'price': opt['ltp'],
                        'bid': opt['bid'],
                        'ask': opt['ask'],
                        'oi': opt.get('oi', 0),
                        'volume': opt.get('volume', 0),
                        'token': opt['fyToken'],
                        'time_to_expiry': self._calculate_time_to_expiry(opt['symbol'])
                    })
            
            # Convert to DataFrame
            self.option_chains_df = pd.DataFrame(options_data)
            self.last_update = current_time
            self.greeks_cache.clear()  # Invalidate cache on new data

    def _extract_expiry_from_symbol(self, symbol: str) -> str:
        """Extract expiry from symbol efficiently"""
        # Example: NSE:SBIN25SEP840CE -> SEP25
        parts = symbol.split(':')[-1]
        if len(parts) >= 9 and parts[4:7].isalpha():
            return f"{parts[4:7]}{parts[3:4]}"  # SEP5 -> SEP25 (year last digit)
        return "UNKNOWN"

    def _calculate_time_to_expiry(self, symbol: str) -> float:
        """Calculate time to expiry in years for a specific option"""
        # Placeholder - you'd implement actual expiry date parsing
        # For now, using fixed values based on your data
        if '25SEP' in symbol:
            return 0.25  # ~3 months
        elif '25OCT' in symbol:
            return 0.33  # ~4 months
        elif '25NOV' in symbol:
            return 0.42  # ~5 months
        return 0.25  # Default

    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True)
    def _black_scholes_vectorized(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
                                 r: float, q: float, sigma: np.ndarray, 
                                 option_type: np.ndarray) -> Tuple:
        """Vectorized Black-Scholes implementation using Numba"""
        n = len(S)
        prices = np.empty(n, dtype=np.float64)
        deltas = np.empty(n, dtype=np.float64)
        gammas = np.empty(n, dtype=np.float64)
        thetas = np.empty(n, dtype=np.float64)
        vegas = np.empty(n, dtype=np.float64)
        
        for i in prange(n):
            # Use local variable to avoid mutating input
            T_local = T[i] if T[i] > 0 else 1e-6
            
            sqrt_T = math.sqrt(T_local)
            d1 = (math.log(S[i] / K[i]) + (r - q + 0.5 * sigma[i] * sigma[i]) * T_local) / (sigma[i] * sqrt_T)
            d2 = d1 - sigma[i] * sqrt_T
            
            pdf_d1 = norm_pdf_numba(d1)
            cdf_d1 = norm_cdf_numba(d1)
            cdf_d2 = norm_cdf_numba(d2)
            
            if option_type[i] == 1:  # Call
                deltas[i] = math.exp(-q * T_local) * cdf_d1
                prices[i] = S[i] * math.exp(-q * T_local) * cdf_d1 - K[i] * math.exp(-r * T_local) * cdf_d2
            else:  # Put
                deltas[i] = math.exp(-q * T_local) * (cdf_d1 - 1)
                prices[i] = K[i] * math.exp(-r * T_local) * norm_cdf_numba(-d2) - S[i] * math.exp(-q * T_local) * norm_cdf_numba(-d1)
            
            gammas[i] = math.exp(-q * T_local) * pdf_d1 / (S[i] * sigma[i] * sqrt_T)
            vegas[i] = S[i] * math.exp(-q * T_local) * pdf_d1 * sqrt_T / 100
            
            # Theta calculation
            if option_type[i] == 1:  # Call
                thetas[i] = -(S[i] * sigma[i] * math.exp(-q * T_local) * pdf_d1) / (2 * sqrt_T) \
                           - r * K[i] * math.exp(-r * T_local) * cdf_d2 \
                           + q * S[i] * math.exp(-q * T_local) * cdf_d1
            else:  # Put
                thetas[i] = -(S[i] * sigma[i] * math.exp(-q * T_local) * pdf_d1) / (2 * sqrt_T) \
                           + r * K[i] * math.exp(-r * T_local) * norm_cdf_numba(-d2) \
                           - q * S[i] * math.exp(-q * T_local) * norm_cdf_numba(-d1)
            
            thetas[i] /= 365  # Daily theta
        
        return prices, deltas, gammas, thetas, vegas

    def compute_greeks(self) -> Optional[Greeks]:
        """Compute Greeks for all options with caching"""
        current_time = time.time()
        
        # Check cache validity
        if (current_time - self.last_update) < self.cache_validity and self.greeks_cache:
            return self.greeks_cache['latest']
        
        with self.lock:
            if self.option_chains_df.empty:
                return None
            
            # Prepare data for vectorized computation - use raw numpy arrays
            S = np.full(len(self.option_chains_df), self.underlying_data['price'], dtype=np.float64)
            K = self.option_chains_df['strike'].values.astype(np.float64)
            T = self.option_chains_df['time_to_expiry'].values.astype(np.float64)
            r = self.risk_free_rate
            q = self.dividend_yield
            sigma = self._estimate_implied_volatility()
            
            # Convert option type to numeric (1 for CE, 0 for PE)
            option_type_numeric = np.where(self.option_chains_df['option_type'] == 'CE', 1, 0).astype(np.int8)
            
            # Compute Greeks
            prices, deltas, gammas, thetas, vegas = self._black_scholes_vectorized(
                S, K, T, r, q, sigma, option_type_numeric
            )
            
            # Store results
            greeks = Greeks(deltas, gammas, thetas, vegas, sigma)
            self.greeks_cache['latest'] = greeks
            self.greeks_cache['timestamp'] = current_time
            
            return greeks

    def _estimate_implied_volatility(self) -> np.ndarray:
        """Fast IV estimation - using historical volatility as placeholder"""
        # For real implementation, use the Newton-Raphson method below
        n = len(self.option_chains_df)
        return np.full(n, 0.25, dtype=np.float64)  # 25% IV placeholder

    def aggregate_greeks(self, portfolio: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """Aggregate Greeks across portfolio with position sizing"""
        greeks = self.compute_greeks()
        if greeks is None:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'net_exposure': 0}
        
        if portfolio is None:
            # Aggregate all options equally
            agg_delta = np.sum(greeks.delta)
            agg_gamma = np.sum(greeks.gamma)
            agg_theta = np.sum(greeks.theta)
            agg_vega = np.sum(greeks.vega)
        else:
            # Weight by portfolio positions
            positions = np.array([portfolio.get(token, 0) for token in self.option_chains_df['token']])
            agg_delta = np.sum(greeks.delta * positions)
            agg_gamma = np.sum(greeks.gamma * positions)
            agg_theta = np.sum(greeks.theta * positions)
            agg_vega = np.sum(greeks.vega * positions)
        
        return {
            'delta': agg_delta,
            'gamma': agg_gamma,
            'theta': agg_theta,
            'vega': agg_vega,
            'net_exposure': agg_delta * self.underlying_data['price']
        }

    def get_vol_surface(self) -> pd.DataFrame:
        """Generate volatility surface for visualization"""
        greeks = self.compute_greeks()
        if greeks is None:
            return pd.DataFrame()
        
        vol_surface = pd.DataFrame({
            'strike': self.option_chains_df['strike'],
            'expiry': self.option_chains_df['expiry'],
            'type': self.option_chains_df['option_type'],
            'iv': greeks.iv,
            'delta': greeks.delta,
            'gamma': greeks.gamma,
            'theta': greeks.theta,
            'vega': greeks.vega,
            'price': self.option_chains_df['price'],
            'oi': self.option_chains_df['oi'],
            'volume': self.option_chains_df['volume']
        })
        
        return vol_surface

# Newton-Raphson IV calculation with Numba
@jit(nopython=True, fastmath=True)
def implied_volatility_newton(S: float, K: float, T: float, r: float, 
                             market_price: float, option_type_int: int, 
                             max_iter: int = 100, precision: float = 1e-6) -> float:
    """Newton-Raphson method for IV calculation - Numba compatible"""
    sigma = 0.3  # Initial guess
    
    for i in range(max_iter):
        if T <= 0:
            T = 1e-6
            
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
        
        pdf_d1 = norm_pdf_numba(d1)
        cdf_d1 = norm_cdf_numba(d1)
        
        if option_type_int == 1:  # Call
            d2 = d1 - sigma * sqrt_T
            cdf_d2 = norm_cdf_numba(d2)
            price = S * cdf_d1 - K * math.exp(-r * T) * cdf_d2
            vega = S * pdf_d1 * sqrt_T
        else:  # Put
            d2 = d1 - sigma * sqrt_T
            price = K * math.exp(-r * T) * norm_cdf_numba(-d2) - S * norm_cdf_numba(-d1)
            vega = S * pdf_d1 * sqrt_T
        
        diff = market_price - price
        
        if abs(diff) < precision:
            return sigma
        
        # Avoid division by very small vega
        if abs(vega) > 1e-10:
            sigma += diff / vega
        else:
            sigma += 0.01 * np.sign(diff)  # Small adjustment if vega is too small
    
    return sigma  # Return best estimate

# Usage with your data
def process_sbin_data():
    """Process your specific SBIN data"""
    raw_data = {
        'code': 200, 
        'data': {
            'callOi': 43992750, 
            'expiryData': [
                {'date': '30-09-2025', 'expiry': '1759226400'}, 
                {'date': '28-10-2025', 'expiry': '1761645600'}, 
                {'date': '25-11-2025', 'expiry': '1764064800'}
            ], 
            'indiavixData': {
                'ask': 0, 'bid': 0, 'description': 'INDIAVIX-INDEX', 
                'ex_symbol': 'INDIAVIX', 'exchange': 'NSE', 
                'fyToken': '101000000026017', 'ltp': 10.03, 
                'ltpch': -0.22, 'ltpchp': -2.15, 'option_type': '', 
                'strike_price': -1, 'symbol': 'NSE:INDIAVIX-INDEX'
            }, 
            'optionsChain': [
                {
                    'ask': 854.95, 'bid': 854.75, 'description': 'STATE BANK OF INDIA', 
                    'ex_symbol': 'SBIN', 'exchange': 'NSE', 'fp': 856.5, 
                    'fpch': -3.55, 'fpchp': -0.41, 'fyToken': '10100000003045', 
                    'ltp': 854.75, 'ltpch': -2.4, 'ltpchp': -0.28, 'option_type': '', 
                    'strike_price': -1, 'symbol': 'NSE:SBIN-EQ'
                }, 
                {
                    'ask': 20.9, 'bid': 20.8, 'fyToken': '1011250930132472', 
                    'ltp': 20.85, 'ltpch': -3.85, 'ltpchp': -15.59, 'oi': 2855250, 
                    'oich': -285000, 'oichp': -9.08, 'option_type': 'CE', 
                    'prev_oi': 3140250, 'strike_price': 840, 'symbol': 'NSE:SBIN25SEP840CE', 
                    'volume': 4756500
                },
                # ... rest of your options data
            ], 
            'putOi': 41502000
        }, 
        'message': '', 
        's': 'ok'
    }
    
    dashboard = VolatilityDashboard()
    dashboard.update_data(raw_data)
    
    # Compute Greeks
    greeks = dashboard.compute_greeks()
    
    # Get aggregated exposure
    aggregated = dashboard.aggregate_greeks()
    
    print(f"Underlying: {dashboard.underlying_data['symbol']} @ {dashboard.underlying_data['price']}")
    print(f"Aggregated Greeks: Delta={aggregated['delta']:.3f}, Gamma={aggregated['gamma']:.6f}")
    print(f"Theta={aggregated['theta']:.3f}, Vega={aggregated['vega']:.3f}")
    
    return dashboard, greeks, aggregated

# Run the processing
if __name__ == "__main__":
    dashboard, greeks, aggregated = process_sbin_data()