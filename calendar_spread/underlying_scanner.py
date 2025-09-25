"""
Underlying Scanner for Calendar Spread Strategy
Scans and ranks potential underlyings based on multiple criteria
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_utils import get_quotes, get_option_chain
from utils.option_symbols import get_options_data
from utils.symbol_utils import nse_symbol
from .config import CalendarConfig

class UnderlyingScanner:
    """
    Scans and ranks potential underlyings for calendar spread strategy
    """
    
    def __init__(self, config: CalendarConfig):
        self.config = config
        self.logger = logging.getLogger("UnderlyingScanner")
        
    def scan_underlyings(self, symbol_list: List[str] = None) -> List[Dict]:
        """
        Scan potential underlyings and return ranked list
        
        Args:
            symbol_list: List of symbols to scan. If None, uses default list
            
        Returns:
            List of dictionaries with underlying data and scores
        """
        if symbol_list is None:
            symbol_list = list(self.config.LOT_SIZE_MAP.keys())
            
        self.logger.info(f"Scanning {len(symbol_list)} underlyings for calendar spreads")
        
        results = []
        for symbol in symbol_list:
            try:
                score_data = self._evaluate_underlying(symbol)
                if score_data:
                    results.append(score_data)
            except Exception as e:
                self.logger.warning(f"Error evaluating {symbol}: {e}")
                continue
                
        # Sort by total score (descending)
        results.sort(key=lambda x: x['total_score'], reverse=True)
        
        self.logger.info(f"Found {len(results)} suitable underlyings")
        return results
    
    def _evaluate_underlying(self, symbol: str) -> Optional[Dict]:
        """
        Evaluate a single underlying for calendar spread suitability
        
        Returns:
            Dictionary with evaluation data or None if unsuitable
        """
        try:
            # Get current price and basic data
            nse_sym = nse_symbol(symbol)
            quotes = get_quotes([nse_sym])
            
            if not quotes or 'd' not in quotes or not quotes['d']:
                return None
                
            quote_data = quotes['d'][0].get('v', {})
            current_price = quote_data.get('lp') or quote_data.get('ltp')
            
            if not current_price:
                return None
                
            current_price = float(current_price)
            
            # Get option chain data
            option_chain = get_option_chain(nse_sym, strikecount=3)
            if not option_chain or option_chain.get('s') != 'ok':
                return None
                
            chain_data = option_chain.get('data', {}).get('optionsChain', [])
            if not chain_data:
                return None
            
            # Calculate scores
            liquidity_score = self._calculate_liquidity_score(chain_data, current_price)
            volatility_score = self._calculate_volatility_score(chain_data, current_price)
            time_decay_score = self._calculate_time_decay_score(chain_data, current_price)
            risk_score = self._calculate_risk_score(symbol, current_price)
            
            # Weighted total score
            total_score = (
                liquidity_score * 0.3 +
                volatility_score * 0.3 +
                time_decay_score * 0.25 +
                risk_score * 0.15
            )
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'liquidity_score': liquidity_score,
                'volatility_score': volatility_score,
                'time_decay_score': time_decay_score,
                'risk_score': risk_score,
                'total_score': total_score,
                'lot_size': self.config.LOT_SIZE_MAP.get(symbol, self.config.DEFAULT_LOT_SIZE),
                'evaluated_at': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating {symbol}: {e}")
            return None
    
    def _calculate_liquidity_score(self, chain_data: List[Dict], current_price: float) -> float:
        """Calculate liquidity score based on volume and bid-ask spreads"""
        try:
            # Find ATM options
            atm_strike = self._find_atm_strike(chain_data, current_price)
            if not atm_strike:
                return 0.0
                
            # Get CE and PE data for ATM strike
            ce_data = next((opt for opt in chain_data 
                          if opt.get('strike_price') == atm_strike and opt.get('option_type') == 'CE'), None)
            pe_data = next((opt for opt in chain_data 
                          if opt.get('strike_price') == atm_strike and opt.get('option_type') == 'PE'), None)
            
            if not ce_data or not pe_data:
                return 0.0
            
            # Calculate bid-ask spread
            ce_bid = float(ce_data.get('bid', 0))
            ce_ask = float(ce_data.get('ask', 0))
            pe_bid = float(pe_data.get('bid', 0))
            pe_ask = float(pe_data.get('ask', 0))
            
            if ce_ask == 0 or pe_ask == 0:
                return 0.0
                
            ce_spread = (ce_ask - ce_bid) / ce_ask if ce_ask > 0 else 1.0
            pe_spread = (pe_ask - pe_bid) / pe_ask if pe_ask > 0 else 1.0
            avg_spread = (ce_spread + pe_spread) / 2
            
            # Score based on spread (lower is better)
            spread_score = max(0, 1 - (avg_spread / self.config.MAX_BID_ASK_SPREAD))
            
            # Volume score (higher is better)
            ce_volume = int(ce_data.get('volume', 0))
            pe_volume = int(pe_data.get('volume', 0))
            total_volume = ce_volume + pe_volume
            
            volume_score = min(1.0, total_volume / self.config.MIN_OPTION_VOLUME)
            
            return (spread_score * 0.6 + volume_score * 0.4)
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return 0.0
    
    def _calculate_volatility_score(self, chain_data: List[Dict], current_price: float) -> float:
        """Calculate volatility score based on IV and IV rank"""
        try:
            # Find ATM options
            atm_strike = self._find_atm_strike(chain_data, current_price)
            if not atm_strike:
                return 0.0
                
            ce_data = next((opt for opt in chain_data 
                          if opt.get('strike_price') == atm_strike and opt.get('option_type') == 'CE'), None)
            
            if not ce_data:
                return 0.0
            
            # Get IV
            iv = float(ce_data.get('iv', 0))
            if iv == 0:
                return 0.0
            
            # Simple IV rank calculation (would need historical data for accurate calculation)
            # For now, use current IV as proxy
            iv_score = min(1.0, iv / 0.5)  # Normalize to 50% IV
            
            # Check if IV is in acceptable range
            if iv < 0.15 or iv > 0.8:  # Too low or too high
                iv_score *= 0.5
                
            return iv_score
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility score: {e}")
            return 0.0
    
    def _calculate_time_decay_score(self, chain_data: List[Dict], current_price: float) -> float:
        """Calculate time decay score based on theta values"""
        try:
            # Find ATM options
            atm_strike = self._find_atm_strike(chain_data, current_price)
            if not atm_strike:
                return 0.0
                
            ce_data = next((opt for opt in chain_data 
                          if opt.get('strike_price') == atm_strike and opt.get('option_type') == 'CE'), None)
            pe_data = next((opt for opt in chain_data 
                          if opt.get('strike_price') == atm_strike and opt.get('option_type') == 'PE'), None)
            
            if not ce_data or not pe_data:
                return 0.0
            
            # Get theta values
            ce_theta = float(ce_data.get('theta', 0))
            pe_theta = float(pe_data.get('theta', 0))
            
            # Higher theta (more negative) is better for short options
            avg_theta = abs((ce_theta + pe_theta) / 2)
            
            # Score based on theta magnitude
            theta_score = min(1.0, avg_theta / 0.1)  # Normalize to 0.1 theta
            
            return theta_score
            
        except Exception as e:
            self.logger.error(f"Error calculating time decay score: {e}")
            return 0.0
    
    def _calculate_risk_score(self, symbol: str, current_price: float) -> float:
        """Calculate risk score based on underlying characteristics"""
        try:
            # Base score
            risk_score = 0.8
            
            # Adjust for lot size (smaller lot size = lower risk)
            lot_size = self.config.LOT_SIZE_MAP.get(symbol, self.config.DEFAULT_LOT_SIZE)
            if lot_size <= 50:
                risk_score += 0.1
            elif lot_size >= 500:
                risk_score -= 0.1
            
            # Adjust for price level (higher price = potentially more stable)
            if current_price > 1000:
                risk_score += 0.1
            elif current_price < 100:
                risk_score -= 0.1
            
            # Index vs individual stock
            if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                risk_score += 0.1  # Indices are generally less risky
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def _find_atm_strike(self, chain_data: List[Dict], current_price: float) -> Optional[float]:
        """Find the ATM strike price from option chain data"""
        try:
            # Get all strikes
            strikes = []
            for opt in chain_data:
                strike = opt.get('strike_price')
                if strike and opt.get('option_type') in ['CE', 'PE']:
                    strikes.append(float(strike))
            
            if not strikes:
                return None
            
            # Find closest strike to current price
            strikes = list(set(strikes))  # Remove duplicates
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))
            
            return atm_strike
            
        except Exception as e:
            self.logger.error(f"Error finding ATM strike: {e}")
            return None
    
    def get_top_candidates(self, symbol_list: List[str] = None, top_n: int = 5) -> List[Dict]:
        """
        Get top N candidates for calendar spread strategy
        
        Args:
            symbol_list: List of symbols to scan
            top_n: Number of top candidates to return
            
        Returns:
            List of top candidates
        """
        all_candidates = self.scan_underlyings(symbol_list)
        return all_candidates[:top_n]
