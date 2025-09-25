"""
Main Calendar Spread Strategy Implementation
Handles position management, entry/exit logic, and risk management
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_utils import get_quotes, place_order, get_positions, get_orderbook
from utils.symbol_utils import nse_symbol, build_option_symbol, build_option_pair
from utils.trade_costs import calculate_total_trade_cost
from .config import CalendarConfig
from .underlying_scanner import UnderlyingScanner

class PositionStatus(Enum):
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"

@dataclass
class CalendarPosition:
    """Data class for calendar spread position"""
    symbol: str
    strike: float
    short_expiry: str
    long_expiry: str
    short_ce_symbol: str
    short_pe_symbol: str
    long_ce_symbol: str
    long_pe_symbol: str
    quantity: int
    entry_price: float
    max_profit: float
    max_loss: float
    entry_time: datetime
    status: PositionStatus
    net_delta: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    current_pnl: float = 0.0
    hedge_quantity: int = 0

class CalendarSpreadStrategy:
    """
    Main Calendar Spread Strategy Class
    
    Responsibilities:
    - Scan and select suitable underlyings
    - Execute calendar spread entries
    - Monitor and manage positions
    - Implement risk management
    - Handle exits based on various criteria
    """
    
    def __init__(self, config: CalendarConfig):
        self.config = config
        self.logger = logging.getLogger("CalendarSpreadStrategy")
        
        # Initialize components
        self.scanner = UnderlyingScanner(config)
        
        # Position tracking
        self.positions: Dict[str, CalendarPosition] = {}
        self.daily_pnl: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        
        # Risk management
        self.daily_hedge_count: Dict[str, int] = {}
        self.last_hedge_time: Dict[str, datetime] = {}
        
    def scan_and_select_underlying(self, symbol_list: List[str] = None) -> Optional[str]:
        """
        Scan underlyings and select the best candidate for calendar spread
        
        Returns:
            Selected symbol or None if no suitable candidate
        """
        try:
            candidates = self.scanner.get_top_candidates(symbol_list, top_n=3)
            
            if not candidates:
                self.logger.warning("No suitable candidates found")
                return None
            
            # Filter out symbols we already have positions in
            available_candidates = [c for c in candidates 
                                 if c['symbol'] not in self.positions]
            
            if not available_candidates:
                self.logger.info("All top candidates already have positions")
                return None
            
            selected = available_candidates[0]
            self.logger.info(f"Selected {selected['symbol']} with score {selected['total_score']:.3f}")
            
            return selected['symbol']
            
        except Exception as e:
            self.logger.error(f"Error in scan_and_select_underlying: {e}")
            return None
    
    def find_optimal_strikes_and_expiries(self, symbol: str) -> Optional[Dict]:
        """
        Find optimal strikes and expiries for calendar spread
        
        Returns:
            Dictionary with strike and expiry information or None
        """
        try:
            nse_sym = nse_symbol(symbol)
            current_quotes = get_quotes([nse_sym])
            
            if not current_quotes or 'd' not in current_quotes:
                return None
                
            current_price = float(current_quotes['d'][0]['v'].get('lp', 0))
            if current_price == 0:
                return None
            
            # Get option chain
            option_chain = get_option_chain(nse_sym, strikecount=5)
            if not option_chain or option_chain.get('s') != 'ok':
                return None
            
            chain_data = option_chain.get('data', {}).get('optionsChain', [])
            if not chain_data:
                return None
            
            # Find ATM strike
            strikes = []
            for opt in chain_data:
                strike = opt.get('strike_price')
                if strike and opt.get('option_type') in ['CE', 'PE']:
                    strikes.append(float(strike))
            
            strikes = list(set(strikes))
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))
            
            # Find suitable expiries
            expiries = self._find_suitable_expiries(chain_data, atm_strike)
            if not expiries:
                return None
            
            return {
                'strike': atm_strike,
                'current_price': current_price,
                'short_expiry': expiries['short'],
                'long_expiry': expiries['long'],
                'short_ce_symbol': self._build_option_symbol(symbol, expiries['short'], atm_strike, 'CE'),
                'short_pe_symbol': self._build_option_symbol(symbol, expiries['short'], atm_strike, 'PE'),
                'long_ce_symbol': self._build_option_symbol(symbol, expiries['long'], atm_strike, 'CE'),
                'long_pe_symbol': self._build_option_symbol(symbol, expiries['long'], atm_strike, 'PE')
            }
            
        except Exception as e:
            self.logger.error(f"Error finding optimal strikes and expiries for {symbol}: {e}")
            return None
    
    def _find_suitable_expiries(self, chain_data: List[Dict], strike: float) -> Optional[Dict]:
        """Find short and long expiries for calendar spread"""
        try:
            # Group options by expiry
            expiry_groups = {}
            for opt in chain_data:
                if (opt.get('strike_price') == strike and 
                    opt.get('option_type') in ['CE', 'PE']):
                    
                    symbol = opt.get('symbol', '')
                    # Extract expiry from symbol (simplified)
                    expiry = self._extract_expiry_from_symbol(symbol)
                    if expiry:
                        if expiry not in expiry_groups:
                            expiry_groups[expiry] = []
                        expiry_groups[expiry].append(opt)
            
            if len(expiry_groups) < 2:
                return None
            
            # Sort expiries by date
            sorted_expiries = sorted(expiry_groups.keys())
            
            # Find short expiry (closest to target days)
            target_short_days = self.config.SHORT_EXPIRY_DAYS
            short_expiry = min(sorted_expiries, 
                             key=lambda x: abs(self._days_to_expiry(x) - target_short_days))
            
            # Find long expiry (closest to target days)
            target_long_days = self.config.LONG_EXPIRY_DAYS
            long_expiry = min(sorted_expiries, 
                            key=lambda x: abs(self._days_to_expiry(x) - target_long_days))
            
            # Ensure long expiry is after short expiry
            if self._days_to_expiry(long_expiry) <= self._days_to_expiry(short_expiry):
                # Find next available expiry after short
                remaining_expiries = [e for e in sorted_expiries 
                                    if self._days_to_expiry(e) > self._days_to_expiry(short_expiry)]
                if remaining_expiries:
                    long_expiry = remaining_expiries[0]
                else:
                    return None
            
            return {
                'short': short_expiry,
                'long': long_expiry
            }
            
        except Exception as e:
            self.logger.error(f"Error finding suitable expiries: {e}")
            return None
    
    def _extract_expiry_from_symbol(self, symbol: str) -> Optional[str]:
        """Extract expiry from option symbol"""
        try:
            # Simplified extraction - would need more robust parsing
            # Format: NSE:SYMBOL{EXPIRY}{STRIKE}{CE/PE}
            import re
            match = re.search(r'(\d+[A-Z]+)', symbol)
            return match.group(1) if match else None
        except:
            return None
    
    def _days_to_expiry(self, expiry_str: str) -> int:
        """Calculate days to expiry (simplified)"""
        try:
            # This is a simplified calculation
            # In practice, you'd parse the expiry string properly
            # For now, return a placeholder
            return 30  # Placeholder
        except:
            return 30
    
    def _build_option_symbol(self, symbol: str, expiry: str, strike: float, right: str) -> str:
        """Build option symbol"""
        return build_option_symbol(symbol, expiry, int(strike), right)
    
    def calculate_position_size(self, symbol: str, option_data: Dict) -> int:
        """
        Calculate appropriate position size based on risk parameters
        
        Returns:
            Number of lots to trade
        """
        try:
            lot_size = self.config.LOT_SIZE_MAP.get(symbol, self.config.DEFAULT_LOT_SIZE)
            
            # Get current quotes for all legs
            symbols = [
                option_data['short_ce_symbol'],
                option_data['short_pe_symbol'],
                option_data['long_ce_symbol'],
                option_data['long_pe_symbol']
            ]
            
            quotes = get_quotes(symbols)
            if not quotes or 'd' not in quotes:
                return 0
            
            # Calculate net debit
            short_ce_price = float(quotes['d'][0]['v'].get('lp', 0))
            short_pe_price = float(quotes['d'][1]['v'].get('lp', 0))
            long_ce_price = float(quotes['d'][2]['v'].get('lp', 0))
            long_pe_price = float(quotes['d'][3]['v'].get('lp', 0))
            
            if any(price == 0 for price in [short_ce_price, short_pe_price, long_ce_price, long_pe_price]):
                return 0
            
            # Net debit = (Long CE + Long PE) - (Short CE + Short PE)
            net_debit = (long_ce_price + long_pe_price) - (short_ce_price + short_pe_price)
            
            if net_debit <= 0:
                self.logger.warning(f"Negative or zero net debit for {symbol}")
                return 0
            
            # Calculate position size based on risk
            max_risk_amount = self.config.MAX_POSITION_SIZE * 100000  # Assuming 1L portfolio
            max_lots = int(max_risk_amount / (net_debit * lot_size))
            
            # Apply additional constraints
            max_lots = min(max_lots, 5)  # Maximum 5 lots per position
            
            return max(1, max_lots) if max_lots > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    def execute_calendar_spread(self, symbol: str, option_data: Dict, quantity: int) -> bool:
        """
        Execute calendar spread entry
        
        Returns:
            True if successful, False otherwise
        """
        try:
            lot_size = self.config.LOT_SIZE_MAP.get(symbol, self.config.DEFAULT_LOT_SIZE)
            qty = quantity * lot_size
            
            # Prepare orders
            orders = [
                # Sell short-term options
                {
                    "symbol": option_data['short_ce_symbol'],
                    "qty": qty,
                    "type": 2,  # Market order
                    "side": -1,  # Sell
                    "productType": "INTRADAY",
                    "validity": "DAY"
                },
                {
                    "symbol": option_data['short_pe_symbol'],
                    "qty": qty,
                    "type": 2,
                    "side": -1,
                    "productType": "INTRADAY",
                    "validity": "DAY"
                },
                # Buy long-term options
                {
                    "symbol": option_data['long_ce_symbol'],
                    "qty": qty,
                    "type": 2,
                    "side": 1,  # Buy
                    "productType": "INTRADAY",
                    "validity": "DAY"
                },
                {
                    "symbol": option_data['long_pe_symbol'],
                    "qty": qty,
                    "type": 2,
                    "side": 1,
                    "productType": "INTRADAY",
                    "validity": "DAY"
                }
            ]
            
            # Execute orders
            response = place_order(orders)
            
            if response and response.get('s') == 'ok':
                # Create position record
                position = CalendarPosition(
                    symbol=symbol,
                    strike=option_data['strike'],
                    short_expiry=option_data['short_expiry'],
                    long_expiry=option_data['long_expiry'],
                    short_ce_symbol=option_data['short_ce_symbol'],
                    short_pe_symbol=option_data['short_pe_symbol'],
                    long_ce_symbol=option_data['long_ce_symbol'],
                    long_pe_symbol=option_data['long_pe_symbol'],
                    quantity=quantity,
                    entry_price=0.0,  # Will be updated
                    max_profit=0.0,   # Will be calculated
                    max_loss=0.0,     # Will be calculated
                    entry_time=datetime.now(),
                    status=PositionStatus.OPEN
                )
                
                self.positions[symbol] = position
                self.total_trades += 1
                
                self.logger.info(f"Successfully opened calendar spread for {symbol}: {response}")
                return True
            else:
                self.logger.error(f"Failed to execute calendar spread for {symbol}: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing calendar spread for {symbol}: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor all open positions and manage risk"""
        try:
            for symbol, position in self.positions.items():
                if position.status == PositionStatus.OPEN:
                    self._update_position_greeks(position)
                    self._check_exit_conditions(position)
                    self._manage_delta_risk(position)
                    
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")
    
    def _update_position_greeks(self, position: CalendarPosition):
        """Update position Greeks and P&L"""
        try:
            symbols = [
                position.short_ce_symbol,
                position.short_pe_symbol,
                position.long_ce_symbol,
                position.long_pe_symbol
            ]
            
            quotes = get_quotes(symbols)
            if not quotes or 'd' not in quotes:
                return
            
            # Calculate net Greeks
            net_delta = 0.0
            net_theta = 0.0
            net_vega = 0.0
            
            for i, symbol in enumerate(symbols):
                quote_data = quotes['d'][i]['v']
                delta = float(quote_data.get('delta', 0))
                theta = float(quote_data.get('theta', 0))
                vega = float(quote_data.get('vega', 0))
                
                # Short options have opposite signs
                if i < 2:  # Short options
                    net_delta -= delta
                    net_theta -= theta
                    net_vega -= vega
                else:  # Long options
                    net_delta += delta
                    net_theta += theta
                    net_vega += vega
            
            # Update position
            position.net_delta = net_delta * position.quantity
            position.net_theta = net_theta * position.quantity
            position.net_vega = net_vega * position.quantity
            
        except Exception as e:
            self.logger.error(f"Error updating position Greeks: {e}")
    
    def _check_exit_conditions(self, position: CalendarPosition):
        """Check if position should be closed"""
        try:
            # Time-based exit
            days_held = (datetime.now() - position.entry_time).days
            if days_held >= self.config.MAX_HOLDING_DAYS:
                self.logger.info(f"Closing {position.symbol} - max holding period reached")
                self._close_position(position, "max_holding_period")
                return
            
            # Days to short expiry
            days_to_short_expiry = self._days_to_expiry(position.short_expiry)
            if days_to_short_expiry <= self.config.CLOSE_BEFORE_EXPIRY:
                self.logger.info(f"Closing {position.symbol} - approaching short expiry")
                self._close_position(position, "approaching_expiry")
                return
            
            # Profit target
            if position.current_pnl >= position.max_profit * self.config.PROFIT_TARGET:
                self.logger.info(f"Closing {position.symbol} - profit target reached")
                self._close_position(position, "profit_target")
                return
            
            # Stop loss
            if position.current_pnl <= -position.max_loss * self.config.MAX_LOSS_MULTIPLIER:
                self.logger.info(f"Closing {position.symbol} - stop loss triggered")
                self._close_position(position, "stop_loss")
                return
                
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
    
    def _manage_delta_risk(self, position: CalendarPosition):
        """Manage delta risk through hedging"""
        try:
            # Check if hedging is needed
            if abs(position.net_delta) > self.config.DELTA_HEDGE_THRESHOLD * position.quantity:
                
                # Check hedge frequency limits
                last_hedge = self.last_hedge_time.get(position.symbol)
                if last_hedge and (datetime.now() - last_hedge).seconds < self.config.HEDGE_INTERVAL_SECONDS:
                    return
                
                # Execute hedge
                self._execute_delta_hedge(position)
                
        except Exception as e:
            self.logger.error(f"Error managing delta risk: {e}")
    
    def _execute_delta_hedge(self, position: CalendarPosition):
        """Execute delta hedge using futures"""
        try:
            if not self.config.USE_FUTURES_HEDGE:
                return
            
            # Build future symbol
            future_symbol = f"NSE:{position.symbol}{position.long_expiry}FUT"
            
            # Calculate hedge quantity
            lot_size = self.config.LOT_SIZE_MAP.get(position.symbol, self.config.DEFAULT_LOT_SIZE)
            hedge_lots = int(abs(position.net_delta) * self.config.HEDGE_RATIO / lot_size)
            
            if hedge_lots == 0:
                return
            
            # Determine hedge direction
            side = 1 if position.net_delta < 0 else -1
            
            # Place hedge order
            hedge_order = [{
                "symbol": future_symbol,
                "qty": hedge_lots * lot_size,
                "type": 2,
                "side": side,
                "productType": "INTRADAY",
                "validity": "DAY"
            }]
            
            response = place_order(hedge_order)
            
            if response and response.get('s') == 'ok':
                position.hedge_quantity += hedge_lots
                self.last_hedge_time[position.symbol] = datetime.now()
                self.daily_hedge_count[position.symbol] = self.daily_hedge_count.get(position.symbol, 0) + 1
                
                self.logger.info(f"Executed delta hedge for {position.symbol}: {hedge_lots} lots")
            else:
                self.logger.error(f"Failed to execute delta hedge for {position.symbol}: {response}")
                
        except Exception as e:
            self.logger.error(f"Error executing delta hedge: {e}")
    
    def _close_position(self, position: CalendarPosition, reason: str):
        """Close calendar spread position"""
        try:
            position.status = PositionStatus.CLOSING
            
            lot_size = self.config.LOT_SIZE_MAP.get(position.symbol, self.config.DEFAULT_LOT_SIZE)
            qty = position.quantity * lot_size
            
            # Prepare close orders (opposite of entry)
            orders = [
                # Buy back short-term options
                {
                    "symbol": position.short_ce_symbol,
                    "qty": qty,
                    "type": 2,
                    "side": 1,  # Buy
                    "productType": "INTRADAY",
                    "validity": "DAY"
                },
                {
                    "symbol": position.short_pe_symbol,
                    "qty": qty,
                    "type": 2,
                    "side": 1,
                    "productType": "INTRADAY",
                    "validity": "DAY"
                },
                # Sell long-term options
                {
                    "symbol": position.long_ce_symbol,
                    "qty": qty,
                    "type": 2,
                    "side": -1,  # Sell
                    "productType": "INTRADAY",
                    "validity": "DAY"
                },
                {
                    "symbol": position.long_pe_symbol,
                    "qty": qty,
                    "type": 2,
                    "side": -1,
                    "productType": "INTRADAY",
                    "validity": "DAY"
                }
            ]
            
            response = place_order(orders)
            
            if response and response.get('s') == 'ok':
                position.status = PositionStatus.CLOSED
                if position.current_pnl > 0:
                    self.winning_trades += 1
                
                self.logger.info(f"Successfully closed {position.symbol} position. Reason: {reason}")
            else:
                position.status = PositionStatus.ERROR
                self.logger.error(f"Failed to close {position.symbol} position: {response}")
                
        except Exception as e:
            self.logger.error(f"Error closing position {position.symbol}: {e}")
            position.status = PositionStatus.ERROR
    
    def run_strategy(self, max_positions: int = 3):
        """
        Main strategy execution loop
        
        Args:
            max_positions: Maximum number of concurrent positions
        """
        try:
            self.logger.info("Starting Calendar Spread Strategy")
            
            while len(self.positions) < max_positions:
                # Scan for new opportunities
                selected_symbol = self.scan_and_select_underlying()
                if not selected_symbol:
                    self.logger.info("No suitable opportunities found")
                    break
                
                # Find optimal strikes and expiries
                option_data = self.find_optimal_strikes_and_expiries(selected_symbol)
                if not option_data:
                    self.logger.warning(f"No suitable options found for {selected_symbol}")
                    continue
                
                # Calculate position size
                quantity = self.calculate_position_size(selected_symbol, option_data)
                if quantity == 0:
                    self.logger.warning(f"Position size is 0 for {selected_symbol}")
                    continue
                
                # Execute calendar spread
                success = self.execute_calendar_spread(selected_symbol, option_data, quantity)
                if not success:
                    self.logger.error(f"Failed to execute calendar spread for {selected_symbol}")
                    continue
                
                self.logger.info(f"Successfully opened position in {selected_symbol}")
                
                # Wait before opening next position
                time.sleep(5)
            
            # Monitor existing positions
            while any(pos.status == PositionStatus.OPEN for pos in self.positions.values()):
                self.monitor_positions()
                time.sleep(self.config.HEDGE_INTERVAL_SECONDS)
            
            self.logger.info("Strategy execution completed")
            
        except Exception as e:
            self.logger.error(f"Error in strategy execution: {e}")
    
    def get_strategy_stats(self) -> Dict:
        """Get strategy performance statistics"""
        try:
            total_positions = len(self.positions)
            open_positions = len([p for p in self.positions.values() if p.status == PositionStatus.OPEN])
            closed_positions = len([p for p in self.positions.values() if p.status == PositionStatus.CLOSED])
            
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            return {
                'total_positions': total_positions,
                'open_positions': open_positions,
                'closed_positions': closed_positions,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'daily_pnl': self.daily_pnl,
                'positions': {symbol: {
                    'status': pos.status.value,
                    'pnl': pos.current_pnl,
                    'net_delta': pos.net_delta,
                    'net_theta': pos.net_theta,
                    'net_vega': pos.net_vega
                } for symbol, pos in self.positions.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting strategy stats: {e}")
            return {}
