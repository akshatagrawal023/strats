"""
Data Processor for Calendar Spread Strategy
Processes real-time market data and generates trading signals
"""
import os
import sys
import logging
import threading
import time
from datetime import datetime, timedelta
from queue import Queue, Empty
from typing import Dict, List, Optional, Any
from collections import deque
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_utils import get_quotes, get_option_chain
from .trading_state import TradingState
from .calendar_spread_strategy import CalendarPosition, PositionStatus

class DataProcessor(threading.Thread):
    """
    Processes real-time market data and generates trading signals for calendar spreads.
    Follows the same pattern as the gamma scalping data processor.
    """
    
    def __init__(self, price_queue: Queue, signal_queue: Queue, config, trading_state: TradingState):
        super().__init__(name="CalendarDataProcessor")
        self.price_queue = price_queue
        self.signal_queue = signal_queue
        self.config = config
        self.trading_state = trading_state
        self.logger = logging.getLogger("CalendarDataProcessor")
        
        # Threading control
        self._running_lock = threading.Lock()
        self._running = False
        
        # Data processing
        self.price_data: Dict[str, deque] = {}
        self.greeks_data: Dict[str, Dict] = {}
        self.volatility_data: Dict[str, deque] = {}
        
        # Signal generation
        self.signal_thresholds = {
            'delta_threshold': 0.10,
            'theta_threshold': 0.5,
            'vega_threshold': 0.1,
            'iv_threshold': 0.02
        }
        
        # Performance tracking
        self.processed_messages = 0
        self.last_processing_time = datetime.now()
        
        # Initialize data structures
        self.init_data_structures()
    
    @property
    def running(self):
        with self._running_lock:
            return self._running
    
    @running.setter
    def running(self, value):
        with self._running_lock:
            self._running = value
    
    def init_data_structures(self):
        """Initialize data structures for all tracked symbols"""
        try:
            # Get all symbols from open positions
            positions = self.trading_state.get_all_positions()
            
            for symbol, position in positions.items():
                # Initialize price data (keep last 1000 ticks)
                self.price_data[symbol] = deque(maxlen=1000)
                
                # Initialize Greeks data
                self.greeks_data[symbol] = {
                    'delta': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'gamma': 0.0,
                    'iv': 0.0,
                    'last_update': datetime.now()
                }
                
                # Initialize volatility data (keep last 100 observations)
                self.volatility_data[symbol] = deque(maxlen=100)
            
            self.logger.info(f"Initialized data structures for {len(positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Error initializing data structures: {e}")
    
    def start(self):
        """Start the data processor"""
        self.logger.info("Starting Calendar Data Processor...")
        self.running = True
        super().start()
    
    def run(self):
        """Main processing loop"""
        try:
            while self.running:
                try:
                    # Process price updates
                    self.process_price_updates()
                    
                    # Update Greeks for all positions
                    self.update_greeks_for_positions()
                    
                    # Generate trading signals
                    self.generate_trading_signals()
                    
                    # Update performance metrics
                    self.update_performance_metrics()
                    
                    # Sleep briefly to prevent excessive CPU usage
                    time.sleep(0.1)
                    
                except Empty:
                    # No data to process, continue
                    continue
                except Exception as e:
                    self.logger.error(f"Error in data processing loop: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in data processor: {e}")
        finally:
            self.logger.info("Calendar Data Processor stopped")
    
    def process_price_updates(self):
        """Process price updates from the queue"""
        try:
            processed_count = 0
            max_batch_size = 100  # Process up to 100 messages at once
            
            while processed_count < max_batch_size:
                try:
                    # Get price update with timeout
                    price_update = self.price_queue.get(timeout=0.1)
                    
                    # Process the update
                    self.process_single_price_update(price_update)
                    
                    # Mark as processed
                    self.price_queue.task_done()
                    processed_count += 1
                    self.processed_messages += 1
                    
                except Empty:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing price update: {e}")
                    continue
            
            if processed_count > 0:
                self.last_processing_time = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Error processing price updates: {e}")
    
    def process_single_price_update(self, price_update: Dict[str, Any]):
        """Process a single price update"""
        try:
            symbol = price_update.get('symbol')
            if not symbol:
                return
            
            # Extract price data
            price_data = {
                'symbol': symbol,
                'bid': float(price_update.get('bid', 0)),
                'ask': float(price_update.get('ask', 0)),
                'ltp': float(price_update.get('ltp', 0)),
                'timestamp': price_update.get('timestamp', datetime.now())
            }
            
            # Store price data
            if symbol not in self.price_data:
                self.price_data[symbol] = deque(maxlen=1000)
            
            self.price_data[symbol].append(price_data)
            
            # Calculate volatility if we have enough data
            if len(self.price_data[symbol]) > 20:
                self.calculate_realized_volatility(symbol)
            
        except Exception as e:
            self.logger.error(f"Error processing single price update: {e}")
    
    def calculate_realized_volatility(self, symbol: str):
        """Calculate realized volatility from price data"""
        try:
            if symbol not in self.price_data or len(self.price_data[symbol]) < 20:
                return
            
            # Get recent prices
            prices = [data['ltp'] for data in list(self.price_data[symbol])[-20:]]
            
            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)
            
            if len(returns) > 1:
                # Calculate realized volatility (annualized)
                realized_vol = np.std(returns) * np.sqrt(252 * 24 * 60)  # Assuming minute data
                
                # Store volatility data
                if symbol not in self.volatility_data:
                    self.volatility_data[symbol] = deque(maxlen=100)
                
                self.volatility_data[symbol].append({
                    'volatility': realized_vol,
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            self.logger.error(f"Error calculating realized volatility for {symbol}: {e}")
    
    def update_greeks_for_positions(self):
        """Update Greeks for all open positions"""
        try:
            positions = self.trading_state.get_all_positions()
            
            for symbol, position in positions.items():
                if position.status == PositionStatus.OPEN:
                    self.update_position_greeks(position)
                    
        except Exception as e:
            self.logger.error(f"Error updating Greeks for positions: {e}")
    
    def update_position_greeks(self, position: CalendarPosition):
        """Update Greeks for a specific position"""
        try:
            # Get quotes for all option legs
            symbols = [
                position.short_ce_symbol,
                position.short_pe_symbol,
                position.long_ce_symbol,
                position.long_pe_symbol
            ]
            
            quotes = get_quotes(symbols)
            if not quotes or 'd' not in quotes or len(quotes['d']) < 4:
                return
            
            # Calculate net Greeks
            net_delta = 0.0
            net_theta = 0.0
            net_vega = 0.0
            net_gamma = 0.0
            avg_iv = 0.0
            
            for i, symbol in enumerate(symbols):
                quote_data = quotes['d'][i]['v']
                
                delta = float(quote_data.get('delta', 0))
                theta = float(quote_data.get('theta', 0))
                vega = float(quote_data.get('vega', 0))
                gamma = float(quote_data.get('gamma', 0))
                iv = float(quote_data.get('iv', 0))
                
                # Short options have opposite signs
                if i < 2:  # Short options
                    net_delta -= delta
                    net_theta -= theta
                    net_vega -= vega
                    net_gamma -= gamma
                else:  # Long options
                    net_delta += delta
                    net_theta += theta
                    net_vega += vega
                    net_gamma += gamma
                
                avg_iv += iv
            
            avg_iv /= 4  # Average IV across all legs
            
            # Update position Greeks
            position.net_delta = net_delta * position.quantity
            position.net_theta = net_theta * position.quantity
            position.net_vega = net_vega * position.quantity
            
            # Update Greeks data
            self.greeks_data[position.symbol] = {
                'delta': net_delta,
                'theta': net_theta,
                'vega': net_vega,
                'gamma': net_gamma,
                'iv': avg_iv,
                'last_update': datetime.now()
            }
            
            # Update position in trading state
            self.trading_state.update_position(position)
            
        except Exception as e:
            self.logger.error(f"Error updating Greeks for position {position.symbol}: {e}")
    
    def generate_trading_signals(self):
        """Generate trading signals based on current market conditions"""
        try:
            positions = self.trading_state.get_all_positions()
            
            for symbol, position in positions.items():
                if position.status != PositionStatus.OPEN:
                    continue
                
                # Check for hedging signals
                self.check_hedging_signals(position)
                
                # Check for exit signals
                self.check_exit_signals(position)
                
                # Check for adjustment signals
                self.check_adjustment_signals(position)
                
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
    
    def check_hedging_signals(self, position: CalendarPosition):
        """Check if hedging is needed"""
        try:
            # Check delta threshold
            if abs(position.net_delta) > self.signal_thresholds['delta_threshold'] * position.quantity:
                signal = {
                    'type': 'HEDGE_DELTA',
                    'symbol': position.symbol,
                    'position': position,
                    'reason': f'Delta exposure: {position.net_delta:.3f}',
                    'timestamp': datetime.now()
                }
                
                self.signal_queue.put(signal)
                self.logger.info(f"Hedging signal generated for {position.symbol}: {signal['reason']}")
            
        except Exception as e:
            self.logger.error(f"Error checking hedging signals for {position.symbol}: {e}")
    
    def check_exit_signals(self, position: CalendarPosition):
        """Check for exit signals"""
        try:
            # Time-based exit
            days_held = (datetime.now() - position.entry_time).days
            if days_held >= self.config.MAX_HOLDING_DAYS:
                signal = {
                    'type': 'EXIT_POSITION',
                    'symbol': position.symbol,
                    'position': position,
                    'reason': 'max_holding_period',
                    'timestamp': datetime.now()
                }
                self.signal_queue.put(signal)
                return
            
            # Profit target
            if position.current_pnl >= position.max_profit * self.config.PROFIT_TARGET:
                signal = {
                    'type': 'EXIT_POSITION',
                    'symbol': position.symbol,
                    'position': position,
                    'reason': 'profit_target',
                    'timestamp': datetime.now()
                }
                self.signal_queue.put(signal)
                return
            
            # Stop loss
            if position.current_pnl <= -position.max_loss * self.config.MAX_LOSS_MULTIPLIER:
                signal = {
                    'type': 'EXIT_POSITION',
                    'symbol': position.symbol,
                    'position': position,
                    'reason': 'stop_loss',
                    'timestamp': datetime.now()
                }
                self.signal_queue.put(signal)
                return
            
            # Volatility-based exit
            if position.symbol in self.volatility_data and len(self.volatility_data[position.symbol]) > 0:
                recent_vol = self.volatility_data[position.symbol][-1]['volatility']
                if recent_vol < 0.1:  # Very low volatility
                    signal = {
                        'type': 'EXIT_POSITION',
                        'symbol': position.symbol,
                        'position': position,
                        'reason': 'low_volatility',
                        'timestamp': datetime.now()
                    }
                    self.signal_queue.put(signal)
                    return
                    
        except Exception as e:
            self.logger.error(f"Error checking exit signals for {position.symbol}: {e}")
    
    def check_adjustment_signals(self, position: CalendarPosition):
        """Check for position adjustment signals"""
        try:
            # Check if position needs adjustment based on Greeks
            if position.symbol in self.greeks_data:
                greeks = self.greeks_data[position.symbol]
                
                # High gamma exposure
                if abs(greeks['gamma']) > 0.5:
                    signal = {
                        'type': 'ADJUST_POSITION',
                        'symbol': position.symbol,
                        'position': position,
                        'reason': f'High gamma exposure: {greeks["gamma"]:.3f}',
                        'timestamp': datetime.now()
                    }
                    self.signal_queue.put(signal)
                
                # High vega exposure
                if abs(greeks['vega']) > self.signal_thresholds['vega_threshold']:
                    signal = {
                        'type': 'ADJUST_POSITION',
                        'symbol': position.symbol,
                        'position': position,
                        'reason': f'High vega exposure: {greeks["vega"]:.3f}',
                        'timestamp': datetime.now()
                    }
                    self.signal_queue.put(signal)
                    
        except Exception as e:
            self.logger.error(f"Error checking adjustment signals for {position.symbol}: {e}")
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate portfolio metrics
            metrics = self.trading_state.calculate_portfolio_metrics()
            
            # Update daily P&L
            self.trading_state.daily_pnl = metrics.get('total_pnl', 0.0)
            
            # Save risk metrics for each position
            for symbol, position in self.trading_state.get_all_positions().items():
                delta_exposure = abs(position.net_delta) / (position.quantity * 100)  # Normalize
                portfolio_exposure = abs(position.current_pnl) / 100000  # Assuming 1L portfolio
                
                self.trading_state.save_risk_metrics(symbol, delta_exposure, portfolio_exposure)
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def get_market_data_summary(self) -> Dict[str, Any]:
        """Get summary of current market data"""
        try:
            summary = {
                'processed_messages': self.processed_messages,
                'last_processing_time': self.last_processing_time,
                'active_symbols': len(self.price_data),
                'positions_tracked': len(self.trading_state.get_all_positions()),
                'signal_queue_size': self.signal_queue.qsize(),
                'price_queue_size': self.price_queue.qsize()
            }
            
            # Add Greeks summary
            greeks_summary = {}
            for symbol, greeks in self.greeks_data.items():
                greeks_summary[symbol] = {
                    'delta': greeks['delta'],
                    'theta': greeks['theta'],
                    'vega': greeks['vega'],
                    'iv': greeks['iv'],
                    'last_update': greeks['last_update'].isoformat()
                }
            
            summary['greeks_data'] = greeks_summary
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting market data summary: {e}")
            return {}
    
    def stop(self):
        """Stop the data processor"""
        self.logger.info("Stopping Calendar Data Processor...")
        self.running = False
        time.sleep(1)  # Give time for current processing to complete
