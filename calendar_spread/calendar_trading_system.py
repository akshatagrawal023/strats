"""
Calendar Trading System - Main Orchestrator
Multi-threaded trading system following the sophisticated architecture from GitHub repository
"""
import os
import sys
import logging
import threading
import time
import signal
from datetime import datetime, timedelta
from queue import Queue, Empty
from typing import Dict, List, Optional, Any
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_utils import get_quotes, place_order, get_positions, get_orderbook
from utils.websocket import WebSocketClient
from .config import CalendarConfig
from .trading_state import TradingState
from .data_processor import DataProcessor
from .trade_manager import TradeManager
from .position_monitor import PositionMonitor
from .underlying_scanner import UnderlyingScanner
from .calendar_spread_strategy import CalendarPosition, PositionStatus

class CalendarTradingSystem:
    """
    Main orchestrator for the Calendar Spread Trading System.
    Follows the sophisticated multi-threaded architecture from the GitHub repository.
    """
    
    def __init__(self, config: CalendarConfig = None):
        self.config = config or CalendarConfig()
        self.logger = logging.getLogger("CalendarTradingSystem")
        
        # System state
        self.running = True
        self.start_time = datetime.now()
        
        # Initialize queues for inter-component communication
        self.price_queue = Queue(maxsize=1000)
        self.signal_queue = Queue(maxsize=50)
        self.order_queue = Queue(maxsize=20)
        
        # Initialize trading state
        self.trading_state = TradingState(self.config)
        
        # Initialize components
        self.data_manager = None
        self.data_processor = None
        self.trade_manager = None
        self.position_monitor = None
        self.underlying_scanner = UnderlyingScanner(self.config)
        
        # System monitoring
        self.system_health = {
            'data_manager': False,
            'data_processor': False,
            'trade_manager': False,
            'position_monitor': False,
            'last_heartbeat': datetime.now()
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'start_time': self.start_time
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        # Initialize system
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing Calendar Trading System...")
            
            # Initialize trading state database
            self.trading_state.ensure_open_positions_table()
            self.trading_state.ensure_closed_positions_table()
            self.trading_state.ensure_performance_table()
            self.trading_state.ensure_risk_metrics_table()
            
            # Rehydrate positions from database
            self.trading_state.rehydrate_from_db()
            open_positions = self.trading_state.get_all_positions()
            self.logger.info(f"Rehydrated {len(open_positions)} positions from database")
            
            # Initialize data manager
            self.data_manager = CalendarDataManager(
                self.price_queue, 
                self.config, 
                self.trading_state
            )
            
            # Initialize data processor
            self.data_processor = DataProcessor(
                self.price_queue,
                self.signal_queue,
                self.config,
                self.trading_state
            )
            
            # Initialize trade manager
            self.trade_manager = TradeManager(
                self.signal_queue,
                self.config,
                self.logger,
                self.trading_state
            )
            
            # Initialize position monitor
            self.position_monitor = PositionMonitor(
                self.trading_state,
                self.config.HEDGE_INTERVAL_SECONDS
            )
            
            self.logger.info("Calendar Trading System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing system: {e}")
            raise
    
    def start(self):
        """Start all system components"""
        try:
            self.logger.info("Starting Calendar Trading System...")
            
            # Start components in order
            self.data_manager.start()
            self.data_processor.start()
            self.trade_manager.start()
            self.position_monitor.start()
            
            # Wait for components to initialize
            time.sleep(2)
            
            # Verify all components are running
            self.verify_system_health()
            
            self.logger.info("All components started successfully")
            
            # Start main system loop
            self.run_main_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            self.shutdown()
    
    def run_main_loop(self):
        """Main system monitoring and control loop"""
        try:
            self.logger.info("Starting main system loop...")
            
            while self.running:
                try:
                    # Monitor system health
                    self.monitor_system_health()
                    
                    # Check for new trading opportunities
                    self.check_trading_opportunities()
                    
                    # Update performance metrics
                    self.update_performance_metrics()
                    
                    # Log system status periodically
                    self.log_system_status()
                    
                    # Sleep for main loop interval
                    time.sleep(30)  # Check every 30 seconds
                    
                except KeyboardInterrupt:
                    self.logger.info("Received keyboard interrupt")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    time.sleep(5)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}")
        finally:
            self.shutdown()
    
    def monitor_system_health(self):
        """Monitor health of all system components"""
        try:
            components = {
                'data_manager': self.data_manager,
                'data_processor': self.data_processor,
                'trade_manager': self.trade_manager,
                'position_monitor': self.position_monitor
            }
            
            for name, component in components.items():
                if component and hasattr(component, 'is_alive'):
                    is_alive = component.is_alive()
                    self.system_health[name] = is_alive
                    
                    if not is_alive:
                        self.logger.error(f"{name} is not running!")
                    else:
                        self.logger.debug(f"{name} is healthy")
            
            # Check queue health
            price_queue_size = self.price_queue.qsize()
            signal_queue_size = self.signal_queue.qsize()
            order_queue_size = self.order_queue.qsize()
            
            if price_queue_size > 800:  # 80% of maxsize
                self.logger.warning(f"Price queue near capacity: {price_queue_size}/1000")
            
            if signal_queue_size > 40:  # 80% of maxsize
                self.logger.warning(f"Signal queue near capacity: {signal_queue_size}/50")
            
            if order_queue_size > 16:  # 80% of maxsize
                self.logger.warning(f"Order queue near capacity: {order_queue_size}/20")
            
            # Update last heartbeat
            self.system_health['last_heartbeat'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error monitoring system health: {e}")
    
    def verify_system_health(self):
        """Verify all components are running properly"""
        try:
            components = {
                'DataManager': self.data_manager,
                'DataProcessor': self.data_processor,
                'TradeManager': self.trade_manager,
                'PositionMonitor': self.position_monitor
            }
            
            for name, component in components.items():
                if not component or not component.is_alive():
                    raise RuntimeError(f"{name} failed to start properly")
                else:
                    self.logger.info(f"{name} is running")
            
            self.logger.info("All components verified and running")
            
        except Exception as e:
            self.logger.error(f"System health verification failed: {e}")
            raise
    
    def check_trading_opportunities(self):
        """Check for new trading opportunities"""
        try:
            # Only check if we have capacity for new positions
            current_positions = len(self.trading_state.get_all_positions())
            if current_positions >= self.config.MAX_POSITIONS_PER_DAY:
                return
            
            # Scan for new opportunities
            selected_symbol = self.underlying_scanner.scan_and_select_underlying()
            if not selected_symbol:
                return
            
            # Check if we already have a position in this symbol
            if selected_symbol in self.trading_state.get_all_positions():
                return
            
            # Find optimal strikes and expiries
            option_data = self.find_optimal_calendar_spread(selected_symbol)
            if not option_data:
                return
            
            # Calculate position size
            quantity = self.calculate_position_size(selected_symbol, option_data)
            if quantity == 0:
                return
            
            # Create entry signal
            entry_signal = {
                'type': 'OPEN_POSITION',
                'symbol': selected_symbol,
                'option_data': option_data,
                'quantity': quantity,
                'timestamp': datetime.now()
            }
            
            # Send signal to trade manager
            self.signal_queue.put(entry_signal)
            
            self.logger.info(f"New trading opportunity identified: {selected_symbol}")
            
        except Exception as e:
            self.logger.error(f"Error checking trading opportunities: {e}")
    
    def find_optimal_calendar_spread(self, symbol: str) -> Optional[Dict]:
        """Find optimal calendar spread for a symbol"""
        try:
            # Get current price
            nse_sym = f"NSE:{symbol}-EQ" if symbol not in ['NIFTY', 'BANKNIFTY', 'FINNIFTY'] else f"NSE:{symbol}-INDEX"
            quotes = get_quotes([nse_sym])
            
            if not quotes or 'd' not in quotes:
                return None
            
            current_price = float(quotes['d'][0]['v'].get('lp', 0))
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
            
            # Find suitable expiries (simplified)
            short_expiry = "25JAN"  # Would need proper expiry calculation
            long_expiry = "25FEB"   # Would need proper expiry calculation
            
            return {
                'strike': atm_strike,
                'current_price': current_price,
                'short_expiry': short_expiry,
                'long_expiry': long_expiry,
                'short_ce_symbol': f'NSE:{symbol}{short_expiry}{int(atm_strike)}CE',
                'short_pe_symbol': f'NSE:{symbol}{short_expiry}{int(atm_strike)}PE',
                'long_ce_symbol': f'NSE:{symbol}{long_expiry}{int(atm_strike)}CE',
                'long_pe_symbol': f'NSE:{symbol}{long_expiry}{int(atm_strike)}PE',
                'max_profit': current_price * 0.1,  # Simplified
                'max_loss': current_price * 0.05    # Simplified
            }
            
        except Exception as e:
            self.logger.error(f"Error finding optimal calendar spread for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, option_data: Dict) -> int:
        """Calculate appropriate position size"""
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
                return 0
            
            # Calculate position size based on risk
            max_risk_amount = self.config.MAX_POSITION_SIZE * 100000  # Assuming 1L portfolio
            max_lots = int(max_risk_amount / (net_debit * lot_size))
            
            # Apply additional constraints
            max_lots = min(max_lots, 3)  # Maximum 3 lots per position
            
            return max(1, max_lots) if max_lots > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    def update_performance_metrics(self):
        """Update system performance metrics"""
        try:
            # Get current portfolio metrics
            portfolio_metrics = self.trading_state.calculate_portfolio_metrics()
            
            # Update performance metrics
            self.performance_metrics.update({
                'total_trades': portfolio_metrics.get('total_trades', 0),
                'total_pnl': portfolio_metrics.get('total_pnl', 0.0),
                'max_drawdown': portfolio_metrics.get('max_drawdown', 0.0)
            })
            
            # Calculate success rate
            total_trades = self.performance_metrics['total_trades']
            if total_trades > 0:
                success_rate = (portfolio_metrics.get('winning_trades', 0) / total_trades) * 100
                self.performance_metrics['success_rate'] = success_rate
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def log_system_status(self):
        """Log system status periodically"""
        try:
            # Log every 5 minutes
            if hasattr(self, '_last_status_log'):
                if datetime.now() - self._last_status_log < timedelta(minutes=5):
                    return
            
            self._last_status_log = datetime.now()
            
            # Get system summary
            portfolio_metrics = self.trading_state.calculate_portfolio_metrics()
            system_health = self.system_health.copy()
            
            self.logger.info("=" * 60)
            self.logger.info("CALENDAR TRADING SYSTEM STATUS")
            self.logger.info("=" * 60)
            self.logger.info(f"Uptime: {datetime.now() - self.start_time}")
            self.logger.info(f"System Health: {system_health}")
            self.logger.info(f"Queue Sizes: Price={self.price_queue.qsize()}, Signal={self.signal_queue.qsize()}")
            self.logger.info(f"Portfolio: {portfolio_metrics}")
            self.logger.info(f"Performance: {self.performance_metrics}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Error logging system status: {e}")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        try:
            portfolio_metrics = self.trading_state.calculate_portfolio_metrics()
            
            summary = {
                'system_info': {
                    'name': 'Calendar Trading System',
                    'version': '1.0.0',
                    'start_time': self.start_time.isoformat(),
                    'uptime': str(datetime.now() - self.start_time),
                    'status': 'RUNNING' if self.running else 'STOPPED'
                },
                'system_health': self.system_health.copy(),
                'queue_status': {
                    'price_queue_size': self.price_queue.qsize(),
                    'signal_queue_size': self.signal_queue.qsize(),
                    'order_queue_size': self.order_queue.qsize()
                },
                'portfolio_metrics': portfolio_metrics,
                'performance_metrics': self.performance_metrics.copy(),
                'positions': {
                    symbol: {
                        'status': pos.status.value,
                        'pnl': pos.current_pnl,
                        'net_delta': pos.net_delta,
                        'net_theta': pos.net_theta,
                        'net_vega': pos.net_vega,
                        'days_held': (datetime.now() - pos.entry_time).days
                    }
                    for symbol, pos in self.trading_state.get_all_positions().items()
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting system summary: {e}")
            return {}
    
    def shutdown(self, signum=None, frame=None):
        """Graceful system shutdown"""
        try:
            if not self.running:
                return
            
            self.logger.info("Shutting down Calendar Trading System...")
            self.running = False
            
            # Stop components in reverse order
            components = [
                self.position_monitor,
                self.trade_manager,
                self.data_processor,
                self.data_manager
            ]
            
            for component in components:
                if component:
                    try:
                        component.stop()
                        component.join(timeout=5)
                        if component.is_alive():
                            self.logger.warning(f"{component.__class__.__name__} did not stop gracefully")
                    except Exception as e:
                        self.logger.error(f"Error stopping {component.__class__.__name__}: {e}")
            
            # Close database connection
            self.trading_state.close()
            
            # Log final summary
            final_summary = self.get_system_summary()
            self.logger.info("Final System Summary:")
            self.logger.info(f"Total Trades: {final_summary.get('performance_metrics', {}).get('total_trades', 0)}")
            self.logger.info(f"Total P&L: {final_summary.get('performance_metrics', {}).get('total_pnl', 0.0)}")
            self.logger.info(f"Success Rate: {final_summary.get('performance_metrics', {}).get('success_rate', 0.0):.2f}%")
            
            self.logger.info("Calendar Trading System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)

class CalendarDataManager(threading.Thread):
    """
    Data Manager for Calendar Spread System
    Handles WebSocket data streaming and option chain fetching
    """
    
    def __init__(self, price_queue: Queue, config: CalendarConfig, trading_state: TradingState):
        super().__init__(name="CalendarDataManager")
        self.price_queue = price_queue
        self.config = config
        self.trading_state = trading_state
        self.logger = logging.getLogger("CalendarDataManager")
        
        # Threading control
        self._running_lock = threading.Lock()
        self._running = False
        
        # WebSocket management
        self.ws = None
        self.ws_thread = None
        self.last_heartbeat = datetime.now()
        
        # Symbol management
        self.subscribed_symbols = set()
        self.all_symbols = self.get_all_symbols()
        
    @property
    def running(self):
        with self._running_lock:
            return self._running
    
    @running.setter
    def running(self, value):
        with self._running_lock:
            self._running = value
    
    def start(self):
        """Start the data manager"""
        self.logger.info("Starting Calendar Data Manager...")
        self.running = True
        super().start()
    
    def run(self):
        """Main data manager loop"""
        try:
            self.connect_websocket()
            
            while self.running:
                self.check_connection_health()
                time.sleep(60)  # Check every minute
                
        except Exception as e:
            self.logger.error(f"Error in data manager: {e}")
        finally:
            self.cleanup()
    
    def get_all_symbols(self):
        """Get all symbols to subscribe to"""
        try:
            symbols = set()
            positions = self.trading_state.get_all_positions()
            
            for position in positions.values():
                symbols.add(position.short_ce_symbol)
                symbols.add(position.short_pe_symbol)
                symbols.add(position.long_ce_symbol)
                symbols.add(position.long_pe_symbol)
            
            self.logger.info(f"Collected {len(symbols)} symbols for subscription")
            return list(symbols)
            
        except Exception as e:
            self.logger.error(f"Error getting symbols: {e}")
            return []
    
    def connect_websocket(self):
        """Connect to WebSocket for real-time data"""
        try:
            if self.ws and hasattr(self.ws, 'data_socket') and self.ws.data_socket.is_connected():
                return
            
            self.ws = WebSocketClient(
                on_message=self.on_message,
                on_connect=self.on_connect,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            data_socket = self.ws.fyers_data_socket()
            self.ws_thread = threading.Thread(
                target=data_socket.connect,
                name="CalendarWebSocket"
            )
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            self.logger.info("WebSocket connection initiated")
            
        except Exception as e:
            self.logger.error(f"Error connecting WebSocket: {e}")
    
    def on_connect(self, client):
        """Handle WebSocket connection"""
        try:
            self.logger.info("WebSocket connected!")
            
            if self.all_symbols:
                client.data_socket.subscribe(
                    data_type="SymbolUpdate",
                    symbols=self.all_symbols
                )
                self.logger.info(f"Subscribed to {len(self.all_symbols)} symbols")
                self.subscribed_symbols.update(self.all_symbols)
            
        except Exception as e:
            self.logger.error(f"Subscription error: {e}")
    
    def on_message(self, message):
        """Handle incoming WebSocket messages"""
        try:
            if message.get("type") != 'sf':
                return
            
            self.last_heartbeat = datetime.now()
            
            # Put message in price queue
            price_update = {
                'symbol': message["symbol"],
                'bid': message["bid_price"],
                'ask': message["ask_price"],
                'ltp': message["ltp"],
                'timestamp': self.last_heartbeat
            }
            
            try:
                self.price_queue.put_nowait(price_update)
            except:
                # Queue full, drop message
                self.logger.warning("Price queue full, dropping message")
                
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
    
    def on_error(self, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {error}")
    
    def on_close(self, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
    
    def check_connection_health(self):
        """Check WebSocket connection health"""
        try:
            now = datetime.now()
            time_since_heartbeat = (now - self.last_heartbeat).total_seconds()
            
            if time_since_heartbeat > 300:  # 5 minutes
                self.logger.warning(f"No data received for {time_since_heartbeat:.0f} seconds")
            
            if self.ws and hasattr(self.ws, 'data_socket'):
                if not self.ws.data_socket.is_connected():
                    self.logger.error("WebSocket disconnected, attempting reconnect...")
                    self.connect_websocket()
                    
        except Exception as e:
            self.logger.error(f"Connection health check failed: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.ws and hasattr(self.ws, 'stop_sockets'):
                self.ws.stop_sockets()
            
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5)
                
            self.logger.info("Data manager cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def stop(self):
        """Stop the data manager"""
        self.logger.info("Stopping Calendar Data Manager...")
        self.running = False
        time.sleep(1)

def main():
    """Main entry point for Calendar Trading System"""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('calendar_trading_system.log', encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        
        # Create and start system
        system = CalendarTradingSystem()
        system.start()
        
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt")
    except Exception as e:
        logging.error(f"Failed to start Calendar Trading System: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
