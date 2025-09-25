"""
Trade Manager for Calendar Spread Strategy
Handles order execution, trade management, and position adjustments
"""
import os
import sys
import logging
import threading
import time
from datetime import datetime, timedelta
from queue import Queue, Empty
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_utils import get_quotes, place_order, get_positions, get_orderbook
from utils.trade_costs import calculate_total_trade_cost
from .trading_state import TradingState
from .calendar_spread_strategy import CalendarPosition, PositionStatus

class OrderType(Enum):
    MARKET = 1
    LIMIT = 2
    STOP = 3
    STOP_LIMIT = 4

class OrderSide(Enum):
    BUY = 1
    SELL = -1

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Data class for order management"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    timestamp: datetime = None
    order_id: str = None
    error_message: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class TradeManager(threading.Thread):
    """
    Manages trade execution, order handling, and position adjustments for calendar spreads.
    Follows the same pattern as the gamma scalping trade manager.
    """
    
    def __init__(self, signal_queue: Queue, config, logger, trading_state: TradingState):
        super().__init__(name="CalendarTradeManager")
        self.signal_queue = signal_queue
        self.config = config
        self.logger = logger
        self.trading_state = trading_state
        
        # Threading control
        self._running_lock = threading.Lock()
        self._running = False
        
        # Order management
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.order_counter = 0
        
        # Trade execution
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_volume': 0,
            'total_commission': 0.0
        }
        
        # Risk management
        self.max_orders_per_minute = 10
        self.order_timestamps: List[datetime] = []
        self.daily_order_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Position management
        self.hedge_orders: Dict[str, List[Order]] = {}
        
    @property
    def running(self):
        with self._running_lock:
            return self._running
    
    @running.setter
    def running(self, value):
        with self._running_lock:
            self._running = value
    
    def start(self):
        """Start the trade manager"""
        self.logger.info("Starting Calendar Trade Manager...")
        self.running = True
        super().start()
    
    def run(self):
        """Main trade management loop"""
        try:
            while self.running:
                try:
                    # Process trading signals
                    self.process_trading_signals()
                    
                    # Monitor pending orders
                    self.monitor_pending_orders()
                    
                    # Update execution statistics
                    self.update_execution_stats()
                    
                    # Sleep briefly
                    time.sleep(0.5)
                    
                except Empty:
                    # No signals to process, continue
                    continue
                except Exception as e:
                    self.logger.error(f"Error in trade management loop: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in trade manager: {e}")
        finally:
            self.logger.info("Calendar Trade Manager stopped")
    
    def process_trading_signals(self):
        """Process trading signals from the queue"""
        try:
            # Process up to 10 signals at once
            processed_count = 0
            max_batch_size = 10
            
            while processed_count < max_batch_size:
                try:
                    signal = self.signal_queue.get(timeout=0.1)
                    self.process_single_signal(signal)
                    self.signal_queue.task_done()
                    processed_count += 1
                    
                except Empty:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing signal: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error processing trading signals: {e}")
    
    def process_single_signal(self, signal: Dict[str, Any]):
        """Process a single trading signal"""
        try:
            signal_type = signal.get('type')
            symbol = signal.get('symbol')
            position = signal.get('position')
            
            self.logger.info(f"Processing signal: {signal_type} for {symbol}")
            
            if signal_type == 'HEDGE_DELTA':
                self.execute_delta_hedge(position, signal.get('reason', ''))
            elif signal_type == 'EXIT_POSITION':
                self.execute_position_exit(position, signal.get('reason', ''))
            elif signal_type == 'ADJUST_POSITION':
                self.execute_position_adjustment(position, signal.get('reason', ''))
            elif signal_type == 'OPEN_POSITION':
                self.execute_position_entry(signal)
            else:
                self.logger.warning(f"Unknown signal type: {signal_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing single signal: {e}")
    
    def execute_delta_hedge(self, position: CalendarPosition, reason: str):
        """Execute delta hedge for a position"""
        try:
            if not self.config.USE_FUTURES_HEDGE:
                self.logger.info(f"Futures hedging disabled for {position.symbol}")
                return
            
            # Check rate limits
            if not self.check_rate_limits():
                self.logger.warning("Rate limit exceeded, skipping hedge")
                return
            
            # Calculate hedge quantity
            lot_size = self.config.LOT_SIZE_MAP.get(position.symbol, self.config.DEFAULT_LOT_SIZE)
            hedge_lots = int(abs(position.net_delta) * self.config.HEDGE_RATIO / lot_size)
            
            if hedge_lots == 0:
                self.logger.info(f"No hedge needed for {position.symbol}")
                return
            
            # Build future symbol
            future_symbol = f"NSE:{position.symbol}{position.long_expiry}FUT"
            
            # Determine hedge direction
            side = OrderSide.BUY if position.net_delta < 0 else OrderSide.SELL
            
            # Create hedge order
            order = self.create_order(
                symbol=future_symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=hedge_lots * lot_size,
                price=0.0  # Market order
            )
            
            # Execute the order
            success = self.execute_order(order)
            
            if success:
                # Update position hedge quantity
                position.hedge_quantity += hedge_lots
                self.trading_state.update_position(position)
                
                # Track hedge order
                if position.symbol not in self.hedge_orders:
                    self.hedge_orders[position.symbol] = []
                self.hedge_orders[position.symbol].append(order)
                
                # Update hedge count
                self.trading_state.daily_hedge_count[position.symbol] = \
                    self.trading_state.daily_hedge_count.get(position.symbol, 0) + 1
                self.trading_state.last_hedge_time[position.symbol] = datetime.now()
                
                self.logger.info(f"Delta hedge executed for {position.symbol}: {hedge_lots} lots")
            else:
                self.logger.error(f"Failed to execute delta hedge for {position.symbol}")
                
        except Exception as e:
            self.logger.error(f"Error executing delta hedge for {position.symbol}: {e}")
    
    def execute_position_exit(self, position: CalendarPosition, reason: str):
        """Execute position exit"""
        try:
            self.logger.info(f"Executing position exit for {position.symbol}, reason: {reason}")
            
            # Check rate limits
            if not self.check_rate_limits():
                self.logger.warning("Rate limit exceeded, skipping exit")
                return
            
            # Create exit orders for all legs
            exit_orders = []
            lot_size = self.config.LOT_SIZE_MAP.get(position.symbol, self.config.DEFAULT_LOT_SIZE)
            qty = position.quantity * lot_size
            
            # Buy back short options
            short_ce_order = self.create_order(
                symbol=position.short_ce_symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=qty,
                price=0.0
            )
            
            short_pe_order = self.create_order(
                symbol=position.short_pe_symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=qty,
                price=0.0
            )
            
            # Sell long options
            long_ce_order = self.create_order(
                symbol=position.long_ce_symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=qty,
                price=0.0
            )
            
            long_pe_order = self.create_order(
                symbol=position.long_pe_symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=qty,
                price=0.0
            )
            
            exit_orders = [short_ce_order, short_pe_order, long_ce_order, long_pe_order]
            
            # Execute all orders
            success_count = 0
            for order in exit_orders:
                if self.execute_order(order):
                    success_count += 1
            
            if success_count == len(exit_orders):
                # All orders executed successfully
                position.status = PositionStatus.CLOSED
                self.trading_state.close_position(position, reason)
                self.trading_state.remove_position(position.symbol)
                
                self.logger.info(f"Successfully closed position {position.symbol}")
            else:
                # Some orders failed
                position.status = PositionStatus.ERROR
                self.trading_state.update_position(position)
                
                self.logger.error(f"Failed to close position {position.symbol} completely")
                
        except Exception as e:
            self.logger.error(f"Error executing position exit for {position.symbol}: {e}")
    
    def execute_position_adjustment(self, position: CalendarPosition, reason: str):
        """Execute position adjustment"""
        try:
            self.logger.info(f"Executing position adjustment for {position.symbol}, reason: {reason}")
            
            # For now, we'll implement basic adjustments
            # In a more sophisticated system, this could involve:
            # - Rolling positions to different strikes
            # - Adjusting position sizes
            # - Adding protective positions
            
            # Check if adjustment is needed based on Greeks
            if position.symbol in self.trading_state.greeks_data:
                greeks = self.trading_state.greeks_data[position.symbol]
                
                # High gamma exposure - consider reducing position size
                if abs(greeks['gamma']) > 0.5:
                    self.logger.info(f"High gamma exposure detected for {position.symbol}")
                    # Could implement position size reduction here
                
                # High vega exposure - consider volatility hedge
                if abs(greeks['vega']) > 0.1:
                    self.logger.info(f"High vega exposure detected for {position.symbol}")
                    # Could implement volatility hedge here
            
        except Exception as e:
            self.logger.error(f"Error executing position adjustment for {position.symbol}: {e}")
    
    def execute_position_entry(self, signal: Dict[str, Any]):
        """Execute new position entry"""
        try:
            symbol = signal.get('symbol')
            option_data = signal.get('option_data')
            quantity = signal.get('quantity')
            
            self.logger.info(f"Executing position entry for {symbol}")
            
            # Check rate limits
            if not self.check_rate_limits():
                self.logger.warning("Rate limit exceeded, skipping entry")
                return
            
            # Create entry orders
            entry_orders = []
            lot_size = self.config.LOT_SIZE_MAP.get(symbol, self.config.DEFAULT_LOT_SIZE)
            qty = quantity * lot_size
            
            # Sell short-term options
            short_ce_order = self.create_order(
                symbol=option_data['short_ce_symbol'],
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=qty,
                price=0.0
            )
            
            short_pe_order = self.create_order(
                symbol=option_data['short_pe_symbol'],
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=qty,
                price=0.0
            )
            
            # Buy long-term options
            long_ce_order = self.create_order(
                symbol=option_data['long_ce_symbol'],
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=qty,
                price=0.0
            )
            
            long_pe_order = self.create_order(
                symbol=option_data['long_pe_symbol'],
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=qty,
                price=0.0
            )
            
            entry_orders = [short_ce_order, short_pe_order, long_ce_order, long_pe_order]
            
            # Execute all orders
            success_count = 0
            for order in entry_orders:
                if self.execute_order(order):
                    success_count += 1
            
            if success_count == len(entry_orders):
                # Create position
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
                    entry_price=option_data.get('current_price', 0.0),
                    max_profit=option_data.get('max_profit', 0.0),
                    max_loss=option_data.get('max_loss', 0.0),
                    entry_time=datetime.now(),
                    status=PositionStatus.OPEN
                )
                
                # Add to trading state
                self.trading_state.add_position(position)
                
                self.logger.info(f"Successfully opened position {symbol}")
            else:
                self.logger.error(f"Failed to open position {symbol} completely")
                
        except Exception as e:
            self.logger.error(f"Error executing position entry for {symbol}: {e}")
    
    def create_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                    quantity: int, price: float = 0.0) -> Order:
        """Create a new order"""
        try:
            self.order_counter += 1
            order_id = f"CAL_{self.order_counter}_{int(datetime.now().timestamp())}"
            
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price
            )
            
            self.pending_orders[order_id] = order
            self.execution_stats['total_orders'] += 1
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            return None
    
    def execute_order(self, order: Order) -> bool:
        """Execute an order through the broker API"""
        try:
            # Prepare order data for Fyers API
            order_data = {
                "symbol": order.symbol,
                "qty": order.quantity,
                "type": order.order_type.value,
                "side": order.side.value,
                "productType": "INTRADAY",
                "validity": "DAY"
            }
            
            # Add price for limit orders
            if order.order_type == OrderType.LIMIT and order.price > 0:
                order_data["price"] = order.price
            
            # Execute order
            response = place_order([order_data])
            
            if response and response.get('s') == 'ok':
                order.status = OrderStatus.SUBMITTED
                order.order_id = response.get('id')
                self.execution_stats['filled_orders'] += 1
                self.execution_stats['total_volume'] += order.quantity
                
                # Calculate commission
                commission = self.calculate_commission(order)
                self.execution_stats['total_commission'] += commission
                
                self.logger.info(f"Order executed successfully: {order.id}")
                return True
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = str(response)
                self.execution_stats['rejected_orders'] += 1
                
                self.logger.error(f"Order rejected: {order.id}, response: {response}")
                return False
                
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self.execution_stats['rejected_orders'] += 1
            
            self.logger.error(f"Error executing order {order.id}: {e}")
            return False
    
    def calculate_commission(self, order: Order) -> float:
        """Calculate commission for an order"""
        try:
            # Use the trade costs utility
            trade_data = {
                'type': 'options',
                'price': order.price if order.price > 0 else 100,  # Default price for commission calc
                'lot_size': order.quantity,
                'is_buy': order.side == OrderSide.BUY,
                'is_sell': order.side == OrderSide.SELL
            }
            
            return calculate_total_trade_cost([trade_data])
            
        except Exception as e:
            self.logger.error(f"Error calculating commission: {e}")
            return 20.0  # Default commission
    
    def monitor_pending_orders(self):
        """Monitor pending orders and update their status"""
        try:
            # Get order book from broker
            orderbook = get_orderbook()
            
            if not orderbook or orderbook.get('s') != 'ok':
                return
            
            orders_data = orderbook.get('data', [])
            
            # Update order statuses
            for order_data in orders_data:
                broker_order_id = order_data.get('id')
                
                # Find matching order
                for order in self.pending_orders.values():
                    if order.order_id == broker_order_id:
                        self.update_order_status(order, order_data)
                        break
                        
        except Exception as e:
            self.logger.error(f"Error monitoring pending orders: {e}")
    
    def update_order_status(self, order: Order, order_data: Dict[str, Any]):
        """Update order status based on broker data"""
        try:
            status = order_data.get('status')
            filled_qty = int(order_data.get('filledQty', 0))
            average_price = float(order_data.get('averagePrice', 0))
            
            if status == 'COMPLETE':
                order.status = OrderStatus.FILLED
                order.filled_quantity = filled_qty
                order.average_price = average_price
                
                # Move to history
                self.order_history.append(order)
                if order.id in self.pending_orders:
                    del self.pending_orders[order.id]
                    
            elif status == 'CANCELLED':
                order.status = OrderStatus.CANCELLED
                self.execution_stats['cancelled_orders'] += 1
                
                # Move to history
                self.order_history.append(order)
                if order.id in self.pending_orders:
                    del self.pending_orders[order.id]
                    
            elif filled_qty > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
                order.filled_quantity = filled_qty
                order.average_price = average_price
                
        except Exception as e:
            self.logger.error(f"Error updating order status: {e}")
    
    def check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        try:
            now = datetime.now()
            
            # Reset daily counter if new day
            if now.date() != self.last_reset_date:
                self.daily_order_count = 0
                self.last_reset_date = now.date()
            
            # Check daily limit
            if self.daily_order_count >= 100:  # Max 100 orders per day
                return False
            
            # Check minute limit
            minute_ago = now - timedelta(minutes=1)
            recent_orders = [ts for ts in self.order_timestamps if ts > minute_ago]
            
            if len(recent_orders) >= self.max_orders_per_minute:
                return False
            
            # Add current order timestamp
            self.order_timestamps.append(now)
            self.daily_order_count += 1
            
            # Clean old timestamps
            self.order_timestamps = [ts for ts in self.order_timestamps if ts > minute_ago]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking rate limits: {e}")
            return False
    
    def update_execution_stats(self):
        """Update execution statistics"""
        try:
            # Calculate fill rate
            total_orders = self.execution_stats['total_orders']
            filled_orders = self.execution_stats['filled_orders']
            
            if total_orders > 0:
                fill_rate = (filled_orders / total_orders) * 100
                self.logger.debug(f"Order fill rate: {fill_rate:.2f}%")
                
        except Exception as e:
            self.logger.error(f"Error updating execution stats: {e}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        try:
            return {
                'execution_stats': self.execution_stats.copy(),
                'pending_orders': len(self.pending_orders),
                'order_history_count': len(self.order_history),
                'daily_order_count': self.daily_order_count,
                'rate_limit_status': self.check_rate_limits()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting execution summary: {e}")
            return {}
    
    def stop(self):
        """Stop the trade manager"""
        self.logger.info("Stopping Calendar Trade Manager...")
        self.running = False
        time.sleep(1)  # Give time for current processing to complete
