"""
Backtesting framework for Calendar Spread Strategy
"""
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calendar_spread.config import CalendarConfig
from calendar_spread.calendar_spread_strategy import CalendarSpreadStrategy, CalendarPosition, PositionStatus

class CalendarBacktest:
    """
    Backtesting framework for Calendar Spread Strategy
    """
    
    def __init__(self, config: CalendarConfig, start_date: datetime, end_date: datetime):
        self.config = config
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger("CalendarBacktest")
        
        # Backtest data
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.trades: List[Dict] = []
        self.daily_pnl: List[Dict] = []
        
        # Performance metrics
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        
    def load_historical_data(self, symbols: List[str]):
        """
        Load historical data for backtesting
        
        Args:
            symbols: List of symbols to load data for
        """
        try:
            self.logger.info(f"Loading historical data for {len(symbols)} symbols")
            
            for symbol in symbols:
                # This would typically load from a data source
                # For now, we'll create mock data
                data = self._create_mock_data(symbol, self.start_date, self.end_date)
                self.historical_data[symbol] = data
                
            self.logger.info("Historical data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
    
    def _create_mock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Create mock historical data for backtesting
        In practice, this would load real historical data
        """
        try:
            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Mock price data (simplified)
            np.random.seed(42)  # For reproducible results
            base_price = 1000 if symbol == 'NIFTY' else 500
            
            prices = []
            current_price = base_price
            
            for date in date_range:
                # Random walk with drift
                change = np.random.normal(0, 0.02)  # 2% daily volatility
                current_price *= (1 + change)
                prices.append(current_price)
            
            # Create DataFrame
            data = pd.DataFrame({
                'date': date_range,
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, len(date_range)),
                'iv': np.random.uniform(0.15, 0.35, len(date_range))  # Mock IV
            })
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error creating mock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, symbols: List[str] = None) -> Dict:
        """
        Run backtest for calendar spread strategy
        
        Args:
            symbols: List of symbols to backtest. If None, uses default list
            
        Returns:
            Dictionary with backtest results
        """
        try:
            if symbols is None:
                symbols = list(self.config.LOT_SIZE_MAP.keys())[:5]  # Test with first 5 symbols
            
            self.logger.info(f"Starting backtest for {len(symbols)} symbols")
            self.logger.info(f"Period: {self.start_date.date()} to {self.end_date.date()}")
            
            # Load historical data
            self.load_historical_data(symbols)
            
            # Initialize strategy
            strategy = CalendarSpreadStrategy(self.config)
            
            # Run backtest simulation
            current_date = self.start_date
            portfolio_value = 100000  # Starting with 1L
            max_portfolio_value = portfolio_value
            
            while current_date <= self.end_date:
                # Check for new opportunities
                if len(strategy.positions) < 3:  # Max 3 positions
                    selected_symbol = self._select_symbol_for_date(symbols, current_date)
                    if selected_symbol:
                        self._simulate_trade_entry(strategy, selected_symbol, current_date)
                
                # Update existing positions
                self._update_positions(strategy, current_date)
                
                # Check exit conditions
                self._check_exit_conditions(strategy, current_date)
                
                # Calculate daily P&L
                daily_pnl = self._calculate_daily_pnl(strategy, current_date)
                portfolio_value += daily_pnl
                
                # Track maximum drawdown
                if portfolio_value > max_portfolio_value:
                    max_portfolio_value = portfolio_value
                
                drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value
                if drawdown > self.max_drawdown:
                    self.max_drawdown = drawdown
                
                # Record daily P&L
                self.daily_pnl.append({
                    'date': current_date,
                    'pnl': daily_pnl,
                    'portfolio_value': portfolio_value,
                    'drawdown': drawdown
                })
                
                current_date += timedelta(days=1)
            
            # Calculate final metrics
            self._calculate_performance_metrics(portfolio_value)
            
            results = {
                'total_return': self.total_return,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': self.sharpe_ratio,
                'win_rate': self.win_rate,
                'total_trades': len(self.trades),
                'final_portfolio_value': portfolio_value,
                'trades': self.trades,
                'daily_pnl': self.daily_pnl
            }
            
            self.logger.info("Backtest completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            return {}
    
    def _select_symbol_for_date(self, symbols: List[str], date: datetime) -> Optional[str]:
        """Select symbol for trade entry on given date"""
        try:
            # Simple selection logic - in practice would use more sophisticated criteria
            available_symbols = [s for s in symbols if s in self.historical_data]
            if not available_symbols:
                return None
            
            # Select symbol with highest volatility on this date
            best_symbol = None
            best_iv = 0
            
            for symbol in available_symbols:
                data = self.historical_data[symbol]
                date_data = data[data['date'] == date]
                if not date_data.empty:
                    iv = date_data['iv'].iloc[0]
                    if iv > best_iv:
                        best_iv = iv
                        best_symbol = symbol
            
            return best_symbol
            
        except Exception as e:
            self.logger.error(f"Error selecting symbol for date {date}: {e}")
            return None
    
    def _simulate_trade_entry(self, strategy: CalendarSpreadStrategy, symbol: str, date: datetime):
        """Simulate trade entry"""
        try:
            # Get price data for the date
            data = self.historical_data[symbol]
            date_data = data[data['date'] == date]
            if date_data.empty:
                return
            
            current_price = date_data['close'].iloc[0]
            
            # Create mock option data
            option_data = {
                'strike': round(current_price / 50) * 50,  # Round to nearest 50
                'current_price': current_price,
                'short_expiry': '25JAN',  # Mock expiry
                'long_expiry': '25FEB',   # Mock expiry
                'short_ce_symbol': f'NSE:{symbol}25JAN{int(current_price)}CE',
                'short_pe_symbol': f'NSE:{symbol}25JAN{int(current_price)}PE',
                'long_ce_symbol': f'NSE:{symbol}25FEB{int(current_price)}CE',
                'long_pe_symbol': f'NSE:{symbol}25FEB{int(current_price)}PE'
            }
            
            # Calculate position size
            quantity = strategy.calculate_position_size(symbol, option_data)
            if quantity == 0:
                return
            
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
                entry_price=current_price,
                max_profit=current_price * 0.1,  # Mock max profit
                max_loss=current_price * 0.05,   # Mock max loss
                entry_time=date,
                status=PositionStatus.OPEN
            )
            
            strategy.positions[symbol] = position
            
            # Record trade
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'entry',
                'strike': option_data['strike'],
                'quantity': quantity,
                'price': current_price
            })
            
            self.logger.info(f"Simulated entry for {symbol} on {date.date()}")
            
        except Exception as e:
            self.logger.error(f"Error simulating trade entry: {e}")
    
    def _update_positions(self, strategy: CalendarSpreadStrategy, date: datetime):
        """Update position values and Greeks"""
        try:
            for symbol, position in strategy.positions.items():
                if position.status != PositionStatus.OPEN:
                    continue
                
                # Get current price
                data = self.historical_data[symbol]
                date_data = data[data['date'] == date]
                if date_data.empty:
                    continue
                
                current_price = date_data['close'].iloc[0]
                
                # Update position Greeks (simplified)
                position.net_delta = (current_price - position.strike) * 0.01 * position.quantity
                position.net_theta = -position.quantity * 0.5  # Mock theta
                position.net_vega = position.quantity * 0.1    # Mock vega
                
                # Update P&L (simplified)
                price_change = current_price - position.entry_price
                position.current_pnl = price_change * position.quantity * 0.1  # Mock P&L calculation
                
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _check_exit_conditions(self, strategy: CalendarSpreadStrategy, date: datetime):
        """Check exit conditions for positions"""
        try:
            positions_to_close = []
            
            for symbol, position in strategy.positions.items():
                if position.status != PositionStatus.OPEN:
                    continue
                
                # Time-based exit
                days_held = (date - position.entry_time).days
                if days_held >= self.config.MAX_HOLDING_DAYS:
                    positions_to_close.append((symbol, "max_holding_period"))
                    continue
                
                # Profit target
                if position.current_pnl >= position.max_profit * self.config.PROFIT_TARGET:
                    positions_to_close.append((symbol, "profit_target"))
                    continue
                
                # Stop loss
                if position.current_pnl <= -position.max_loss * self.config.MAX_LOSS_MULTIPLIER:
                    positions_to_close.append((symbol, "stop_loss"))
                    continue
            
            # Close positions
            for symbol, reason in positions_to_close:
                position = strategy.positions[symbol]
                position.status = PositionStatus.CLOSED
                
                # Record trade
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'exit',
                    'reason': reason,
                    'pnl': position.current_pnl
                })
                
                self.logger.info(f"Closed position {symbol} on {date.date()}, reason: {reason}")
                
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
    
    def _calculate_daily_pnl(self, strategy: CalendarSpreadStrategy, date: datetime) -> float:
        """Calculate daily P&L"""
        try:
            total_pnl = 0.0
            
            for position in strategy.positions.values():
                if position.status == PositionStatus.OPEN:
                    total_pnl += position.current_pnl
            
            return total_pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating daily P&L: {e}")
            return 0.0
    
    def _calculate_performance_metrics(self, final_portfolio_value: float):
        """Calculate performance metrics"""
        try:
            # Total return
            self.total_return = (final_portfolio_value - 100000) / 100000
            
            # Win rate
            winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
            total_trades = len([t for t in self.trades if t['action'] == 'exit'])
            self.win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Sharpe ratio (simplified)
            if self.daily_pnl:
                daily_returns = [d['pnl'] / 100000 for d in self.daily_pnl]
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                self.sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
    
    def print_results(self, results: Dict):
        """Print backtest results"""
        try:
            print("\n" + "=" * 60)
            print("CALENDAR SPREAD STRATEGY BACKTEST RESULTS")
            print("=" * 60)
            print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
            print(f"Total Return: {results.get('total_return', 0):.2%}")
            print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
            print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"Win Rate: {results.get('win_rate', 0):.2f}%")
            print(f"Total Trades: {results.get('total_trades', 0)}")
            print(f"Final Portfolio Value: ₹{results.get('final_portfolio_value', 0):,.2f}")
            print("=" * 60)
            
            # Print trade summary
            if results.get('trades'):
                print("\nTRADE SUMMARY:")
                print("-" * 40)
                for trade in results['trades']:
                    if trade['action'] == 'exit':
                        print(f"{trade['date'].date()} | {trade['symbol']} | "
                              f"Exit ({trade['reason']}) | P&L: ₹{trade.get('pnl', 0):,.2f}")
            
        except Exception as e:
            self.logger.error(f"Error printing results: {e}")

def main():
    """Main backtest execution"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize configuration
        config = CalendarConfig()
        
        # Set backtest period
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 6, 30)
        
        # Initialize backtest
        backtest = CalendarBacktest(config, start_date, end_date)
        
        # Run backtest
        results = backtest.run_backtest()
        
        # Print results
        backtest.print_results(results)
        
    except Exception as e:
        print(f"Error in backtest: {e}")

if __name__ == "__main__":
    main()
