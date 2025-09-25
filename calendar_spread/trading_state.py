"""
Trading State Management for Calendar Spread Strategy
Handles position persistence, database operations, and state management
"""
import os
import sys
import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .calendar_spread_strategy import CalendarPosition, PositionStatus

class TradingState:
    """
    Centralized state management for calendar spread trading system.
    Handles position persistence, database operations, and state recovery.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("TradingState")
        
        # Database setup
        self.db_path = "calendar_spread.db"
        self.connection = None
        
        # State tracking
        self.open_positions: Dict[str, CalendarPosition] = {}
        self.closed_positions: List[CalendarPosition] = []
        self.daily_pnl: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        
        # Performance tracking
        self.performance_metrics = {
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_trade_duration': 0.0,
            'total_volume': 0.0
        }
        
        # Risk management
        self.daily_hedge_count: Dict[str, int] = {}
        self.last_hedge_time: Dict[str, datetime] = {}
        self.risk_limits = {
            'max_daily_loss': 10000,
            'max_positions': 5,
            'max_delta_exposure': 0.20,
            'max_portfolio_exposure': 0.20
        }
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for position persistence"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            # Create tables
            self.ensure_open_positions_table()
            self.ensure_closed_positions_table()
            self.ensure_performance_table()
            self.ensure_risk_metrics_table()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def ensure_open_positions_table(self):
        """Create open positions table if it doesn't exist"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS open_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strike REAL NOT NULL,
                    short_expiry TEXT NOT NULL,
                    long_expiry TEXT NOT NULL,
                    short_ce_symbol TEXT NOT NULL,
                    short_pe_symbol TEXT NOT NULL,
                    long_ce_symbol TEXT NOT NULL,
                    long_pe_symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    max_profit REAL NOT NULL,
                    max_loss REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    status TEXT NOT NULL,
                    net_delta REAL DEFAULT 0.0,
                    net_theta REAL DEFAULT 0.0,
                    net_vega REAL DEFAULT 0.0,
                    current_pnl REAL DEFAULT 0.0,
                    hedge_quantity INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.commit()
            self.logger.info("Open positions table ensured")
            
        except Exception as e:
            self.logger.error(f"Error creating open positions table: {e}")
            raise
    
    def ensure_closed_positions_table(self):
        """Create closed positions table if it doesn't exist"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS closed_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strike REAL NOT NULL,
                    short_expiry TEXT NOT NULL,
                    long_expiry TEXT NOT NULL,
                    short_ce_symbol TEXT NOT NULL,
                    short_pe_symbol TEXT NOT NULL,
                    long_ce_symbol TEXT NOT NULL,
                    long_pe_symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    max_profit REAL NOT NULL,
                    max_loss REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    status TEXT NOT NULL,
                    exit_reason TEXT,
                    final_pnl REAL,
                    duration_days INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.commit()
            self.logger.info("Closed positions table ensured")
            
        except Exception as e:
            self.logger.error(f"Error creating closed positions table: {e}")
            raise
    
    def ensure_performance_table(self):
        """Create performance metrics table"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_return REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0,
                    sharpe_ratio REAL DEFAULT 0.0,
                    win_rate REAL DEFAULT 0.0,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    daily_pnl REAL DEFAULT 0.0,
                    portfolio_value REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.commit()
            self.logger.info("Performance metrics table ensured")
            
        except Exception as e:
            self.logger.error(f"Error creating performance table: {e}")
            raise
    
    def ensure_risk_metrics_table(self):
        """Create risk metrics table"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    delta_exposure REAL DEFAULT 0.0,
                    portfolio_exposure REAL DEFAULT 0.0,
                    hedge_count INTEGER DEFAULT 0,
                    risk_level TEXT DEFAULT 'LOW',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.commit()
            self.logger.info("Risk metrics table ensured")
            
        except Exception as e:
            self.logger.error(f"Error creating risk metrics table: {e}")
            raise
    
    def save_position(self, position: CalendarPosition) -> bool:
        """Save position to database"""
        try:
            cursor = self.connection.cursor()
            
            # Check if position already exists
            cursor.execute("SELECT id FROM open_positions WHERE symbol = ?", (position.symbol,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing position
                cursor.execute("""
                    UPDATE open_positions SET
                        net_delta = ?, net_theta = ?, net_vega = ?, 
                        current_pnl = ?, hedge_quantity = ?, status = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ?
                """, (position.net_delta, position.net_theta, position.net_vega,
                      position.current_pnl, position.hedge_quantity, position.status.value,
                      position.symbol))
            else:
                # Insert new position
                cursor.execute("""
                    INSERT INTO open_positions (
                        symbol, strike, short_expiry, long_expiry,
                        short_ce_symbol, short_pe_symbol, long_ce_symbol, long_pe_symbol,
                        quantity, entry_price, max_profit, max_loss, entry_time, status,
                        net_delta, net_theta, net_vega, current_pnl, hedge_quantity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position.symbol, position.strike, position.short_expiry, position.long_expiry,
                    position.short_ce_symbol, position.short_pe_symbol, 
                    position.long_ce_symbol, position.long_pe_symbol,
                    position.quantity, position.entry_price, position.max_profit, position.max_loss,
                    position.entry_time.isoformat(), position.status.value,
                    position.net_delta, position.net_theta, position.net_vega, 
                    position.current_pnl, position.hedge_quantity
                ))
            
            self.connection.commit()
            self.logger.debug(f"Position saved for {position.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving position {position.symbol}: {e}")
            return False
    
    def close_position(self, position: CalendarPosition, exit_reason: str = None) -> bool:
        """Move position from open to closed"""
        try:
            cursor = self.connection.cursor()
            
            # Calculate duration
            duration = (datetime.now() - position.entry_time).days
            
            # Insert into closed positions
            cursor.execute("""
                INSERT INTO closed_positions (
                    symbol, strike, short_expiry, long_expiry,
                    short_ce_symbol, short_pe_symbol, long_ce_symbol, long_pe_symbol,
                    quantity, entry_price, exit_price, max_profit, max_loss,
                    entry_time, exit_time, status, exit_reason, final_pnl, duration_days
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.symbol, position.strike, position.short_expiry, position.long_expiry,
                position.short_ce_symbol, position.short_pe_symbol,
                position.long_ce_symbol, position.long_pe_symbol,
                position.quantity, position.entry_price, position.current_pnl,
                position.max_profit, position.max_loss,
                position.entry_time.isoformat(), datetime.now().isoformat(),
                position.status.value, exit_reason, position.current_pnl, duration
            ))
            
            # Remove from open positions
            cursor.execute("DELETE FROM open_positions WHERE symbol = ?", (position.symbol,))
            
            self.connection.commit()
            
            # Update statistics
            self.total_trades += 1
            if position.current_pnl > 0:
                self.winning_trades += 1
            
            self.logger.info(f"Position closed for {position.symbol}, reason: {exit_reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position {position.symbol}: {e}")
            return False
    
    def rehydrate_from_db(self):
        """Rehydrate positions from database on startup"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM open_positions")
            rows = cursor.fetchall()
            
            for row in rows:
                position = CalendarPosition(
                    symbol=row['symbol'],
                    strike=row['strike'],
                    short_expiry=row['short_expiry'],
                    long_expiry=row['long_expiry'],
                    short_ce_symbol=row['short_ce_symbol'],
                    short_pe_symbol=row['short_pe_symbol'],
                    long_ce_symbol=row['long_ce_symbol'],
                    long_pe_symbol=row['long_pe_symbol'],
                    quantity=row['quantity'],
                    entry_price=row['entry_price'],
                    max_profit=row['max_profit'],
                    max_loss=row['max_loss'],
                    entry_time=datetime.fromisoformat(row['entry_time']),
                    status=PositionStatus(row['status']),
                    net_delta=row['net_delta'],
                    net_theta=row['net_theta'],
                    net_vega=row['net_vega'],
                    current_pnl=row['current_pnl'],
                    hedge_quantity=row['hedge_quantity']
                )
                
                self.open_positions[position.symbol] = position
            
            self.logger.info(f"Rehydrated {len(self.open_positions)} positions from database")
            
        except Exception as e:
            self.logger.error(f"Error rehydrating positions: {e}")
    
    def get_position(self, symbol: str) -> Optional[CalendarPosition]:
        """Get position by symbol"""
        return self.open_positions.get(symbol)
    
    def add_position(self, position: CalendarPosition):
        """Add new position to state"""
        self.open_positions[position.symbol] = position
        self.save_position(position)
    
    def remove_position(self, symbol: str):
        """Remove position from state"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
    
    def update_position(self, position: CalendarPosition):
        """Update existing position"""
        if position.symbol in self.open_positions:
            self.open_positions[position.symbol] = position
            self.save_position(position)
    
    def get_all_positions(self) -> Dict[str, CalendarPosition]:
        """Get all open positions"""
        return self.open_positions.copy()
    
    def get_positions_by_status(self, status: PositionStatus) -> List[CalendarPosition]:
        """Get positions by status"""
        return [pos for pos in self.open_positions.values() if pos.status == status]
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio-level metrics"""
        try:
            total_positions = len(self.open_positions)
            open_positions = len([p for p in self.open_positions.values() if p.status == PositionStatus.OPEN])
            
            total_delta = sum(pos.net_delta for pos in self.open_positions.values())
            total_theta = sum(pos.net_theta for pos in self.open_positions.values())
            total_vega = sum(pos.net_vega for pos in self.open_positions.values())
            total_pnl = sum(pos.current_pnl for pos in self.open_positions.values())
            
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            return {
                'total_positions': total_positions,
                'open_positions': open_positions,
                'total_delta': total_delta,
                'total_theta': total_theta,
                'total_vega': total_vega,
                'total_pnl': total_pnl,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'daily_pnl': self.daily_pnl
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def save_performance_metrics(self):
        """Save daily performance metrics"""
        try:
            metrics = self.calculate_portfolio_metrics()
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO performance_metrics (
                    date, total_return, max_drawdown, sharpe_ratio, win_rate,
                    total_trades, winning_trades, daily_pnl, portfolio_value
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().date().isoformat(),
                metrics.get('total_return', 0.0),
                metrics.get('max_drawdown', 0.0),
                metrics.get('sharpe_ratio', 0.0),
                metrics.get('win_rate', 0.0),
                metrics.get('total_trades', 0),
                metrics.get('winning_trades', 0),
                metrics.get('daily_pnl', 0.0),
                100000 + metrics.get('total_pnl', 0.0)  # Assuming 1L base
            ))
            
            self.connection.commit()
            self.logger.info("Performance metrics saved")
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")
    
    def save_risk_metrics(self, symbol: str, delta_exposure: float, portfolio_exposure: float):
        """Save risk metrics for a symbol"""
        try:
            cursor = self.connection.cursor()
            
            # Determine risk level
            risk_level = 'LOW'
            if abs(delta_exposure) > 0.15 or portfolio_exposure > 0.15:
                risk_level = 'HIGH'
            elif abs(delta_exposure) > 0.10 or portfolio_exposure > 0.10:
                risk_level = 'MEDIUM'
            
            cursor.execute("""
                INSERT INTO risk_metrics (
                    date, symbol, delta_exposure, portfolio_exposure, 
                    hedge_count, risk_level
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().date().isoformat(),
                symbol, delta_exposure, portfolio_exposure,
                self.daily_hedge_count.get(symbol, 0), risk_level
            ))
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving risk metrics for {symbol}: {e}")
    
    def check_risk_limits(self) -> Dict[str, bool]:
        """Check if risk limits are breached"""
        try:
            metrics = self.calculate_portfolio_metrics()
            
            limits_check = {
                'max_positions': len(self.open_positions) < self.risk_limits['max_positions'],
                'max_daily_loss': self.daily_pnl > -self.risk_limits['max_daily_loss'],
                'max_delta_exposure': abs(metrics.get('total_delta', 0)) < self.risk_limits['max_delta_exposure'],
                'max_portfolio_exposure': abs(metrics.get('total_pnl', 0)) < self.risk_limits['max_portfolio_exposure']
            }
            
            return limits_check
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data from database"""
        try:
            cursor = self.connection.cursor()
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Clean up old closed positions
            cursor.execute("DELETE FROM closed_positions WHERE created_at < ?", (cutoff_date,))
            
            # Clean up old performance metrics
            cursor.execute("DELETE FROM performance_metrics WHERE created_at < ?", (cutoff_date,))
            
            # Clean up old risk metrics
            cursor.execute("DELETE FROM risk_metrics WHERE created_at < ?", (cutoff_date,))
            
            self.connection.commit()
            self.logger.info(f"Cleaned up data older than {days} days")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def close(self):
        """Close database connection"""
        try:
            if self.connection:
                self.connection.close()
                self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database: {e}")
