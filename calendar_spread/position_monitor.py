"""
Enhanced Position Monitor for Calendar Spread Strategy
Monitors positions, manages risk, and handles position lifecycle
"""
import os
import sys
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .trading_state import TradingState
from .calendar_spread_strategy import CalendarPosition, PositionStatus

class PositionMonitor(threading.Thread):
    """
    Enhanced position monitor that tracks positions, manages risk, and handles lifecycle events.
    Follows the same pattern as the gamma scalping position monitor but enhanced for calendar spreads.
    """
    
    def __init__(self, trading_state: TradingState, monitor_interval: int = 30):
        super().__init__(name="CalendarPositionMonitor")
        self.trading_state = trading_state
        self.monitor_interval = monitor_interval
        self.logger = logging.getLogger("CalendarPositionMonitor")
        
        # Threading control
        self._running_lock = threading.Lock()
        self._running = False
        
        # Monitoring data
        self.position_history: Dict[str, List[Dict]] = defaultdict(list)
        self.risk_alerts: List[Dict] = []
        self.performance_alerts: List[Dict] = []
        
        # Risk thresholds
        self.risk_thresholds = {
            'max_delta_exposure': 0.20,
            'max_portfolio_exposure': 0.20,
            'max_daily_loss': 10000,
            'max_position_duration': 30,  # days
            'min_theta_threshold': -0.5,
            'max_vega_exposure': 0.1
        }
        
        # Performance tracking
        self.daily_performance = {
            'start_value': 100000,  # Assuming 1L portfolio
            'current_value': 100000,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_value': 100000
        }
        
        # Alert system
        self.alert_counters = defaultdict(int)
        self.last_alert_time = defaultdict(lambda: datetime.min)
        
    @property
    def running(self):
        with self._running_lock:
            return self._running
    
    @running.setter
    def running(self, value):
        with self._running_lock:
            self._running = value
    
    def start(self):
        """Start the position monitor"""
        self.logger.info("Starting Calendar Position Monitor...")
        self.running = True
        super().start()
    
    def run(self):
        """Main monitoring loop"""
        try:
            while self.running:
                try:
                    # Monitor all positions
                    self.monitor_all_positions()
                    
                    # Check risk limits
                    self.check_risk_limits()
                    
                    # Update performance metrics
                    self.update_performance_metrics()
                    
                    # Generate alerts
                    self.generate_alerts()
                    
                    # Clean up old data
                    self.cleanup_old_data()
                    
                    # Sleep for monitor interval
                    time.sleep(self.monitor_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in position monitoring loop: {e}")
                    time.sleep(5)  # Wait before retrying
                    
        except Exception as e:
            self.logger.error(f"Fatal error in position monitor: {e}")
        finally:
            self.logger.info("Calendar Position Monitor stopped")
    
    def monitor_all_positions(self):
        """Monitor all open positions"""
        try:
            positions = self.trading_state.get_all_positions()
            
            for symbol, position in positions.items():
                if position.status == PositionStatus.OPEN:
                    self.monitor_single_position(position)
                    
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")
    
    def monitor_single_position(self, position: CalendarPosition):
        """Monitor a single position"""
        try:
            # Record position snapshot
            snapshot = {
                'timestamp': datetime.now(),
                'symbol': position.symbol,
                'net_delta': position.net_delta,
                'net_theta': position.net_theta,
                'net_vega': position.net_vega,
                'current_pnl': position.current_pnl,
                'hedge_quantity': position.hedge_quantity,
                'days_held': (datetime.now() - position.entry_time).days
            }
            
            self.position_history[position.symbol].append(snapshot)
            
            # Keep only last 1000 snapshots per position
            if len(self.position_history[position.symbol]) > 1000:
                self.position_history[position.symbol] = \
                    self.position_history[position.symbol][-1000:]
            
            # Check position-specific risks
            self.check_position_risks(position)
            
            # Check position lifecycle
            self.check_position_lifecycle(position)
            
        except Exception as e:
            self.logger.error(f"Error monitoring position {position.symbol}: {e}")
    
    def check_position_risks(self, position: CalendarPosition):
        """Check risks for a specific position"""
        try:
            # Delta exposure risk
            delta_exposure = abs(position.net_delta) / (position.quantity * 100)
            if delta_exposure > self.risk_thresholds['max_delta_exposure']:
                self.create_risk_alert(
                    position.symbol,
                    'HIGH_DELTA_EXPOSURE',
                    f'Delta exposure: {delta_exposure:.3f}',
                    'HIGH'
                )
            
            # Theta decay risk
            if position.net_theta < self.risk_thresholds['min_theta_threshold']:
                self.create_risk_alert(
                    position.symbol,
                    'HIGH_THETA_DECAY',
                    f'Theta decay: {position.net_theta:.3f}',
                    'MEDIUM'
                )
            
            # Vega exposure risk
            vega_exposure = abs(position.net_vega) / (position.quantity * 100)
            if vega_exposure > self.risk_thresholds['max_vega_exposure']:
                self.create_risk_alert(
                    position.symbol,
                    'HIGH_VEGA_EXPOSURE',
                    f'Vega exposure: {vega_exposure:.3f}',
                    'MEDIUM'
                )
            
            # Position duration risk
            days_held = (datetime.now() - position.entry_time).days
            if days_held > self.risk_thresholds['max_position_duration']:
                self.create_risk_alert(
                    position.symbol,
                    'POSITION_TOO_OLD',
                    f'Position held for {days_held} days',
                    'HIGH'
                )
            
        except Exception as e:
            self.logger.error(f"Error checking position risks for {position.symbol}: {e}")
    
    def check_position_lifecycle(self, position: CalendarPosition):
        """Check position lifecycle events"""
        try:
            days_held = (datetime.now() - position.entry_time).days
            
            # Check for expiry approaching
            # This would need to be calculated based on actual expiry dates
            # For now, we'll use a simple time-based approach
            
            # Check profit/loss thresholds
            if position.current_pnl > 0:
                profit_percentage = (position.current_pnl / position.max_profit) * 100
                if profit_percentage > 50:  # 50% of max profit
                    self.create_performance_alert(
                        position.symbol,
                        'PROFIT_TARGET_APPROACHING',
                        f'Profit: {profit_percentage:.1f}% of max',
                        'INFO'
                    )
            
            if position.current_pnl < 0:
                loss_percentage = abs(position.current_pnl / position.max_loss) * 100
                if loss_percentage > 80:  # 80% of max loss
                    self.create_performance_alert(
                        position.symbol,
                        'STOP_LOSS_APPROACHING',
                        f'Loss: {loss_percentage:.1f}% of max',
                        'WARNING'
                    )
            
        except Exception as e:
            self.logger.error(f"Error checking position lifecycle for {position.symbol}: {e}")
    
    def check_risk_limits(self):
        """Check portfolio-level risk limits"""
        try:
            # Get portfolio metrics
            metrics = self.trading_state.calculate_portfolio_metrics()
            
            # Check daily loss limit
            if self.daily_performance['daily_pnl'] < -self.risk_thresholds['max_daily_loss']:
                self.create_risk_alert(
                    'PORTFOLIO',
                    'DAILY_LOSS_LIMIT',
                    f'Daily loss: {self.daily_performance["daily_pnl"]:.2f}',
                    'CRITICAL'
                )
            
            # Check portfolio exposure
            total_exposure = abs(metrics.get('total_pnl', 0)) / self.daily_performance['current_value']
            if total_exposure > self.risk_thresholds['max_portfolio_exposure']:
                self.create_risk_alert(
                    'PORTFOLIO',
                    'HIGH_PORTFOLIO_EXPOSURE',
                    f'Portfolio exposure: {total_exposure:.3f}',
                    'HIGH'
                )
            
            # Check maximum drawdown
            current_drawdown = (self.daily_performance['peak_value'] - self.daily_performance['current_value']) / self.daily_performance['peak_value']
            if current_drawdown > 0.15:  # 15% drawdown
                self.create_risk_alert(
                    'PORTFOLIO',
                    'HIGH_DRAWDOWN',
                    f'Drawdown: {current_drawdown:.3f}',
                    'HIGH'
                )
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Get current portfolio value
            metrics = self.trading_state.calculate_portfolio_metrics()
            current_pnl = metrics.get('total_pnl', 0.0)
            
            # Update daily performance
            self.daily_performance['current_value'] = self.daily_performance['start_value'] + current_pnl
            self.daily_performance['daily_pnl'] = current_pnl
            
            # Update peak value
            if self.daily_performance['current_value'] > self.daily_performance['peak_value']:
                self.daily_performance['peak_value'] = self.daily_performance['current_value']
            
            # Calculate maximum drawdown
            current_drawdown = (self.daily_performance['peak_value'] - self.daily_performance['current_value']) / self.daily_performance['peak_value']
            if current_drawdown > self.daily_performance['max_drawdown']:
                self.daily_performance['max_drawdown'] = current_drawdown
            
            # Save performance metrics to database
            self.trading_state.save_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def generate_alerts(self):
        """Generate and process alerts"""
        try:
            # Process risk alerts
            for alert in self.risk_alerts[-10:]:  # Last 10 alerts
                if self.should_send_alert(alert):
                    self.send_alert(alert)
            
            # Process performance alerts
            for alert in self.performance_alerts[-10:]:  # Last 10 alerts
                if self.should_send_alert(alert):
                    self.send_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error generating alerts: {e}")
    
    def create_risk_alert(self, symbol: str, alert_type: str, message: str, severity: str):
        """Create a risk alert"""
        try:
            alert = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'type': alert_type,
                'message': message,
                'severity': severity,
                'category': 'RISK'
            }
            
            self.risk_alerts.append(alert)
            
            # Log the alert
            self.logger.warning(f"RISK ALERT [{severity}] {symbol}: {message}")
            
        except Exception as e:
            self.logger.error(f"Error creating risk alert: {e}")
    
    def create_performance_alert(self, symbol: str, alert_type: str, message: str, severity: str):
        """Create a performance alert"""
        try:
            alert = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'type': alert_type,
                'message': message,
                'severity': severity,
                'category': 'PERFORMANCE'
            }
            
            self.performance_alerts.append(alert)
            
            # Log the alert
            self.logger.info(f"PERFORMANCE ALERT [{severity}] {symbol}: {message}")
            
        except Exception as e:
            self.logger.error(f"Error creating performance alert: {e}")
    
    def should_send_alert(self, alert: Dict[str, Any]) -> bool:
        """Check if alert should be sent (rate limiting)"""
        try:
            alert_key = f"{alert['symbol']}_{alert['type']}"
            now = datetime.now()
            
            # Check if we've sent this alert recently (within 1 hour)
            if now - self.last_alert_time[alert_key] < timedelta(hours=1):
                return False
            
            # Check alert count (max 5 per hour per alert type)
            if self.alert_counters[alert_key] >= 5:
                return False
            
            # Update counters
            self.alert_counters[alert_key] += 1
            self.last_alert_time[alert_key] = now
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking alert rate limit: {e}")
            return False
    
    def send_alert(self, alert: Dict[str, Any]):
        """Send alert (placeholder for actual alert system)"""
        try:
            # In a real implementation, this would send alerts via:
            # - Email
            # - SMS
            # - Slack/Discord
            # - Push notifications
            # - Database logging
            
            self.logger.critical(f"ALERT SENT: {alert}")
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
    
    def cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            
            # Clean up old position history
            for symbol in list(self.position_history.keys()):
                self.position_history[symbol] = [
                    snapshot for snapshot in self.position_history[symbol]
                    if snapshot['timestamp'] > cutoff_time
                ]
                
                # Remove empty entries
                if not self.position_history[symbol]:
                    del self.position_history[symbol]
            
            # Clean up old alerts
            self.risk_alerts = [
                alert for alert in self.risk_alerts
                if alert['timestamp'] > cutoff_time
            ]
            
            self.performance_alerts = [
                alert for alert in self.performance_alerts
                if alert['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary"""
        try:
            positions = self.trading_state.get_all_positions()
            
            summary = {
                'monitor_status': 'RUNNING' if self.running else 'STOPPED',
                'monitor_interval': self.monitor_interval,
                'total_positions': len(positions),
                'open_positions': len([p for p in positions.values() if p.status == PositionStatus.OPEN]),
                'daily_performance': self.daily_performance.copy(),
                'recent_risk_alerts': len([a for a in self.risk_alerts if a['timestamp'] > datetime.now() - timedelta(hours=1)]),
                'recent_performance_alerts': len([a for a in self.performance_alerts if a['timestamp'] > datetime.now() - timedelta(hours=1)]),
                'position_summaries': {}
            }
            
            # Add position summaries
            for symbol, position in positions.items():
                if position.status == PositionStatus.OPEN:
                    summary['position_summaries'][symbol] = {
                        'net_delta': position.net_delta,
                        'net_theta': position.net_theta,
                        'net_vega': position.net_vega,
                        'current_pnl': position.current_pnl,
                        'days_held': (datetime.now() - position.entry_time).days,
                        'hedge_quantity': position.hedge_quantity
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting monitoring summary: {e}")
            return {}
    
    def get_position_analytics(self, symbol: str) -> Dict[str, Any]:
        """Get detailed analytics for a specific position"""
        try:
            if symbol not in self.position_history:
                return {}
            
            history = self.position_history[symbol]
            if not history:
                return {}
            
            # Calculate analytics
            deltas = [snapshot['net_delta'] for snapshot in history]
            thetas = [snapshot['net_theta'] for snapshot in history]
            vegas = [snapshot['net_vega'] for snapshot in history]
            pnls = [snapshot['current_pnl'] for snapshot in history]
            
            analytics = {
                'symbol': symbol,
                'data_points': len(history),
                'time_range': {
                    'start': history[0]['timestamp'].isoformat(),
                    'end': history[-1]['timestamp'].isoformat()
                },
                'delta_analytics': {
                    'current': deltas[-1] if deltas else 0,
                    'average': sum(deltas) / len(deltas) if deltas else 0,
                    'min': min(deltas) if deltas else 0,
                    'max': max(deltas) if deltas else 0,
                    'volatility': self.calculate_volatility(deltas) if len(deltas) > 1 else 0
                },
                'theta_analytics': {
                    'current': thetas[-1] if thetas else 0,
                    'average': sum(thetas) / len(thetas) if thetas else 0,
                    'min': min(thetas) if thetas else 0,
                    'max': max(thetas) if thetas else 0
                },
                'vega_analytics': {
                    'current': vegas[-1] if vegas else 0,
                    'average': sum(vegas) / len(vegas) if vegas else 0,
                    'min': min(vegas) if vegas else 0,
                    'max': max(vegas) if vegas else 0
                },
                'pnl_analytics': {
                    'current': pnls[-1] if pnls else 0,
                    'average': sum(pnls) / len(pnls) if pnls else 0,
                    'min': min(pnls) if pnls else 0,
                    'max': max(pnls) if pnls else 0,
                    'total_return': ((pnls[-1] - pnls[0]) / abs(pnls[0]) * 100) if pnls and pnls[0] != 0 else 0
                }
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting position analytics for {symbol}: {e}")
            return {}
    
    def calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility of a series of values"""
        try:
            if len(values) < 2:
                return 0.0
            
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            return variance ** 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def stop(self):
        """Stop the position monitor"""
        self.logger.info("Stopping Calendar Position Monitor...")
        self.running = False
        time.sleep(1)  # Give time for current monitoring to complete
