"""
Results Analyzer for Calendar Spread Strategy Backtesting
Analyzes backtest results and provides performance insights
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ResultsAnalyzer:
    """
    Analyzes backtest results and provides comprehensive performance insights
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ResultsAnalyzer")
        
        # Analysis results
        self.analysis_results = {}
        
        # Performance metrics
        self.performance_metrics = {}
        
        # Risk metrics
        self.risk_metrics = {}
    
    def analyze_results(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze backtest results comprehensively
        
        Args:
            backtest_results: Results from CalendarBacktest.run_backtest()
            
        Returns:
            Comprehensive analysis results
        """
        try:
            self.logger.info("Starting comprehensive results analysis...")
            
            # Extract data
            trades = backtest_results.get('trades', [])
            daily_pnl = backtest_results.get('daily_pnl', [])
            
            # Perform various analyses
            analysis = {
                'overview': self.analyze_overview(backtest_results),
                'performance': self.analyze_performance(backtest_results),
                'risk_analysis': self.analyze_risk(backtest_results),
                'trade_analysis': self.analyze_trades(trades),
                'drawdown_analysis': self.analyze_drawdowns(daily_pnl),
                'monthly_analysis': self.analyze_monthly_performance(daily_pnl),
                'recommendations': self.generate_recommendations(backtest_results)
            }
            
            self.analysis_results = analysis
            self.logger.info("Results analysis completed successfully")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing results: {e}")
            return {}
    
    def analyze_overview(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze basic overview metrics"""
        try:
            total_return = results.get('total_return', 0)
            max_drawdown = results.get('max_drawdown', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)
            win_rate = results.get('win_rate', 0)
            total_trades = results.get('total_trades', 0)
            final_portfolio_value = results.get('final_portfolio_value', 100000)
            
            return {
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'final_portfolio_value': final_portfolio_value,
                'initial_portfolio_value': 100000,
                'net_profit': final_portfolio_value - 100000
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing overview: {e}")
            return {}
    
    def analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        try:
            daily_pnl = results.get('daily_pnl', [])
            
            if not daily_pnl:
                return {}
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(daily_pnl)
            df['date'] = pd.to_datetime(df['date'])
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
            # Calculate performance metrics
            total_return = df['cumulative_pnl'].iloc[-1] / 100000  # Assuming 1L initial
            
            # Annualized return
            days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
            annualized_return = (1 + total_return) ** (365 / days) - 1
            
            # Volatility
            daily_returns = df['pnl'] / 100000
            volatility = daily_returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = 0.06  # 6% risk-free rate
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = daily_returns[daily_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            
            # Calmar ratio
            max_drawdown = results.get('max_drawdown', 0)
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'best_day': df['pnl'].max(),
                'worst_day': df['pnl'].min(),
                'avg_daily_return': df['pnl'].mean(),
                'positive_days': (df['pnl'] > 0).sum(),
                'negative_days': (df['pnl'] < 0).sum(),
                'win_rate_daily': (df['pnl'] > 0).mean() * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
            return {}
    
    def analyze_risk(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk metrics"""
        try:
            daily_pnl = results.get('daily_pnl', [])
            
            if not daily_pnl:
                return {}
            
            df = pd.DataFrame(daily_pnl)
            df['date'] = pd.to_datetime(df['date'])
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
            # Calculate risk metrics
            daily_returns = df['pnl'] / 100000
            
            # Value at Risk (VaR)
            var_95 = np.percentile(daily_returns, 5)
            var_99 = np.percentile(daily_returns, 1)
            
            # Expected Shortfall (CVaR)
            cvar_95 = daily_returns[daily_returns <= var_95].mean()
            cvar_99 = daily_returns[daily_returns <= var_99].mean()
            
            # Maximum drawdown analysis
            df['running_max'] = df['cumulative_pnl'].expanding().max()
            df['drawdown'] = df['cumulative_pnl'] - df['running_max']
            df['drawdown_pct'] = df['drawdown'] / df['running_max'] * 100
            
            max_drawdown = df['drawdown_pct'].min()
            max_drawdown_duration = self.calculate_max_drawdown_duration(df)
            
            # Tail risk
            tail_ratio = abs(daily_returns[daily_returns < var_95].mean()) / abs(daily_returns[daily_returns > np.percentile(daily_returns, 95)].mean())
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': max_drawdown_duration,
                'tail_ratio': tail_ratio,
                'volatility': daily_returns.std() * np.sqrt(252),
                'skewness': daily_returns.skew(),
                'kurtosis': daily_returns.kurtosis()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk: {e}")
            return {}
    
    def analyze_trades(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze individual trades"""
        try:
            if not trades:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Filter exit trades
            exit_trades = df[df['action'] == 'exit'].copy()
            
            if exit_trades.empty:
                return {}
            
            # Calculate trade metrics
            winning_trades = exit_trades[exit_trades['pnl'] > 0]
            losing_trades = exit_trades[exit_trades['pnl'] < 0]
            
            avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
            
            win_rate = len(winning_trades) / len(exit_trades) * 100
            
            # Profit factor
            total_wins = winning_trades['pnl'].sum() if not winning_trades.empty else 0
            total_losses = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Average trade duration
            exit_trades['entry_date'] = pd.to_datetime(exit_trades['date'])
            # This would need entry dates from the trades data
            
            return {
                'total_trades': len(exit_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': exit_trades['pnl'].max(),
                'largest_loss': exit_trades['pnl'].min(),
                'total_wins': total_wins,
                'total_losses': total_losses,
                'profit_factor': profit_factor,
                'expectancy': (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trades: {e}")
            return {}
    
    def analyze_drawdowns(self, daily_pnl: List[Dict]) -> Dict[str, Any]:
        """Analyze drawdown patterns"""
        try:
            if not daily_pnl:
                return {}
            
            df = pd.DataFrame(daily_pnl)
            df['date'] = pd.to_datetime(df['date'])
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
            # Calculate drawdowns
            df['running_max'] = df['cumulative_pnl'].expanding().max()
            df['drawdown'] = df['cumulative_pnl'] - df['running_max']
            df['drawdown_pct'] = df['drawdown'] / df['running_max'] * 100
            
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_date = None
            
            for idx, row in df.iterrows():
                if row['drawdown_pct'] < -1:  # 1% threshold
                    if not in_drawdown:
                        in_drawdown = True
                        start_date = row['date']
                else:
                    if in_drawdown:
                        in_drawdown = False
                        end_date = row['date']
                        duration = (end_date - start_date).days
                        max_dd = df.loc[start_date:end_date, 'drawdown_pct'].min()
                        drawdown_periods.append({
                            'start': start_date,
                            'end': end_date,
                            'duration': duration,
                            'max_drawdown': max_dd
                        })
            
            # Calculate drawdown statistics
            if drawdown_periods:
                durations = [dd['duration'] for dd in drawdown_periods]
                max_drawdowns = [dd['max_drawdown'] for dd in drawdown_periods]
                
                return {
                    'total_drawdowns': len(drawdown_periods),
                    'avg_drawdown_duration': np.mean(durations),
                    'max_drawdown_duration': max(durations),
                    'avg_drawdown_magnitude': np.mean(max_drawdowns),
                    'max_drawdown_magnitude': min(max_drawdowns),
                    'drawdown_periods': drawdown_periods
                }
            else:
                return {
                    'total_drawdowns': 0,
                    'avg_drawdown_duration': 0,
                    'max_drawdown_duration': 0,
                    'avg_drawdown_magnitude': 0,
                    'max_drawdown_magnitude': 0,
                    'drawdown_periods': []
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing drawdowns: {e}")
            return {}
    
    def analyze_monthly_performance(self, daily_pnl: List[Dict]) -> Dict[str, Any]:
        """Analyze monthly performance"""
        try:
            if not daily_pnl:
                return {}
            
            df = pd.DataFrame(daily_pnl)
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M')
            
            # Group by month
            monthly_pnl = df.groupby('month')['pnl'].sum()
            monthly_returns = monthly_pnl / 100000  # Assuming 1L initial
            
            return {
                'best_month': monthly_returns.max(),
                'worst_month': monthly_returns.min(),
                'avg_monthly_return': monthly_returns.mean(),
                'monthly_volatility': monthly_returns.std(),
                'positive_months': (monthly_returns > 0).sum(),
                'negative_months': (monthly_returns < 0).sum(),
                'monthly_win_rate': (monthly_returns > 0).mean() * 100,
                'monthly_returns': monthly_returns.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing monthly performance: {e}")
            return {}
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate strategy recommendations"""
        try:
            recommendations = []
            
            # Performance-based recommendations
            total_return = results.get('total_return', 0)
            max_drawdown = results.get('max_drawdown', 0)
            win_rate = results.get('win_rate', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)
            
            if total_return < 0:
                recommendations.append("Strategy shows negative returns. Consider reviewing entry criteria and risk management.")
            
            if abs(max_drawdown) > 0.20:  # 20% drawdown
                recommendations.append("High maximum drawdown detected. Consider reducing position sizes or improving stop-loss mechanisms.")
            
            if win_rate < 40:
                recommendations.append("Low win rate. Consider improving entry timing or exit strategies.")
            
            if sharpe_ratio < 1.0:
                recommendations.append("Low Sharpe ratio. Consider optimizing risk-adjusted returns.")
            
            if total_return > 0.20 and sharpe_ratio > 1.5:
                recommendations.append("Strong performance metrics. Strategy shows promise for live trading.")
            
            # Risk-based recommendations
            if abs(max_drawdown) < 0.10 and win_rate > 60:
                recommendations.append("Excellent risk management. Consider increasing position sizes gradually.")
            
            # General recommendations
            recommendations.append("Consider paper trading before live implementation.")
            recommendations.append("Monitor strategy performance regularly and adjust parameters as needed.")
            recommendations.append("Ensure proper position sizing and risk management in live trading.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def calculate_max_drawdown_duration(self, df: pd.DataFrame) -> int:
        """Calculate maximum drawdown duration in days"""
        try:
            # Find the longest continuous drawdown period
            in_drawdown = df['drawdown_pct'] < -1  # 1% threshold
            
            max_duration = 0
            current_duration = 0
            
            for is_dd in in_drawdown:
                if is_dd:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
            
            return max_duration
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown duration: {e}")
            return 0
    
    def print_analysis(self, analysis: Dict[str, Any]):
        """Print comprehensive analysis results"""
        try:
            print("\n" + "=" * 80)
            print("CALENDAR SPREAD STRATEGY - BACKTEST ANALYSIS")
            print("=" * 80)
            
            # Overview
            overview = analysis.get('overview', {})
            print("\n📊 OVERVIEW")
            print("-" * 40)
            print(f"Total Return: {overview.get('total_return_pct', 0):.2f}%")
            print(f"Final Portfolio Value: ₹{overview.get('final_portfolio_value', 0):,.2f}")
            print(f"Net Profit: ₹{overview.get('net_profit', 0):,.2f}")
            print(f"Total Trades: {overview.get('total_trades', 0)}")
            print(f"Win Rate: {overview.get('win_rate', 0):.2f}%")
            
            # Performance
            performance = analysis.get('performance', {})
            print("\n📈 PERFORMANCE METRICS")
            print("-" * 40)
            print(f"Annualized Return: {performance.get('annualized_return', 0) * 100:.2f}%")
            print(f"Volatility: {performance.get('volatility', 0) * 100:.2f}%")
            print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
            print(f"Sortino Ratio: {performance.get('sortino_ratio', 0):.2f}")
            print(f"Calmar Ratio: {performance.get('calmar_ratio', 0):.2f}")
            print(f"Best Day: ₹{performance.get('best_day', 0):,.2f}")
            print(f"Worst Day: ₹{performance.get('worst_day', 0):,.2f}")
            
            # Risk Analysis
            risk = analysis.get('risk_analysis', {})
            print("\n⚠️ RISK ANALYSIS")
            print("-" * 40)
            print(f"Max Drawdown: {risk.get('max_drawdown', 0):.2f}%")
            print(f"Max Drawdown Duration: {risk.get('max_drawdown_duration', 0)} days")
            print(f"VaR (95%): {risk.get('var_95', 0) * 100:.2f}%")
            print(f"VaR (99%): {risk.get('var_99', 0) * 100:.2f}%")
            print(f"CVaR (95%): {risk.get('cvar_95', 0) * 100:.2f}%")
            print(f"Skewness: {risk.get('skewness', 0):.2f}")
            print(f"Kurtosis: {risk.get('kurtosis', 0):.2f}")
            
            # Trade Analysis
            trades = analysis.get('trade_analysis', {})
            print("\n💼 TRADE ANALYSIS")
            print("-" * 40)
            print(f"Total Trades: {trades.get('total_trades', 0)}")
            print(f"Winning Trades: {trades.get('winning_trades', 0)}")
            print(f"Losing Trades: {trades.get('losing_trades', 0)}")
            print(f"Win Rate: {trades.get('win_rate', 0):.2f}%")
            print(f"Average Win: ₹{trades.get('avg_win', 0):,.2f}")
            print(f"Average Loss: ₹{trades.get('avg_loss', 0):,.2f}")
            print(f"Profit Factor: {trades.get('profit_factor', 0):.2f}")
            print(f"Expectancy: ₹{trades.get('expectancy', 0):,.2f}")
            
            # Monthly Analysis
            monthly = analysis.get('monthly_analysis', {})
            print("\n📅 MONTHLY ANALYSIS")
            print("-" * 40)
            print(f"Best Month: {monthly.get('best_month', 0) * 100:.2f}%")
            print(f"Worst Month: {monthly.get('worst_month', 0) * 100:.2f}%")
            print(f"Average Monthly Return: {monthly.get('avg_monthly_return', 0) * 100:.2f}%")
            print(f"Monthly Win Rate: {monthly.get('monthly_win_rate', 0):.2f}%")
            print(f"Positive Months: {monthly.get('positive_months', 0)}")
            print(f"Negative Months: {monthly.get('negative_months', 0)}")
            
            # Recommendations
            recommendations = analysis.get('recommendations', [])
            print("\n💡 RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
            
            print("\n" + "=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error printing analysis: {e}")
    
    def export_analysis(self, analysis: Dict[str, Any], filename: str = "backtest_analysis.json"):
        """Export analysis results to JSON file"""
        try:
            import json
            
            with open(filename, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            self.logger.info(f"Analysis exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis: {e}")

def main():
    """Test the results analyzer"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create analyzer
        analyzer = ResultsAnalyzer()
        
        # Mock results for testing
        mock_results = {
            'total_return': 0.15,
            'max_drawdown': -0.08,
            'sharpe_ratio': 1.2,
            'win_rate': 65.0,
            'total_trades': 25,
            'final_portfolio_value': 115000,
            'trades': [
                {'action': 'exit', 'pnl': 1000, 'date': '2024-01-15'},
                {'action': 'exit', 'pnl': -500, 'date': '2024-01-20'},
                {'action': 'exit', 'pnl': 800, 'date': '2024-02-01'},
            ],
            'daily_pnl': [
                {'date': '2024-01-01', 'pnl': 100, 'portfolio_value': 100100},
                {'date': '2024-01-02', 'pnl': -50, 'portfolio_value': 100050},
                {'date': '2024-01-03', 'pnl': 200, 'portfolio_value': 100250},
            ]
        }
        
        # Analyze results
        analysis = analyzer.analyze_results(mock_results)
        
        # Print analysis
        analyzer.print_analysis(analysis)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
