"""
Calendar Spread Strategy - Backtest Runner
Simple script to run backtests with historical data
"""
import os
import sys
import logging
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calendar_spread.config import CalendarConfig
from calendar_spread.backtest import CalendarBacktest
from calendar_spread.historical_data_manager import HistoricalDataManager
from calendar_spread.results_analyzer import ResultsAnalyzer

def setup_logging():
    """Setup logging for backtesting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/backtest.log', encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def download_historical_data(symbols: list, start_date: datetime, end_date: datetime):
    """Download historical data for backtesting"""
    try:
        print("📥 Downloading historical data...")
        
        # Create data manager
        config = CalendarConfig()
        data_manager = HistoricalDataManager(config)
        
        # Download data for all symbols
        results = data_manager.download_multiple_symbols(symbols, start_date, end_date)
        
        print(f"✅ Downloaded data for {len(results)} symbols:")
        for symbol, data in results.items():
            print(f"   - {symbol}: {len(data)} records")
        
        return data_manager
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

def run_backtest(symbols: list, start_date: datetime, end_date: datetime, config: CalendarConfig):
    """Run the calendar spread backtest"""
    try:
        print("\n🚀 Starting Calendar Spread Backtest...")
        print(f"   Symbols: {', '.join(symbols)}")
        print(f"   Period: {start_date.date()} to {end_date.date()}")
        print(f"   Initial Capital: ₹1,00,000")
        
        # Create backtest instance
        backtest = CalendarBacktest(config, start_date, end_date)
        
        # Run backtest
        results = backtest.run_backtest(symbols)
        
        print("✅ Backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return None

def analyze_results(results: dict):
    """Analyze backtest results"""
    try:
        print("\n📊 Analyzing results...")
        
        # Create analyzer
        analyzer = ResultsAnalyzer()
        
        # Analyze results
        analysis = analyzer.analyze_results(results)
        
        # Print comprehensive analysis
        analyzer.print_analysis(analysis)
        
        # Export analysis
        analyzer.export_analysis(analysis, "backtest_analysis.json")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return None

def main():
    """Main backtest execution"""
    try:
        # Setup logging
        setup_logging()
        os.makedirs("logs", exist_ok=True)
        
        print("🎯 Calendar Spread Strategy - Backtest Runner")
        print("=" * 60)
        
        # Configuration
        config = CalendarConfig()
        
        # Backtest parameters
        symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']  # Add your stocks here
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 6, 30)
        
        print(f"📋 Configuration:")
        print(f"   Strategy: {config.STRATEGY_NAME}")
        print(f"   Max Position Size: {config.MAX_POSITION_SIZE * 100}%")
        print(f"   Profit Target: {config.PROFIT_TARGET * 100}%")
        print(f"   Stop Loss: {config.MAX_LOSS_MULTIPLIER}x")
        print(f"   Short Expiry: {config.SHORT_EXPIRY_DAYS} days")
        print(f"   Long Expiry: {config.LONG_EXPIRY_DAYS} days")
        
        # Step 1: Download historical data
        data_manager = download_historical_data(symbols, start_date, end_date)
        if not data_manager:
            return
        
        # Step 2: Run backtest
        results = run_backtest(symbols, start_date, end_date, config)
        if not results:
            return
        
        # Step 3: Analyze results
        analysis = analyze_results(results)
        if not analysis:
            return
        
        # Step 4: Summary
        print("\n🎉 Backtest Summary:")
        print("=" * 60)
        
        overview = analysis.get('overview', {})
        performance = analysis.get('performance', {})
        
        print(f"Total Return: {overview.get('total_return_pct', 0):.2f}%")
        print(f"Final Value: ₹{overview.get('final_portfolio_value', 0):,.2f}")
        print(f"Net Profit: ₹{overview.get('net_profit', 0):,.2f}")
        print(f"Max Drawdown: {analysis.get('risk_analysis', {}).get('max_drawdown', 0):.2f}%")
        print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"Win Rate: {overview.get('win_rate', 0):.2f}%")
        print(f"Total Trades: {overview.get('total_trades', 0)}")
        
        print("\n📁 Files Generated:")
        print("   - logs/backtest.log (detailed logs)")
        print("   - backtest_analysis.json (analysis results)")
        print("   - historical_data/ (downloaded data)")
        
        print("\n✅ Backtest completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Backtest interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logging.error(f"Fatal error in main: {e}")

if __name__ == "__main__":
    main()
