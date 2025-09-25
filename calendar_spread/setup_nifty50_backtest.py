"""
Simple Setup for Nifty 50 Calendar Spread Backtest
Reads NiftyFifty.csv, gets lot sizes from FnO_lot_structured.csv, and sets up backtesting
"""
import os
import sys
import logging
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calendar_spread.nifty50_processor import Nifty50Processor

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Main setup function"""
    try:
        setup_logging()
        
        print("🎯 Nifty 50 Calendar Spread Strategy Setup")
        print("=" * 60)
        
        # Initialize processor
        processor = Nifty50Processor()
        
        # Print summary
        processor.print_summary()
        
        # Export data for backtesting
        print("\n📊 Exporting data for backtesting...")
        backtest_data = processor.export_for_backtest("nifty50_backtest_data.json")
        
        if backtest_data:
            print(f"\n✅ Setup completed successfully!")
            print(f"📊 Ready to backtest {len(backtest_data)} Nifty 50 stocks")
            print(f"📁 Data exported to: nifty50_backtest_data.json")
            
            print(f"\n🚀 Next Steps:")
            print(f"1. Review the exported data in nifty50_backtest_data.json")
            print(f"2. Run your backtesting with the selected stocks")
            print(f"3. Use the ATM strikes and lot sizes provided")
            
            # Show sample data
            print(f"\n📋 Sample Backtest Data:")
            for i, stock in enumerate(backtest_data[:5], 1):
                print(f"   {i}. {stock['company_name']} ({stock['symbol']})")
                print(f"      Strike: {stock['atm_strike']}, Price: {stock['underlying_price']}")
                print(f"      CE: {stock['ce_symbol']}")
                print(f"      PE: {stock['pe_symbol']}")
                print(f"      Lot Size: {stock['lot_size']}")
                print()
        else:
            print(f"\n❌ Setup failed - no data exported!")
        
    except Exception as e:
        print(f"❌ Error in setup: {e}")
        logging.error(f"Setup error: {e}")

if __name__ == "__main__":
    main()