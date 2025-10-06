"""
Simple Data Viewer for Historical Data
Shows summary and details of fetched historical data
"""

import json
import pandas as pd
from typing import Dict, List
from datetime import datetime

class DataViewer:
    def __init__(self, data_file: str = "calendar_spread/historical_data.json"):
        self.data_file = data_file
        self.data = self.load_data()
    
    def load_data(self) -> Dict:
        """Load historical data from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return {}
    
    def get_data_summary(self) -> Dict:
        """Get summary of fetched data"""
        try:
            summary = {
                'total_stocks': len(self.data),
                'stocks_with_underlying': 0,
                'stocks_with_options': 0,
                'stocks_with_complete_data': 0,
                'date_ranges': {}
            }
            
            for symbol, stock_data in self.data.items():
                has_underlying = stock_data.get('underlying_data') is not None
                has_options = stock_data.get('options_data') is not None
                
                if has_underlying:
                    summary['stocks_with_underlying'] += 1
                
                if has_options:
                    summary['stocks_with_options'] += 1
                
                if has_underlying and has_options:
                    summary['stocks_with_complete_data'] += 1
                
                # Get date range for underlying data
                if has_underlying and 'underlying_data' in stock_data:
                    underlying_data = stock_data['underlying_data']
                    if 'dates' in underlying_data and underlying_data['dates']:
                        dates = underlying_data['dates']
                        summary['date_ranges'][symbol] = {
                            'start': min(dates),
                            'end': max(dates),
                            'days': len(dates)
                        }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error getting data summary: {e}")
            return {}
    
    def print_detailed_summary(self):
        """Print detailed summary of the data"""
        summary = self.get_data_summary()
        
        print("=" * 80)
        print("📊 HISTORICAL DATA FETCH RESULTS")
        print("=" * 80)
        
        print(f"Total stocks processed: {summary['total_stocks']}")
        print(f"Stocks with underlying data: {summary['stocks_with_underlying']}")
        print(f"Stocks with options data: {summary['stocks_with_options']}")
        print(f"Stocks with complete data: {summary['stocks_with_complete_data']}")
        print(f"Success rate: {summary['stocks_with_complete_data']/summary['total_stocks']*100:.1f}%")
        
        print(f"\n📈 DETAILED BREAKDOWN:")
        print("-" * 80)
        
        for symbol, stock_data in self.data.items():
            company_name = stock_data.get('company_name', 'Unknown')
            atm_strike = stock_data.get('atm_strike', 'N/A')
            
            # Underlying data info
            underlying_days = 0
            if stock_data.get('underlying_data'):
                underlying_days = len(stock_data['underlying_data'].get('dates', []))
            
            # Options data info
            ce_days = 0
            pe_days = 0
            if stock_data.get('options_data'):
                ce_data = stock_data['options_data'].get('CE', {})
                pe_data = stock_data['options_data'].get('PE', {})
                ce_days = len(ce_data.get('dates', []))
                pe_days = len(pe_data.get('dates', []))
            
            print(f"{company_name:<25} ({symbol:<12}) | Strike: {atm_strike:<6} | Underlying: {underlying_days:2d}d | CE: {ce_days:2d}d | PE: {pe_days:2d}d")
        
        # Show date ranges
        if summary['date_ranges']:
            print(f"\n📅 DATE RANGES:")
            print("-" * 80)
            for symbol, date_info in summary['date_ranges'].items():
                print(f"{symbol:<12}: {date_info['start']} to {date_info['end']} ({date_info['days']} days)")
        
        print("=" * 80)
    
    def show_sample_data(self, symbol: str = None):
        """Show sample data for a specific stock or first available stock"""
        if not symbol:
            # Get first available stock
            available_stocks = [s for s, data in self.data.items() 
                              if data.get('underlying_data') and data.get('options_data')]
            if not available_stocks:
                print("❌ No stocks with complete data found")
                return
            symbol = available_stocks[0]
        
        if symbol not in self.data:
            print(f"❌ Stock {symbol} not found")
            return
        
        stock_data = self.data[symbol]
        company_name = stock_data.get('company_name', 'Unknown')
        
        print(f"\n🔍 SAMPLE DATA FOR {company_name} ({symbol}):")
        print("=" * 60)
        
        # Show underlying data sample
        if stock_data.get('underlying_data'):
            underlying = stock_data['underlying_data']
            print(f"📈 UNDERLYING DATA ({len(underlying['dates'])} days):")
            print(f"   Date Range: {underlying['dates'][0]} to {underlying['dates'][-1]}")
            print(f"   Latest Close: ₹{underlying['close'][-1]:.2f}")
            print(f"   Latest Volume: {underlying['volume'][-1]:,}")
        
        # Show options data sample
        if stock_data.get('options_data'):
            options = stock_data['options_data']
            print(f"\n📊 OPTIONS DATA:")
            
            if 'CE' in options:
                ce_data = options['CE']
                print(f"   CE Options ({len(ce_data['dates'])} days):")
                print(f"   Latest Close: ₹{ce_data['close'][-1]:.2f}")
                print(f"   Latest Volume: {ce_data['volume'][-1]:,}")
            
            if 'PE' in options:
                pe_data = options['PE']
                print(f"   PE Options ({len(pe_data['dates'])} days):")
                print(f"   Latest Close: ₹{pe_data['close'][-1]:.2f}")
                print(f"   Latest Volume: {pe_data['volume'][-1]:,}")
        
        print("=" * 60)

def main():
    """Main function to view data"""
    viewer = DataViewer()
    
    if not viewer.data:
        print("❌ No data found. Run historical_data_fetcher.py first.")
        return
    
    # Print detailed summary
    viewer.print_detailed_summary()
    
    # Show sample data
    viewer.show_sample_data()

if __name__ == "__main__":
    main()
