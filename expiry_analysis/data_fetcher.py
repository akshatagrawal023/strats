import os
import sys
import json
import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.historical_data import hist_df, hist_data
from utils.fyers_instance import FyersInstance

class DataFetcher:
    def __init__(self, backtest_data_file: str = "nifty50_backtest_data.json"):
        self.logger = logging.getLogger("CalendarSpreadDataFetcher")
        
        # Load backtest data
        self.backtest_data = self.load_backtest_data()
        
        # Initialize Fyers instance
        try:
            self.fyers = FyersInstance.get_instance()
            self.logger.info("✅ Connected to Fyers API")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Fyers API: {e}")
            raise
    
    def load_backtest_data(self) -> List[Dict]:
        """Load the backtest data from JSON file"""
        try:
            with open(self.backtest_data_file, 'r') as f:
                data = json.load(f)
            self.logger.info(f"📊 Loaded {len(data)} stocks for backtesting")
            return data
        except Exception as e:
            self.logger.error(f"❌ Error loading backtest data: {e}")
            return []
    
    def fetch_underlying_data(self, symbol: str, nse_symbol: str, timeframe: str, resolution: str = "D") -> Optional[pd.DataFrame]:
        """Fetch historical data for underlying asset"""
        try:
            self.logger.info(f"📈 Fetching underlying data for {symbol}...")
            # "01/01/2024 // 10/02/2024"
            # Use existing hist_df function
            df = hist_df(nse_symbol, timeframe, resolution=resolution)
            
            if df is not None and not df.empty:
                # Add metadata
                df['symbol'] = symbol
                df['nse_symbol'] = nse_symbol
                df['asset_type'] = 'underlying'
                
                self.logger.info(f"✅ Got {len(df)} days of underlying data for {symbol}")
                return df
            else:
                self.logger.warning(f"⚠️ No underlying data for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching underlying data for {symbol}: {e}")
            return None
    
    def fetch_options_data(self, ce_symbol: str, pe_symbol: str, timeframe: str, resolution: str = "D") -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch historical data for both CE and PE options"""
        try:
            self.logger.info(f"📊 Fetching options data for {ce_symbol} and {pe_symbol}...")
            
            options_data = {}
            
            # Fetch CE data
            try:
                ce_df = hist_df(ce_symbol, timeframe, resolution=resolution)
                if ce_df is not None and not ce_df.empty:
                    ce_df['option_type'] = 'CE'
                    ce_df['option_symbol'] = ce_symbol
                    ce_df['asset_type'] = 'option'
                    options_data['CE'] = ce_df
                    self.logger.info(f"✅ Got {len(ce_df)} days of CE data")
                else:
                    self.logger.warning(f"⚠️ No CE data for {ce_symbol}")
            except Exception as e:
                self.logger.warning(f"⚠️ Error fetching CE data for {ce_symbol}: {e}")
            
            # Fetch PE data
            try:
                pe_df = hist_df(pe_symbol, timeframe, resolution=resolution)
                if pe_df is not None and not pe_df.empty:
                    pe_df['option_type'] = 'PE'
                    pe_df['option_symbol'] = pe_symbol
                    pe_df['asset_type'] = 'option'
                    options_data['PE'] = pe_df
                    self.logger.info(f"✅ Got {len(pe_df)} days of PE data")
                else:
                    self.logger.warning(f"⚠️ No PE data for {pe_symbol}")
            except Exception as e:
                self.logger.warning(f"⚠️ Error fetching PE data for {pe_symbol}: {e}")
            
            return options_data if options_data else None
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching options data: {e}")
            return None
    
    def fetch_all_data(self, timeframe: str, resolution: str = "D", delay: float = 1.0) -> Dict:
        """Fetch historical data for all stocks in backtest data"""
        try:
            all_data = {}
            total_stocks = len(self.backtest_data)
            
            self.logger.info(f"🚀 Starting data fetch for {total_stocks} stocks...")
            self.logger.info(f"📅 Timeframe: {timeframe}")
            self.logger.info(f"⏱️ Resolution: {resolution}")
            
            for i, stock in enumerate(self.backtest_data, 1):
                symbol = stock['symbol']
                nse_symbol = stock['nse_symbol']
                ce_symbol = stock['ce_symbol']
                pe_symbol = stock['pe_symbol']
                
                self.logger.info(f"\n📊 Processing {i}/{total_stocks}: {stock['company_name']} ({symbol})")
                
                # Fetch underlying data
                underlying_data = self.fetch_underlying_data(symbol, nse_symbol, timeframe, resolution)
                
                # Fetch options data
                options_data = self.fetch_options_data(ce_symbol, pe_symbol, timeframe, resolution)
                
                # Store data
                all_data[symbol] = {
                    'company_name': stock['company_name'],
                    'underlying_data': underlying_data,
                    'options_data': options_data,
                    'atm_strike': stock['atm_strike'],
                    'ce_symbol': ce_symbol,
                    'pe_symbol': pe_symbol,
                    'underlying_price': stock.get('underlying_price', 0)
                }
                
                # Add delay to avoid rate limiting
                if i < total_stocks:
                    time.sleep(delay)
            
            self.logger.info(f"\n✅ Completed data fetch for {len(all_data)} stocks")
            return all_data
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching all data: {e}")
            return {}
    
    def save_data(self, data: Dict, output_file: str = "calendar_spread/historical_data.json"):
        """Save historical data to JSON file"""
        try:
            # Convert DataFrames to dictionaries for JSON serialization
            json_data = {}
            
            for symbol, stock_data in data.items():
                json_data[symbol] = {
                    'company_name': stock_data['company_name'],
                    'atm_strike': stock_data['atm_strike'],
                    'ce_symbol': stock_data['ce_symbol'],
                    'pe_symbol': stock_data['pe_symbol'],
                    'underlying_price': stock_data['underlying_price']
                }
                
                # Convert underlying data
                if stock_data['underlying_data'] is not None:
                    underlying_df = stock_data['underlying_data']
                    json_data[symbol]['underlying_data'] = {
                        'dates': underlying_df['Time'].dt.strftime('%Y-%m-%d').tolist(),
                        'open': underlying_df['Open'].tolist(),
                        'high': underlying_df['High'].tolist(),
                        'low': underlying_df['Low'].tolist(),
                        'close': underlying_df['Close'].tolist(),
                        'volume': underlying_df['Volume'].tolist()
                    }
                
                # Convert options data
                if stock_data['options_data']:
                    json_data[symbol]['options_data'] = {}
                    
                    for option_type, option_df in stock_data['options_data'].items():
                        json_data[symbol]['options_data'][option_type] = {
                            'dates': option_df['Time'].dt.strftime('%Y-%m-%d').tolist(),
                            'open': option_df['Open'].tolist(),
                            'high': option_df['High'].tolist(),
                            'low': option_df['Low'].tolist(),
                            'close': option_df['Close'].tolist(),
                            'volume': option_df['Volume'].tolist()
                        }
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            self.logger.info(f"💾 Saved data to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving data: {e}")
            return False
    
    def get_data_summary(self, data: Dict) -> Dict:
        """Get summary of fetched data"""
        try:
            summary = {
                'total_stocks': len(data),
                'stocks_with_underlying': 0,
                'stocks_with_options': 0,
                'stocks_with_complete_data': 0,
                'date_ranges': {}
            }
            
            for symbol, stock_data in data.items():
                has_underlying = stock_data.get('underlying_data') is not None
                has_options = stock_data.get('options_data') is not None
                
                if has_underlying:
                    summary['stocks_with_underlying'] += 1
                
                if has_options:
                    summary['stocks_with_options'] += 1
                
                if has_underlying and has_options:
                    summary['stocks_with_complete_data'] += 1
                
                # Get date range for underlying data
                if has_underlying:
                    underlying_dates = stock_data['underlying_data']['dates']
                    if underlying_dates:
                        summary['date_ranges'][symbol] = {
                            'start': min(underlying_dates),
                            'end': max(underlying_dates),
                            'days': len(underlying_dates)
                        }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error getting data summary: {e}")
            return {}

def main():
    """Main function to fetch historical data"""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create fetcher
        fetcher = CalendarSpreadDataFetcher()
        
        if not fetcher.backtest_data:
            print("❌ No backtest data found. Run nifty50_processor.py first.")
            return
        
        # Define timeframe (last 3 months)
        timeframe = "90d"  # Using your existing format
        
        print(f"🚀 Starting historical data fetch...")
        print(f"📅 Timeframe: {timeframe}")
        print(f"📊 Stocks to process: {len(fetcher.backtest_data)}")
        
        # Fetch all data
        historical_data = fetcher.fetch_all_data(timeframe, resolution="D", delay=1.0)
        
        if historical_data:
            # Save data
            fetcher.save_data(historical_data)
            
            # Show summary
            summary = fetcher.get_data_summary(historical_data)
            print(f"\n📈 DATA FETCH SUMMARY:")
            print(f"   Total stocks: {summary['total_stocks']}")
            print(f"   With underlying data: {summary['stocks_with_underlying']}")
            print(f"   With options data: {summary['stocks_with_options']}")
            print(f"   With complete data: {summary['stocks_with_complete_data']}")
            
            print(f"\n✅ Historical data fetch completed!")
            print(f"💾 Data saved to: calendar_spread/historical_data.json")
        else:
            print("❌ No data fetched")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
