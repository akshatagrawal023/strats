"""
Expiry Day Data Fetcher
Fetches historical data for all Nifty 50 stocks on expiry day
Analyzes volume, price movements, and option chain data
"""

import os
import sys
import json
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.historical_data import hist_df, hist_data
from utils.fyers_instance import FyersInstance
from utils.symbol_database import get_symbol_db, get_all_symbols
from utils.api_utils import get_option_chain

class ExpiryDataFetcher:
    def __init__(self):
        self.logger = logging.getLogger("ExpiryDataFetcher")
        self.db = get_symbol_db()
        
        # Initialize Fyers instance
        try:
            self.fyers = FyersInstance.get_instance()
            self.logger.info("✅ Connected to Fyers API")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Fyers API: {e}")
            raise
    
    def get_expiry_date(self, days_back: int = 1) -> str:
        """Get expiry date (yesterday by default)"""
        expiry_date = datetime.now() - timedelta(days=days_back)
        return expiry_date.strftime("%d/%m/%Y")
    
    def get_nifty50_symbols(self) -> List[Dict]:
        """Get all Nifty 50 symbols from database"""
        try:
            all_mappings = get_all_symbols()
            nifty50_data = []
            
            for company_name, symbol in all_mappings.items():
                nifty50_data.append({
                    'company_name': company_name,
                    'symbol': symbol,
                    'nse_symbol': f"NSE:{symbol}-EQ"
                })
            
            self.logger.info(f"📊 Loaded {len(nifty50_data)} Nifty 50 stocks")
            return nifty50_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading Nifty 50 symbols: {e}")
            return []
    
    def fetch_stock_data(self, symbol: str, nse_symbol: str, expiry_date: str, resolution: str = "5") -> Optional[Dict]:
        """Fetch historical data for a single stock on expiry day"""
        try:
            self.logger.info(f"📈 Fetching data for {symbol} on {expiry_date}...")
            
            # Get 5-minute data for the expiry day
            df = hist_df(nse_symbol, expiry_date, resolution=resolution)
            
            if df is not None and not df.empty:
                # Calculate key metrics
                data = {
                    'symbol': symbol,
                    'nse_symbol': nse_symbol,
                    'expiry_date': expiry_date,
                    'data_points': len(df),
                    'open_price': df['Open'].iloc[0],
                    'high_price': df['High'].max(),
                    'low_price': df['Low'].min(),
                    'close_price': df['Close'].iloc[-1],
                    'total_volume': df['Volume'].sum(),
                    'avg_volume': df['Volume'].mean(),
                    'max_volume': df['Volume'].max(),
                    'price_change': df['Close'].iloc[-1] - df['Open'].iloc[0],
                    'price_change_pct': ((df['Close'].iloc[-1] - df['Open'].iloc[0]) / df['Open'].iloc[0]) * 100,
                    'intraday_high_low_pct': ((df['High'].max() - df['Low'].min()) / df['Open'].iloc[0]) * 100,
                    'volume_spike_ratio': df['Volume'].max() / df['Volume'].mean() if df['Volume'].mean() > 0 else 0,
                    'raw_data': df.to_dict('records')
                }
                
                self.logger.info(f"✅ Got {len(df)} data points for {symbol}")
                return data
            else:
                self.logger.warning(f"⚠️ No data for {symbol} on {expiry_date}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_option_chain_data(self, symbol: str, nse_symbol: str) -> Optional[Dict]:
        """Fetch option chain data for a stock"""
        try:
            self.logger.info(f"📊 Fetching option chain for {symbol}...")
            
            # Get option chain data
            chain_data = get_option_chain(nse_symbol, strikecount=5)
            
            if chain_data and chain_data.get('s') == 'ok':
                options_chain = chain_data.get('data', {}).get('optionsChain', [])
                
                # Analyze option chain
                option_analysis = {
                    'symbol': symbol,
                    'total_options': len(options_chain),
                    'ce_options': 0,
                    'pe_options': 0,
                    'atm_strike': None,
                    'max_oi_ce': {'strike': None, 'oi': 0},
                    'max_oi_pe': {'strike': None, 'oi': 0},
                    'max_volume_ce': {'strike': None, 'volume': 0},
                    'max_volume_pe': {'strike': None, 'volume': 0},
                    'put_call_ratio': 0,
                    'total_ce_oi': 0,
                    'total_pe_oi': 0,
                    'total_ce_volume': 0,
                    'total_pe_volume': 0
                }
                
                # Get underlying price
                underlying_price = None
                for option in options_chain:
                    if option.get('strike_price') == -1:  # Underlying
                        underlying_price = option.get('fp') or option.get('ltp')
                        break
                
                if underlying_price:
                    option_analysis['underlying_price'] = underlying_price
                    
                    # Find ATM strike
                    atm_strike = None
                    min_distance = float('inf')
                    
                    for option in options_chain:
                        strike = option.get('strike_price')
                        if strike and strike > 0:
                            distance = abs(strike - underlying_price)
                            if distance < min_distance:
                                min_distance = distance
                                atm_strike = strike
                    
                    option_analysis['atm_strike'] = atm_strike
                
                # Analyze options
                for option in options_chain:
                    strike = option.get('strike_price')
                    option_type = option.get('option_type')
                    oi = option.get('open_interest', 0)
                    volume = option.get('volume', 0)
                    
                    if strike and strike > 0:  # Skip underlying
                        if option_type == 'CE':
                            option_analysis['ce_options'] += 1
                            option_analysis['total_ce_oi'] += oi
                            option_analysis['total_ce_volume'] += volume
                            
                            if oi > option_analysis['max_oi_ce']['oi']:
                                option_analysis['max_oi_ce'] = {'strike': strike, 'oi': oi}
                            
                            if volume > option_analysis['max_volume_ce']['volume']:
                                option_analysis['max_volume_ce'] = {'strike': strike, 'volume': volume}
                        
                        elif option_type == 'PE':
                            option_analysis['pe_options'] += 1
                            option_analysis['total_pe_oi'] += oi
                            option_analysis['total_pe_volume'] += volume
                            
                            if oi > option_analysis['max_oi_pe']['oi']:
                                option_analysis['max_oi_pe'] = {'strike': strike, 'oi': oi}
                            
                            if volume > option_analysis['max_volume_pe']['volume']:
                                option_analysis['max_volume_pe'] = {'strike': strike, 'volume': volume}
                
                # Calculate put-call ratio
                if option_analysis['total_ce_oi'] > 0:
                    option_analysis['put_call_ratio'] = option_analysis['total_pe_oi'] / option_analysis['total_ce_oi']
                
                self.logger.info(f"✅ Got option chain data for {symbol}")
                return option_analysis
            else:
                self.logger.warning(f"⚠️ No option chain data for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching option chain for {symbol}: {e}")
            return None
    
    def fetch_all_expiry_data(self, expiry_date: str = None, delay: float = 1.0) -> Dict:
        """Fetch expiry day data for all Nifty 50 stocks"""
        try:
            if not expiry_date:
                expiry_date = self.get_expiry_date()
            
            nifty50_stocks = self.get_nifty50_symbols()
            total_stocks = len(nifty50_stocks)
            
            self.logger.info(f"🚀 Starting expiry data fetch for {total_stocks} stocks...")
            self.logger.info(f"📅 Expiry date: {expiry_date}")
            
            all_data = {}
            
            for i, stock in enumerate(nifty50_stocks, 1):
                symbol = stock['symbol']
                nse_symbol = stock['nse_symbol']
                company_name = stock['company_name']
                
                self.logger.info(f"\n📊 Processing {i}/{total_stocks}: {company_name} ({symbol})")
                
                # Fetch stock data
                stock_data = self.fetch_stock_data(symbol, nse_symbol, expiry_date)
                
                # Fetch option chain data
                option_data = self.fetch_option_chain_data(symbol, nse_symbol)
                
                # Combine data
                all_data[symbol] = {
                    'company_name': company_name,
                    'stock_data': stock_data,
                    'option_data': option_data,
                    'expiry_date': expiry_date
                }
                
                # Add delay to avoid rate limiting
                if i < total_stocks:
                    time.sleep(delay)
            
            self.logger.info(f"\n✅ Completed expiry data fetch for {len(all_data)} stocks")
            return all_data
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching all expiry data: {e}")
            return {}
    
    def save_expiry_data(self, data: Dict, output_file: str = "expiry_analysis/expiry_data.json"):
        """Save expiry data to JSON file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Convert DataFrames to dictionaries for JSON serialization
            json_data = {}
            
            for symbol, stock_data in data.items():
                json_data[symbol] = {
                    'company_name': stock_data['company_name'],
                    'expiry_date': stock_data['expiry_date'],
                    'stock_data': stock_data['stock_data'],
                    'option_data': stock_data['option_data']
                }
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            self.logger.info(f"💾 Saved expiry data to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving expiry data: {e}")
            return False
    
    def get_expiry_summary(self, data: Dict) -> Dict:
        """Get summary of expiry day data"""
        try:
            summary = {
                'total_stocks': len(data),
                'stocks_with_data': 0,
                'stocks_with_options': 0,
                'wild_movers': [],
                'volume_spikers': [],
                'top_gainers': [],
                'top_losers': [],
                'high_put_call_ratio': [],
                'low_put_call_ratio': []
            }
            
            for symbol, stock_data in data.items():
                company_name = stock_data['company_name']
                stock_info = stock_data.get('stock_data')
                option_info = stock_data.get('option_data')
                
                if stock_info:
                    summary['stocks_with_data'] += 1
                    
                    # Identify wild movers (high intraday range)
                    if stock_info.get('intraday_high_low_pct', 0) > 5:
                        summary['wild_movers'].append({
                            'symbol': symbol,
                            'company_name': company_name,
                            'intraday_range_pct': stock_info['intraday_high_low_pct'],
                            'price_change_pct': stock_info['price_change_pct']
                        })
                    
                    # Identify volume spikers
                    if stock_info.get('volume_spike_ratio', 0) > 3:
                        summary['volume_spikers'].append({
                            'symbol': symbol,
                            'company_name': company_name,
                            'volume_spike_ratio': stock_info['volume_spike_ratio'],
                            'total_volume': stock_info['total_volume']
                        })
                    
                    # Top gainers and losers
                    price_change = stock_info.get('price_change_pct', 0)
                    if price_change > 2:
                        summary['top_gainers'].append({
                            'symbol': symbol,
                            'company_name': company_name,
                            'price_change_pct': price_change
                        })
                    elif price_change < -2:
                        summary['top_losers'].append({
                            'symbol': symbol,
                            'company_name': company_name,
                            'price_change_pct': price_change
                        })
                
                if option_info:
                    summary['stocks_with_options'] += 1
                    
                    # High and low put-call ratios
                    pcr = option_info.get('put_call_ratio', 0)
                    if pcr > 1.5:
                        summary['high_put_call_ratio'].append({
                            'symbol': symbol,
                            'company_name': company_name,
                            'put_call_ratio': pcr
                        })
                    elif pcr < 0.5:
                        summary['low_put_call_ratio'].append({
                            'symbol': symbol,
                            'company_name': company_name,
                            'put_call_ratio': pcr
                        })
            
            # Sort lists
            summary['wild_movers'].sort(key=lambda x: x['intraday_range_pct'], reverse=True)
            summary['volume_spikers'].sort(key=lambda x: x['volume_spike_ratio'], reverse=True)
            summary['top_gainers'].sort(key=lambda x: x['price_change_pct'], reverse=True)
            summary['top_losers'].sort(key=lambda x: x['price_change_pct'])
            summary['high_put_call_ratio'].sort(key=lambda x: x['put_call_ratio'], reverse=True)
            summary['low_put_call_ratio'].sort(key=lambda x: x['put_call_ratio'])
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error getting expiry summary: {e}")
            return {}

def main():
    """Main function to fetch expiry data"""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create fetcher
        fetcher = ExpiryDataFetcher()
        
        # Get expiry date (yesterday)
        expiry_date = fetcher.get_expiry_date()
        
        print(f"🚀 Starting expiry day data fetch...")
        print(f"📅 Expiry date: {expiry_date}")
        
        # Fetch all data
        expiry_data = fetcher.fetch_all_expiry_data(expiry_date, delay=1.0)
        
        if expiry_data:
            # Save data
            fetcher.save_expiry_data(expiry_data)
            
            # Get summary
            summary = fetcher.get_expiry_summary(expiry_data)
            
            print(f"\n📈 EXPIRY DAY SUMMARY:")
            print(f"   Total stocks: {summary['total_stocks']}")
            print(f"   Stocks with data: {summary['stocks_with_data']}")
            print(f"   Stocks with options: {summary['stocks_with_options']}")
            print(f"   Wild movers (>5% range): {len(summary['wild_movers'])}")
            print(f"   Volume spikers (>3x avg): {len(summary['volume_spikers'])}")
            print(f"   Top gainers (>2%): {len(summary['top_gainers'])}")
            print(f"   Top losers (<-2%): {len(summary['top_losers'])}")
            
            print(f"\n✅ Expiry data fetch completed!")
            print(f"💾 Data saved to: expiry_analysis/expiry_data.json")
        else:
            print("❌ No data fetched")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
