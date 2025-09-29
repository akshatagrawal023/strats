"""
Streamlined Nifty 50 Processor using SQLite Database
Fast and efficient processing for calendar spread strategy
"""

import os
import sys
import logging
import pandas as pd
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.symbol_database import get_symbol_db, get_symbol
from utils.api_utils import get_option_chain

class Nifty50Processor:
    def __init__(self, csv_path: str = "utils/NiftyFifty.csv"):
        self.csv_path = csv_path
        self.logger = logging.getLogger("Nifty50Processor")
        
        # Load data
        self.nifty_data = None
        self.db = get_symbol_db()
        
        self.load_nifty_data()
    
    def load_nifty_data(self):
        """Load Nifty 50 data from CSV"""
        try:
            if not os.path.exists(self.csv_path):
                self.logger.error(f"CSV file not found: {self.csv_path}")
                return
            
            self.nifty_data = pd.read_csv(self.csv_path)
            self.nifty_data['Name'] = self.nifty_data['Name'].str.strip()
            self.logger.info(f"Loaded {len(self.nifty_data)} Nifty 50 stocks")
            
        except Exception as e:
            self.logger.error(f"Error loading Nifty 50 data: {e}")
    
    def get_nse_symbol(self, symbol: str) -> str:
        """Convert symbol to NSE format"""
        if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
            return f"NSE:{symbol}-INDEX"
        else:
            return f"NSE:{symbol}-EQ"
    
    def get_liquid_stocks(self, min_volume: float = 1000000) -> List[Dict]:
        """Get liquid stocks based on volume"""
        try:
            # Convert volume to numeric
            self.nifty_data['Volume_Numeric'] = self.nifty_data['Volume'].astype(str).str.replace(',', '').astype(float)
            
            # Filter by volume and get symbols from database
            liquid_stocks = []
            for _, row in self.nifty_data.iterrows():
                if row['Volume_Numeric'] >= min_volume:
                    company_name = row['Name']
                    symbol = get_symbol(company_name)
                    
                    if symbol:
                        liquid_stocks.append({
                            'company_name': company_name,
                            'symbol': symbol,
                            'nse_symbol': self.get_nse_symbol(symbol),
                            'volume': row['Volume_Numeric'],
                            'market_cap': row.get('Market Cap (Cr.)', 0),
                            'pe_ratio': row.get('PE Ratio', 0)
                        })
            
            return liquid_stocks
            
        except Exception as e:
            self.logger.error(f"Error getting liquid stocks: {e}")
            return []
    
    def get_recommended_stocks(self, limit: int = 15) -> List[Dict]:
        """Get recommended stocks for calendar spread (high volume + reasonable PE)"""
        try:
            # Get liquid stocks
            liquid_stocks = self.get_liquid_stocks()
            
            # Filter by reasonable PE ratio (5-40)
            recommended = []
            for stock in liquid_stocks:
                pe_ratio = stock.get('pe_ratio', 0)
                if 5 <= pe_ratio <= 40:
                    recommended.append(stock)
            
            # Sort by volume (descending) and limit
            recommended.sort(key=lambda x: x['volume'], reverse=True)
            return recommended[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting recommended stocks: {e}")
            return []
    
    def get_atm_strike(self, symbol: str) -> Optional[Dict]:
        """Get ATM strike and option symbols for a symbol"""
        try:
            nse_symbol = self.get_nse_symbol(symbol)
            chain_data = get_option_chain(nse_symbol, strikecount=2)
            
            if not chain_data or chain_data.get('s') != 'ok':
                return None
            
            options_chain = chain_data.get('data', {}).get('optionsChain', [])
            
            # Get underlying price
            underlying_price = None
            for option in options_chain:
                if option.get('strike_price') == -1:  # Underlying
                    underlying_price = option.get('fp') or option.get('ltp')
                    break
            
            if not underlying_price:
                return None
            
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
            
            if not atm_strike:
                return None
            
            # Find CE and PE symbols for ATM strike
            ce_symbol = None
            pe_symbol = None
            
            for option in options_chain:
                if option.get('strike_price') == atm_strike:
                    if option.get('option_type') == 'CE':
                        ce_symbol = option.get('symbol')
                    elif option.get('option_type') == 'PE':
                        pe_symbol = option.get('symbol')
            
            if not ce_symbol or not pe_symbol:
                return None
            
            return {
                'symbol': symbol,
                'nse_symbol': nse_symbol,
                'underlying_price': underlying_price,
                'atm_strike': atm_strike,
                'ce_symbol': ce_symbol,
                'pe_symbol': pe_symbol
            }
            
        except Exception as e:
            self.logger.error(f"Error getting ATM strike for {symbol}: {e}")
            return None
    
    def export_for_backtest(self, filename: str = "nifty50_backtest_data.json") -> List[Dict]:
        """Export recommended stocks for backtesting"""
        try:
            import json
            
            # Get recommended stocks
            recommended = self.get_recommended_stocks(15)
            
            # Get ATM strikes for recommended stocks
            backtest_data = []
            for stock in recommended:
                symbol = stock['symbol']
                atm_data = self.get_atm_strike(symbol)
                
                if atm_data:
                    backtest_data.append({
                        'company_name': stock['company_name'],
                        'symbol': symbol,
                        'nse_symbol': stock['nse_symbol'],
                        'volume': stock['volume'],
                        'market_cap': stock['market_cap'],
                        'pe_ratio': stock['pe_ratio'],
                        'underlying_price': atm_data['underlying_price'],
                        'atm_strike': atm_data['atm_strike'],
                        'ce_symbol': atm_data['ce_symbol'],
                        'pe_symbol': atm_data['pe_symbol']
                    })
            
            # Export to JSON
            with open(filename, 'w') as f:
                json.dump(backtest_data, f, indent=2)
            
            self.logger.info(f"Exported {len(backtest_data)} stocks to {filename}")
            return backtest_data
            
        except Exception as e:
            self.logger.error(f"Error exporting for backtest: {e}")
            return []
    
    def print_summary(self):
        """Print summary of processed data"""
        try:
            print("\n" + "=" * 80)
            print("NIFTY 50 PROCESSOR - SUMMARY")
            print("=" * 80)
            
            print(f"📊 Total Companies: {len(self.nifty_data)}")
            
            # Show recommended stocks
            recommended = self.get_recommended_stocks(15)
            print(f"\n🎯 Top {len(recommended)} Recommended for Calendar Spread:")
            for i, stock in enumerate(recommended, 1):
                print(f"   {i:2d}. {stock['company_name']} ({stock['symbol']}) - Vol: {stock['volume']:,.0f}, PE: {stock['pe_ratio']:.1f}")
            
            print("\n" + "=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error printing summary: {e}")

def main():
    """Test the Nifty 50 processor"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create processor
        processor = Nifty50Processor()
        
        # Print summary
        processor.print_summary()
        
        # Export for backtest
        backtest_data = processor.export_for_backtest()
        
        print(f"\n✅ Ready for backtesting with {len(backtest_data)} stocks")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()