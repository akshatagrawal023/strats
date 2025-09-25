"""
Simple Nifty 50 Processor for Calendar Spread Backtesting
Reads NiftyFifty.csv, gets lot sizes from FnO_lot_structured.csv, and uses option chain for strikes
"""
import os
import sys
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fuzzy_name import search_symbol
from utils.api_utils import get_option_chain

class Nifty50Processor:
    """
    Simple processor for Nifty 50 stocks - focuses only on backtesting needs
    """
    
    def __init__(self, csv_path: str = "utils/NiftyFifty.csv", lot_csv_path: str = "utils/FnO_lot_structured.csv"):
        self.csv_path = csv_path
        self.lot_csv_path = lot_csv_path
        self.logger = logging.getLogger("Nifty50Processor")
        
        # Load data
        self.nifty_data = None
        self.lot_sizes = {}
        self.symbol_mapping = {}
        
        self.load_nifty_data()
        self.load_lot_sizes()
        self.process_symbols()
    
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
    
    def load_lot_sizes(self):
        """Load lot sizes from FnO_lot_structured.csv"""
        try:
            if not os.path.exists(self.lot_csv_path):
                self.logger.error(f"Lot sizes CSV not found: {self.lot_csv_path}")
                return
            
            lot_data = pd.read_csv(self.lot_csv_path)
            
            # Create mapping from symbol to lot size
            for _, row in lot_data.iterrows():
                symbol = row.get('Symbol', '').strip()
                lot_size = row.get('Lot Size', 0)
                if symbol and lot_size > 0:
                    self.lot_sizes[symbol] = int(lot_size)
            
            self.logger.info(f"Loaded lot sizes for {len(self.lot_sizes)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error loading lot sizes: {e}")
    
    def process_symbols(self):
        """Convert company names to symbols"""
        try:
            self.logger.info("Converting company names to symbols...")
            
            for _, row in self.nifty_data.iterrows():
                company_name = row['Name']
                symbol = search_symbol(company_name, interactive=False)
                
                if symbol:
                    self.symbol_mapping[company_name] = symbol
                    self.logger.debug(f"✅ {company_name} → {symbol}")
                else:
                    self.logger.warning(f"❌ Could not find symbol for: {company_name}")
            
            self.logger.info(f"Successfully processed {len(self.symbol_mapping)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error processing symbols: {e}")
    
    def get_nse_symbol(self, symbol: str) -> str:
        """Convert symbol to NSE format (e.g., TCS -> NSE:TCS-EQ)"""
        if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
            return f"NSE:{symbol}-INDEX"
        else:
            return f"NSE:{symbol}-EQ"
    
    def get_lot_size(self, symbol: str) -> int:
        """Get lot size for symbol from CSV"""
        return self.lot_sizes.get(symbol, 100)  # Default to 100 if not found
    
    def get_option_chain_data(self, symbol: str, strikecount: int = 2) -> Optional[Dict]:
        """Get option chain data for a symbol"""
        try:
            nse_symbol = self.get_nse_symbol(symbol)
            chain_data = get_option_chain(nse_symbol, strikecount)
            
            if chain_data and chain_data.get('s') == 'ok':
                return chain_data.get('data', {})
            else:
                self.logger.warning(f"Failed to get option chain for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting option chain for {symbol}: {e}")
            return None
    
    def get_atm_strike(self, symbol: str) -> Optional[Dict]:
        """Get ATM strike and option symbols for a symbol"""
        try:
            chain_data = self.get_option_chain_data(symbol, strikecount=2)
            if not chain_data:
                return None
            
            # Get underlying price
            underlying_price = None
            for option in chain_data.get('optionsChain', []):
                if option.get('strike_price') == -1:  # Underlying
                    underlying_price = option.get('fp') or option.get('ltp')
                    break
            
            if not underlying_price:
                self.logger.warning(f"Could not get underlying price for {symbol}")
                return None
            
            # Find ATM strike
            atm_strike = None
            min_distance = float('inf')
            
            for option in chain_data.get('optionsChain', []):
                strike = option.get('strike_price')
                if strike and strike > 0:
                    distance = abs(strike - underlying_price)
                    if distance < min_distance:
                        min_distance = distance
                        atm_strike = strike
            
            if not atm_strike:
                self.logger.warning(f"Could not find ATM strike for {symbol}")
                return None
            
            # Find CE and PE symbols for ATM strike
            ce_symbol = None
            pe_symbol = None
            
            for option in chain_data.get('optionsChain', []):
                if option.get('strike_price') == atm_strike:
                    if option.get('option_type') == 'CE':
                        ce_symbol = option.get('symbol')
                    elif option.get('option_type') == 'PE':
                        pe_symbol = option.get('symbol')
            
            if not ce_symbol or not pe_symbol:
                self.logger.warning(f"Could not find CE/PE symbols for {symbol} at strike {atm_strike}")
                return None
            
            return {
                'symbol': symbol,
                'nse_symbol': self.get_nse_symbol(symbol),
                'underlying_price': underlying_price,
                'atm_strike': atm_strike,
                'ce_symbol': ce_symbol,
                'pe_symbol': pe_symbol,
                'lot_size': self.get_lot_size(symbol)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting ATM strike for {symbol}: {e}")
            return None
    
    def get_all_atm_strikes(self) -> List[Dict]:
        """Get ATM strikes for all Nifty 50 stocks"""
        try:
            atm_strikes = []
            
            for company_name, symbol in self.symbol_mapping.items():
                self.logger.info(f"Getting ATM strike for {symbol}...")
                atm_data = self.get_atm_strike(symbol)
                
                if atm_data:
                    atm_data['company_name'] = company_name
                    atm_strikes.append(atm_data)
                    self.logger.info(f"✅ {symbol}: Strike {atm_data['atm_strike']}, Price {atm_data['underlying_price']}")
                else:
                    self.logger.warning(f"❌ Failed to get ATM strike for {symbol}")
            
            self.logger.info(f"Successfully got ATM strikes for {len(atm_strikes)} stocks")
            return atm_strikes
            
        except Exception as e:
            self.logger.error(f"Error getting all ATM strikes: {e}")
            return []
    
    def get_liquid_stocks(self, min_volume: float = 1000000) -> List[Dict]:
        """Get liquid stocks based on volume"""
        try:
            # Convert volume to numeric
            self.nifty_data['Volume_Numeric'] = self.nifty_data['Volume'].astype(str).str.replace(',', '').astype(float)
            
            # Filter by volume
            liquid_stocks = self.nifty_data[self.nifty_data['Volume_Numeric'] >= min_volume]
            
            results = []
            for _, row in liquid_stocks.iterrows():
                company_name = row['Name']
                if company_name in self.symbol_mapping:
                    symbol = self.symbol_mapping[company_name]
                    results.append({
                        'company_name': company_name,
                        'symbol': symbol,
                        'nse_symbol': self.get_nse_symbol(symbol),
                        'volume': row['Volume_Numeric'],
                        'lot_size': self.get_lot_size(symbol)
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting liquid stocks: {e}")
            return []
    
    def get_recommended_stocks(self, limit: int = 15) -> List[Dict]:
        """Get recommended stocks for calendar spread (high volume + reasonable PE)"""
        try:
            # Get liquid stocks
            liquid_stocks = self.get_liquid_stocks()
            
            # Filter by reasonable PE ratio (5-40)
            self.nifty_data['PE_Numeric'] = pd.to_numeric(self.nifty_data['PE Ratio'], errors='coerce')
            reasonable_pe = self.nifty_data[
                (self.nifty_data['PE_Numeric'] >= 5) & 
                (self.nifty_data['PE_Numeric'] <= 40)
            ]
            
            # Get intersection
            recommended = []
            for stock in liquid_stocks:
                company_name = stock['company_name']
                if company_name in reasonable_pe['Name'].values:
                    recommended.append(stock)
            
            # Limit to requested number
            return recommended[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting recommended stocks: {e}")
            return []
    
    def export_for_backtest(self, filename: str = "nifty50_backtest_data.json"):
        """Export data for backtesting"""
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
                        'lot_size': stock['lot_size'],
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
            print(f"✅ Symbols Converted: {len(self.symbol_mapping)}")
            print(f"📋 Lot Sizes Loaded: {len(self.lot_sizes)}")
            
            # Show recommended stocks
            recommended = self.get_recommended_stocks(10)
            print(f"\n🎯 Top 10 Recommended for Calendar Spread:")
            for i, stock in enumerate(recommended, 1):
                print(f"   {i:2d}. {stock['company_name']} ({stock['symbol']}) - Lot: {stock['lot_size']}")
            
            # Test ATM strike for a few stocks
            print(f"\n🔍 Sample ATM Strikes:")
            test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
            for symbol in test_symbols:
                if symbol in self.symbol_mapping.values():
                    atm_data = self.get_atm_strike(symbol)
                    if atm_data:
                        print(f"   {symbol}: Strike {atm_data['atm_strike']}, Price {atm_data['underlying_price']}")
            
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