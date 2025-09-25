"""
Historical Data Manager for Calendar Spread Strategy
Downloads and manages historical data for backtesting
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yfinance as yf
import requests
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import CalendarConfig

class HistoricalDataManager:
    """
    Manages historical data download and storage for calendar spread backtesting.
    Supports multiple data sources and formats.
    """
    
    def __init__(self, config: CalendarConfig):
        self.config = config
        self.logger = logging.getLogger("HistoricalDataManager")
        
        # Data storage
        self.data_dir = "historical_data"
        self.options_data_dir = os.path.join(self.data_dir, "options")
        self.stocks_data_dir = os.path.join(self.data_dir, "stocks")
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.options_data_dir, exist_ok=True)
        os.makedirs(self.stocks_data_dir, exist_ok=True)
        
        # Data cache
        self.stock_data_cache: Dict[str, pd.DataFrame] = {}
        self.options_data_cache: Dict[str, Dict] = {}
        
        # Data sources
        self.data_sources = {
            'yfinance': self.download_from_yfinance,
            'nse': self.download_from_nse,
            'manual': self.load_manual_data
        }
    
    def download_stock_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                           source: str = 'yfinance') -> Optional[pd.DataFrame]:
        """
        Download historical stock data
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            start_date: Start date for data
            end_date: End date for data
            source: Data source ('yfinance', 'nse', 'manual')
            
        Returns:
            DataFrame with historical data or None if failed
        """
        try:
            self.logger.info(f"Downloading stock data for {symbol} from {start_date} to {end_date}")
            
            # Check cache first
            cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
            if cache_key in self.stock_data_cache:
                self.logger.info(f"Using cached data for {symbol}")
                return self.stock_data_cache[cache_key]
            
            # Download data
            if source in self.data_sources:
                data = self.data_sources[source](symbol, start_date, end_date, 'stock')
            else:
                raise ValueError(f"Unknown data source: {source}")
            
            if data is not None and not data.empty:
                # Clean and validate data
                data = self.clean_stock_data(data)
                
                # Save to file
                filename = os.path.join(self.stocks_data_dir, f"{symbol}_historical.csv")
                data.to_csv(filename)
                
                # Cache data
                self.stock_data_cache[cache_key] = data
                
                self.logger.info(f"Successfully downloaded {len(data)} records for {symbol}")
                return data
            else:
                self.logger.error(f"No data downloaded for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error downloading stock data for {symbol}: {e}")
            return None
    
    def download_options_data(self, symbol: str, start_date: datetime, end_date: datetime,
                             source: str = 'nse') -> Optional[Dict]:
        """
        Download historical options data
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            source: Data source
            
        Returns:
            Dictionary with options data or None if failed
        """
        try:
            self.logger.info(f"Downloading options data for {symbol}")
            
            # Check cache
            cache_key = f"{symbol}_options_{start_date.date()}_{end_date.date()}"
            if cache_key in self.options_data_cache:
                return self.options_data_cache[cache_key]
            
            # Download data
            if source in self.data_sources:
                data = self.data_sources[source](symbol, start_date, end_date, 'options')
            else:
                raise ValueError(f"Unknown data source: {source}")
            
            if data:
                # Save to file
                filename = os.path.join(self.options_data_dir, f"{symbol}_options.json")
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                
                # Cache data
                self.options_data_cache[cache_key] = data
                
                self.logger.info(f"Successfully downloaded options data for {symbol}")
                return data
            else:
                self.logger.error(f"No options data downloaded for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error downloading options data for {symbol}: {e}")
            return None
    
    def download_from_yfinance(self, symbol: str, start_date: datetime, end_date: datetime, 
                              data_type: str) -> Optional[pd.DataFrame]:
        """Download data from Yahoo Finance"""
        try:
            if data_type == 'stock':
                # Add .NS suffix for NSE stocks
                ticker_symbol = f"{symbol}.NS" if symbol not in ['NIFTY', 'BANKNIFTY', 'FINNIFTY'] else f"{symbol}.NSE"
                
                ticker = yf.Ticker(ticker_symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Rename columns to standard format
                    data.columns = [col.lower() for col in data.columns]
                    data = data.reset_index()
                    data['date'] = data['Date']
                    data = data.drop('Date', axis=1)
                    
                    return data
                else:
                    self.logger.warning(f"No data found for {ticker_symbol}")
                    return None
            else:
                self.logger.warning("Yahoo Finance doesn't support options data")
                return None
                
        except Exception as e:
            self.logger.error(f"Error downloading from Yahoo Finance: {e}")
            return None
    
    def download_from_nse(self, symbol: str, start_date: datetime, end_date: datetime,
                         data_type: str) -> Optional[Any]:
        """Download data from NSE (placeholder - would need NSE API)"""
        try:
            if data_type == 'stock':
                # This would need to be implemented with actual NSE API
                self.logger.warning("NSE API not implemented, using mock data")
                return self.create_mock_stock_data(symbol, start_date, end_date)
            else:
                # Options data from NSE
                self.logger.warning("NSE options API not implemented, using mock data")
                return self.create_mock_options_data(symbol, start_date, end_date)
                
        except Exception as e:
            self.logger.error(f"Error downloading from NSE: {e}")
            return None
    
    def load_manual_data(self, symbol: str, start_date: datetime, end_date: datetime,
                        data_type: str) -> Optional[Any]:
        """Load manually prepared data"""
        try:
            if data_type == 'stock':
                filename = os.path.join(self.stocks_data_dir, f"{symbol}_manual.csv")
                if os.path.exists(filename):
                    data = pd.read_csv(filename)
                    data['date'] = pd.to_datetime(data['date'])
                    return data
                else:
                    self.logger.warning(f"Manual data file not found: {filename}")
                    return None
            else:
                filename = os.path.join(self.options_data_dir, f"{symbol}_options_manual.json")
                if os.path.exists(filename):
                    with open(filename, 'r') as f:
                        return json.load(f)
                else:
                    self.logger.warning(f"Manual options file not found: {filename}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error loading manual data: {e}")
            return None
    
    def create_mock_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create mock stock data for testing"""
        try:
            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Mock price data (simplified random walk)
            np.random.seed(42)  # For reproducible results
            base_price = 1000 if symbol == 'NIFTY' else 500
            
            prices = []
            volumes = []
            current_price = base_price
            
            for date in date_range:
                # Random walk with drift
                change = np.random.normal(0, 0.02)  # 2% daily volatility
                current_price *= (1 + change)
                prices.append(current_price)
                volumes.append(np.random.randint(1000000, 5000000))
            
            # Create DataFrame
            data = pd.DataFrame({
                'date': date_range,
                'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
                'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
                'close': prices,
                'volume': volumes,
                'adj_close': prices
            })
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error creating mock stock data: {e}")
            return pd.DataFrame()
    
    def create_mock_options_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """Create mock options data for testing"""
        try:
            # This would create mock options data with strikes, expiries, etc.
            # For now, return a simple structure
            return {
                'symbol': symbol,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'options_data': {
                    'strikes': [100, 105, 110, 115, 120],
                    'expiries': ['25JAN', '25FEB', '25MAR'],
                    'iv_data': {
                        '25JAN': {'100': 0.25, '105': 0.24, '110': 0.23},
                        '25FEB': {'100': 0.30, '105': 0.29, '110': 0.28}
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error creating mock options data: {e}")
            return {}
    
    def clean_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate stock data"""
        try:
            # Remove any rows with NaN values
            data = data.dropna()
            
            # Ensure date column is datetime
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            # Sort by date
            data = data.sort_values('date').reset_index(drop=True)
            
            # Validate required columns
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.warning(f"Missing required column: {col}")
            
            # Remove any negative prices
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    data = data[data[col] > 0]
            
            # Remove any negative volumes
            if 'volume' in data.columns:
                data = data[data['volume'] > 0]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error cleaning stock data: {e}")
            return data
    
    def load_stock_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> Optional[pd.DataFrame]:
        """Load stock data from file"""
        try:
            filename = os.path.join(self.stocks_data_dir, f"{symbol}_historical.csv")
            
            if not os.path.exists(filename):
                self.logger.warning(f"Stock data file not found: {filename}")
                return None
            
            data = pd.read_csv(filename)
            data['date'] = pd.to_datetime(data['date'])
            
            # Filter by date range if provided
            if start_date:
                data = data[data['date'] >= start_date]
            if end_date:
                data = data[data['date'] <= end_date]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading stock data for {symbol}: {e}")
            return None
    
    def load_options_data(self, symbol: str) -> Optional[Dict]:
        """Load options data from file"""
        try:
            filename = os.path.join(self.options_data_dir, f"{symbol}_options.json")
            
            if not os.path.exists(filename):
                self.logger.warning(f"Options data file not found: {filename}")
                return None
            
            with open(filename, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading options data for {symbol}: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols"""
        try:
            symbols = []
            
            # Get stock symbols
            for filename in os.listdir(self.stocks_data_dir):
                if filename.endswith('_historical.csv'):
                    symbol = filename.replace('_historical.csv', '')
                    symbols.append(symbol)
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data"""
        try:
            symbols = self.get_available_symbols()
            
            summary = {
                'total_symbols': len(symbols),
                'symbols': symbols,
                'data_directory': self.data_dir,
                'stocks_directory': self.stocks_data_dir,
                'options_directory': self.options_data_dir
            }
            
            # Add data range for each symbol
            symbol_ranges = {}
            for symbol in symbols:
                data = self.load_stock_data(symbol)
                if data is not None and not data.empty:
                    symbol_ranges[symbol] = {
                        'start_date': data['date'].min().isoformat(),
                        'end_date': data['date'].max().isoformat(),
                        'total_records': len(data)
                    }
            
            summary['symbol_ranges'] = symbol_ranges
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting data summary: {e}")
            return {}
    
    def download_multiple_symbols(self, symbols: List[str], start_date: datetime, 
                                 end_date: datetime, source: str = 'yfinance') -> Dict[str, pd.DataFrame]:
        """Download data for multiple symbols"""
        try:
            results = {}
            
            for symbol in symbols:
                self.logger.info(f"Downloading data for {symbol}...")
                data = self.download_stock_data(symbol, start_date, end_date, source)
                
                if data is not None:
                    results[symbol] = data
                    self.logger.info(f"Successfully downloaded {len(data)} records for {symbol}")
                else:
                    self.logger.error(f"Failed to download data for {symbol}")
            
            self.logger.info(f"Downloaded data for {len(results)} out of {len(symbols)} symbols")
            return results
            
        except Exception as e:
            self.logger.error(f"Error downloading multiple symbols: {e}")
            return {}
    
    def validate_data(self, symbol: str) -> Dict[str, Any]:
        """Validate data quality for a symbol"""
        try:
            data = self.load_stock_data(symbol)
            
            if data is None or data.empty:
                return {'valid': False, 'error': 'No data available'}
            
            validation_results = {
                'valid': True,
                'total_records': len(data),
                'date_range': {
                    'start': data['date'].min().isoformat(),
                    'end': data['date'].max().isoformat()
                },
                'missing_data': {},
                'data_quality': {}
            }
            
            # Check for missing data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    missing_count = data[col].isna().sum()
                    validation_results['missing_data'][col] = missing_count
            
            # Check data quality
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    negative_prices = (data[col] <= 0).sum()
                    validation_results['data_quality'][f'{col}_negative'] = negative_prices
            
            # Check for gaps in dates
            date_diff = data['date'].diff().dt.days
            gaps = (date_diff > 1).sum()
            validation_results['data_quality']['date_gaps'] = gaps
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating data for {symbol}: {e}")
            return {'valid': False, 'error': str(e)}

def main():
    """Test the historical data manager"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create manager
        from .config import CalendarConfig
        config = CalendarConfig()
        manager = HistoricalDataManager(config)
        
        # Test data download
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 6, 30)
        
        symbols = ['RELIANCE', 'TCS', 'INFY']
        
        print("Downloading historical data...")
        results = manager.download_multiple_symbols(symbols, start_date, end_date)
        
        print(f"Downloaded data for {len(results)} symbols")
        
        # Get summary
        summary = manager.get_data_summary()
        print("Data Summary:")
        print(json.dumps(summary, indent=2))
        
        # Validate data
        for symbol in symbols:
            validation = manager.validate_data(symbol)
            print(f"Validation for {symbol}: {validation}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
