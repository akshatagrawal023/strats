"""
Dynamic Lot Size Processor for Calendar Spread Strategy
Processes lot sizes from options symbols CSV and provides dynamic lot size management
"""
import os
import sys
import logging
import csv
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.option_symbols import get_options_data, fetch_options_data

class LotSizeProcessor:
    """
    Dynamic lot size processor that extracts and manages lot sizes from options symbols data.
    Follows the pattern from the GitHub repository for processing options symbols CSV.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("LotSizeProcessor")
        
        # Lot size data
        self.lot_size_map: Dict[str, int] = {}
        self.symbol_details: Dict[str, Dict] = {}
        self.last_update_time = None
        
        # Cache settings
        self.cache_duration = 24 * 3600  # 24 hours
        self.cache_file = "lot_size_cache.json"
        
        # Initialize lot sizes
        self.initialize_lot_sizes()
    
    def initialize_lot_sizes(self):
        """Initialize lot sizes from various sources"""
        try:
            # Try to load from cache first
            if self.load_from_cache():
                self.logger.info("Loaded lot sizes from cache")
                return
            
            # Load from options symbols data
            self.load_from_options_data()
            
            # Load from configuration
            self.load_from_config()
            
            # Save to cache
            self.save_to_cache()
            
            self.logger.info(f"Initialized lot sizes for {len(self.lot_size_map)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error initializing lot sizes: {e}")
            # Fallback to default
            self.load_default_lot_sizes()
    
    def load_from_cache(self) -> bool:
        """Load lot sizes from cache file"""
        try:
            if not os.path.exists(self.cache_file):
                return False
            
            # Check cache age
            cache_time = os.path.getmtime(self.cache_file)
            if time.time() - cache_time > self.cache_duration:
                self.logger.info("Cache expired, refreshing lot sizes")
                return False
            
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            self.lot_size_map = cache_data.get('lot_size_map', {})
            self.symbol_details = cache_data.get('symbol_details', {})
            self.last_update_time = datetime.fromisoformat(cache_data.get('last_update_time', ''))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading from cache: {e}")
            return False
    
    def save_to_cache(self):
        """Save lot sizes to cache file"""
        try:
            cache_data = {
                'lot_size_map': self.lot_size_map,
                'symbol_details': self.symbol_details,
                'last_update_time': datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.info("Lot sizes saved to cache")
            
        except Exception as e:
            self.logger.error(f"Error saving to cache: {e}")
    
    def load_from_options_data(self):
        """Load lot sizes from options symbols data"""
        try:
            self.logger.info("Loading lot sizes from options symbols data...")
            
            # Get options data
            options_data = get_options_data(use_cache=True)
            
            if not options_data:
                self.logger.warning("No options data available, fetching fresh data...")
                options_data = fetch_options_data()
            
            # Process each symbol
            for symbol, details in options_data.items():
                try:
                    # Extract lot size from symbol details
                    lot_size = self.extract_lot_size_from_symbol(symbol, details)
                    
                    if lot_size:
                        self.lot_size_map[symbol] = lot_size
                        self.symbol_details[symbol] = {
                            'lot_size': lot_size,
                            'strikes': details.get('strikes', []),
                            'expiries': details.get('expiries', []),
                            'has_futures': bool(details.get('futures', [])),
                            'has_options': bool(details.get('strikes', [])),
                            'last_updated': datetime.now().isoformat()
                        }
                    
                except Exception as e:
                    self.logger.warning(f"Error processing symbol {symbol}: {e}")
                    continue
            
            self.logger.info(f"Processed {len(self.lot_size_map)} symbols from options data")
            
        except Exception as e:
            self.logger.error(f"Error loading from options data: {e}")
    
    def extract_lot_size_from_symbol(self, symbol: str, details: Dict) -> Optional[int]:
        """Extract lot size from symbol details"""
        try:
            # Method 1: Try to get from futures data
            futures = details.get('futures', [])
            if futures:
                # Extract lot size from futures symbol
                lot_size = self.extract_lot_size_from_futures_symbol(futures[0])
                if lot_size:
                    return lot_size
            
            # Method 2: Use known lot sizes for common symbols
            known_lot_sizes = {
                'NIFTY': 50,
                'BANKNIFTY': 25,
                'FINNIFTY': 40,
                'RELIANCE': 250,
                'TCS': 125,
                'INFY': 200,
                'HDFC': 250,
                'ICICIBANK': 275,
                'SBIN': 1500,
                'BHARTIARTL': 400,
                'ITC': 800,
                'LT': 200,
                'MARUTI': 25,
                'ASIANPAINT': 25,
                'NESTLEIND': 25,
                'ULTRACEMCO': 25,
                'SUNPHARMA': 200,
                'TITAN': 50,
                'WIPRO': 400,
                'AXISBANK': 300
            }
            
            if symbol in known_lot_sizes:
                return known_lot_sizes[symbol]
            
            # Method 3: Estimate based on symbol characteristics
            return self.estimate_lot_size_from_symbol(symbol, details)
            
        except Exception as e:
            self.logger.error(f"Error extracting lot size for {symbol}: {e}")
            return None
    
    def extract_lot_size_from_futures_symbol(self, futures_symbol: str) -> Optional[int]:
        """Extract lot size from futures symbol"""
        try:
            # This would need to be implemented based on the actual futures symbol format
            # For now, return None to use other methods
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting lot size from futures symbol {futures_symbol}: {e}")
            return None
    
    def estimate_lot_size_from_symbol(self, symbol: str, details: Dict) -> Optional[int]:
        """Estimate lot size based on symbol characteristics"""
        try:
            # Get strikes to estimate price range
            strikes = details.get('strikes', [])
            if not strikes:
                return None
            
            # Convert strikes to integers
            strike_values = []
            for strike in strikes:
                try:
                    strike_values.append(int(strike))
                except:
                    continue
            
            if not strike_values:
                return None
            
            # Estimate based on strike price range
            min_strike = min(strike_values)
            max_strike = max(strike_values)
            avg_strike = (min_strike + max_strike) / 2
            
            # Estimate lot size based on price range
            if avg_strike < 100:
                return 1000  # High lot size for low-priced stocks
            elif avg_strike < 500:
                return 500   # Medium lot size
            elif avg_strike < 1000:
                return 250  # Lower lot size for higher-priced stocks
            else:
                return 100  # Very low lot size for very high-priced stocks
            
        except Exception as e:
            self.logger.error(f"Error estimating lot size for {symbol}: {e}")
            return None
    
    def load_from_config(self):
        """Load lot sizes from configuration"""
        try:
            config_lot_sizes = getattr(self.config, 'LOT_SIZE_MAP', {})
            
            for symbol, lot_size in config_lot_sizes.items():
                if symbol not in self.lot_size_map:
                    self.lot_size_map[symbol] = lot_size
                    self.logger.info(f"Added lot size from config: {symbol} = {lot_size}")
            
        except Exception as e:
            self.logger.error(f"Error loading from config: {e}")
    
    def load_default_lot_sizes(self):
        """Load default lot sizes as fallback"""
        try:
            default_lot_sizes = {
                'NIFTY': 50,
                'BANKNIFTY': 25,
                'FINNIFTY': 40,
                'RELIANCE': 250,
                'TCS': 125,
                'INFY': 200,
                'HDFC': 250,
                'ICICIBANK': 275,
                'SBIN': 1500,
                'BHARTIARTL': 400,
                'ITC': 800,
                'LT': 200,
                'MARUTI': 25,
                'ASIANPAINT': 25,
                'NESTLEIND': 25,
                'ULTRACEMCO': 25,
                'SUNPHARMA': 200,
                'TITAN': 50,
                'WIPRO': 400,
                'AXISBANK': 300
            }
            
            self.lot_size_map.update(default_lot_sizes)
            self.logger.info("Loaded default lot sizes")
            
        except Exception as e:
            self.logger.error(f"Error loading default lot sizes: {e}")
    
    def get_lot_size(self, symbol: str) -> int:
        """Get lot size for a symbol"""
        try:
            # Check if we have the symbol
            if symbol in self.lot_size_map:
                return self.lot_size_map[symbol]
            
            # Try to find similar symbol
            similar_symbol = self.find_similar_symbol(symbol)
            if similar_symbol:
                return self.lot_size_map[similar_symbol]
            
            # Return default lot size
            return getattr(self.config, 'DEFAULT_LOT_SIZE', 50)
            
        except Exception as e:
            self.logger.error(f"Error getting lot size for {symbol}: {e}")
            return getattr(self.config, 'DEFAULT_LOT_SIZE', 50)
    
    def find_similar_symbol(self, symbol: str) -> Optional[str]:
        """Find similar symbol in lot size map"""
        try:
            # Try exact match first
            if symbol in self.lot_size_map:
                return symbol
            
            # Try case-insensitive match
            for key in self.lot_size_map.keys():
                if key.upper() == symbol.upper():
                    return key
            
            # Try partial match
            for key in self.lot_size_map.keys():
                if symbol.upper() in key.upper() or key.upper() in symbol.upper():
                    return key
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding similar symbol for {symbol}: {e}")
            return None
    
    def update_lot_size(self, symbol: str, lot_size: int):
        """Update lot size for a symbol"""
        try:
            self.lot_size_map[symbol] = lot_size
            
            # Update symbol details
            if symbol in self.symbol_details:
                self.symbol_details[symbol]['lot_size'] = lot_size
                self.symbol_details[symbol]['last_updated'] = datetime.now().isoformat()
            else:
                self.symbol_details[symbol] = {
                    'lot_size': lot_size,
                    'last_updated': datetime.now().isoformat()
                }
            
            # Save to cache
            self.save_to_cache()
            
            self.logger.info(f"Updated lot size for {symbol}: {lot_size}")
            
        except Exception as e:
            self.logger.error(f"Error updating lot size for {symbol}: {e}")
    
    def refresh_lot_sizes(self):
        """Refresh lot sizes from source data"""
        try:
            self.logger.info("Refreshing lot sizes...")
            
            # Clear existing data
            self.lot_size_map.clear()
            self.symbol_details.clear()
            
            # Reload from sources
            self.load_from_options_data()
            self.load_from_config()
            
            # Save to cache
            self.save_to_cache()
            
            self.last_update_time = datetime.now()
            
            self.logger.info(f"Refreshed lot sizes for {len(self.lot_size_map)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error refreshing lot sizes: {e}")
    
    def get_all_lot_sizes(self) -> Dict[str, int]:
        """Get all lot sizes"""
        return self.lot_size_map.copy()
    
    def get_symbol_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed information for a symbol"""
        return self.symbol_details.get(symbol)
    
    def get_all_symbol_details(self) -> Dict[str, Dict]:
        """Get all symbol details"""
        return self.symbol_details.copy()
    
    def export_lot_sizes_to_csv(self, filename: str = "lot_sizes.csv"):
        """Export lot sizes to CSV file"""
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Symbol', 'Lot Size', 'Last Updated'])
                
                for symbol, details in self.symbol_details.items():
                    lot_size = details.get('lot_size', 0)
                    last_updated = details.get('last_updated', '')
                    writer.writerow([symbol, lot_size, last_updated])
            
            self.logger.info(f"Exported lot sizes to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting lot sizes: {e}")
    
    def import_lot_sizes_from_csv(self, filename: str):
        """Import lot sizes from CSV file"""
        try:
            if not os.path.exists(filename):
                self.logger.error(f"File {filename} not found")
                return
            
            with open(filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    symbol = row['Symbol']
                    lot_size = int(row['Lot Size'])
                    last_updated = row.get('Last Updated', datetime.now().isoformat())
                    
                    self.lot_size_map[symbol] = lot_size
                    self.symbol_details[symbol] = {
                        'lot_size': lot_size,
                        'last_updated': last_updated
                    }
            
            # Save to cache
            self.save_to_cache()
            
            self.logger.info(f"Imported lot sizes from {filename}")
            
        except Exception as e:
            self.logger.error(f"Error importing lot sizes: {e}")
    
    def get_lot_size_summary(self) -> Dict[str, Any]:
        """Get summary of lot size data"""
        try:
            return {
                'total_symbols': len(self.lot_size_map),
                'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
                'cache_file': self.cache_file,
                'cache_exists': os.path.exists(self.cache_file),
                'lot_size_range': {
                    'min': min(self.lot_size_map.values()) if self.lot_size_map else 0,
                    'max': max(self.lot_size_map.values()) if self.lot_size_map else 0,
                    'average': sum(self.lot_size_map.values()) / len(self.lot_size_map) if self.lot_size_map else 0
                },
                'top_symbols': dict(sorted(self.lot_size_map.items(), key=lambda x: x[1], reverse=True)[:10])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting lot size summary: {e}")
            return {}
    
    def validate_lot_sizes(self) -> Dict[str, List[str]]:
        """Validate lot sizes and return issues"""
        try:
            issues = {
                'missing_lot_sizes': [],
                'invalid_lot_sizes': [],
                'suspicious_lot_sizes': []
            }
            
            for symbol, lot_size in self.lot_size_map.items():
                # Check for missing lot sizes
                if not lot_size or lot_size <= 0:
                    issues['missing_lot_sizes'].append(symbol)
                
                # Check for invalid lot sizes
                if not isinstance(lot_size, int):
                    issues['invalid_lot_sizes'].append(symbol)
                
                # Check for suspicious lot sizes (too high or too low)
                if lot_size > 10000 or lot_size < 1:
                    issues['suspicious_lot_sizes'].append(symbol)
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Error validating lot sizes: {e}")
            return {}

def main():
    """Test the lot size processor"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create processor
        from .config import CalendarConfig
        config = CalendarConfig()
        processor = LotSizeProcessor(config)
        
        # Print summary
        summary = processor.get_lot_size_summary()
        print("Lot Size Summary:")
        print(json.dumps(summary, indent=2))
        
        # Test getting lot sizes
        test_symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'UNKNOWN']
        print("\nLot Sizes:")
        for symbol in test_symbols:
            lot_size = processor.get_lot_size(symbol)
            print(f"{symbol}: {lot_size}")
        
        # Validate lot sizes
        issues = processor.validate_lot_sizes()
        if any(issues.values()):
            print("\nValidation Issues:")
            print(json.dumps(issues, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
