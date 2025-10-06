import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(".."))

def get_option_symbols_in_range(stock_symbol, underlying_price, percentile_range=5, expiry=None):
    """
    Extract CE and PE option symbols within ±percentile_range of underlying price
    
    Args:
        stock_symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
        underlying_price: Current underlying price
        percentile_range: Percentage range (default 5%)
        expiry: Optional expiry code (e.g., '25SEP'); if None, picks front/nearest
    
    Returns:
        dict: {'ce_symbols': [...], 'pe_symbols': [...], 'expiry': 'XXMMM', 'future_symbol': 'NSE:XXX...FUT'}
    """
    
    # Load options data
    with open('utils/options_details_cache.json', 'r') as f:
        options_data = json.load(f)
    
    if stock_symbol not in options_data:
        return {'ce_symbols': [], 'pe_symbols': []}
    
    stock_data = options_data[stock_symbol]
    strikes = [float(s) for s in stock_data['strikes']]
    expiries = stock_data.get('expiries', [])
    if not expiries:
        return {'ce_symbols': [], 'pe_symbols': []}
    
    # Choose a single expiry: if provided and valid use it; otherwise assume the last entry is front/nearest
    chosen_expiry = expiry if (expiry and expiry in expiries) else expiries[-1]
    
    # Choose nearest future: try to match chosen_expiry; fallback to first available
    futures = stock_data.get('futures', [])
    future_symbol = None
    if futures:
        matching_futures = [f for f in futures if chosen_expiry in f]
        future_symbol = matching_futures[0] if matching_futures else futures[0]
    
    # Calculate price range
    lower_bound = underlying_price * (1 - percentile_range/100)
    upper_bound = underlying_price * (1 + percentile_range/100)
    
    # Find strikes in range
    strikes_in_range = [s for s in strikes if lower_bound <= s <= upper_bound]
    
    # Get CE and PE symbols for strikes in range
    ce_symbols = []
    pe_symbols = []
    
    for strike in strikes_in_range:
        strike_str = str(int(strike))
        ce_symbol = f"NSE:{stock_symbol}{chosen_expiry}{strike_str}CE"
        pe_symbol = f"NSE:{stock_symbol}{chosen_expiry}{strike_str}PE"
        
        if ce_symbol in stock_data['ce_symbols']:
            ce_symbols.append(ce_symbol)
        if pe_symbol in stock_data['pe_symbols']:
            pe_symbols.append(pe_symbol)
    
    return {
        'ce_symbols': ce_symbols,
        'pe_symbols': pe_symbols,
        'strikes_in_range': strikes_in_range,
        'price_range': (lower_bound, upper_bound),
        'expiry': chosen_expiry,
        'future_symbol': future_symbol
    }

# Example usage
if __name__ == "__main__":
    # Example: Get options for RELIANCE at price 2500
    result = get_option_symbols_in_range('RELIANCE', 2500, 5)
    
    print(f"CE Symbols: {len(result['ce_symbols'])}")
    print(f"PE Symbols: {len(result['pe_symbols'])}")
    print(f"Strikes in range: {result['strikes_in_range']}")
    print(f"Price range: {result['price_range']}")
    print(f"Nearest future: {result['future_symbol']}")
    
    # Show first few symbols
    print(f"\nFirst 5 CE symbols: {result['ce_symbols'][:5]}")
    print(f"First 5 PE symbols: {result['pe_symbols'][:5]}")
