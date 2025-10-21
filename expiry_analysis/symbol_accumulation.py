import sqlite3
import pandas as pd
import os, sys
from datetime import datetime
from pandas.tseries.offsets import BDay
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.historical_data import hist_df, hist_data

conn = sqlite3.connect("utils/symbol_mappings.db")
df = pd.read_sql_query("SELECT * FROM symbol_mappings WHERE is_active = 1", conn)
conn.close()

prev_day = 1
today = pd.date_range(start=datetime.today(), end = datetime.today())
yes = (today-BDay(prev_day)).strftime('%d/%m/%Y')[0] 

def yes_close(symbol):
    sym = f"NSE:{symbol}-EQ"
    print(sym)
    data_for_day = hist_df(sym, yes, "1D")  # 5-minute candles for the specified day
    print(data_for_day)
    return data_for_day['Close'][0]

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
    with open(r'C:\Users\akshatx.agrawal\Strategies\utils\options_details_cache.json', 'r') as f:
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

for i in df['symbol'][:2]:
    underlying_price = yes_close(i)
    print(underlying_price)
    result = get_option_symbols_in_range(i, underlying_price, percentile_range=5)
    print(result)
