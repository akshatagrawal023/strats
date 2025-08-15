import re
import requests
import json
import time
import os
import csv

def fetch_options_data():
    print("Fetching option symbol data from web")
    # Fetch JSON data
    try:
        URL = "https://public.fyers.in/sym_details/NSE_FO_sym_master.json"
        response = requests.get(URL)

        if response.status_code == 200:
            print("API Response received")
            data = response.json()
            # Extract all symbols
            all_symbols = list(data.keys())
            option_symbols = [sym for sym in all_symbols if 'CE' in sym or 'PE' in sym]
            future_symbols = [sym for sym in all_symbols if 'FUT' in sym]
            print(f"Found {len(option_symbols)} option symbols and {len(future_symbols)} future symbols")

            symbol_details = {}
            symbol_pattern = r'NSE:([A-Z]+)'
            
            # Process options
            for option in option_symbols:
                match = re.match(symbol_pattern, option)
                if match:
                    symbol = match.group(1)
                    
                    # Initialize for new symbols
                    if symbol not in symbol_details:
                        symbol_details[symbol] = {
                            'strikes': set(),
                            'expiries': set(),
                            'futures': [],
                            'ce_symbols': set(),
                            'pe_symbols': set()
                        }
                    
                    # Extract expiry and strike
                    details_pattern = r'NSE:' + symbol + r'(\d+[A-Z]+)(\d+)([CP]E)'
                    details_match = re.match(details_pattern, option)
                    
                    if details_match:
                        expiry = details_match.group(1)  # Expiry (e.g., 25MAY)
                        strike = details_match.group(2)  # Strike price (e.g., 4400)
                        
                        # Add to sets in symbol_details
                        symbol_details[symbol]['strikes'].add(strike)
                        symbol_details[symbol]['expiries'].add(expiry)
                    
                    # Add to ce_symbols or pe_symbols
                    if option.endswith('CE'):
                        symbol_details[symbol]['ce_symbols'].add(option)
                    elif option.endswith('PE'):
                        symbol_details[symbol]['pe_symbols'].add(option)
            
            # Process futures - just group them by symbol
            for future in future_symbols:
                match = re.match(symbol_pattern, future)
                if match:
                    symbol = match.group(1)
                    
                    # Add to symbol_details if it exists or create new entry
                    if symbol in symbol_details:
                        symbol_details[symbol]['futures'].append(future)
                    else:
                        # Symbol has futures but no options
                        symbol_details[symbol] = {
                            'strikes': set(),
                            'expiries': set(),
                            'futures': [future],
                            'ce_symbols': set(),
                            'pe_symbols': set()
                        }
            
            # Convert sets to sorted lists for better readability
            for symbol in symbol_details:
                symbol_details[symbol]['strikes'] = sorted(list(symbol_details[symbol]['strikes']), key=int)
                symbol_details[symbol]['expiries'] = sorted(list(symbol_details[symbol]['expiries']))
                symbol_details[symbol]['ce_symbols'] = sorted(list(symbol_details[symbol]['ce_symbols']))
                symbol_details[symbol]['pe_symbols'] = sorted(list(symbol_details[symbol]['pe_symbols']))
            
            # Write to CSV
            with open('utils/FnO_list.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Stocks'])  # Write header
                for stock in symbol_details.keys():
                    writer.writerow([stock])
    
        else:
            print(f"API returned status code: {response.status_code}")
            symbol_details = load_from_cache()
        
    except Exception as e:
        print(f"Error fetching data: {e}")

        

    return symbol_details

def get_options_data(symbol=None, data_type = 'details', use_cache=True):
    """
    Get options data for all symbols or a specific symbol.
    Returns:
        - If symbol is None: symbol_details dict
        - If symbol is provided: (ce_symbols, pe_symbols, details_dict) tuple for that symbol
    """

    symbol_details = {}

    details_cache_file = 'utils/options_details_cache.json'

    if use_cache and os.path.exists(details_cache_file):
        try:
            cache_time = os.path.getmtime(details_cache_file)
            # If cache is less than 1 day old, use it
            if (time.time() - cache_time) < 24 * 3600:
                with open(details_cache_file, 'r') as f:
                    symbol_details = json.load(f)
                print(f"Loaded data for {len(symbol_details)} symbols from cache")
            
            else:
                symbol_details = fetch_options_data()
                
        except Exception as e:
            print(f"Error loading from cache: {e}")

    # Fetch fresh data if cache not available or not fresh
    else:
        symbol_details = fetch_options_data()

    # Save to cache
    with open(details_cache_file, 'w') as f:
        json.dump(symbol_details, f)

    if symbol is None:
        return symbol_details

    symbol_data = symbol_details.get(symbol, {})

    # Filter based on data_type
    match data_type:
        case "ce":
            return symbol_data.get("ce_symbols", [])
        case "pe":
            return symbol_data.get("pe_symbols", [])
        case "futures":
            return symbol_data.get("futures", [])
        case "option_symbols":
            return {
                "ce_symbols": symbol_data.get("ce_symbols", []),
                "pe_symbols": symbol_data.get("pe_symbols", [])
            }
        case "strikes":
            return symbol_data.get("strikes", [])
        case "expires":
            return symbol_data.get("expires", [])
        case "details":
            return symbol_data
        case _:
            raise ValueError(f"Unknown data_type: {data_type}")

# Simple example
if __name__ == "__main__":
    # Get all data in a single call
    symbol_details = get_options_data()
    print(f"Found data for {len(symbol_details)} symbols")
    
    # Get data for a specific stock (e.g., NIFTY)
    symbol = "NIFTY"
    print(get_options_data(symbol).get('futures'))

   
    
    # if details:
    #     print(f"\n{symbol} details:")
    #     print(f"CE symbols count: {len(ce_symbols)}")
    #     print(f"PE symbols count: {len(pe_symbols)}")
        
    #     if details.get('expiries'):
    #         print(f"Expiries: {details['expiries']}")
        
    #     if details.get('strikes'):
    #         print(f"Strikes: {details['strikes']}")
        
    #     if details.get('futures'):
    #         print(f"Futures: {details['futures']}")
    # else:
    #     print(f"\nNo data found for {symbol}")
        
    # # Find symbols with both options and futures - add safety check
    # symbols_with_both = []
    # for sym, sym_details in symbol_details.items():
    #     has_futures = sym_details.get('futures', [])
    #     has_options = sym_details.get('strikes', []) or sym_details.get('expiries', [])
    #     if has_futures and has_options:
    #         symbols_with_both.append(sym)
    
    # print(f"\nFound {len(symbols_with_both)} symbols with both options and futures") 
