from fuzzywuzzy import process
import re

# Fixed dictionary of Nifty 50 stocks
symbol_name_map = {
    "RELIANCE": "Reliance Industries",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "INFY": "Infosys",
    "TCS": "Tata Consultancy Services",
    "LT": "Larsen & Toubro",
    "AXISBANK": "Axis Bank",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel",
    "BAJFINANCE": "Bajaj Finance",
    "ITC": "ITC Limited",
    "HINDUNILVR": "Hindustan Unilever",
    "HCLTECH": "HCL Technologies",
    "SUNPHARMA": "Sun Pharmaceutical",
    "MARUTI": "Maruti Suzuki",
    "ULTRACEMCO": "UltraTech Cement",
    "M&M": "Mahindra & Mahindra",
    "NTPC": "NTPC Limited",
    "TITAN": "Titan Company",
    "POWERGRID": "Power Grid Corporation",
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports and SEZ",
    "WIPRO": "Wipro Limited",
    "JSWSTEEL": "JSW Steel",
    "COALINDIA": "Coal India",
    "TATAMOTORS": "Tata Motors",
    "NESTLEIND": "Nestle India",
    "ASIANPAINT": "Asian Paints",
    "BAJAJ-AUTO": "Bajaj Auto",
    "BEL": "Bharat Electronics",
    "GRASIM": "Grasim Industries",
    "TRENT": "Trent Limited",
    "TATASTEEL": "Tata Steel",
    "SBILIFE": "SBI Life Insurance",
    "JIOFIN": "Jio Financial Services",
    "EICHERMOT": "Eicher Motors",
    "HDFCLIFE": "HDFC Life",
    "TECHM": "Tech Mahindra",
    "HINDALCO": "Hindalco Industries",
    "CIPLA": "Cipla Limited",
    "SHRIRAMFIN": "Shriram Finance",
    "TATACONSUM": "Tata Consumer Products",
    "APOLLOHOSP": "Apollo Hospitals",
    "DRREDDY": "Dr. Reddy's Laboratories",
    "HEROMOTOCO": "Hero Motocorp",
    "INDUSINDBK": "IndusInd Bank"
}

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9]', '', text.lower())

def search_symbol(user_input, symbol_name_map):
    user_input_clean = clean_text(user_input)

    # 1. Direct match with SYMBOL
    for symbol in symbol_name_map.keys():
        if user_input_clean == clean_text(symbol):
            print(f"Exact symbol match found: {symbol} ({symbol_name_map[symbol]})")
            return symbol

    # 2. Substring match with company names
    matched_stocks = {}
    for symbol, company_name in symbol_name_map.items():
        clean_company_name = clean_text(company_name)
        if user_input_clean in clean_company_name:
            matched_stocks[symbol] = company_name

    if matched_stocks:
        if len(matched_stocks) == 1:
            symbol, company_name = list(matched_stocks.items())[0]
            print(f"Direct substring match found: {company_name} ({symbol})")
            return symbol
        else:
            print("Multiple substring matches found:")
            for idx, (symbol, company) in enumerate(matched_stocks.items()):
                print(f"{idx + 1}. {company} ({symbol})")
            choice = int(input("Enter the number corresponding to your choice: ")) - 1
            selected_symbol = list(matched_stocks.keys())[choice]
            return selected_symbol
    else:
        # 3. Fallback: fuzzy match on company names
        company_names = list(symbol_name_map.values())
        matches = process.extract(user_input, company_names, limit=5)
        matches = [match for match in matches if match[1] > 60]

        if not matches:
            print("No close matches found. Please refine your input.")
            return None
        
        if len(matches) == 1:
            best_match = matches[0][0]
            for symbol, company in symbol_name_map.items():
                if company == best_match:
                    print(f"Fuzzy match found: {best_match} ({symbol})")
                    return symbol
        else:
            print("Multiple fuzzy matches found:")
            for idx, match in enumerate(matches):
                print(f"{idx + 1}. {match[0]} ({match[1]}% match)")
            choice = int(input("Enter the number corresponding to your choice: ")) - 1
            best_match = matches[choice][0]
            for symbol, company in symbol_name_map.items():
                if company == best_match:
                    return symbol

if __name__ == "__main__":
    print("Nifty 50 Stocks Loaded Successfully.")

    while True:
        query = input("\nEnter company name or symbol (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        symbol = search_symbol(query, symbol_name_map)
        if symbol:
            print(f"Mapped symbol: {symbol}")
