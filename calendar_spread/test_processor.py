"""
Test the simplified Nifty 50 processor
"""
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calendar_spread.nifty50_processor import Nifty50Processor

def main():
    """Test the processor"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        print("🧪 Testing Nifty 50 Processor")
        print("=" * 50)
        
        # Create processor
        processor = Nifty50Processor()
        
        # Test basic functionality
        print("\n1. Testing symbol conversion...")
        test_companies = ['Reliance Industries', 'TCS', 'HDFC Bank']
        for company in test_companies:
            if company in processor.symbol_mapping:
                symbol = processor.symbol_mapping[company]
                print(f"   ✅ {company} → {symbol}")
            else:
                print(f"   ❌ {company} → Not found")
        
        print("\n2. Testing lot size lookup...")
        test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK']
        for symbol in test_symbols:
            lot_size = processor.get_lot_size(symbol)
            print(f"   {symbol}: Lot size = {lot_size}")
        
        print("\n3. Testing NSE symbol format...")
        for symbol in test_symbols:
            nse_symbol = processor.get_nse_symbol(symbol)
            print(f"   {symbol} → {nse_symbol}")
        
        print("\n4. Testing option chain (sample)...")
        # Test with one symbol
        test_symbol = 'TCS'
        print(f"   Getting option chain for {test_symbol}...")
        chain_data = processor.get_option_chain_data(test_symbol, strikecount=2)
        
        if chain_data:
            print(f"   ✅ Got option chain data")
            print(f"   Expiries: {len(chain_data.get('expiryData', []))}")
            print(f"   Options: {len(chain_data.get('optionsChain', []))}")
        else:
            print(f"   ❌ Failed to get option chain data")
        
        print("\n5. Testing ATM strike (sample)...")
        atm_data = processor.get_atm_strike(test_symbol)
        if atm_data:
            print(f"   ✅ {test_symbol}:")
            print(f"      Underlying Price: {atm_data['underlying_price']}")
            print(f"      ATM Strike: {atm_data['atm_strike']}")
            print(f"      CE Symbol: {atm_data['ce_symbol']}")
            print(f"      PE Symbol: {atm_data['pe_symbol']}")
            print(f"      Lot Size: {atm_data['lot_size']}")
        else:
            print(f"   ❌ Failed to get ATM strike for {test_symbol}")
        
        print("\n6. Testing recommended stocks...")
        recommended = processor.get_recommended_stocks(5)
        print(f"   Found {len(recommended)} recommended stocks:")
        for stock in recommended:
            print(f"      {stock['company_name']} ({stock['symbol']}) - Lot: {stock['lot_size']}")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
