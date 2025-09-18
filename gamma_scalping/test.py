import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# Fix the import path - go up one level to find utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def extract_underlying_info(options_chain: List[Dict]) -> Dict[str, Any]:
    """Extract underlying price and market info"""
    for record in options_chain:
        if record.get('option_type') == '' and record.get('strike_price') == -1:
            return {
                'symbol': record.get('symbol'),
                'ltp': record.get('ltp'),
                'fp': record.get('fp'),  # Futures price (more relevant for options)
                'ltpch': record.get('ltpch'),  # Price change
                'ltpchp': record.get('ltpchp'),  # Price change %
                'description': record.get('description')
            }
    return {}

def extract_vix_info(data: Dict) -> Dict[str, Any]:
    """Extract India VIX information"""
    vix_data = data.get('indiavixData', {})
    return {
        'ltp': vix_data.get('ltp'),
        'ltpch': vix_data.get('ltpch'),
        'ltpchp': vix_data.get('ltpchp'),
        'bid': vix_data.get('bid'),
        'ask': vix_data.get('ask')
    }

def extract_option_data(options_chain: List[Dict]) -> Dict[str, List[Dict]]:
    """Extract structured option data by strike"""
    options_by_strike = {}
    
    for record in options_chain:
        if record.get('option_type') in ['CE', 'PE']:
            strike = record.get('strike_price')
            option_type = record.get('option_type')
            
            if strike not in options_by_strike:
                options_by_strike[strike] = {'CE': None, 'PE': None}
            
            option_data = {
                'symbol': record.get('symbol'),
                'option_type': option_type,
                'strike': strike,
                'ltp': record.get('ltp'),
                'bid': record.get('bid'),
                'ask': record.get('ask'),
                'oi': record.get('oi'),
                'oich': record.get('oich'),  # OI change
                'oichp': record.get('oichp'),  # OI change %
                'volume': record.get('volume'),
                'ltpch': record.get('ltpch'),  # Price change
                'ltpchp': record.get('ltpchp'),  # Price change %
                'spread': (record.get('ask', 0) - record.get('bid', 0)) if record.get('ask') and record.get('bid') else None,
                'spread_pct': ((record.get('ask', 0) - record.get('bid', 0)) / record.get('ltp', 1) * 100) if record.get('ask') and record.get('bid') and record.get('ltp') else None
            }
            
            options_by_strike[strike][option_type] = option_data
    
    return options_by_strike

def calculate_atm_strike(underlying_price: float, strike_step: int = 50) -> int:
    """Calculate ATM strike based on underlying price"""
    return int(round(underlying_price / strike_step) * strike_step)

def analyze_liquidity(options_by_strike: Dict) -> Dict[str, Any]:
    """Analyze liquidity metrics"""
    liquidity_analysis = {
        'total_ce_oi': 0,
        'total_pe_oi': 0,
        'total_ce_volume': 0,
        'total_pe_volume': 0,
        'avg_ce_spread': 0,
        'avg_pe_spread': 0,
        'ce_count': 0,
        'pe_count': 0,
        'strikes_analyzed': len(options_by_strike)
    }
    
    ce_spreads = []
    pe_spreads = []
    
    for strike, options in options_by_strike.items():
        if options['CE']:
            liquidity_analysis['total_ce_oi'] += options['CE']['oi'] or 0
            liquidity_analysis['total_ce_volume'] += options['CE']['volume'] or 0
            liquidity_analysis['ce_count'] += 1
            if options['CE']['spread']:
                ce_spreads.append(options['CE']['spread'])
        
        if options['PE']:
            liquidity_analysis['total_pe_oi'] += options['PE']['oi'] or 0
            liquidity_analysis['total_pe_volume'] += options['PE']['volume'] or 0
            liquidity_analysis['pe_count'] += 1
            if options['PE']['spread']:
                pe_spreads.append(options['PE']['spread'])
    
    liquidity_analysis['avg_ce_spread'] = sum(ce_spreads) / len(ce_spreads) if ce_spreads else 0
    liquidity_analysis['avg_pe_spread'] = sum(pe_spreads) / len(pe_spreads) if pe_spreads else 0
    
    return liquidity_analysis

def extract_expiry_info(data: Dict) -> List[Dict]:
    """Extract expiry information"""
    expiry_data = data.get('expiryData', [])
    return [
        {
            'date': exp.get('date'),
            'expiry': exp.get('expiry'),
            'days_to_expiry': (datetime.strptime(exp.get('date'), '%d-%m-%Y') - datetime.now()).days if exp.get('date') else None
        }
        for exp in expiry_data
    ]

def extract_option_chain_analysis(symbol: str, strike_count: int = 3) -> Dict[str, Any]:
    """Main function to extract all relevant option chain data"""
    try:
        from utils.api_utils import get_option_chain
        
        print(f"Fetching option chain for {symbol}...")
        response = get_option_chain(symbol, strike_count)
        
        if response.get('s') != 'ok':
            return {'error': f"API failed: {response}"}
        
        data = response.get('data', {})
        options_chain = data.get('optionsChain', [])
        
        # Extract underlying info
        underlying = extract_underlying_info(options_chain)
        
        # Calculate ATM strike
        atm_strike = calculate_atm_strike(underlying.get('fp', underlying.get('ltp', 0)))
        
        # Extract option data by strike
        options_by_strike = extract_option_data(options_chain)
        
        # Analyze liquidity
        liquidity_analysis = analyze_liquidity(options_by_strike)
        
        # Extract VIX info
        vix_info = extract_vix_info(data)
        
        # Extract expiry info
        expiry_info = extract_expiry_info(data)
        
        # Create structured analysis
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'underlying': underlying,
            'atm_strike': atm_strike,
            'vix_info': vix_info,
            'liquidity_analysis': liquidity_analysis,
            'expiry_info': expiry_info,
            'options_by_strike': options_by_strike,
            'market_summary': {
                'current_price': underlying.get('fp') or underlying.get('ltp'),
                'futures_premium': (underlying.get('fp', 0) - underlying.get('ltp', 0)) if underlying.get('fp') and underlying.get('ltp') else 0,
                'vix_level': vix_info.get('ltp'),
                'vix_change': vix_info.get('ltpch'),
                'atm_strike': atm_strike,
                'strikes_available': len(options_by_strike),
                'ce_pe_ratio': liquidity_analysis['total_ce_oi'] / liquidity_analysis['total_pe_oi'] if liquidity_analysis['total_pe_oi'] > 0 else float('inf')
            }
        }
        
        return analysis
        
    except Exception as e:
        return {'error': f"Extraction failed: {str(e)}"}

def print_analysis_summary(analysis: Dict[str, Any]):
    """Print a clean summary of the analysis"""
    if 'error' in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    print("\n" + "="*60)
    print("OPTION CHAIN ANALYSIS SUMMARY")
    print("="*60)
    
    # Market Overview
    underlying = analysis['underlying']
    market_summary = analysis['market_summary']
    
    print(f"📊 MARKET OVERVIEW")
    print(f"   Symbol: {analysis['symbol']}")
    print(f"   Current Price: {underlying.get('ltp'):.2f}")
    print(f"   Futures Price: {underlying.get('fp'):.2f}")
    print(f"   Futures Premium: {market_summary['futures_premium']:.2f}")
    print(f"   ATM Strike: {analysis['atm_strike']}")
    print(f"   India VIX: {market_summary['vix_level']:.2f} ({market_summary['vix_change']:+.2f})")
    
    # Liquidity Analysis
    liq = analysis['liquidity_analysis']
    print(f"\n💰 LIQUIDITY ANALYSIS")
    print(f"   Total CE OI: {liq['total_ce_oi']:,}")
    print(f"   Total PE OI: {liq['total_pe_oi']:,}")
    print(f"   CE/PE OI Ratio: {market_summary['ce_pe_ratio']:.2f}")
    print(f"   Avg CE Spread: {liq['avg_ce_spread']:.2f}")
    print(f"   Avg PE Spread: {liq['avg_pe_spread']:.2f}")
    print(f"   Total CE Volume: {liq['total_ce_volume']:,}")
    print(f"   Total PE Volume: {liq['total_pe_volume']:,}")
    
    # Option Details by Strike
    print(f"\n📈 OPTION DETAILS BY STRIKE")
    for strike, options in analysis['options_by_strike'].items():
        print(f"\n   Strike: {strike}")
        if options['CE']:
            ce = options['CE']
            print(f"     CE: {ce['ltp']:.2f} (B:{ce['bid']:.2f}/A:{ce['ask']:.2f}) | OI:{ce['oi']:,} | Vol:{ce['volume']:,} | Spread:{ce['spread']:.2f}")
        if options['PE']:
            pe = options['PE']
            print(f"     PE: {pe['ltp']:.2f} (B:{pe['bid']:.2f}/A:{pe['ask']:.2f}) | OI:{pe['oi']:,} | Vol:{pe['volume']:,} | Spread:{pe['spread']:.2f}")
    
    # Trading Recommendations
    print(f"\n🎯 TRADING INSIGHTS")
    if market_summary['ce_pe_ratio'] > 1.5:
        print(f"   ⚠️  Heavy CE activity - potential upside pressure")
    elif market_summary['ce_pe_ratio'] < 0.7:
        print(f"   ⚠️  Heavy PE activity - potential downside pressure")
    else:
        print(f"   ✅ Balanced CE/PE activity")
    
    if liq['avg_ce_spread'] > liq['avg_pe_spread'] * 1.5:
        print(f"   ⚠️  CE spreads wider - expect slippage on CE side")
    
    if market_summary['vix_level'] > 15:
        print(f"   ⚠️  High VIX ({market_summary['vix_level']:.2f}) - expect high volatility")
    elif market_summary['vix_level'] < 10:
        print(f"   ✅ Low VIX ({market_summary['vix_level']:.2f}) - low volatility expected")

# Test the extraction
if __name__ == "__main__":
    try:
        # Test with NIFTY
        analysis = extract_option_chain_analysis("NSE:NIFTY50-INDEX", 3)
        print_analysis_summary(analysis)
        
        # Test with BANKNIFTY
        print("\n" + "="*60)
        print("TESTING BANKNIFTY")
        print("="*60)
        banknifty_analysis = extract_option_chain_analysis("NSE:NIFTYBANK-INDEX", 3)
        print_analysis_summary(banknifty_analysis)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")