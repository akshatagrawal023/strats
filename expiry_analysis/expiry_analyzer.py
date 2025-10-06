"""
Expiry Day Analyzer
Analyzes patterns in expiry day movements, volume, and option chain data
"""

import json
import pandas as pd
from typing import Dict, List
from datetime import datetime

class ExpiryAnalyzer:
    def __init__(self, data_file: str = "expiry_analysis/expiry_data.json"):
        self.data_file = data_file
        self.data = self.load_data()
    
    def load_data(self) -> Dict:
        """Load expiry data from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return {}
    
    def analyze_wild_movers(self) -> pd.DataFrame:
        """Analyze stocks with wild movements"""
        wild_movers = []
        
        for symbol, stock_data in self.data.items():
            stock_info = stock_data.get('stock_data')
            option_info = stock_data.get('option_data')
            
            if stock_info and stock_info.get('intraday_high_low_pct', 0) > 3:
                wild_movers.append({
                    'symbol': symbol,
                    'company_name': stock_data['company_name'],
                    'intraday_range_pct': stock_info['intraday_high_low_pct'],
                    'price_change_pct': stock_info['price_change_pct'],
                    'volume_spike_ratio': stock_info['volume_spike_ratio'],
                    'total_volume': stock_info['total_volume'],
                    'put_call_ratio': option_info.get('put_call_ratio', 0) if option_info else 0,
                    'max_ce_oi': option_info.get('max_oi_ce', {}).get('oi', 0) if option_info else 0,
                    'max_pe_oi': option_info.get('max_oi_pe', {}).get('oi', 0) if option_info else 0
                })
        
        df = pd.DataFrame(wild_movers)
        return df.sort_values('intraday_range_pct', ascending=False)
    
    def analyze_volume_patterns(self) -> pd.DataFrame:
        """Analyze volume patterns"""
        volume_data = []
        
        for symbol, stock_data in self.data.items():
            stock_info = stock_data.get('stock_data')
            
            if stock_info:
                volume_data.append({
                    'symbol': symbol,
                    'company_name': stock_data['company_name'],
                    'total_volume': stock_info['total_volume'],
                    'avg_volume': stock_info['avg_volume'],
                    'max_volume': stock_info['max_volume'],
                    'volume_spike_ratio': stock_info['volume_spike_ratio'],
                    'price_change_pct': stock_info['price_change_pct'],
                    'intraday_range_pct': stock_info['intraday_high_low_pct']
                })
        
        df = pd.DataFrame(volume_data)
        return df.sort_values('volume_spike_ratio', ascending=False)
    
    def analyze_option_chain_patterns(self) -> pd.DataFrame:
        """Analyze option chain patterns"""
        option_data = []
        
        for symbol, stock_data in self.data.items():
            option_info = stock_data.get('option_data')
            stock_info = stock_data.get('stock_data')
            
            if option_info and stock_info:
                option_data.append({
                    'symbol': symbol,
                    'company_name': stock_data['company_name'],
                    'put_call_ratio': option_info['put_call_ratio'],
                    'total_ce_oi': option_info['total_ce_oi'],
                    'total_pe_oi': option_info['total_pe_oi'],
                    'total_ce_volume': option_info['total_ce_volume'],
                    'total_pe_volume': option_info['total_pe_volume'],
                    'price_change_pct': stock_info['price_change_pct'],
                    'volume_spike_ratio': stock_info['volume_spike_ratio']
                })
        
        df = pd.DataFrame(option_data)
        return df.sort_values('put_call_ratio', ascending=False)
    
    def print_analysis_report(self):
        """Print comprehensive analysis report"""
        print("=" * 80)
        print("📊 EXPIRY DAY ANALYSIS REPORT")
        print("=" * 80)
        
        # Wild movers analysis
        wild_movers_df = self.analyze_wild_movers()
        if not wild_movers_df.empty:
            print(f"\n🎢 WILD MOVERS (Intraday Range > 3%):")
            print("-" * 80)
            for _, row in wild_movers_df.head(10).iterrows():
                print(f"{row['company_name']:<25} ({row['symbol']:<12}) | Range: {row['intraday_range_pct']:5.1f}% | Change: {row['price_change_pct']:6.1f}% | Vol Spike: {row['volume_spike_ratio']:4.1f}x | PCR: {row['put_call_ratio']:4.2f}")
        
        # Volume analysis
        volume_df = self.analyze_volume_patterns()
        if not volume_df.empty:
            print(f"\n📈 VOLUME SPIKERS (Volume Spike > 2x):")
            print("-" * 80)
            volume_spikers = volume_df[volume_df['volume_spike_ratio'] > 2]
            for _, row in volume_spikers.head(10).iterrows():
                print(f"{row['company_name']:<25} ({row['symbol']:<12}) | Vol Spike: {row['volume_spike_ratio']:4.1f}x | Total Vol: {row['total_volume']:>10,} | Change: {row['price_change_pct']:6.1f}%")
        
        # Option chain analysis
        option_df = self.analyze_option_chain_patterns()
        if not option_df.empty:
            print(f"\n📊 OPTION CHAIN PATTERNS:")
            print("-" * 80)
            
            # High PCR stocks
            high_pcr = option_df[option_df['put_call_ratio'] > 1.2]
            if not high_pcr.empty:
                print(f"High Put-Call Ratio (>1.2):")
                for _, row in high_pcr.head(5).iterrows():
                    print(f"  {row['company_name']:<25} ({row['symbol']:<12}) | PCR: {row['put_call_ratio']:4.2f} | Change: {row['price_change_pct']:6.1f}%")
            
            # Low PCR stocks
            low_pcr = option_df[option_df['put_call_ratio'] < 0.8]
            if not low_pcr.empty:
                print(f"\nLow Put-Call Ratio (<0.8):")
                for _, row in low_pcr.head(5).iterrows():
                    print(f"  {row['company_name']:<25} ({row['symbol']:<12}) | PCR: {row['put_call_ratio']:4.2f} | Change: {row['price_change_pct']:6.1f}%")
        
        # Correlation analysis
        print(f"\n🔍 CORRELATION INSIGHTS:")
        print("-" * 80)
        
        if not option_df.empty:
            # Volume spike vs price movement correlation
            volume_price_corr = volume_df['volume_spike_ratio'].corr(volume_df['price_change_pct'])
            print(f"Volume Spike vs Price Change Correlation: {volume_price_corr:.3f}")
            
            # PCR vs price movement correlation
            pcr_price_corr = option_df['put_call_ratio'].corr(option_df['price_change_pct'])
            print(f"Put-Call Ratio vs Price Change Correlation: {pcr_price_corr:.3f}")
            
            # Volume vs PCR correlation
            volume_pcr_corr = volume_df['volume_spike_ratio'].corr(option_df['put_call_ratio'])
            print(f"Volume Spike vs Put-Call Ratio Correlation: {volume_pcr_corr:.3f}")
        
        print("=" * 80)

def main():
    """Main function to run expiry analysis"""
    analyzer = ExpiryAnalyzer()
    
    if not analyzer.data:
        print("❌ No expiry data found. Run expiry_data_fetcher.py first.")
        return
    
    analyzer.print_analysis_report()

if __name__ == "__main__":
    main()
