"""
Simple script to add new companies to the symbol database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from symbol_database import add_symbol, get_symbol_db

def add_new_company():
    """Interactive script to add new companies"""
    print("🏢 ADD NEW COMPANY TO SYMBOL DATABASE")
    print("=" * 50)
    
    company_name = input("Enter company name: ").strip()
    if not company_name:
        print("❌ Company name cannot be empty")
        return
    
    symbol = input("Enter symbol: ").strip().upper()
    if not symbol:
        print("❌ Symbol cannot be empty")
        return
    
    sector = input("Enter sector (optional): ").strip() or None
    
    market_cap_input = input("Enter market cap in Cr (optional): ").strip()
    market_cap = None
    if market_cap_input:
        try:
            market_cap = float(market_cap_input.replace(',', ''))
        except:
            print("⚠️  Invalid market cap format, skipping...")
    
    lot_size_input = input("Enter lot size (optional): ").strip()
    lot_size = None
    if lot_size_input:
        try:
            lot_size = int(lot_size_input)
        except:
            print("⚠️  Invalid lot size format, skipping...")
    
    # Add to database
    if add_symbol(company_name, symbol, sector, market_cap, None, lot_size):
        print(f"✅ Successfully added: {company_name} → {symbol}")
        
        # Show updated database stats
        db = get_symbol_db()
        total = len(db.get_all_mappings())
        print(f"📊 Total companies in database: {total}")
    else:
        print("❌ Failed to add company")

if __name__ == "__main__":
    add_new_company()
