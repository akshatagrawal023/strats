"""
Universal SQLite Database Manager for Symbol Mappings
Streamlined version for managing symbol mappings across all trading strategies
"""

import sqlite3
import os
import logging
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime

class SymbolDatabase:
    """Streamlined symbol database manager using SQLite"""
    
    def __init__(self, db_path: str = "utils/symbol_mappings.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("SymbolDatabase")
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS symbol_mappings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_name TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        sector TEXT,
                        market_cap REAL,
                        pe_ratio REAL,
                        lot_size INTEGER,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_company_name ON symbol_mappings(company_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON symbol_mappings(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_active ON symbol_mappings(is_active)')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def add_symbol(self, company_name: str, symbol: str, sector: str = None, 
                   market_cap: float = None, pe_ratio: float = None, 
                   lot_size: int = None) -> bool:
        """Add or update a symbol mapping"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO symbol_mappings 
                    (company_name, symbol, sector, market_cap, pe_ratio, lot_size, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (company_name.strip(), symbol.strip(), sector, market_cap, pe_ratio, lot_size))
                conn.commit()
                self.logger.info(f"Added/Updated: {company_name} → {symbol}")
                return True
        except Exception as e:
            self.logger.error(f"Error adding symbol {company_name}: {e}")
            return False
    
    def get_symbol(self, company_name: str) -> Optional[str]:
        """Get symbol for a company name with fuzzy matching"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Direct lookup
                cursor.execute('''
                    SELECT symbol FROM symbol_mappings 
                    WHERE company_name = ? AND is_active = 1
                ''', (company_name.strip(),))
                
                result = cursor.fetchone()
                if result:
                    return result[0]
                
                # Try variations
                variations = [
                    company_name.replace(" Limited", ""),
                    company_name.replace(" Ltd", ""),
                    company_name.replace(" Corporation", ""),
                    company_name.replace(" Corp", ""),
                    company_name.replace(" Industries", ""),
                    company_name.replace(" & ", " "),
                    company_name.replace(" and ", " ")
                ]
                
                for variation in variations:
                    if variation.strip():
                        cursor.execute('''
                            SELECT symbol FROM symbol_mappings 
                            WHERE company_name = ? AND is_active = 1
                        ''', (variation.strip(),))
                        
                        result = cursor.fetchone()
                        if result:
                            return result[0]
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting symbol for {company_name}: {e}")
            return None
    
    def get_all_mappings(self) -> Dict[str, str]:
        """Get all active symbol mappings"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT company_name, symbol FROM symbol_mappings 
                    WHERE is_active = 1 ORDER BY company_name
                ''')
                return dict(cursor.fetchall())
        except Exception as e:
            self.logger.error(f"Error getting all mappings: {e}")
            return {}
    
    def search_companies(self, query: str) -> List[Dict]:
        """Search for companies by name or symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT company_name, symbol, sector, market_cap, lot_size 
                    FROM symbol_mappings 
                    WHERE (company_name LIKE ? OR symbol LIKE ?) AND is_active = 1
                    ORDER BY company_name
                ''', (f'%{query}%', f'%{query}%'))
                
                columns = ['company_name', 'symbol', 'sector', 'market_cap', 'lot_size']
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                return results
        except Exception as e:
            self.logger.error(f"Error searching companies: {e}")
            return []
    
    def bulk_insert(self, mappings: List[Tuple]):
        """Bulk insert symbol mappings"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany('''
                    INSERT OR REPLACE INTO symbol_mappings 
                    (company_name, symbol, sector, market_cap, pe_ratio, lot_size, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', mappings)
                conn.commit()
                self.logger.info(f"Bulk inserted {len(mappings)} mappings")
        except Exception as e:
            self.logger.error(f"Error in bulk insert: {e}")
    
    def export_to_csv(self, output_path: str = "utils/symbol_mappings_export.csv"):
        """Export all mappings to CSV"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('''
                    SELECT company_name, symbol, sector, market_cap, pe_ratio, lot_size 
                    FROM symbol_mappings WHERE is_active = 1 ORDER BY company_name
                ''', conn)
                df.to_csv(output_path, index=False)
                self.logger.info(f"Exported {len(df)} mappings to {output_path}")
                return True
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return False

def initialize_nifty50_data():
    """Initialize database with Nifty 50 companies"""
    try:
        nifty_df = pd.read_csv("utils/NiftyFifty.csv")
        
        # Symbol mappings
        symbol_mapping = {
            "Reliance Industries": "RELIANCE", "HDFC Bank": "HDFCBANK", "Bharti Airtel": "BHARTIARTL",
            "Tata Consultancy Services": "TCS", "ICICI Bank": "ICICIBANK", "State Bank of India": "SBIN",
            "Bajaj Finance ": "BAJFINANCE", "Infosys": "INFY", "Hindustan Unilever": "HINDUNILVR",
            "Maruti Suzuki": "MARUTI", "Larsen & Toubro": "LT", "ITC": "ITC",
            "Mahindra & Mahindra": "M&M", "Kotak Bank": "KOTAKBANK", "HCL Technologies": "HCLTECH",
            "Sun Pharmaceutical": "SUNPHARMA", "UltraTech Cement": "ULTRACEMCO", "Axis Bank": "AXISBANK",
            "NTPC": "NTPC", "Bajaj Finserv": "BAJAJFINSV", "Eternal": "ETERNAL",
            "Adani Ports & SEZ": "ADANIPORTS", "Titan": "TITAN", "Adani Enterprises": "ADANIENT",
            "Oil & Natural Gas Corporation": "ONGC", "Bharat Electronics": "BEL", "JSW Steel": "JSWSTEEL",
            "Power Grid Corporation of India": "POWERGRID", "Wipro": "WIPRO", "Tata Motors": "TATAMOTORS",
            "Bajaj Auto": "BAJAJ-AUTO", "Coal India": "COALINDIA", "Asian Paints": "ASIANPAINT",
            "Nestle": "NESTLEIND", "Tata Steel": "TATASTEEL", "Jio Financial Services": "JIOFIN",
            "Eicher Motors": "EICHERMOT", "Grasim Industries": "GRASIM", "SBI Life Insurance": "SBILIFE",
            "Trent": "TRENT", "Hindalco Industries": "HINDALCO", "HDFC Life Insurance": "HDFCLIFE",
            "Tech Mahindra": "TECHM", "Cipla": "CIPLA", "Shriram Finance": "SHRIRAMFIN",
            "Tata Consumer Products": "TATACONSUM", "Apollo Hospitals": "APOLLOHOSP",
            "Dr Reddys Laboratories": "DRREDDY", "Hero Motocorp": "HEROMOTOCO", "Indusind Bank": "INDUSINDBK"
        }
        
        # Sector mappings
        sector_mapping = {
            "RELIANCE": "Oil & Gas", "HDFCBANK": "Banking", "BHARTIARTL": "Telecom", "TCS": "IT",
            "ICICIBANK": "Banking", "SBIN": "Banking", "BAJFINANCE": "NBFC", "INFY": "IT",
            "HINDUNILVR": "FMCG", "MARUTI": "Automobile", "LT": "Engineering", "ITC": "FMCG",
            "M&M": "Automobile", "KOTAKBANK": "Banking", "HCLTECH": "IT", "SUNPHARMA": "Pharma",
            "ULTRACEMCO": "Cement", "AXISBANK": "Banking", "NTPC": "Power", "BAJAJFINSV": "NBFC",
            "ETERNAL": "Chemicals", "ADANIPORTS": "Infrastructure", "TITAN": "Consumer Goods",
            "ADANIENT": "Infrastructure", "ONGC": "Oil & Gas", "BEL": "Defense", "JSWSTEEL": "Steel",
            "POWERGRID": "Power", "WIPRO": "IT", "TATAMOTORS": "Automobile", "BAJAJ-AUTO": "Automobile",
            "COALINDIA": "Mining", "ASIANPAINT": "Paints", "NESTLEIND": "FMCG", "TATASTEEL": "Steel",
            "JIOFIN": "NBFC", "EICHERMOT": "Automobile", "GRASIM": "Textiles", "SBILIFE": "Insurance",
            "TRENT": "Retail", "HINDALCO": "Metals", "HDFCLIFE": "Insurance", "TECHM": "IT",
            "CIPLA": "Pharma", "SHRIRAMFIN": "NBFC", "TATACONSUM": "FMCG", "APOLLOHOSP": "Healthcare",
            "DRREDDY": "Pharma", "HEROMOTOCO": "Automobile", "INDUSINDBK": "Banking"
        }
        
        # Lot sizes from FnO CSV
        lot_sizes = {}
        try:
            lot_df = pd.read_csv("utils/FnO_lot_structured.csv")
            lot_sizes = lot_df.groupby('Symbol')['LotSize'].first().to_dict()
        except:
            pass
        
        mappings = []
        for _, row in nifty_df.iterrows():
            company_name = row['Name']
            symbol = symbol_mapping.get(company_name)
            
            if symbol:
                # Parse market cap and PE ratio
                market_cap = None
                pe_ratio = None
                try:
                    market_cap = float(str(row['Market Cap (Cr.)']).replace(',', ''))
                    pe_ratio = float(row['PE Ratio'])
                except:
                    pass
                
                mapping = (
                    company_name, symbol, sector_mapping.get(symbol),
                    market_cap, pe_ratio, lot_sizes.get(symbol)
                )
                mappings.append(mapping)
        
        db = SymbolDatabase()
        db.bulk_insert(mappings)
        return db
        
    except Exception as e:
        print(f"Error initializing Nifty 50 data: {e}")
        return None

# Global instance
_symbol_db = None

def get_symbol_db() -> SymbolDatabase:
    """Get the global symbol database instance"""
    global _symbol_db
    if _symbol_db is None:
        _symbol_db = SymbolDatabase()
        if not _symbol_db.get_all_mappings():
            initialize_nifty50_data()
    return _symbol_db

def get_symbol(company_name: str) -> Optional[str]:
    """Convenience function to get symbol"""
    return get_symbol_db().get_symbol(company_name)

def add_symbol(company_name: str, symbol: str, sector: str = None, 
               market_cap: float = None, pe_ratio: float = None, 
               lot_size: int = None) -> bool:
    """Convenience function to add symbol"""
    return get_symbol_db().add_symbol(company_name, symbol, sector, market_cap, pe_ratio, lot_size)

# CLI interface
def main():
    """Command line interface for database management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Symbol Database Manager')
    parser.add_argument('--init', action='store_true', help='Initialize database with Nifty 50 data')
    parser.add_argument('--add', nargs=2, metavar=('COMPANY', 'SYMBOL'), help='Add a new symbol mapping')
    parser.add_argument('--search', metavar='QUERY', help='Search for companies')
    parser.add_argument('--list', action='store_true', help='List all mappings')
    parser.add_argument('--export', metavar='FILE', help='Export to CSV')
    
    args = parser.parse_args()
    
    db = get_symbol_db()
    
    if args.init:
        print("Initializing database with Nifty 50 data...")
        initialize_nifty50_data()
        print("Database initialized successfully!")
    
    elif args.add:
        company, symbol = args.add
        if db.add_symbol(company, symbol):
            print(f"Added mapping: {company} → {symbol}")
        else:
            print("Failed to add mapping")
    
    elif args.search:
        results = db.search_companies(args.search)
        if results:
            print(f"Found {len(results)} matches:")
            for result in results:
                print(f"  {result['company_name']} → {result['symbol']} ({result.get('sector', 'N/A')})")
        else:
            print("No matches found")
    
    elif args.list:
        mappings = db.get_all_mappings()
        print(f"Total mappings: {len(mappings)}")
        for company, symbol in mappings.items():
            print(f"  {company} → {symbol}")
    
    elif args.export:
        if db.export_to_csv(args.export):
            print(f"Exported to {args.export}")
        else:
            print("Export failed")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
