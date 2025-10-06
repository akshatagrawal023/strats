import sqlite3
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.historical_data import hist_df, hist_data
# Connect to database
conn = sqlite3.connect("utils/symbol_mappings.db")

# Read all data
df = pd.read_sql_query("SELECT * FROM symbol_mappings WHERE is_active = 1", conn)

# Display the data
print(f"Total records: {len(df)}")
print("\nData:")
print(df[['company_name', 'symbol', 'sector', 'market_cap', 'pe_ratio', 'lot_size']].to_string(index=False))

# Close connection
conn.close()