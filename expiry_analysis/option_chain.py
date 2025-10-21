import sqlite3
import pandas as pd
import os, sys
from datetime import datetime
from pandas.tseries.offsets import BDay
import json
from stream_store import StreamStore
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.historical_data import hist_df, hist_data

from utils.api_utils import get_option_chain

conn = sqlite3.connect("utils/symbol_mappings.db")
df = pd.read_sql_query("SELECT * FROM symbol_mappings WHERE is_active = 1", conn)
conn.close()

prev_day = 1
today = pd.date_range(start=datetime.today(), end = datetime.today())
yes = (today-BDay(prev_day)).strftime('%d/%m/%Y')[0] 

while True:
    for i in df['symbol'][:2]:
        # print(i)
        sym = f"NSE:{i}-EQ"
        result = get_option_chain(sym)
        print(result)

# 1) Initialize store; this controls your in-memory queue size for chains
store = StreamStore(db_path="expiry_analysis/stream_data.db",
                    max_chain_snapshots=300)  # set your fixed window here

def fetch_and_store_chain(ueq: str):
    resp = get_option_chain(ueq, strikecount=50)  # tune strikecount
    ts = datetime.now()
    store.ingest_option_chain_payload(ueq, resp, ts)

