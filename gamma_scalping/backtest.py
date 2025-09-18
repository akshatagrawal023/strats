import os
import sys
import json
from datetime import datetime, time as dtime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.historical_data import hist_data
from utils.symbol_utils import get_future_symbols
# from utils.option_symbols import get_options_data

def get_futures_data(symbol, timeframe):    
    futures = hist_data(symbol, timeframe)
    return futures

timeframe1 = "18/08/2025" 
futures = get_future_symbols('NIFTY')
data = hist_data(futures[0], timeframe1, "1") 
print(data)

