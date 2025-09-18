import sys, os
from datetime import datetime, time as dtime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.historical_data import hist_data
from utils.symbol_utils import get_future_symbols
from utils.api_utils import get_option_chain

symbol = "NSE:SBIN-EQ"  
data = get_option_chain(symbol, strikecount=1)
print(data)







