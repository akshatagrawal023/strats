import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_utils import get_option_chain

print(get_option_chain("NSE:SBIN-EQ", strikecount=1))