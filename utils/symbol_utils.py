import pandas as pd
from typing import Iterable, List, Optional, Union
import os
import json

def nse_symbol(stocks):
	DEFAULT_INDICES = {
	"MIDCPNIFTY",
	"NIFTY",
	"BANKNIFTY",
	"FINNIFTY",
	"NIFTYNXT50",
	}
	# Add NSE and BSE columns
	stock = "NSE:" + stocks + "-EQ"
	if stocks in DEFAULT_INDICES:
		stock = f"NSE:{stocks}-INDEX"
	return stock

def bse_symbol(stocks):
	DEFAULT_INDICES = {
	"SENSEX"
	}
	# Add NSE and BSE columns
	stock = "BSE:" + stocks + "-A"
	if stocks in DEFAULT_INDICES:
		stock = f"NSE:{stocks}-INDEX"
	return stock

def get_future_symbols(symbol):
    try:
        # Get absolute path to the JSON file
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'options_details_cache.json')
        print(f"Looking for JSON file at: {json_path}")
        if not os.path.exists(json_path):
            print(f"JSON file not found at {json_path}")
            return []
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            symbol_data = data.get(symbol, {})
            futures = symbol_data.get('futures', [])
            # print(f"Found futures for {symbol}: {futures}")
            return futures
    except Exception as e:
        print(f"Error reading options_details_cache.json: {e}")
        return []

def get_option_symbols(symbol):
    try:
        # Get absolute path to the JSON file
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'options_details_cache.json')
        print(f"Looking for JSON file at: {json_path}")
        if not os.path.exists(json_path):
            print(f"JSON file not found at {json_path}")
            return []
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            symbol_data = data.get(symbol, {})
            futures = symbol_data.get('futures', [])
            # print(f"Found futures for {symbol}: {futures}")
            return futures
    except Exception as e:
        print(f"Error reading options_details_cache.json: {e}")
        return []

import re

def build_option_symbol(base: str, expiry_code: str, strike: int, right: str) -> str:
	"""Return FO option symbol like 'NSE:{BASE}{EXPIRY}{STRIKE}{RIGHT}'"""
	base = base.upper()
	right = right.upper()
	return f"NSE:{base}{expiry_code}{int(strike)}{right}"

def build_option_pair(base: str, expiry_code: str, strike: int):
	"""Given base, expiry code (e.g., '25AUG') and strike, return (CE, PE) symbols."""
	ce = build_option_symbol(base, expiry_code, strike, 'CE')
	pe = build_option_symbol(base, expiry_code, strike, 'PE')
	return ce, pe

OPTION_SYMBOL_RE = re.compile(r"^NSE:([A-Z]+)(\d+[A-Z]+)(\d+)(CE|PE)$")
FUTURE_SYMBOL_RE = re.compile(r"^NSE:([A-Z]+)(\d+[A-Z]+)FUT$")

def option_pair_from_sample(sample_symbol: str):
	"""
	Given a sample option symbol like 'NSE:NCC25AUG135CE', parse and return (CE, PE).
	"""
	m = OPTION_SYMBOL_RE.match(sample_symbol.strip().upper())
	if not m:
		raise ValueError(f"Unrecognized option symbol format: {sample_symbol}")
	base, expiry_code, strike, _right = m.groups()
	return build_option_pair(base, expiry_code, int(strike))

def option_pair_from_future(future_symbol: str, strike: int):
	"""
	Given a future symbol like 'NSE:NCC25AUGFUT' and a strike, return (CE, PE) symbols
	for the same base and expiry code.
	"""
	m = FUTURE_SYMBOL_RE.match(future_symbol.strip().upper())
	if not m:
		raise ValueError(f"Unrecognized future symbol format: {future_symbol}")
	base, expiry_code = m.groups()
	return build_option_pair(base, expiry_code, int(strike))

# Example:
# ce, pe = build_option_pair('NCC', '25AUG', 135)
# ce2, pe2 = option_pair_from_sample('NSE:NCC25AUG135CE')