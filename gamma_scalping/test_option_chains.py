import os
import sys
from datetime import datetime, time as dtime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.historical_data import hist_data
from utils.option_symbols import get_options_data


def read_expiry_calendar(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = [c.strip().capitalize() for c in df.columns]
    # Expecting columns like: Symbol, Expiry
    df['Expiry_dt'] = pd.to_datetime(df['Expiry'], format='%d-%b-%Y')
    return df


def next_weekly_expiry_code(df: pd.DataFrame, base: str, target_date: datetime) -> Optional[str]:
    rows = df[df['Symbol'].str.upper() == base.upper()].copy()
    if rows.empty:
        return None
    # Next expiry on/after target_date
    rows = rows[rows['Expiry_dt'] >= target_date]
    if rows.empty:
        return None
    expiry_dt = rows.sort_values('Expiry_dt').iloc[0]['Expiry_dt']
    # Convert to code like 21AUG
    code = expiry_dt.strftime('%d%b').upper()
    return code


def pick_near_month_future_symbol(base: str, target_date: datetime) -> Optional[str]:
    futures: List[str] = get_options_data(base, data_type='futures') or []
    if not futures:
        return None
    # Heuristic: choose future whose expiry month matches target_date month if available, else first
    month_token = target_date.strftime('%b').upper()
    candidates = [f for f in futures if month_token in f]
    return (candidates[0] if candidates else futures[0])


def fetch_entry_underlying_price(fut_symbol: str, day_str: str, entry_time: dtime = dtime(9, 20)) -> Optional[float]:
    # Fetch 1-minute candles for the given day; hist_data returns dict with 'candles'
    resp = hist_data(fut_symbol, day_str, resolution="1", date_format=0, cont_flag=1)
    if not isinstance(resp, dict) or resp.get('s') != 'ok':
        return None
    candles = resp.get('candles', [])
    if not candles:
        return None
    # candles: [epoch_ts, open, high, low, close, volume] with IST epoch already adjusted by util
    # Pick bar at entry_time or nearest after
    target_minutes = entry_time.hour * 60 + entry_time.minute
    best = None
    for ts, o, h, l, c, v in candles:
        dt = datetime.fromtimestamp(ts)  # already IST naive per utils
        minutes = dt.hour * 60 + dt.minute
        if minutes >= target_minutes:
            best = c
            break
    if best is None:
        # fallback last close
        best = candles[0][4]
    return float(best)


def find_nearest_strike_from_master(base: str, strike_price: float) -> Optional[int]:
    strikes = get_options_data(base, data_type='strikes') or []
    if not strikes:
        return None
    strikes_int = [int(s) for s in strikes if str(s).isdigit()]
    return min(strikes_int, key=lambda k: abs(k - strike_price)) if strikes_int else None


def resolve_atm_option_symbols(base: str, expiry_code: str, atm_strike: int) -> Tuple[Optional[str], Optional[str]]:
    option_syms = get_options_data(base, data_type='option_symbols') or {}
    ce_list = option_syms.get('ce_symbols', []) or []
    pe_list = option_syms.get('pe_symbols', []) or []
    ce_candidate = f"NSE:{base}{expiry_code}{atm_strike}CE"
    pe_candidate = f"NSE:{base}{expiry_code}{atm_strike}PE"
    ce_symbol = next((s for s in ce_list if s == ce_candidate), None)
    pe_symbol = next((s for s in pe_list if s == pe_candidate), None)
    return ce_symbol, pe_symbol


def fetch_intraday_series(symbol: str, day_str: str, resolution: str = "1") -> Dict[str, Any]:
    resp = hist_data(symbol, day_str, resolution=resolution, date_format=0, cont_flag=1)
    return resp if isinstance(resp, dict) else {"s": "error", "error": "bad response"}


def prepare_day_dataset(base: str, day_str: str, entry_time: dtime = dtime(9, 20)) -> Dict[str, Any]:
    # 1) Expiry code from calendar
    csv_path = os.path.join('utils', 'FnO_lot_structured.csv')
    cal_df = read_expiry_calendar(csv_path)
    target_date = datetime.strptime(day_str, '%d/%m/%Y')
    expiry_code = next_weekly_expiry_code(cal_df, base, target_date)
    if not expiry_code:
        return {"s": "error", "error": f"No expiry code for {base} on {day_str}"}

    # 2) Near-month future symbol
    fut_symbol = pick_near_month_future_symbol(base, target_date)
    if not fut_symbol:
        return {"s": "error", "error": f"No futures for {base}"}

    # 3) Underlying entry price from futures
    entry_price = fetch_entry_underlying_price(fut_symbol, day_str, entry_time)
    if entry_price is None:
        return {"s": "error", "error": f"No entry price for {fut_symbol}"}

    # 4) ATM strike from master (nearest available strike)
    atm_strike = find_nearest_strike_from_master(base, entry_price)
    if atm_strike is None:
        return {"s": "error", "error": f"No strikes in master for {base}"}

    # 5) Resolve option symbols for ATM strike
    ce_symbol, pe_symbol = resolve_atm_option_symbols(base, expiry_code, atm_strike)
    if not ce_symbol or not pe_symbol:
        return {"s": "error", "error": f"ATM symbols not found for {base} {expiry_code} {atm_strike}"}

    # 6) Fetch intraday candles for futures and options (1-min)
    fut_series = fetch_intraday_series(fut_symbol, day_str, resolution="1")
    ce_series = fetch_intraday_series(ce_symbol, day_str, resolution="1")
    pe_series = fetch_intraday_series(pe_symbol, day_str, resolution="1")

    return {
        "s": "ok",
        "base": base,
        "day": day_str,
        "expiry_code": expiry_code,
        "entry_price": entry_price,
        "atm_strike": atm_strike,
        "fut_symbol": fut_symbol,
        "ce_symbol": ce_symbol,
        "pe_symbol": pe_symbol,
        "fut_series": fut_series,
        "ce_series": ce_series,
        "pe_series": pe_series,
    }


def main():
    # Pick a backtest day (DD/MM/YYYY)
    day_str = "21/08/2025"  # change as needed

    # BANKNIFTY derivatives use base "BANKNIFTY" in FO master; SBI uses "SBIN"
    bases = ["BANKNIFTY", "SBIN"]

    for base in bases:
        print("\n" + "=" * 80)
        print(f"Preparing dataset for {base} on {day_str}")
        ds = prepare_day_dataset(base, day_str)
        if ds.get("s") != "ok":
            print(f"Error: {ds.get('error')}")
            continue
        print(f"Expiry code: {ds['expiry_code']}")
        print(f"Entry price (fut): {ds['entry_price']:.2f}")
        print(f"ATM strike: {ds['atm_strike']}")
        print(f"FUT: {ds['fut_symbol']}\nCE : {ds['ce_symbol']}\nPE : {ds['pe_symbol']}")
        # Summaries
        for label in ("fut_series", "ce_series", "pe_series"):
            series = ds[label]
            status = series.get('s')
            count = len(series.get('candles', [])) if status == 'ok' else 0
            print(f"{label}: status={status}, bars={count}")


if __name__ == "__main__":
    main()