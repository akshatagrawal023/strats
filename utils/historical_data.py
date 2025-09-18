import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fyers_instance import FyersInstance
from utils.option_symbols import get_options_data
from datetime import datetime, timedelta
import time
import pytz
import re
import pandas as pd
# Define Indian timezone
IST = pytz.timezone("Asia/Kolkata")

def get_timestamps(input_str, date_f = 0):
    
    match = re.match(r"(\d+)([smhd])$", input_str.strip())

    if match:
        num, unit = int(match.group(1)), match.group(2).lower()  
        now = datetime.now(IST)

        if unit == "s":  # Seconds
            start_date = now - timedelta(seconds=num)
        elif unit == "m":  # Minutes
            start_date = now - timedelta(minutes=num)
        elif unit == "h":  # Hours
            start_date = now - timedelta(hours=num)
        elif unit == "d":  # Days
            start_date = now - timedelta(days=num)

        end_date = now  # End date is always current time

    elif "//" in input_str:
        # Extract specific date range
        start_str, end_str = input_str.split("//")
        start_date = datetime.strptime(start_str.strip(), "%d/%m/%Y")
        end_date = datetime.strptime(end_str.strip(), "%d/%m/%Y")

        # Convert to IST timezone
        start_date = IST.localize(start_date)
        end_date = IST.localize(end_date)
    
    elif re.match(r"\d{2}/\d{2}/\d{4}$", input_str.strip()):
        # Single date format (DD/MM/YYYY)
        date_str = input_str.strip()
        date = datetime.strptime(date_str, "%d/%m/%Y")
        
        # Convert to IST timezone
        date = IST.localize(date)
        
        # Check if it's a market day (weekday, not Saturday or Sunday)
        if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            # Find the most recent market day
            current = date
            while current.weekday() >= 5:
                current = current - timedelta(days=1)
            date = current
            
        # Set start to market open (9:15 AM), end to market close (3:30 PM)
        start_date = date.replace(hour=9, minute=15, second=0, microsecond=0)
        end_date = date.replace(hour=15, minute=30, second=0, microsecond=0)

    else:
        raise ValueError("Invalid format. Use 'Xd', 'Xm', 'Xs', 'Xh', 'DD/MM/YYYY' or 'DD/MM/YYYY // DD/MM/YYYY'")

    if date_f == 1:
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    else:
        # Convert to Unix timestamp using the timestamp() method which respects timezone
        return int(start_date.timestamp()), int(end_date.timestamp())


def get_params(sym, range, res = "D", date_f = 0, cont_flag = 1):

    range_from, range_to = get_timestamps(range, date_f)

    if date_f == 0:

        params = {
            "symbol": sym,
            "resolution": res,
            "date_format": date_f,
            "range_from": str(range_from),
            "range_to": str(range_to),
            "cont_flag": cont_flag
        }
        return params
    
    else:
        params = {
            "symbol": sym,
            "resolution": res,
            "date_format": date_f,
            "range_from": str(range_from),
            "range_to": str(range_to),
            "cont_flag": cont_flag
        }
        return params

def convert_to_ist(ts):
    utc_dt = datetime.fromtimestamp(ts, tz=pytz.utc)  # Updated line
    ist_dt = utc_dt.astimezone(IST)  # Ensure IST is used
    return ist_dt

def hist_df(symbol, timeframe, resolution="D", date_format=0, cont_flag=1):
    response = hist_data(symbol, timeframe, resolution=resolution, date_format=date_format, cont_flag=cont_flag)
    if response['s'] == 'ok':
        print('ok')
        columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.DataFrame(response['candles'], columns=columns)
        # Convert to IST but then remove timezone info for display purposes
        df['Time'] = df['Time'].apply(lambda ts: convert_to_ist(ts).replace(tzinfo=None))
        return df

def hist_data(symbol, timeframe, resolution="D", date_format=0, cont_flag=1):
    """
    Get historical data for a symbol with a single function call.
    """
    # Get the Fyers instance using the singleton pattern
    fyers = FyersInstance.get_instance()
    # Get parameters for the API call
    parameters = get_params(symbol, timeframe, resolution, date_format, cont_flag)
    # Make the API call
    response = fyers.history(data=parameters)
    
    # Convert timestamps to IST if data is available and in the expected format
    if isinstance(response, dict) and response.get('s') == 'ok' and 'candles' in response:
        for candle in response['candles']:
            if len(candle) > 0:  # Make sure there's at least one item (timestamp)
                # Convert the timestamp (first element) to IST
                utc_dt = datetime.fromtimestamp(candle[0], tz=pytz.utc)
                ist_dt = utc_dt.astimezone(IST)
                candle[0] = int(ist_dt.timestamp())
    
    return response

# Example Usage:
if __name__ == "__main__":
    try:
        # Get Fyers instance using the singleton pattern
        fyers = FyersInstance.get_instance()
        print("Connected to Fyers API successfully.")

        # timeframe1 = "01/01/2024 // 10/02/2024"  # Examples: "60m", "1h", "30s", "01/01/2024 // 10/02/2024"
        
        # # Method 1: Using individual steps
        # df = hist_df("NSE:SBIN-EQ", timeframe1)
        # print(df)
        
        # timeframe2 = "13/04/2025"
        # # Method 2: Using the convenience function
        # data = hist_df("NSE:AARTIIND29MAY2025FUT", timeframe2, "45S")
        # print(data)
        
        # # Example of using a single date (will get data for that specific day)
        # date = "22/08/2024"  # Format: DD/MM/YYYY
        # print(f"\nGetting data for: {date}")
        # data_for_day = hist_data("NSE:SBIN-EQ", date, "5")  # 5-minute candles for the specified day
        # print(data_for_day)

        # # Example of quotes API
        # data = {
        # "symbols":"NSE:SBIN-EQ,NSE:IDEA-EQ"
        # }
        # response = fyers.quotes(data=data)
        # print(response)
        timeframe2 = "01/04/2025 // 10/04/2025"
        stocks = pd.read_csv('FnO_list.csv')['Stocks'].tolist()
        stock = stocks[3]
        futures = get_options_data(stock, "futures")
        near, far = futures[0], futures[1]
        print(near, far)
        tick = "NSE:ABB25APRFUT"
        data = hist_df(tick, timeframe2, "45S")
        print(data)
    except Exception as e:
            print(f"Error: {e}")
    """
    Accepted formats for timeframe:
    1. "Xd"  -> Last X days
    2. "Xm"  -> Last X minutes
    3. "Xs"  -> Last X seconds
    4. "Xh"  -> Last X hours
    5. "DD/MM/YYYY // DD/MM/YYYY" -> Specific date range
    6. "DD/MM/YYYY" -> Single date (defaults to market hours 9:15 AM to 3:30 PM IST for that day)
                      If the date is a weekend, it automatically finds the most recent market day.
    """

    
