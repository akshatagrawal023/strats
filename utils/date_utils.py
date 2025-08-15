from datetime import datetime, timedelta
import pandas as pd
import time
import re
import pytz

# Define Indian timezone
IST = pytz.timezone("Asia/Kolkata")

def last_thursday(year, month):
    last_day = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    while last_day.weekday() != 3:
        last_day -= timedelta(days=1)
    return last_day

def get_month_years():
    today = datetime.now()
    last_thursday_date = last_thursday(today.year, today.month)
    months_ahead = 3 if today <= last_thursday_date else 4

    month_year_pairs = []
    for i in range(months_ahead):
        target_date = today + pd.DateOffset(months=i)
        month_mmm = target_date.strftime('%b').upper()
        year_yy = target_date.strftime('%y')
        month_year_pairs.append((month_mmm, year_yy))

    return month_year_pairs

# Function to get the number of days to the last Thursday of this month and the next two months
def days_to_last_thursdays():
    today = datetime.now()
    current_year = today.year
    last_thursday_date = last_thursday(today.year, today.month)
    current_month = today.month if today <= last_thursday_date else today.month+1
    
    # Calculate last Thursdays
    last_thursday_current = last_thursday(current_year, current_month)
    last_thursday_next = last_thursday(current_year, current_month + 1)
    last_thursday_next_to_next = last_thursday(current_year, current_month + 2)
    
    # Calculate days to each last Thursday
    days_to_current = (last_thursday_current - today).days
    days_to_next = (last_thursday_next - today).days
    days_to_next_to_next = (last_thursday_next_to_next - today).days
    
    return days_to_current, days_to_next, days_to_next_to_next

def get_timestamps(input_str, date_f=0):
    """
    Convert user input (e.g., '60d', '1h', '01/01/2024 // 10/02/2024') to timestamps or date strings.
    
    Args:
        input_str (str): Time range in format like "60d", "1h" or "DD/MM/YYYY // DD/MM/YYYY"
        date_f (int): Date format (0 for epoch, 1 for YYYY-MM-DD)
        
    Returns:
        Tuple of (from_timestamp, to_timestamp) or (from_date, to_date)
    """
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

    else:
        raise ValueError("Invalid format. Use 'Xd', 'Xm', 'Xs', 'Xh' or 'DD/MM/YYYY // DD/MM/YYYY'")

    if date_f == 1:
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    else:
        return int(time.mktime(start_date.timetuple())), int(time.mktime(end_date.timetuple()))

def convert_to_ist(ts):
    """
    Convert timestamp to IST datetime.
    
    Args:
        ts (int): Timestamp
        
    Returns:
        datetime: Timestamp converted to IST timezone
    """
    utc_dt = datetime.fromtimestamp(ts, tz=pytz.utc)
    ist_dt = utc_dt.astimezone(IST)
    return ist_dt