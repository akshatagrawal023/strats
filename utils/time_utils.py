import time
import datetime

def get_trading_time_fraction(current_ts: float, expiry_ts: float, holidays: list = None) -> float:
    """
    Calculate the exact Trading Time Fraction remaining until expiration.
    This uses 'Trading Time' (Market open only) rather than 'Calendar Time'.
    
    NSE Market Hours:
    Open: 09:15 IST
    Close: 15:30 IST
    Total Seconds per Trading Day: 6.25 hours = 22,500 seconds.
    
    Args:
        current_ts: Current UNIX timestamp.
        expiry_ts: Expiration UNIX timestamp (typically 15:30 IST on expiry day).
        holidays: List of 'YYYY-MM-DD' strings for public holidays to skip.
        
    Returns:
        float: Time to expiration (T) in fully annualized Trading Years 
               (where 1 Trading Year = 252 days * 22,500 seconds).
    """
    if holidays is None:
        # Example 2026 holidays can be added here
        holidays = []
        
    start_dt = datetime.datetime.fromtimestamp(current_ts)
    end_dt = datetime.datetime.fromtimestamp(expiry_ts)
    
    # Ensure minimum 60 seconds (1 minute) to avoid division by zero near 0DTE closes
    if expiry_ts - current_ts <= 60.0:
        return 60.0 / (252 * 22500)
    
    trading_seconds = 0.0
    current_iter = start_dt
    
    while current_iter < end_dt:
        # Check if it's a weekend (5=Saturday, 6=Sunday) or a holiday
        if current_iter.weekday() >= 5 or current_iter.strftime('%Y-%m-%d') in holidays:
            # Skip to exactly the next day 00:00:00
            current_iter = (current_iter + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            continue
            
        # Define market open and close for the current iteration day
        market_open = current_iter.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = current_iter.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Determine overlap between our time window and market operating hours
        overlap_start = max(current_iter, market_open)
        
        if end_dt.date() == current_iter.date():
            overlap_end = min(end_dt, market_close) # Final day clamp
        else:
            overlap_end = market_close # Normal day completely to the close
            
        if overlap_end > overlap_start:
            trading_seconds += (overlap_end - overlap_start).total_seconds()
            
        # Move forward to the next absolute day
        current_iter = (current_iter + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        
    # Apply safety floor again just in case the entire period was closed (e.g. weekend requests)
    trading_seconds = max(trading_seconds, 60.0)
    
    # Return annualized trading time (assuming 252 trading days per year, 22,500 active seconds per day)
    return trading_seconds / (252.0 * 22500.0)
