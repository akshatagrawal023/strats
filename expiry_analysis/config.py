"""
Configuration for expiry analysis system
"""
# Trading parameters
UNDERLYINGS = [
    "RELIANCE", "HDFCBANK", "INFY", "TCS", "HINDUNILVR",
    "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK"
]  # 10 stocks × 20 calls/min = 200 calls/min
STRIKE_COUNT = 3
POLL_INTERVAL = 3.0  # seconds

# Greeks calculation parameters
RISK_FREE_RATE = 0.08  # 6.5%
DAYS_TO_EXPIRY = 4

# Data storage parameters
WINDOW_SIZE = 300  # Number of snapshots to keep in memory

