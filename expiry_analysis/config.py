"""
Configuration for expiry analysis system
"""
# Trading parameters
UNDERLYINGS = [
    "RELIANCE", "HDFCBANK", "INFY", "TCS", "HINDUNILVR",
    "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"
]
STRIKE_COUNT = 3
POLL_INTERVAL = 3.0  # seconds (11 stocks × 20 calls/min = 220 calls/min, but API limit is 200)

# Greeks calculation parameters
RISK_FREE_RATE = 0.065  # 6.5%
DAYS_TO_EXPIRY = 7

# Data storage parameters
WINDOW_SIZE = 300  # Number of snapshots to keep in memory

