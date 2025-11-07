"""
Configuration for expiry analysis system
"""
# Trading parameters
UNDERLYINGS = ["RELIANCE", "HDFCBANK"]
STRIKE_COUNT = 3
POLL_INTERVAL = 2.0  # seconds

# Greeks calculation parameters
RISK_FREE_RATE = 0.065  # 6.5%
DAYS_TO_EXPIRY = 7

# Data storage parameters
WINDOW_SIZE = 300  # Number of snapshots to keep in memory

