"""
Configuration for Calendar Spread Strategy
"""
class CalendarConfig:
    # Strategy Parameters
    STRATEGY_NAME = "CalendarSpread"
    
    # Underlying Selection Criteria
    MIN_DAILY_VOLUME = 1000000  # Minimum daily volume for stocks
    MIN_OPTION_VOLUME = 1000    # Minimum option volume
    MAX_BID_ASK_SPREAD = 0.02   # 2% max bid-ask spread
    MIN_IV_RANK = 0.3           # Minimum IV rank (30th percentile)
    MAX_IV_RANK = 0.8           # Maximum IV rank (80th percentile)
    
    # Position Sizing
    MAX_POSITION_SIZE = 0.05    # 5% of portfolio per underlying
    MAX_TOTAL_EXPOSURE = 0.20   # 20% total portfolio exposure
    MIN_PREMIUM_RECEIVED = 1000 # Minimum premium to receive
    
    # Calendar Spread Parameters
    SHORT_EXPIRY_DAYS = 14      # Days to short expiry (7-21 range)
    LONG_EXPIRY_DAYS = 45       # Days to long expiry (30-90 range)
    STRIKE_SELECTION = "ATM"    # ATM, OTM_5, OTM_10
    
    # Risk Management
    MAX_DELTA_EXPOSURE = 0.20   # 20% of position value
    DELTA_HEDGE_THRESHOLD = 0.10 # Hedge when delta exceeds 10%
    MAX_LOSS_MULTIPLIER = 2.0   # Stop loss at 2x initial debit
    PROFIT_TARGET = 0.50        # Take profit at 50% of max profit
    
    # Time-based Exits
    CLOSE_BEFORE_EXPIRY = 7     # Close 7 days before short expiry
    MAX_HOLDING_DAYS = 30       # Maximum holding period
    
    # Hedging Parameters
    HEDGE_INTERVAL_SECONDS = 30 # Check for hedging every 30 seconds
    HEDGE_RATIO = 0.5           # Partial hedging ratio
    USE_FUTURES_HEDGE = True    # Use futures for delta hedging
    
    # Lot Sizes (update based on current exchange specs)
    LOT_SIZE_MAP = {
        "NIFTY": 50,
        "BANKNIFTY": 25,
        "FINNIFTY": 40,
        "RELIANCE": 250,
        "TCS": 125,
        "INFY": 200,
        "HDFC": 250,
        "ICICIBANK": 275,
        "SBIN": 1500,
        "BHARTIARTL": 400,
        "ITC": 800,
        "LT": 200,
        "MARUTI": 25,
        "ASIANPAINT": 25,
        "NESTLEIND": 25,
        "ULTRACEMCO": 25,
        "SUNPHARMA": 200,
        "TITAN": 50,
        "WIPRO": 400,
        "AXISBANK": 300
    }
    DEFAULT_LOT_SIZE = 50
    
    # Option Chain Parameters
    USE_OPTIONCHAIN_FOR_ATM = True
    CHAIN_STRIKECOUNT = 5
    OPTION_STRIKE_STEP = 50
    
    # Session Management
    SESSION_START_TIME = "09:15"
    SESSION_END_TIME = "15:30"
    MAX_POSITIONS_PER_DAY = 5
    
    # Logging and Monitoring
    LOG_LEVEL = "INFO"
    SAVE_TRADES_TO_FILE = True
    TRADES_LOG_FILE = "calendar_trades.log"
    
    # Backtesting Parameters
    COMMISSION_PER_LOT = 20
    SLIPPAGE_BPS = 5  # 5 basis points slippage
    MARGIN_REQUIREMENT = 0.15  # 15% margin requirement
