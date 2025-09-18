"""
Live configuration for the gamma scalper targeting NIFTY.

Adjust expiry, lot sizes, and symbols per your broker/exchange.
"""
class Config:
    # Underlying selection
    GAMMA_BASE = "NIFTY"  # NIFTY index
    # Expiry string should match your broker symbol format used by futures/options
    # Example weekly format: '24AUG' for 2024 Aug weekly contracts
    GAMMA_EXPIRY = "24AUG"

    # Position sizing
    GAMMA_LOTS = 1

    # Option chain selection
    # For NIFTY, most brokers accept an index-level chain symbol. Verify the exact
    # symbol with your Fyers account. Common candidates include 'NSE:NIFTY50'.
    # If optionchain fails, the code will fall back to FUT LTP-based ATM selection.
    USE_OPTIONCHAIN_FOR_ATM = True
    GAMMA_CHAIN_SYMBOL = "NSE:NIFTY50"  # Confirm exact symbol for your API
    CHAIN_STRIKECOUNT = 3

    # Hedging parameters
    # DELTA_BAND is interpreted as a fraction of one contract delta
    DELTA_BAND = 0.10
    HEDGE_INTERVAL_SECONDS = 3
    MAX_HEDGES_PER_DAY = 150
    SESSION_MINUTES = 90

    # Lot sizes (ensure these match current exchange specs)
    # If unsure, confirm with your broker and update accordingly
    LOT_SIZE_MAP = {
        "NIFTY": 50,  # Update if the official NIFTY lot size changes
    }
    DEFAULT_LOT_SIZE = 50

    # Option strike step used for fallback ATM selection via FUT LTP
    OPTION_STRIKE_STEP = 50

