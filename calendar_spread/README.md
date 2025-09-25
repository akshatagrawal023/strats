# Calendar Spread Strategy - Nifty 50 Backtesting

A focused calendar spread trading strategy for Nifty 50 stocks that profits from time decay differentials and volatility expansion.

## Strategy Overview

A calendar spread involves:
- **Selling** short-term options (7-30 days to expiry)
- **Buying** long-term options (30-90 days to expiry)
- **Same strike price** for both legs
- **Same underlying asset**

### Key Benefits
- Limited risk (maximum loss = net debit paid)
- Time decay advantage (short options decay faster)
- Volatility benefit (profit from IV expansion)
- Delta neutral approach with hedging

## Quick Start - Nifty 50 Backtesting

### Step 1: Setup
```bash
python calendar_spread/setup_nifty50_backtest.py
```

### Step 2: Run Backtest
```bash
python calendar_spread/run_nifty50_backtest.py
```

## Core Components

### 1. Configuration (`config.py`)
- Strategy parameters and risk limits
- Nifty 50 stock symbols and lot sizes
- Entry/exit criteria and hedging parameters

### 2. Nifty 50 Processor (`nifty50_processor.py`)
- Reads your `utils/NiftyFifty.csv` file
- Converts company names to stock symbols
- Selects liquid stocks suitable for calendar spreads

### 3. Historical Data Manager (`historical_data_manager.py`)
- Downloads real historical data using yfinance
- Data validation and cleaning
- Caching system for efficiency

### 4. Backtesting Engine (`backtest.py`)
- Historical performance testing
- Trade simulation with realistic costs
- Performance metrics calculation

### 5. Results Analyzer (`results_analyzer.py`)
- Comprehensive performance analysis
- Risk metrics (VaR, drawdown, Sharpe ratio)
- Strategy recommendations

## Files Structure

```
calendar_spread/
├── config.py                    # Strategy configuration
├── nifty50_processor.py         # Nifty 50 CSV processor
├── historical_data_manager.py   # Data download & management
├── backtest.py                  # Backtesting engine
├── results_analyzer.py          # Results analysis
├── setup_nifty50_backtest.py    # Setup script
├── run_nifty50_backtest.py      # Generated backtest script
└── README.md                    # This file
```

## Entry Logic

### Underlying Selection Criteria
1. **High IV Rank** (30th-80th percentile)
2. **Adequate Liquidity** (>1M daily volume, tight spreads)
3. **Multiple Expiries** available
4. **Stable Fundamentals**

### Strike Selection
- **ATM (At-The-Money)** strikes preferred
- Maximum time decay benefit
- Balanced risk profile

### Expiry Selection
- **Short Expiry**: 7-21 days (accelerated theta decay)
- **Long Expiry**: 30-90 days (volatility exposure)
- **Minimum 2-week gap** between expiries

## Risk Management

### Delta Management
- **Threshold**: Hedge when delta exceeds 10% of position
- **Method**: Futures hedging or opposing options
- **Frequency**: Check every 30 seconds during market hours

### Position Sizing
- **Maximum 5%** of portfolio per underlying
- **Maximum 20%** total portfolio exposure
- **Risk-based sizing** based on net debit

### Exit Strategies
1. **Profit Target**: 50% of maximum profit
2. **Stop Loss**: 2x initial debit
3. **Time Exit**: Close 7 days before short expiry
4. **Maximum Hold**: 30 days maximum

## Usage

### Live Trading
```python
from calendar_spread.config import CalendarConfig
from calendar_spread.calendar_spread_strategy import CalendarSpreadStrategy

# Initialize configuration
config = CalendarConfig()

# Initialize strategy
strategy = CalendarSpreadStrategy(config)

# Run strategy
strategy.run_strategy(max_positions=3)
```

### Backtesting
```python
from calendar_spread.backtest import CalendarBacktest
from datetime import datetime

# Set backtest period
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 6, 30)

# Run backtest
backtest = CalendarBacktest(config, start_date, end_date)
results = backtest.run_backtest()
backtest.print_results(results)
```

## Key Features

### Automated Execution
- **Real-time scanning** for opportunities
- **Automated entry/exit** based on criteria
- **Dynamic hedging** for risk management
- **Position monitoring** with alerts

### Risk Controls
- **Delta limits** with automatic hedging
- **Position size limits** per underlying
- **Daily loss limits** and drawdown controls
- **Time-based exits** to avoid assignment risk

### Performance Tracking
- **Real-time P&L** monitoring
- **Greeks tracking** (delta, theta, vega)
- **Trade logging** and analysis
- **Performance metrics** calculation

## Configuration Parameters

### Strategy Parameters
- `SHORT_EXPIRY_DAYS`: Days to short expiry (default: 14)
- `LONG_EXPIRY_DAYS`: Days to long expiry (default: 45)
- `MAX_POSITION_SIZE`: Max position size (default: 5% of portfolio)
- `DELTA_HEDGE_THRESHOLD`: Delta hedging threshold (default: 10%)

### Risk Management
- `MAX_LOSS_MULTIPLIER`: Stop loss multiplier (default: 2.0)
- `PROFIT_TARGET`: Profit target percentage (default: 50%)
- `MAX_HOLDING_DAYS`: Maximum holding period (default: 30)
- `CLOSE_BEFORE_EXPIRY`: Days before expiry to close (default: 7)

### Selection Criteria
- `MIN_DAILY_VOLUME`: Minimum daily volume (default: 1M)
- `MIN_OPTION_VOLUME`: Minimum option volume (default: 1K)
- `MAX_BID_ASK_SPREAD`: Maximum bid-ask spread (default: 2%)
- `MIN_IV_RANK`: Minimum IV rank (default: 30%)

## Monitoring and Alerts

### Real-time Monitoring
- Position status and P&L
- Delta exposure and hedging needs
- Time to expiry warnings
- Market condition alerts

### Performance Metrics
- Win rate and average P&L
- Maximum drawdown
- Sharpe ratio
- Trade frequency and duration

## Dependencies

- `utils.api_utils`: Fyers API integration
- `utils.option_symbols`: Option symbol management
- `utils.symbol_utils`: Symbol utilities
- `utils.trade_costs`: Cost calculation
- `pandas`, `numpy`: Data analysis
- `logging`: Logging and monitoring

## Risk Disclaimer

This strategy involves options trading which carries significant risk. Past performance does not guarantee future results. Always:
- Test thoroughly with paper trading
- Start with small position sizes
- Monitor positions actively
- Have proper risk management in place
- Understand assignment risks
- Be prepared for margin requirements

## Support

For questions or issues:
1. Check the logs for error messages
2. Verify API connectivity and permissions
3. Ensure sufficient margin and capital
4. Review configuration parameters
5. Test with small position sizes first
