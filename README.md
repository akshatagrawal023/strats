# Fyers Trading Strategies

Automated trading strategies using Fyers API with real-time data streaming, advanced risk management, and intelligent token management.

## 🚀 Features

- **Gamma Scalping Strategy**: Delta-hedged options trading with automated hedging
- **Real-time Data Streaming**: WebSocket-based market data for live trading
- **Intelligent Token Management**: Automatic refresh token usage (15-day validity)
- **Risk Management**: Built-in position limits, time stops, and delta bands
- **Multi-strategy Support**: Framework for running multiple strategies simultaneously
- **Fyers API Integration**: Seamless integration with Fyers trading platform
- **Automated Authentication**: Smart token refresh without manual login

## ️ Project Structure

```markdown
<code_block_to_apply_changes_from>
```
Strategies/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── config.py                # Main configuration
├── config_live.py           # Live trading configuration
├── option_symbols.py        # Options symbol utilities
├── strategies/              # Trading strategies
│   ├── __init__.py
│   ├── gamma_scalping/      # Gamma scalping strategy
│   │   ├── __init__.py
│   │   ├── gamma_scalper.py
│   │   └── run_gamma_scalper.py
│   └── calendar_spread/     # Calendar spread strategy
│       ├── __init__.py
│       └── README.md
├── utils/                    # Common utilities
│   ├── __init__.py
│   ├── fyers_api.py         # Fyers API with smart token management
│   ├── fyers_instance.py    # Singleton Fyers instance
│   ├── api_utils.py         # API utilities and rate limiting
│   ├── date_utils.py        # Date and time utilities
│   ├── fyers_orders.py      # Order management utilities
│   ├── trade_costs.py       # Trading cost calculations
│   ├── fuzzy_name.py        # Fuzzy matching utilities
│   ├── option_symbols.py    # Options symbol utilities
│   └── websocket.py         # WebSocket manager for real-time data
├── logs/                     # Log files
└── tests/                    # Test files
```

## ⚡ Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your Fyers credentials
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key
```