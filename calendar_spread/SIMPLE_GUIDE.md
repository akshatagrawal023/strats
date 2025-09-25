# Calendar Spread Strategy - Simple Guide

## 🎯 What You Have Now

A clean, focused calendar spread backtesting system for Nifty 50 stocks.

## 📁 Essential Files Only

```
calendar_spread/
├── config.py                    # Edit this to adjust parameters
├── nifty50_processor.py         # Reads your NiftyFifty.csv
├── historical_data_manager.py   # Downloads real data
├── backtest.py                  # Runs backtests
├── results_analyzer.py          # Analyzes results
├── setup_nifty50_backtest.py    # Setup script
└── README.md                    # Documentation
```

## 🚀 How to Use (2 Steps)

### Step 1: Setup
```bash
python calendar_spread/setup_nifty50_backtest.py
```
This will:
- Read your `utils/NiftyFifty.csv` file
- Load lot sizes from `utils/FnO_lot_structured.csv`
- Convert company names to symbols using fuzzy matching
- Get ATM strikes using option chain API
- Export data to `nifty50_backtest_data.json`

### Step 2: Test the Processor
```bash
python calendar_spread/test_processor.py
```
This will:
- Test symbol conversion
- Test lot size lookup
- Test option chain fetching
- Test ATM strike calculation
- Show recommended stocks

## 📊 What You'll Get

- **Company name to symbol conversion** using fuzzy matching
- **Dynamic lot sizes** from your FnO_lot_structured.csv
- **Real-time ATM strikes** using option chain API
- **Liquid stock selection** based on volume and PE ratios
- **Ready-to-use data** in JSON format for backtesting

## 🔧 Key Features

- **Dynamic lot sizes** from your CSV (no hardcoding)
- **Real-time option chain** data for accurate strikes
- **Fuzzy matching** for company name conversion
- **Liquid stock filtering** based on volume and PE
- **ATM strike calculation** for all 50 stocks

## 🎯 That's It!

No complex setup, no redundant files. Just run the two commands above and you'll have:
- ✅ Company names converted to symbols
- ✅ Lot sizes loaded from your CSV
- ✅ ATM strikes calculated using option chain
- ✅ Liquid stocks selected for backtesting
- ✅ Data exported in JSON format
