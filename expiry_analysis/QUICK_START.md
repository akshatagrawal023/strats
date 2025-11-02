# Quick Start Guide

## 🚀 3-Step Setup

### Step 1: Import and Initialize
```python
from expiry_analysis.main_orchestrator import TradingSystem

system = TradingSystem(
    underlyings=["RELIANCE", "HDFCBANK"],
    strike_count=3,
    poll_interval=2.0  # Poll API every 2 seconds
)
```

### Step 2: Add Your Signal Handler (Optional)
```python
def my_signal_handler(underlying, signal_dict):
    if signal_dict['signal'] == 'BUY':
        # Place buy order
        pass
    elif signal_dict['signal'] == 'SELL':
        # Place sell order
        pass

system.on_signal = my_signal_handler
```

### Step 3: Start the System
```python
system.start()  # Blocking - press Ctrl+C to stop
```

**That's it!** The system will:
- ✅ Fetch data from API
- ✅ Store in memory (ring buffers)
- ✅ Write to database (async)
- ✅ Compute Greeks/Features (async)
- ✅ Generate signals
- ✅ Call your handler

---

## 📁 File Organization

### Core Files (You Don't Need to Touch)
- `chain_processor.py` - Ingests & stores data
- `runtime_pipeline.py` - Async DB + Feature worker
- `feature_pipeline.py` - Feature generation
- `matrix_greeks.py` - Greeks calculation

### Files You Use
- `main_orchestrator.py` - **Start here!** Wires everything together
- `system_arch.md` - Architecture details

---

## 🔌 Integration with Your Model

### Option A: Simple Rule-Based (Built-in)
```python
signal = system.get_signal("RELIANCE")  # Uses PCR/momentum rules
```

### Option B: Your Custom Model
```python
import torch

class MyModel(torch.nn.Module):
    # Your CNN/RNN model
    pass

model = MyModel.load_state_dict(torch.load("model.pth"))

def model_wrapper(tensor_batch):
    # tensor_batch shape: (1, time, channels, strikes)
    with torch.no_grad():
        output = model(torch.from_numpy(tensor_batch).float())
    return output.numpy()

signal = system.get_signal("RELIANCE", model=model_wrapper)
```

---

## 📊 Accessing Features Directly

### Get DL Tensor (for model inference)
```python
tensor, features_df = system.get_latest_features("RELIANCE", window=64)
# tensor shape: (time_steps, channels, strikes)
# channels: CE_MID, PE_MID, CE_IV, PE_IV, CE_DELTA, PE_DELTA, CE_OI, PE_OI, CE_OICH, PE_OICH, UNDERLYING_LTP
```

### Get Compact Features (for analysis)
```python
features_df = system.feature_gen.build_compact_features("RELIANCE", window=20)
print(features_df['pcr_atm'].iloc[0])  # Put-Call Ratio at ATM
```

---

## 🔄 WebSocket Integration

If you have a WebSocket handler:

```python
from expiry_analysis.data_socket import DataManager

# Your WebSocket handler
socket = DataManager()

# Process each message
def on_websocket_message(message):
    underlying = extract_underlying(message)
    resp = parse_to_option_chain_format(message)
    system.process_snapshot(underlying, resp)

socket.on_message = on_websocket_message
socket.start()
```

---

## 📈 System Status

```python
system.status()  # Prints snapshot counts, etc.
```

---

## 🛑 Stopping

```python
system.stop()  # Gracefully shuts down all threads
```

---

## 📝 What Gets Stored

- **Memory**: Last 300 snapshots (configurable) in ring buffers
- **Database**: All raw option chain responses in SQLite (`expiry_analysis/stream_data.db`)
- **Features**: Computed on-demand or via async callback

---

## 🎯 Minimal Example

```python
from expiry_analysis.main_orchestrator import TradingSystem

system = TradingSystem(underlyings=["RELIANCE"])
system.start()  # That's it!
```

Press Ctrl+C when done.

---

## ❓ Common Questions

**Q: How do I add more underlyings?**
```python
system = TradingSystem(underlyings=["RELIANCE", "HDFCBANK", "TATAMOTORS"])
```

**Q: How do I change database location?**
```python
system = TradingSystem(
    underlyings=["RELIANCE"],
    db_path="custom/path/data.db"
)
```

**Q: How do I adjust Greeks parameters?**
```python
system = TradingSystem(
    underlyings=["RELIANCE"],
    risk_free_rate=0.065,  # 6.5%
    days_to_expiry=7      # 7 days to expiry
)
```

**Q: Can I process data manually without polling?**
```python
resp = get_option_chain("NSE:RELIANCE-EQ", 3)
system.process_snapshot("RELIANCE", resp)
```

---

## 🐛 Troubleshooting

**Problem**: "No matrix data for X"
- **Solution**: Wait a few seconds - system needs to accumulate snapshots first

**Problem**: "Invalid response"
- **Solution**: Check API connection, verify symbol format

**Problem**: High latency
- **Solution**: Reduce `poll_interval` or use WebSocket instead of polling

---

## 📚 Next Steps

1. Run `python expiry_analysis/main_orchestrator.py` to see it work
2. Add your model in `get_signal()` method
3. Add trading logic in `on_signal` callback
4. Monitor with `system.status()`

That's all you need! 🎉

