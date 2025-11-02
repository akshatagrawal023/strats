# Trading System Architecture

## File Structure & Responsibilities

```
expiry_analysis/
├── chain_processor.py          # Core: Ingests API data → Matrix storage (ring buffers)
├── runtime_pipeline.py         # Async: DB writer + Feature worker (background threads)
├── feature_pipeline.py         # Features: Generates DL tensors + compact features
├── stream_store.py             # Database: SQLite storage for raw chains (existing)
├── main_orchestrator.py        # ⭐ START HERE: Wires everything together
└── system_arch.md              # This file

vol/
└── matrix_greeks.py            # Greeks: IV + Delta/Gamma/Theta/Vega calculation

utils/
├── api_utils.py                # API: get_option_chain() wrapper
└── ...
```

## Data Flow

```
API Response
    ↓
[chain_processor.py] → Matrix (ring buffer) + Callback
    ↓
[runtime_pipeline.py] → Parallel async:
    ├── DB Writer → stream_store.py (SQLite)
    └── Feature Worker → matrix_greeks.py → Extended Matrix (23 channels)
        ↓
[feature_pipeline.py] → DL Tensor + Compact Features
    ↓
Your Model → Trading Signal
```

## Component Details

### 1. chain_processor.py (Ingestion Layer)
- **What**: Stores option chain data in preallocated circular buffers
- **Input**: Raw API response from `get_option_chain()`
- **Output**: Matrix (11 channels × strikes) in memory
- **Key Method**: `process_option_chain(underlying, resp)`
- **Feature**: `on_snapshot` callback for async dispatch

### 2. runtime_pipeline.py (Async Processing)
- **What**: Background workers that don't block ingestion
- **DB Writer**: Batches raw responses → SQLite (every 250ms or 64 items)
- **Feature Worker**: Adds IV/Greeks → Extended matrix (23 channels)
- **Key Method**: `handle_snapshot()` - called by processor callback

### 3. feature_pipeline.py (Feature Generation)
- **What**: Converts matrices to model-ready features
- **DL Tensor**: Normalized (time, channels, strikes) for CNN/RNN
- **Compact Features**: Aggregated indicators (PCR, momentum, etc.)
- **Key Methods**: `build_dl_tensor()`, `build_compact_features()`

### 4. matrix_greeks.py (Greeks Calculator)
- **What**: Computes IV + Greeks from option prices
- **Input**: Matrix (13 channels) with prices
- **Output**: Extended matrix (23 channels) with Greeks added
- **Key Method**: `add_greeks_to_matrix()`

## How to Run

### Option 1: Polling Mode (API-based)
```python
from expiry_analysis.main_orchestrator import TradingSystem

system = TradingSystem(
    underlyings=["RELIANCE", "HDFCBANK"],
    poll_interval=2.0
)
system.start()
```

### Option 2: WebSocket Mode (Real-time)
```python
# Use your existing data_socket.py with system
from expiry_analysis.data_socket import DataManager
from expiry_analysis.main_orchestrator import TradingSystem

socket = DataManager()  # Your WebSocket handler
system = TradingSystem.from_websocket(socket)
system.start()
```

## Integration Points

1. **Data Source**: Update `main_orchestrator.py` to use your WebSocket/API source
2. **Model Inference**: Add your model in `on_features_ready()` callback
3. **Trading Logic**: Add signal processing in `on_signal()` callback
4. **Monitoring**: Extend `_monitor()` method for custom metrics

## Minimal Setup (3 steps)

1. **Initialize**: `system = TradingSystem(underlyings=["RELIANCE"])`
2. **Start**: `system.start()`
3. **Process signals**: Add your model/signal handler in `on_signal()` callback

That's it! All wiring is done in `main_orchestrator.py`.

