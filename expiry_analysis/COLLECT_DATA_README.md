# Data Collection Script

Minimal script to collect live option chain data and Greek matrices for training.

## Features

- ✅ **Parallel collection** for multiple underlyings
- ✅ **No windowing** - saves each snapshot separately
- ✅ **Separate files** per underlying
- ✅ **NPZ format** - efficient NumPy compressed storage
- ✅ **Metadata tracking** - JSON file with all snapshot info

## Usage

### 1. Configure

Edit `expiry_analysis/config.py`:
```python
UNDERLYINGS = ["RELIANCE", "HDFCBANK", "TATAMOTORS"]
STRIKE_COUNT = 3
POLL_INTERVAL = 2.0  # seconds
RISK_FREE_RATE = 0.065
DAYS_TO_EXPIRY = 7
```

### 2. Run Collection

```bash
python expiry_analysis/collect_training_data.py
```

The script will:
- Create `training_data/` directory
- Create subdirectories for each underlying
- Save each snapshot as `snapshot_<timestamp>.npz`
- Save metadata in `metadata.json`
- Print progress every 10 snapshots

### 3. Stop Collection

Press `Ctrl+C` to stop. The script will:
- Save final metadata
- Print summary of collected data

## Data Structure

```
training_data/
├── RELIANCE/
│   ├── metadata.json          # All snapshot metadata
│   ├── snapshot_1234567890.npz
│   ├── snapshot_1234567892.npz
│   └── ...
└── HDFCBANK/
    ├── metadata.json
    ├── snapshot_1234567890.npz
    └── ...
```

Each NPZ file contains:
- `base_matrix`: (13, strikes) - Raw option chain data
- `greeks_matrix`: (10, strikes) - Calculated Greeks
- `combined_matrix`: (23, strikes) - Combined matrix
- `timestamp`: Unix timestamp

## Loading Data

```python
from expiry_analysis.load_training_data import load_underlying_data, load_all_underlyings

# Load one underlying
data = load_underlying_data("training_data/RELIANCE")
print(f"Loaded {len(data)} snapshots")

# Access a snapshot
snapshot = data[0]
print(f"Matrix shape: {snapshot['combined_matrix'].shape}")  # (23, 7)

# Load all underlyings
all_data = load_all_underlyings("training_data")
for underlying, snapshots in all_data.items():
    print(f"{underlying}: {len(snapshots)} snapshots")
```

## Creating Training Windows

```python
from expiry_analysis.load_training_data import create_training_windows

# Create rolling windows of 300 timesteps
windows = create_training_windows(data, window_size=300, step_size=1)
print(f"Created {len(windows)} windows")
print(f"Window shape: {windows[0].shape}")  # (300, 23, 7)
```

## Notes

- **Storage**: Each snapshot is ~5-10 KB (compressed)
- **Rate**: Can collect ~0.5 snapshots/second per underlying (limited by API)
- **Resume**: Script can resume - it loads existing metadata on start
- **Thread-safe**: Each underlying saves to its own directory

## Example Output

```
[Start] Collecting data for ['RELIANCE', 'HDFCBANK']
Output directory: training_data
Interval: 2.0s
Press Ctrl+C to stop

[RELIANCE] Started data collection
[HDFCBANK] Started data collection
[RELIANCE] Saved 10 snapshots | Rate: 0.5/s
[HDFCBANK] Saved 10 snapshots | Rate: 0.5/s
...
^C
[Stop] Shutting down...
[RELIANCE] Stopped data collection. Total: 150 snapshots
[HDFCBANK] Stopped data collection. Total: 150 snapshots

[Summary]
  RELIANCE: 150 snapshots in training_data\RELIANCE
  HDFCBANK: 150 snapshots in training_data\HDFCBANK
```

