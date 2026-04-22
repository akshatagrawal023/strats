"""
hdf5_archiver.py
----------------
Background HDF5 writer for the paper trading pipeline.

Stores a single unified row per timestamp containing:
  - All scalar features (spot, T, atm_iv, skew, iv_z, panic_flag, spot_ewm_vol)
  - MTM PnL columns per simulation ID
  - Index references into raw_matrices and greeks_matrices numpy datasets

Flushes to disk every 60 seconds or on clean shutdown.
Never touches the hot asyncio path.
"""

import numpy as np
import threading
import time
import os
import logging
from datetime import datetime
from typing import Optional

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logging.warning("[HDF5Archiver] h5py not installed. Run: pip install h5py. Falling back to CSV.")

logger = logging.getLogger(__name__)


class HDF5Archiver:
    """
    Thread-safe background archiver that persists option chain data per tick
    into a unified HDF5 file (one file per trading day).

    File structure:
        NIFTY_<YYYYMMDD>.h5
          /ticks          — DataFrame: one row per timestamp (scalars + MTM)
          /raw_matrices   — 3D numpy: (n_ticks, n_channels, n_strikes)
          /greeks_matrices— 3D numpy: (n_ticks, 8, n_strikes)
    """

    def __init__(self, output_dir: str = "paper_trading_logs", flush_interval: int = 60):
        self.output_dir = output_dir
        self.flush_interval = flush_interval  # seconds

        os.makedirs(output_dir, exist_ok=True)

        # In-memory row buffers
        self._scalar_rows = []        # list of dicts
        self._raw_matrices = []       # list of np.ndarray (n_channels, n_strikes)
        self._greeks_matrices = []    # list of np.ndarray (8, n_strikes)

        self._lock = threading.Lock()
        self._shutdown = threading.Event()

        # Start background flush thread
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

        logger.info(f"[HDF5Archiver] Initialized. Output: {output_dir}, Flush: {flush_interval}s")

    def _get_filepath(self) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.output_dir, f"NIFTY_{date_str}.h5")

    def record_tick(
        self,
        timestamp: float,
        spot: float,
        expiry_ts: float,
        T: float,
        atm_iv: float,
        skew: float,
        iv_z_score: float,
        panic_flag: bool,
        spot_ewm_vol: float,
        raw_matrix: np.ndarray,         # shape (n_channels, n_strikes)
        greeks_matrix: np.ndarray,      # shape (8, n_strikes)
        mtm_pnl: Optional[dict] = None, # {'IB_W50': -700.0, 'IB_W100': -600.0, ...}
    ):
        """
        Queue one tick's data for background archival.
        This is the only method called from the hot asyncio loop.
        It is non-blocking — just appends to in-memory lists under a lock.
        """
        if not HDF5_AVAILABLE:
            return

        row = {
            'timestamp': timestamp,
            'spot': spot,
            'expiry_ts': expiry_ts,
            'T': T,
            'atm_iv': atm_iv,
            'skew': skew,
            'iv_z_score': iv_z_score,
            'panic_flag': 1 if panic_flag else 0,
            'spot_ewm_vol': spot_ewm_vol,
            'raw_matrix_idx': -1,    # will be updated at flush time
            'greeks_matrix_idx': -1,
        }

        # Append MTM columns
        if mtm_pnl:
            for sim_id, pnl in mtm_pnl.items():
                row[f'mtm_{sim_id}'] = pnl

        with self._lock:
            idx = len(self._raw_matrices)
            row['raw_matrix_idx'] = idx
            row['greeks_matrix_idx'] = idx

            self._scalar_rows.append(row)
            self._raw_matrices.append(raw_matrix.copy())
            self._greeks_matrices.append(greeks_matrix.copy())

    def _flush_loop(self):
        """Background thread: flush to HDF5 every flush_interval seconds."""
        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=self.flush_interval)
            self._flush_to_disk()

    def _flush_to_disk(self):
        with self._lock:
            if not self._scalar_rows:
                return

            rows = self._scalar_rows.copy()
            raw_mats = list(self._raw_matrices)
            greek_mats = list(self._greeks_matrices)

            self._scalar_rows.clear()
            self._raw_matrices.clear()
            self._greeks_matrices.clear()

        filepath = self._get_filepath()

        try:
            with h5py.File(filepath, 'a') as f:
                n_new = len(rows)

                # ----- /raw_matrices -----
                raw_stack = np.stack(raw_mats, axis=0)  # (n_new, n_ch, n_strikes)
                if 'raw_matrices' in f:
                    old = f['raw_matrices']
                    old_n = old.shape[0]
                    old.resize(old_n + n_new, axis=0)
                    old[old_n:] = raw_stack
                else:
                    f.create_dataset('raw_matrices', data=raw_stack,
                                     maxshape=(None, raw_stack.shape[1], raw_stack.shape[2]),
                                     compression='gzip', compression_opts=4)

                # ----- /greeks_matrices -----
                greek_stack = np.stack(greek_mats, axis=0)
                if 'greeks_matrices' in f:
                    old = f['greeks_matrices']
                    old_n = old.shape[0]
                    old.resize(old_n + n_new, axis=0)
                    old[old_n:] = greek_stack
                else:
                    f.create_dataset('greeks_matrices', data=greek_stack,
                                     maxshape=(None, greek_stack.shape[1], greek_stack.shape[2]),
                                     compression='gzip', compression_opts=4)

                # ----- /ticks (scalar features + MTM) -----
                # Build typed numpy record array from row dicts
                all_keys = list(rows[0].keys())
                for row in rows[1:]:
                    for k in row:
                        if k not in all_keys:
                            all_keys.append(k)

                dtype_map = []
                for k in all_keys:
                    # Determine dtype from first non-None value
                    sample = rows[0].get(k, np.nan)
                    if isinstance(sample, (bool, int)) and k == 'panic_flag':
                        dtype_map.append((k, np.int8))
                    else:
                        dtype_map.append((k, np.float64))

                records = np.full(n_new, fill_value=np.nan,
                                  dtype=np.dtype(dtype_map))

                for i, row in enumerate(rows):
                    for k in all_keys:
                        records[i][k] = row.get(k, np.nan)

                if 'ticks' in f:
                    old = f['ticks']
                    old_n = old.shape[0]
                    old.resize(old_n + n_new, axis=0)
                    old[old_n:] = records
                else:
                    f.create_dataset('ticks', data=records,
                                     maxshape=(None,),
                                     compression='gzip', compression_opts=4)

            logger.info(f"[HDF5Archiver] Flushed {n_new} ticks → {filepath}")

        except Exception as e:
            logger.error(f"[HDF5Archiver] Flush failed: {e}")

    def shutdown(self):
        """Flush remaining buffer and stop background thread cleanly."""
        logger.info("[HDF5Archiver] Shutdown: flushing remaining buffer...")
        self._shutdown.set()
        self._flush_to_disk()
        logger.info("[HDF5Archiver] Shutdown complete.")
