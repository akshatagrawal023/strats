import threading
import queue
import time
from typing import Optional, Callable
import numpy as np
from expiry_analysis.stream_store import StreamStore
from greeks.matrix_greeks import MatrixGreeksCalculator

class AsyncSinks:
    """
    Background sinks for low-latency ingestion:
    - DB writer: persists raw option-chain responses via StreamStore
    - Feature worker: computes Greeks/features from matrices

    Usage:
      sinks = AsyncSinks(db_path="expiry_analysis/stream_data.db")
      processor.on_snapshot = sinks.handle_snapshot
      sinks.start()
    """

    def __init__(self, db_path: str,
                 risk_free_rate: float = 0.065,
                 days_to_expiry: int = 7,
                 feature_callback: Optional[Callable] = None):
        self.db = StreamStore(db_path=db_path)
        self.greeks_calc = MatrixGreeksCalculator(risk_free_rate=risk_free_rate,
                                                  days_to_expiry=days_to_expiry)

        self._q_db = queue.Queue(maxsize=2048)
        self._q_feat = queue.Queue(maxsize=2048)
        self._stop = threading.Event()

        self._t_db = threading.Thread(target=self._db_loop, name="DBWriter", daemon=True)
        self._t_feat = threading.Thread(target=self._feat_loop, name="FeatureWorker", daemon=True)

        # Optional user callback: def cb(underlying, ts, extended_matrix, metrics_dict)
        self.feature_callback = feature_callback

    def start(self):
        self._t_db.start()
        self._t_feat.start()

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        self._t_db.join(timeout=timeout)
        self._t_feat.join(timeout=timeout)

    def handle_snapshot(self, underlying: str, ts: float, base_mat, spot: float, future_pr: float, raw_resp: dict):
        # Enqueue for DB write (raw API resp is cheapest to persist without extra parsing here)
        try:
            self._q_db.put_nowait((underlying, raw_resp))
        except queue.Full:
            pass

        # Enqueue for feature/greeks (use base matrix + spot/future for speed)
        try:
            self._q_feat.put_nowait((underlying, ts, base_mat, spot, future_pr))
        except queue.Full:
            pass

    # --------------------------------- internal loops ---------------------------------

    def _db_loop(self):
        # Use small batching for SQLite throughput
        batch = []
        last_flush = time.time()
        while not self._stop.is_set():
            flush_due = (time.time() - last_flush) > 0.25  # flush every 250ms
            try:
                item = self._q_db.get(timeout=0.05)
                batch.append(item)
            except queue.Empty:
                pass

            if batch and (flush_due or len(batch) >= 64):
                # Write batch
                try:
                    for underlying, raw_resp in batch:
                        self.db.ingest_option_chain_payload(underlying, raw_resp)
                except Exception:
                    # Drop on error; keep loop alive
                    pass
                finally:
                    batch.clear()
                    last_flush = time.time()

    def _feat_loop(self):
        # Lightweight compute loop; debounce to avoid redundant work
        last_ts_by_und = {}
        while not self._stop.is_set():
            try:
                underlying, ts, base_mat, spot, future_pr = self._q_feat.get(timeout=0.05)
            except queue.Empty:
                continue

            # Simple debounce per underlying
            prev_ts = last_ts_by_und.get(underlying)
            if prev_ts is not None and ts <= prev_ts:
                continue
            last_ts_by_und[underlying] = ts

            # Extend matrix by broadcasting spot/future
            # base_mat shape: (11, strikes)
            try:
                n_strikes = base_mat.shape[1]
                spot_row = spot if spot is not None else float('nan')
                fut_row = future_pr if future_pr is not None else float('nan')
                spot_mat = (np.ones((1, n_strikes)) * spot_row).astype(float)
                fut_mat = (np.ones((1, n_strikes)) * fut_row).astype(float)
                extended = np.vstack([base_mat, spot_mat, fut_mat])  # (13, strikes)

                # Compute IV/Greeks
                extended_with_greeks = self.greeks_calc.add_greeks_to_matrix(extended)

                if callable(self.feature_callback):
                    # Provide a minimal metrics dict (can be expanded later)
                    atm = self.greeks_calc.get_atm_greeks(extended_with_greeks)
                    self.feature_callback(underlying, ts, extended_with_greeks, {'atm': atm})
            except Exception:
                # Compute errors are ignored to avoid impacting ingestion
                pass


