import sqlite3
import threading
from collections import deque, defaultdict
from datetime import datetime


class StreamStore:
    """
    Stores streaming quotes and option-chain snapshots in SQLite (durable) and
    in-memory ring buffers (low latency). Designed for one-underlying-at-a-time
    inference, but supports multiple symbols.
    """

    def __init__(self, db_path: str = "expiry_analysis/stream_data.db",
                 max_seconds_snapshots: int = 300,
                 max_minute_bars: int = 180,
                 max_chain_snapshots: int = 180):
        self.db_path = db_path
        self.lock = threading.Lock()

        # In-memory buffers
        self.latest = {}  # symbol -> last snapshot (dict)
        self.snapshots_1s = defaultdict(lambda: deque(maxlen=max_seconds_snapshots))  # symbol -> deque of 1s quotes
        self.minute_bars = defaultdict(lambda: deque(maxlen=max_minute_bars))        # symbol -> deque of minute bars
        self.chain_snapshots = defaultdict(lambda: deque(maxlen=max_chain_snapshots)) # symbol -> deque of chain snapshots

        self._init_db()

    # ---------- SQLite setup ----------
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Quotes table (underlying/future/option point-in-time snapshot)
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS quotes (
                    ts INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    underlying TEXT,
                    instrument TEXT, -- EQ, FUT, OPT
                    ltp REAL,
                    bid REAL,
                    ask REAL,
                    volume INTEGER,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    prev_close REAL,
                    atp REAL,
                    PRIMARY KEY (ts, symbol)
                )
                """
            )

            # Minute bars table
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS minute_bars (
                    ts_min INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    underlying TEXT,
                    o REAL, h REAL, l REAL, c REAL, v INTEGER,
                    PRIMARY KEY (ts_min, symbol)
                )
                """
            )

            # Option-chain per-strike table
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS option_chain (
                    ts INTEGER NOT NULL,
                    underlying TEXT NOT NULL,
                    expiry TEXT,
                    strike REAL,
                    type TEXT,   -- 'CE' or 'PE'
                    ltp REAL,
                    bid REAL,
                    ask REAL,
                    volume INTEGER,
                    oi INTEGER,
                    oich INTEGER,
                    prev_oi INTEGER,
                    iv REAL,
                    delta REAL, gamma REAL, theta REAL, vega REAL,
                    PRIMARY KEY (ts, underlying, expiry, strike, type)
                )
                """
            )

            # Option-chain aggregates
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS option_chain_agg (
                    ts INTEGER NOT NULL,
                    underlying TEXT NOT NULL,
                    expiry TEXT,
                    call_oi INTEGER,
                    put_oi INTEGER,
                    pcr REAL,
                    PRIMARY KEY (ts, underlying, expiry)
                )
                """
            )

            # Lightweight migration: ensure new columns exist
            def ensure_column(table, column, decl):
                try:
                    c.execute(f"PRAGMA table_info({table})")
                    cols = [row[1] for row in c.fetchall()]
                    if column not in cols:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                except Exception:
                    pass

            ensure_column('quotes', 'underlying', 'TEXT')
            ensure_column('quotes', 'instrument', 'TEXT')
            ensure_column('minute_bars', 'underlying', 'TEXT')

            conn.commit()

    # ---------- Helpers ----------
    @staticmethod
    def _sec(ts):
        # ts may be str or ms; normalize to int seconds
        try:
            if isinstance(ts, str):
                ts = int(float(ts))
            if ts > 10**12:
                ts = int(ts / 1000)
            return int(ts)
        except Exception:
            return int(datetime.utcnow().timestamp())

    @staticmethod
    def _minute_bucket(ts_sec):
        return int(ts_sec - (ts_sec % 60))

    # ---------- Symbol parsing ----------
    @staticmethod
    def _classify_symbol(symbol: str):
        """
        Return (underlying, instrument) for Fyers-style symbols like
        'NSE:RELIANCE-EQ', 'NSE:RELIANCE25SEPFUT', 'NSE:RELIANCE25SEP2500CE'.
        Best-effort using simple rules; callers can override if needed.
        """
        if not symbol:
            return None, None
        try:
            # Strip exchange prefix
            core = symbol.split(':', 1)[1] if ':' in symbol else symbol
            # Options
            if core.endswith('CE') or core.endswith('PE'):
                # underlying = leading letters
                u = ''
                for ch in core:
                    if ch.isalpha():
                        u += ch
                    else:
                        break
                return u, 'OPT'
            # Futures
            if core.endswith('FUT'):
                u = ''
                for ch in core:
                    if ch.isalpha():
                        u += ch
                    else:
                        break
                return u, 'FUT'
            # Equity
            if core.endswith('-EQ'):
                u = core[:-3]  # drop -EQ
                return u, 'EQ'
            # Fallback
            # take leading letters as underlying
            u = ''
            for ch in core:
                if ch.isalpha():
                    u += ch
                else:
                    break
            return u or core, None
        except Exception:
            return None, None

    # ---------- Ingestion: quotes (get_quotes payload) ----------
    def ingest_quotes_payload(self, resp: dict):
        if not resp or resp.get('s') != 'ok':
            return
        items = resp.get('d', [])
        rows = []
        with self.lock:
            for it in items:
                v = it.get('v', {})
                symbol = v.get('symbol') or it.get('n')
                if not symbol:
                    continue
                ts = self._sec(v.get('tt') or datetime.utcnow().timestamp())
                ltp = v.get('lp')
                bid = v.get('bid')
                ask = v.get('ask')
                vol = v.get('volume')
                opn = v.get('open_price')
                high = v.get('high_price')
                low = v.get('low_price')
                prev = v.get('prev_close_price')
                atp = v.get('atp')
                underlying, instrument = self._classify_symbol(symbol)

                # latest
                self.latest[symbol] = {
                    'ts': ts, 'ltp': ltp, 'bid': bid, 'ask': ask, 'volume': vol,
                    'underlying': underlying, 'instrument': instrument
                }

                # 1s snapshots
                self.snapshots_1s[symbol].append({'ts': ts, 'ltp': ltp, 'bid': bid, 'ask': ask, 'volume': vol})

                # minute bars (naive on 1s snapshots)
                ts_min = self._minute_bucket(ts)
                bars = self.minute_bars[symbol]
                if not bars or bars[-1]['ts_min'] != ts_min:
                    # start new bar
                    bars.append({'ts_min': ts_min, 'underlying': underlying, 'o': ltp, 'h': ltp, 'l': ltp, 'c': ltp, 'v': vol})
                else:
                    bar = bars[-1]
                    if ltp is not None:
                        bar['h'] = ltp if bar['h'] is None else max(bar['h'], ltp)
                        bar['l'] = ltp if bar['l'] is None else min(bar['l'], ltp)
                        bar['c'] = ltp
                    bar['v'] = vol

                rows.append((ts, symbol, underlying, instrument, ltp, bid, ask, vol, opn, high, low, prev, atp))

        if rows:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.executemany(
                    """
                    INSERT OR REPLACE INTO quotes
                    (ts, symbol, underlying, instrument, ltp, bid, ask, volume, open_price, high_price, low_price, prev_close, atp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()

    # ---------- Ingestion: option-chain (api_utils.get_option_chain payload) ----------
    def ingest_option_chain_payload(self, underlying_symbol: str, optionchain_resp: dict, ts):
        if not optionchain_resp or optionchain_resp.get('s') != 'ok':
            return

        data = optionchain_resp.get('data', {})
        options = data.get('optionsChain', [])

        # Derive expiry per strike row if encoded in symbol; else leave None
        per_strike_rows = []
        ce_oi_total = 0
        pe_oi_total = 0

        for row in options:
            sym = row.get('symbol', '')
            if row.get('option_type') in ('CE', 'PE'):
                strike = row.get('strike_price')
                otype = row.get('option_type')
                ltp = row.get('ltp')
                bid = row.get('bid')
                ask = row.get('ask')
                vol = row.get('volume')
                oi = row.get('oi')
                oich = row.get('oich')
                prev_oi = row.get('prev_oi')
                # naive expiry parse (broker formats vary); leave None if unknown
                expiry = None
                per_strike_rows.append((ts, underlying_symbol, expiry, strike, otype, ltp, bid, ask, vol, oi, oich, prev_oi))

                if oi is not None:
                    if otype == 'CE':
                        ce_oi_total += int(oi)
                    elif otype == 'PE':
                        pe_oi_total += int(oi)

        # Insert per-strike rows
        if per_strike_rows:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.executemany(
                    """
                    INSERT OR REPLACE INTO option_chain
                    (ts, underlying, expiry, strike, type, ltp, bid, ask, volume, oi, oich, prev_oi)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    per_strike_rows,
                )
                # Aggregates
                pcr = (pe_oi_total / ce_oi_total) if ce_oi_total else None
                c.execute(
                    """
                    INSERT OR REPLACE INTO option_chain_agg
                    (ts, underlying, expiry, call_oi, put_oi, pcr)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (ts, underlying_symbol, None, ce_oi_total or 0, pe_oi_total or 0, pcr),
                )
                conn.commit()

        # In-memory snapshot for low latency
        with self.lock:
            self.chain_snapshots[underlying_symbol].append({
                'ts': ts,
                'options': options,
                'call_oi': ce_oi_total,
                'put_oi': pe_oi_total,
                'pcr': (pe_oi_total / ce_oi_total) if ce_oi_total else None,
            })

    # ---------- Accessors for model ----------
    def get_latest_feature_snapshot(self, underlying_symbol: str):
        with self.lock:
            return {
                'latest_quotes': {underlying_symbol: self.latest.get(underlying_symbol)},
                'latest_future': None,  # populate from quotes ingestor using your FUT symbol
                'minute_bars': list(self.minute_bars.get(underlying_symbol, []))[-5:],
                'chain': self.chain_snapshots.get(underlying_symbol, deque())[-1] if self.chain_snapshots.get(underlying_symbol) else None,
            }


