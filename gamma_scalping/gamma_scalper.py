import os
import sys
import time
import logging
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fyers_instance import FyersInstance
from utils.api_utils import get_quotes, place_order


class GammaScalper:
    """
    Minimal delta-hedged long gamma scaffolding.

    Responsibilities:
      - Select ATM straddle (or 25Δ strangle) for a target underlying and expiry
      - Track live greeks/IV from quotes (proxy via LTP/IV fields if available)
      - Maintain delta-neutrality by hedging with futures when |Δ| > threshold
      - Basic risk controls: time stop, max hedges/day, slippage budget
      - Logging of PnL components placeholders (theta, hedge PnL)
    """

    def __init__(self, config):
        self.config = config
        self.fyers = FyersInstance.get_instance()
        self.logger = logging.getLogger("GammaScalper")

        # Rolling windows for realized volatility estimation (optional)
        self.price_window = deque(maxlen=600)  # ~10 minutes at 1s cadence

        # State
        self.current_positions: Dict[str, Any] = {}
        self.last_hedge_time: Dict[str, float] = {}
        self.hedge_count: Dict[str, int] = {}

        # Defaults with safe fallbacks
        self.strike_step = getattr(self.config, 'OPTION_STRIKE_STEP', 50)
        self.delta_band = getattr(self.config, 'DELTA_BAND', 0.10)  # as fraction of one contract delta
        self.hedge_interval = getattr(self.config, 'HEDGE_INTERVAL_SECONDS', 5)
        self.session_minutes = getattr(self.config, 'SESSION_MINUTES', 60)
        self.max_hedges = getattr(self.config, 'MAX_HEDGES_PER_DAY', 50)
        self.lot_size_map = getattr(self.config, 'LOT_SIZE_MAP', {})
        self.default_lot_size = getattr(self.config, 'DEFAULT_LOT_SIZE', 50)

    # ---------- Symbol helpers ----------
    def build_symbol(self, base: str, expiry: str, strike: int, right: str) -> str:
        """Return Fyers symbol like 'NSE:NIFTY24AUG20000CE'"""
        return f"NSE:{base}{expiry}{strike}{right}"

    def build_future(self, base: str, expiry: str) -> str:
        return f"NSE:{base}{expiry}FUT"

    # ---------- Strategy lifecycle ----------
    def choose_atm_straddle(self, base: str, expiry: str) -> Tuple[str, str, int]:
        """Pick ATM CE/PE using option chain (preferred), fallback to futures LTP."""
        try:
            if getattr(self.config, 'USE_OPTIONCHAIN_FOR_ATM', True):
                ce, pe, atm = self._choose_atm_from_optionchain(base, expiry)
                if ce and pe:
                    return ce, pe, atm
        except Exception as e:
            self.logger.warning(f"Optionchain ATM selection failed, falling back. err={e}")

        fut_symbol = self.build_future(base, expiry)
        q = get_quotes([fut_symbol])
        fut_ltp = q.get('d', [{}])[0].get('v', {}).get('lp') or q.get('d', [{}])[0].get('v', {}).get('ltp')
        if not fut_ltp:
            raise RuntimeError(f"No LTP for {fut_symbol}")
        step = self.strike_step
        atm = int(round(float(fut_ltp) / step) * step)
        ce = self.build_symbol(base, expiry, atm, 'CE')
        pe = self.build_symbol(base, expiry, atm, 'PE')
        return ce, pe, atm

    def _choose_atm_from_optionchain(self, base: str, expiry: str) -> Tuple[str, str, int]:
        """Query Fyers optionchain and select ATM CE/PE symbols for given expiry."""
        # Determine underlying symbol for optionchain API (EQ vs INDEX). Allow override.
        oc_symbol = getattr(self.config, 'GAMMA_CHAIN_SYMBOL', None)
        if not oc_symbol:
            # Heuristic: try equity format; users can override in Config
            oc_symbol = f"NSE:{base}-EQ"
        data = {
            "symbol": oc_symbol,
            "strikecount": getattr(self.config, 'CHAIN_STRIKECOUNT', 2),
            "timestamp": ""
        }
        fy = self.fyers
        resp = fy.optionchain(data=data)
        if not resp or resp.get('s') != 'ok':
            raise RuntimeError(f"optionchain failed for {oc_symbol}: {resp}")
        chain = resp.get('data', {}).get('optionsChain', [])
        if not chain:
            raise RuntimeError("No optionsChain in response")
        # Underlying record: option_type == '' and strike_price == -1
        und = next((r for r in chain if r.get('option_type', '') == ''), None)
        und_ltp = und.get('ltp') if und else None
        if und_ltp is None:
            # fallback to fp
            und_ltp = und.get('fp') if und else None
        if und_ltp is None:
            raise RuntimeError("No underlying price in optionchain response")
        # Filter to target expiry if present in symbol string
        def is_expiry_match(sym: str) -> bool:
            return expiry in sym if expiry else True
        ce_rows = [r for r in chain if r.get('option_type') == 'CE' and is_expiry_match(r.get('symbol', ''))]
        pe_rows = [r for r in chain if r.get('option_type') == 'PE' and is_expiry_match(r.get('symbol', ''))]
        if not ce_rows or not pe_rows:
            raise RuntimeError("No CE/PE rows for requested expiry")
        # Pick strike closest to underlying
        target_strike = min({*{r.get('strike_price') for r in ce_rows}, *{r.get('strike_price') for r in pe_rows}},
                            key=lambda k: abs(float(k) - float(und_ltp)))
        ce_row = min(ce_rows, key=lambda r: abs(float(r.get('strike_price')) - float(target_strike)))
        pe_row = min(pe_rows, key=lambda r: abs(float(r.get('strike_price')) - float(target_strike)))
        ce_sym = ce_row.get('symbol')
        pe_sym = pe_row.get('symbol')
        return ce_sym, pe_sym, int(target_strike)

    def open_straddle(self, base: str, expiry: str, lots: int):
        ce, pe, atm = self.choose_atm_straddle(base, expiry)
        qty = lots * self.lot_size_map.get(base, self.default_lot_size)

        orders = [
            {"symbol": ce, "qty": qty, "type": 2, "side": 1, "productType": "INTRADAY", "validity": "DAY"},
            {"symbol": pe, "qty": qty, "type": 2, "side": 1, "productType": "INTRADAY", "validity": "DAY"},
        ]
        resp = place_order(orders)
        self.logger.info(f"Opened straddle {base} {expiry} ATM {atm}: {resp}")
        self.current_positions[base] = {
            "expiry": expiry,
            "atm": atm,
            "qty": qty,
            "legs": {"CE": ce, "PE": pe},
            "delta": 0.0,
            "opened_at": datetime.now(),
        }
        self.hedge_count[base] = 0

    # ---------- Hedging loop ----------
    def estimate_delta(self, base: str) -> float:
        pos = self.current_positions.get(base)
        if not pos:
            return 0.0
        ce = pos["legs"]["CE"]
        pe = pos["legs"]["PE"]
        fut = self.build_future(base, pos["expiry"])

        q = get_quotes([ce, pe, fut])
        data = q.get('d', [])
        if len(data) < 3:
            return 0.0
        v0 = data[0].get('v', {})
        v1 = data[1].get('v', {})
        # Prefer greeks if available
        ce_delta = v0.get('delta')
        pe_delta = v1.get('delta')
        if ce_delta is not None and pe_delta is not None:
            gross_delta = (float(ce_delta) + float(pe_delta)) * pos["qty"]
        else:
            # Fallback: no hedge if greeks unavailable
            gross_delta = 0.0
        pos["delta"] = gross_delta
        return gross_delta

    def hedge_delta(self, base: str):
        pos = self.current_positions.get(base)
        if not pos:
            return
        delta = self.estimate_delta(base)
        band = self.delta_band
        if abs(delta) < band * pos["qty"]:
            return
        # Place futures hedge
        fut = self.build_future(base, pos["expiry"])  # use same-month future
        lot = self.lot_size_map.get(base, self.default_lot_size)
        if lot <= 0:
            return
        hedge_lots = int(round(-delta / lot))
        hedge_qty = abs(hedge_lots)
        if hedge_qty == 0:
            return

        side = 1 if hedge_lots > 0 else -1
        orders = [{"symbol": fut, "qty": hedge_qty, "type": 2, "side": side, "productType": "INTRADAY", "validity": "DAY"}]
        resp = place_order(orders)
        self.hedge_count[base] = self.hedge_count.get(base, 0) + 1
        self.logger.info(f"HEDGE {base} qty={hedge_qty} resp={resp}")

    def manage(self, base: str, expiry: str, lots: int):
        if base not in self.current_positions:
            self.open_straddle(base, expiry, lots)
        start = datetime.now()
        while (datetime.now() - start) < timedelta(minutes=self.session_minutes):
            try:
                self.hedge_delta(base)
                time.sleep(self.hedge_interval)
                if self.hedge_count.get(base, 0) >= self.max_hedges:
                    self.logger.warning(f"Max hedges reached for {base}")
                    break
            except Exception as e:
                self.logger.error(f"Manage loop error: {e}")
                time.sleep(1)

        self.logger.info(f"Session end for {base}")
        # Attempt to close straddle legs conservatively (marketable IOC would be better)
        try:
            pos = self.current_positions.get(base)
            if pos:
                ce = pos["legs"]["CE"]
                pe = pos["legs"]["PE"]
                qty = pos["qty"]
                close_orders = [
                    {"symbol": ce, "qty": qty, "type": 2, "side": -1, "productType": "INTRADAY", "validity": "DAY"},
                    {"symbol": pe, "qty": qty, "type": 2, "side": -1, "productType": "INTRADAY", "validity": "DAY"},
                ]
                resp = place_order(close_orders)
                self.logger.info(f"Closed straddle {base} resp={resp}")
        except Exception as e:
            self.logger.error(f"Close legs error: {e}")

