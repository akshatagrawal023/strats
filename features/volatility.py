# features/volatility.py
import numpy as np
from scipy import stats
from .base import FeatureExtractor
from ..greeks.models import OptionChainGreeks

class VolatilityFeatures(FeatureExtractor):
    """Extract volatility‑related features from an option chain."""
    
    def __init__(self, historical_iv_store=None):
        """
        Args:
            historical_iv_store: Optional object with a `get_percentile(symbol, iv)` method.
        """
        self.historical = historical_iv_store

    def compute(self, chain: OptionChainGreeks) -> dict:
        f = {}
        S = chain.underlying_price
        strikes = chain.strikes
        call = chain.calls
        put = chain.puts

        # --- ATM IV (average) ---
        atm_idx = np.argmin(np.abs(strikes - S))
        f['atm_iv'] = (call.iv[atm_idx] + put.iv[atm_idx]) / 2.0

        # --- IV percentile (if historical store provided) ---
        if self.historical:
            p = self.historical.get_percentile(chain.symbol, f['atm_iv'])
            f['iv_percentile'] = p

        # --- Moneyness and OTM strikes ---
        moneyness = strikes / S
        otm_put_mask = moneyness < 0.95   # roughly 5% OTM put
        otm_call_mask = moneyness > 1.05  # roughly 5% OTM call

        # --- Risk reversal (simplified: use nearest strikes) ---
        if np.any(otm_put_mask):
            otm_put_idx = np.where(otm_put_mask)[0][-1]   # farthest OTM put
            otm_call_idx = np.where(otm_call_mask)[0][0] if np.any(otm_call_mask) else atm_idx
            f['risk_reversal'] = call.iv[otm_call_idx] - put.iv[otm_put_idx]
        else:
            f['risk_reversal'] = np.nan

        # --- Butterfly (strangle IV - ATM IV) ---
        if np.any(otm_put_mask) and np.any(otm_call_mask):
            strangle_iv = (put.iv[otm_put_idx] + call.iv[otm_call_idx]) / 2.0
            f['butterfly'] = strangle_iv - f['atm_iv']
        else:
            f['butterfly'] = np.nan

        # --- Skew slope (linear regression on puts) ---
        put_mask = moneyness <= 1.0
        if put_mask.sum() >= 2:
            slope_put, _, _, _, _ = stats.linregress(moneyness[put_mask], put.iv[put_mask])
            f['skew_slope_put'] = slope_put
        else:
            f['skew_slope_put'] = np.nan

        call_mask = moneyness >= 1.0
        if call_mask.sum() >= 2:
            slope_call, _, _, _, _ = stats.linregress(moneyness[call_mask], call.iv[call_mask])
            f['skew_slope_call'] = slope_call
        else:
            f['skew_slope_call'] = np.nan

        # --- Vanna and Volga at ATM (if available) ---
        if hasattr(call, 'vanna') and hasattr(put, 'vanna'):
            f['atm_vanna'] = call.vanna[atm_idx]   # or average? decide
            f['atm_volga'] = call.volga[atm_idx]
        else:
            f['atm_vanna'] = np.nan
            f['atm_volga'] = np.nan

        # --- Put/call open interest ratio (if data exists) ---
        # This assumes your chain object holds OI arrays.
        if hasattr(chain, 'oi_put') and hasattr(chain, 'oi_call'):
            total_oi_put = np.nansum(chain.oi_put)
            total_oi_call = np.nansum(chain.oi_call)
            f['oi_ratio'] = total_oi_put / total_oi_call if total_oi_call != 0 else np.nan

        return f