"""
Feature generation from matrix + Greeks for DL models (CNN/RNN) and classic ML.

Inputs:
- OptionDataProcessor (provides rolling matrices via get_matrix)
- MatrixGreeksCalculator (adds IV + Greeks channels)

Outputs:
- Tensors for DL: (time_steps, channels, strikes) with normalized channels
- Compact feature vectors for ML: dict/Series with aggregated indicators
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict

from greeks.matrix_greeks import MatrixGreeksCalculator


class FeatureGenerator:
    def __init__(self, processor, risk_free_rate: float = 0.065, days_to_expiry: int = 7):
        self.processor = processor
        self.greeks_calc = MatrixGreeksCalculator(risk_free_rate=risk_free_rate,
                                                  days_to_expiry=days_to_expiry)

    def get_extended_matrix(self, underlying: str, window: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (timestamps, extended_matrices) where extended_matrices shape =
        (time, 23, strikes): original 13 + 10 Greeks channels.
        """
        ts, mats = self.processor.get_matrix(underlying, window)
        if mats is None:
            return None, None
        # Add IV/Greeks per time step
        extended = []
        for snap in mats:  # snap: (13, strikes)
            try:
                ext = self.greeks_calc.add_greeks_to_matrix(snap)
            except Exception:
                # Fallback: keep original shape if Greeks fail
                ext = np.concatenate([snap, np.full((10, snap.shape[1]), np.nan)], axis=0)
            extended.append(ext)
        return np.array(ts), np.stack(extended, axis=0)

    def build_dl_tensor(self, underlying: str, window: int = 64,
                        channels: Optional[list] = None,
                        normalize: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Build normalized tensor for DL models.
        - Returns (timestamps, tensor) with shape (time, channels, strikes)
        - Default channels: MID_PRICE (CE/PE), IV, DELTA, OI, OICH (all normalized)
        - Bid/ask removed - only mid prices used
        """
        default_channels = [
            'CE_MID', 'PE_MID',           # Computed mid prices (not raw channels)
            'CE_IV', 'PE_IV',             # 13-14
            'CE_DELTA', 'PE_DELTA',       # 15-16
            'CE_OI', 'PE_OI',             # 6-7
            'CE_OICH', 'PE_OICH',         # 8-9
            'UNDERLYING_LTP'              # 11
        ]
        from greeks.matrix_greeks import CHANNEL_NAMES
        ch_names = CHANNEL_NAMES
        
        ts, ext = self.get_extended_matrix(underlying, window)
        if ext is None:
            return None, None
        
        # Compute mid prices from bid/ask
        ce_bid = ext[:, ch_names.index('CE_BID'), :]
        ce_ask = ext[:, ch_names.index('CE_ASK'), :]
        pe_bid = ext[:, ch_names.index('PE_BID'), :]
        pe_ask = ext[:, ch_names.index('PE_ASK'), :]
        ce_mid = (ce_bid + ce_ask) / 2.0
        pe_mid = (pe_bid + pe_ask) / 2.0
        
        # Build tensor from selected channels
        sel = channels or default_channels
        tensor_channels = []
        
        for ch_name in sel:
            if ch_name == 'CE_MID':
                tensor_channels.append(ce_mid)
            elif ch_name == 'PE_MID':
                tensor_channels.append(pe_mid)
            else:
                # Direct channel lookup
                try:
                    idx = ch_names.index(ch_name)
                    tensor_channels.append(ext[:, idx, :])
                except ValueError:
                    # Unknown channel - fill with NaN
                    tensor_channels.append(np.full((ext.shape[0], ext.shape[2]), np.nan))
        
        tensor = np.stack(tensor_channels, axis=1)  # (time, channels, strikes)

        if normalize:
            # Normalization by channel type
            spot = ext[:, ch_names.index('UNDERLYING_LTP'), 0].reshape(-1, 1, 1)
            spot = np.where(np.isfinite(spot) & (spot > 0), spot, 1.0)

            for j, name in enumerate(sel):
                ch = tensor[:, j, :]
                
                if name in {'CE_MID', 'PE_MID'}:
                    # Normalize mid prices by spot
                    tensor[:, j, :] = ch / spot.squeeze(-1)
                elif name in {'CE_OI', 'PE_OI', 'CE_OICH', 'PE_OICH'}:
                    # Normalize OI by max across strikes
                    denom = np.nanmax(np.abs(ch), axis=1, keepdims=True)
                    tensor[:, j, :] = ch / np.where(denom > 0, denom, 1.0)
                elif name in {'CE_IV', 'PE_IV'}:
                    # IV already in decimal (0-1), no normalization needed
                    tensor[:, j, :] = ch
                elif name in {'CE_DELTA', 'PE_DELTA'}:
                    # Delta already in -1..+1 range
                    tensor[:, j, :] = ch
                elif name == 'UNDERLYING_LTP':
                    # Already normalized (divided by itself = 1.0), but keep as-is for reference
                    tensor[:, j, :] = ch / spot.squeeze(-1)

        return ts, tensor

    def build_compact_features(self, underlying: str, window: int = 20) -> Optional[pd.DataFrame]:
        """
        Build a single-row DataFrame of aggregated features over the trailing window.
        Includes PCRs, OI/volume momentum, max-pain distance, and ATM Greeks.
        """
        ts, ext = self.get_extended_matrix(underlying, window)
        if ext is None:
            return None

        from greeks.matrix_greeks import CHANNEL_NAMES as N
        def ch(name):
            return ext[:, N.index(name), :]

        strikes = ch('STRIKE')
        spot = ch('UNDERLYING_LTP')[:, :, 0]
        ce_oi, pe_oi = ch('CE_OI'), ch('PE_OI')
        ce_vol, pe_vol = ch('CE_VOL'), ch('PE_VOL')
        ce_iv, pe_iv = ch('CE_IV'), ch('PE_IV')
        ce_delta, pe_delta = ch('CE_DELTA'), ch('PE_DELTA')

        # Latest snapshot
        s_last = spot[-1, 0]
        strikes_last = strikes[-1]
        atm_idx = int(np.nanargmin(np.abs(strikes_last - s_last)))

        # PCRs
        pcr_atm = pe_oi[-1, atm_idx] / (ce_oi[-1, atm_idx] + 1e-6)
        pcr_total = np.nansum(pe_oi[-1]) / (np.nansum(ce_oi[-1]) + 1e-6)

        # Momentum (10-step if available)
        k = min(10, ext.shape[0] - 1)
        ce_oi_mom = (np.nanmean(ce_oi[-1]) - np.nanmean(ce_oi[-1 - k])) / (np.nanmean(ce_oi[-1 - k]) + 1e-6)
        pe_oi_mom = (np.nanmean(pe_oi[-1]) - np.nanmean(pe_oi[-1 - k])) / (np.nanmean(pe_oi[-1 - k]) + 1e-6)
        ce_vol_mom = (np.nanmean(ce_vol[-1]) - np.nanmean(ce_vol[-1 - k])) / (np.nanmean(ce_vol[-1 - k]) + 1e-6)
        pe_vol_mom = (np.nanmean(pe_vol[-1]) - np.nanmean(pe_vol[-1 - k])) / (np.nanmean(pe_vol[-1 - k]) + 1e-6)

        # Max pain distance
        total_oi = ce_oi[-1] + pe_oi[-1]
        max_oi_idx = int(np.nanargmax(total_oi))
        max_pain_dist = (strikes_last[max_oi_idx] - s_last) / s_last

        # ATM Greeks/IV
        ce_iv_atm = ce_iv[-1, atm_idx]
        pe_iv_atm = pe_iv[-1, atm_idx]
        ce_delta_atm = ce_delta[-1, atm_idx]
        pe_delta_atm = pe_delta[-1, atm_idx]

        row = {
            'timestamp': pd.to_datetime(ts[-1], unit='s'),
            'spot': s_last,
            'pcr_atm': float(pcr_atm),
            'pcr_total': float(pcr_total),
            'ce_oi_momentum': float(ce_oi_mom),
            'pe_oi_momentum': float(pe_oi_mom),
            'ce_vol_momentum': float(ce_vol_mom),
            'pe_vol_momentum': float(pe_vol_mom),
            'max_pain_distance': float(max_pain_dist),
            'ce_iv_atm': float(ce_iv_atm),
            'pe_iv_atm': float(pe_iv_atm),
            'ce_delta_atm': float(ce_delta_atm),
            'pe_delta_atm': float(pe_delta_atm),
        }
        return pd.DataFrame([row]).set_index('timestamp')


