import numpy as np
from load_data import load_training_data


def validate_matrix(symbol: str, data_dir: str = None):
    """Validate that matrices make sense with increasing strikes."""
    data = load_training_data(symbol, data_dir)
    
    # Channel names
    feature_channels = [
        'CE_BID', 'CE_ASK', 'PE_BID', 'PE_ASK',
        'CE_VOL', 'PE_VOL', 'CE_OI', 'PE_OI',
        'CE_OICH', 'PE_OICH', 'STRIKE'
    ]
    
    greeks_channels = [
        'CE_IV', 'PE_IV', 'CE_DELTA', 'PE_DELTA',
        'CE_GAMMA', 'PE_GAMMA', 'CE_THETA', 'PE_THETA',
        'CE_VEGA', 'PE_VEGA'
    ]
    
    print(f"\n=== Matrix Validation for {symbol} ===\n")
    
    # Check first snapshot
    features = data['features'][0]  # (11, 7)
    greeks = data['greeks'][0]      # (10, 7)
    strikes = features[10, :]       # STRIKE channel
    
    print(f"Strikes: {strikes}")
    print(f"Strike order: {'✓ Increasing' if np.all(np.diff(strikes) > 0) else '✗ NOT increasing'}\n")
    
    # Validate Features
    print("=== Features Validation ===")
    # CE_BID, CE_ASK should decrease (OTM calls cheaper)
    ce_bid = features[0, :]
    ce_ask = features[1, :]
    print(f"CE_BID: {ce_bid}")
    print(f"  Trend: {'✓ Decreasing' if np.all(np.diff(ce_bid) < 0) else '✗ NOT decreasing'}")
    print(f"CE_ASK: {ce_ask}")
    print(f"  Trend: {'✓ Decreasing' if np.all(np.diff(ce_ask) < 0) else '✗ NOT decreasing'}")
    
    # PE_BID, PE_ASK should increase (OTM puts more expensive)
    pe_bid = features[2, :]
    pe_ask = features[3, :]
    print(f"\nPE_BID: {pe_bid}")
    print(f"  Trend: {'✓ Increasing' if np.all(np.diff(pe_bid) > 0) else '✗ NOT increasing'}")
    print(f"PE_ASK: {pe_ask}")
    print(f"  Trend: {'✓ Increasing' if np.all(np.diff(pe_ask) > 0) else '✗ NOT increasing'}")
    
    # Validate Greeks
    print("\n=== Greeks Validation ===")
    # CE_DELTA should decrease (OTM calls have lower delta)
    ce_delta = greeks[2, :]
    print(f"CE_DELTA: {ce_delta}")
    print(f"  Trend: {'✓ Decreasing' if np.all(np.diff(ce_delta) < 0) else '✗ NOT decreasing'}")
    print(f"  Range: [{ce_delta.min():.3f}, {ce_delta.max():.3f}] (should be 0-1)")
    
    # PE_DELTA should decrease (become more negative) as strikes increase
    # Lower strikes (OTM) → delta closer to 0, Higher strikes (ITM) → delta closer to -1
    pe_delta = greeks[3, :]
    print(f"\nPE_DELTA: {pe_delta}")
    print(f"  Trend: {'✓ Decreasing (more negative)' if np.all(np.diff(pe_delta) < 0) else '✗ NOT decreasing'}")
    print(f"  Range: [{pe_delta.min():.3f}, {pe_delta.max():.3f}] (should be -1 to 0)")
    
    # Gamma should peak near ATM
    ce_gamma = greeks[4, :]
    pe_gamma = greeks[5, :]
    print(f"\nCE_GAMMA: {ce_gamma}")
    print(f"  Max at strike {strikes[np.argmax(ce_gamma)]:.0f}")
    print(f"PE_GAMMA: {pe_gamma}")
    print(f"  Max at strike {strikes[np.argmax(pe_gamma)]:.0f}")
    
    # IV should be reasonable (typically 0.1-2.0 = 10%-200%)
    ce_iv = greeks[0, :]
    pe_iv = greeks[1, :]
    print(f"\nCE_IV: {ce_iv}")
    print(f"  Range: [{ce_iv.min():.2%}, {ce_iv.max():.2%}]")
    print(f"PE_IV: {pe_iv}")
    print(f"  Range: [{pe_iv.min():.2%}, {pe_iv.max():.2%}]")
    
    # Check for NaN values
    print("\n=== Data Quality ===")
    nan_features = np.isnan(features).sum()
    nan_greeks = np.isnan(greeks).sum()
    print(f"NaN in features: {nan_features}")
    print(f"NaN in greeks: {nan_greeks}")
    
    return {
        'strikes_valid': np.all(np.diff(strikes) > 0),
        'ce_prices_decreasing': np.all(np.diff(ce_bid) < 0) and np.all(np.diff(ce_ask) < 0),
        'pe_prices_increasing': np.all(np.diff(pe_bid) > 0) and np.all(np.diff(pe_ask) > 0),
        'ce_delta_valid': np.all(np.diff(ce_delta) < 0) and np.all((ce_delta >= 0) & (ce_delta <= 1)),
        'pe_delta_valid': np.all(np.diff(pe_delta) < 0) and np.all((pe_delta >= -1) & (pe_delta <= 0)),
        'iv_reasonable': np.all((ce_iv > 0) & (ce_iv < 2)) and np.all((pe_iv > 0) & (pe_iv < 2))
    }