import numpy as np
from option_chain_buffer import OptionChainBuffer

def create_mock_chain(underlying_price, strikes):
    """Creates a mock raw_matrix (11, len(strikes)) where channel 10 is the strike."""
    matrix = np.zeros((11, len(strikes)))
    matrix[10, :] = strikes  # Channel 10 is STRIKE
    matrix[0, :] = np.random.rand(len(strikes)) * 10  # Mock CE_BID
    return matrix

def test_moneyness_buffer():
    print("Initializing OptionChainBuffer with Unadulterated Strikes Grid...")
    # Buffer expects raw strikes, padding to `max_strikes` size 
    buffer = OptionChainBuffer(buffer_seconds=30, interval_seconds=10, n_channels=11, max_strikes=11)
    
    # Time T1: Underlying is 22000
    # API returns strikes centered around 22000
    print("\n--- Time T1 ---")
    spot_T1 = 22000
    strikes_T1 = [21500, 21600, 21700, 21800, 21900, 22000, 22100, 22200, 22300, 22400, 22500]
    matrix_T1 = create_mock_chain(spot_T1, strikes_T1)
    buffer.add_matrix(matrix_T1, timestamp=100.0, underlying=spot_T1, expiry="2024-05-30")
    
    # Time T2: Underlying dropped to 21900
    # API shifted strikes to center around 21900 (dropped 22500, added 21400)
    print("--- Time T2 ---")
    spot_T2 = 21900
    strikes_T2 = [21400, 21500, 21600, 21700, 21800, 21900, 22000, 22100, 22200, 22300, 22400]
    matrix_T2 = create_mock_chain(spot_T2, strikes_T2)
    buffer.add_matrix(matrix_T2, timestamp=110.0, underlying=spot_T2, expiry="2024-05-30")
    
    # Time T3: Underlying dropped to 21800
    # API shifted strikes to center around 21800
    print("--- Time T3 ---")
    spot_T3 = 21800
    strikes_T3 = [21300, 21400, 21500, 21600, 21700, 21800, 21900, 22000, 22100, 22200, 22300]
    matrix_T3 = create_mock_chain(spot_T3, strikes_T3)
    buffer.add_matrix(matrix_T3, timestamp=120.0, underlying=spot_T3, expiry="2024-05-30")
    
    # Extract the slice exactly at Moneyness = 1.00 and 1.05
    # The new expected output is (time, moneyness_targets, channels)
    moneyness_slice = buffer.get_strike_slice(moneyness_targets=[1.00, 1.05], lookback=3)
    
    print("\n=== Validation Results ===")
    print(f"Shape of extracted moneyness_slice: {moneyness_slice.shape} -> (time, targets, channels)")
    
    # Check the strikes returned for the ATM target (index 0)
    atm_strikes_over_time = moneyness_slice[:, 0, 10]
    
    print("\nATM Moneyness Target (1.00) mapped to strikes over time:")
    print(f"T1 (Spot 22000): Mapped Strike = {atm_strikes_over_time[0]}")
    print(f"T2 (Spot 21900): Mapped Strike = {atm_strikes_over_time[1]}")
    print(f"T3 (Spot 21800): Mapped Strike = {atm_strikes_over_time[2]}")
    
    assert atm_strikes_over_time[0] == 22000, "T1 ATM mapping failed"
    assert atm_strikes_over_time[1] == 21900, "T2 ATM mapping failed"
    assert atm_strikes_over_time[2] == 21800, "T3 ATM mapping failed"
    
    print("\nSUCCESS! The raw strike buffer successfully interpolated exact moneyness points exactly when asked by get_strike_slice().")

if __name__ == "__main__":
    test_moneyness_buffer()
