import numpy as np

class OptionDataProcessor:
    """
    Stateless parser to convert raw option chain JSON from the API into a 2D numpy matrix.
    Memory and history are handled upstream by OptionChainBuffer.
    """
    def __init__(self, strike_count: int = 25):
        self.strike_count = strike_count
        self.num_strikes = 2 * strike_count + 1

    def create_matrix_from_response(self, resp):
        """
        Create a single matrix from option chain response.
        Returns: (matrix, underlying_ltp, future_price, vix, pcr, expiry_ts) or (None, None, None, None, None, None) if invalid.
        Matrix shape: (11, num_strikes)
        
        Channels mappings (Aligned with vol_trade.py):
        0: CE_BID, 1: CE_ASK, 2: PE_BID, 3: PE_ASK
        4: CE_VOL, 5: PE_VOL, 6: CE_OI, 7: PE_OI,
        8: CE_OICH, 9: PE_OICH, 10: STRIKE
        """
        if not resp or resp.get('s') != 'ok':
            return None, None, None, None, None, None
            
        data = resp.get('data', {})
        options = data.get('optionsChain', [])
        
        # Create matrix with NaNs
        mat = np.full((11, self.num_strikes), np.nan, dtype=float)
        
        # Extract macro data
        underlying_ltp = np.nan
        future_price = np.nan
        vix = data.get('indiavixData', {}).get('ltp', np.nan)
        
        call_oi = data.get('callOi', 0)
        put_oi = data.get('putOi', 0)
        pcr = (put_oi / call_oi) if call_oi > 0 else np.nan
        
        # Extract active expiry timestamp from the list (defaults to closest if not specifically queried differently by a Calendar spread logic)
        expiry_ts = np.nan
        expiry_data = data.get('expiryData', [])
        if len(expiry_data) > 0:
            try:
                # Assuming first item in expiryData list matches the current optionsChain we downloaded
                expiry_ts = float(expiry_data[0].get('expiry', np.nan))
            except (ValueError, TypeError):
                pass
        
        if len(options) > 0:
            first_row = options[0]
            underlying_ltp = first_row.get('ltp', np.nan)
            future_price = first_row.get('fp', np.nan)
        
        # Process options in pairs (CE, PE) - typical of Fyers/NSE responses
        si = 0
        for i in range(1, len(options), 2):
            if i + 1 >= len(options) or si >= self.num_strikes:
                break
            
            # The API usually provides rows in alternating Call/Put fashion for a given strike
            a = options[i]
            b = options[i + 1]
            at = a.get('option_type')
            bt = b.get('option_type')
            
            # Ensure we correctly assign the Call and Put side
            if at == 'CE' and bt == 'PE':
                ce_row, pe_row = a, b
            elif at == 'PE' and bt == 'CE':
                ce_row, pe_row = b, a
            else:
                continue
            
            strike = ce_row.get('strike_price')
            if strike is None:
                continue
            
            # Map the parsed data exactly how our buffer and downstream Greeks calculators expect it
            mat[0, si] = ce_row.get('bid', np.nan)
            mat[1, si] = ce_row.get('ask', np.nan)
            mat[2, si] = pe_row.get('bid', np.nan)
            mat[3, si] = pe_row.get('ask', np.nan)
            
            mat[4, si] = ce_row.get('volume', np.nan)
            mat[5, si] = pe_row.get('volume', np.nan)
            
            mat[6, si] = ce_row.get('oi', np.nan)
            mat[7, si] = pe_row.get('oi', np.nan)
            
            mat[8, si] = ce_row.get('oich', np.nan)
            mat[9, si] = pe_row.get('oich', np.nan)
            
            mat[10, si] = strike
            
            si += 1
            
        return mat, underlying_ltp, future_price, vix, pcr, expiry_ts
