import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the local fyers_api module
from utils.fyers_api import get_access_token
from utils.config import client_id
from fyers_apiv3 import fyersModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class OptionChainFetcher:
    """
    Fetch option chains for NIFTY and BANKNIFTY to determine ATM strikes
    and generate symbols for straddle positions.
    """
    
    def __init__(self):
        # Get access token and create Fyers instance
        access_token = get_access_token()
        self.fyers = fyersModel.FyersModel(
            token=access_token,
            is_async=False,
            client_id=client_id
        )
        
        # Strike steps for different instruments
        self.strike_step_map = {
            "NIFTY": 50,
            "BANKNIFTY": 100,
        }
        
        # Lot sizes
        self.lot_size_map = {
            "NIFTY": 50,
            "BANKNIFTY": 25,
        }
        
        # Option chain symbols for indices
        self.chain_symbol_map = {
            "NIFTY": "NSE:NIFTY50",
            "BANKNIFTY": "NSE:NIFTYBANK",
        }

    def fetch_option_chain(self, base: str, expiry: str, strike_count: int = 3) -> Dict[str, Any]:
        """
        Fetch option chain using Fyers API.
        
        Args:
            base: Underlying symbol (NIFTY, BANKNIFTY)
            expiry: Expiry string (e.g., '25APR')
            strike_count: Number of strikes to fetch around ATM
            
        Returns:
            Option chain response data
        """
        chain_symbol = self.chain_symbol_map.get(base)
        if not chain_symbol:
            raise ValueError(f"No chain symbol configured for {base}")
        
        data = {
            "symbol": chain_symbol,
            "strikecount": strike_count,
            "timestamp": ""
        }
        
        logger.info(f"Fetching option chain for {base} ({chain_symbol}) with {strike_count} strikes")
        
        try:
            response = self.fyers.optionchain(data=data)
            
            if response.get("s") != "ok":
                logger.error(f"Option chain API failed for {base}: {response}")
                raise RuntimeError(f"Option chain failed for {base}: {response}")
            
            logger.info(f"Successfully fetched option chain for {base}")
            return response["data"]
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {base}: {e}")
            raise

    def get_atm_strike(self, chain_data: Dict[str, Any], base: str) -> int:
        """
        Extract ATM strike from option chain response.
        
        Args:
            chain_data: Option chain response data
            base: Underlying symbol for strike step lookup
            
        Returns:
            ATM strike price
        """
        # Find underlying record (option_type == '' and strike_price == -1)
        underlying = None
        for record in chain_data.get("optionsChain", []):
            if (record.get("option_type") == "" and 
                record.get("strike_price") == -1):
                underlying = record
                break
        
        if not underlying:
            raise RuntimeError("No underlying price found in option chain")
        
        # Use fp (futures price) or ltp for ATM calculation
        underlying_price = underlying.get("fp") or underlying.get("ltp")
        if not underlying_price:
            raise RuntimeError("No valid price for underlying")
        
        underlying_price = float(underlying_price)
        strike_step = self.strike_step_map.get(base, 50)
        
        # Round to nearest strike step
        atm_strike = int(round(underlying_price / strike_step) * strike_step)
        
        logger.info(f"Underlying price: {underlying_price}, ATM strike: {atm_strike}")
        return atm_strike

    def find_atm_options(self, chain_data: Dict[str, Any], atm_strike: int) -> Tuple[str, str]:
        """
        Find CE and PE symbols for the ATM strike.
        
        Args:
            chain_data: Option chain response data
            atm_strike: ATM strike price
            
        Returns:
            Tuple of (CE_symbol, PE_symbol)
        """
        ce_symbol = None
        pe_symbol = None
        
        for record in chain_data.get("optionsChain", []):
            if record.get("strike_price") == atm_strike:
                option_type = record.get("option_type")
                symbol = record.get("symbol")
                
                if option_type == "CE":
                    ce_symbol = symbol
                elif option_type == "PE":
                    pe_symbol = symbol
                
                if ce_symbol and pe_symbol:
                    break
        
        if not ce_symbol or not pe_symbol:
            raise RuntimeError(f"Could not find CE/PE options for strike {atm_strike}")
        
        logger.info(f"Found ATM options - CE: {ce_symbol}, PE: {pe_symbol}")
        return ce_symbol, pe_symbol

    def generate_straddle_symbols(self, base: str, expiry: str) -> Dict[str, Any]:
        """
        Generate all symbols needed for a straddle position.
        
        Args:
            base: Underlying symbol (NIFTY, BANKNIFTY)
            expiry: Expiry string (e.g., '25APR')
            
        Returns:
            Dictionary with position details and symbols
        """
        logger.info(f"Generating straddle symbols for {base} {expiry}")
        
        # 1. Fetch option chain
        chain_data = self.fetch_option_chain(base, expiry)
        
        # 2. Determine ATM strike
        atm_strike = self.get_atm_strike(chain_data, base)
        
        # 3. Find ATM CE and PE symbols
        ce_symbol, pe_symbol = self.find_atm_options(chain_data, atm_strike)
        
        # 4. Build futures symbol
        fut_symbol = f"NSE:{base}{expiry}FUT"
        
        # 5. Create position info
        position = {
            "base": base,
            "expiry": expiry,
            "atm_strike": atm_strike,
            "ce_symbol": ce_symbol,
            "pe_symbol": pe_symbol,
            "fut_symbol": fut_symbol,
            "lot_size": self.lot_size_map.get(base, 50),
            "strike_step": self.strike_step_map.get(base, 50),
            "underlying_price": None,  # Will be filled from chain data
            "chain_data": chain_data
        }
        
        # Extract underlying price
        for record in chain_data.get("optionsChain", []):
            if (record.get("option_type") == "" and 
                record.get("strike_price") == -1):
                position["underlying_price"] = record.get("fp") or record.get("ltp")
                break
        
        logger.info(f"Generated straddle position for {base}:")
        logger.info(f"  ATM Strike: {atm_strike}")
        logger.info(f"  CE Symbol: {ce_symbol}")
        logger.info(f"  PE Symbol: {pe_symbol}")
        logger.info(f"  FUT Symbol: {fut_symbol}")
        logger.info(f"  Lot Size: {position['lot_size']}")
        logger.info(f"  Underlying Price: {position['underlying_price']}")
        
        return position

    def get_all_symbols_for_subscription(self, positions: List[Dict[str, Any]]) -> List[str]:
        """
        Get all symbols that need to be subscribed for websocket data.
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            List of all symbols to subscribe
        """
        all_symbols = []
        
        for position in positions:
            all_symbols.extend([
                position["ce_symbol"],
                position["pe_symbol"],
                position["fut_symbol"]
            ])
        
        logger.info(f"Symbols for websocket subscription: {all_symbols}")
        return all_symbols


def main():
    """Test the option chain fetcher for NIFTY and BANKNIFTY."""
    fetcher = OptionChainFetcher()
    
    # Test symbols
    test_positions = [
        {"base": "NIFTY", "expiry": "25APR"},
        {"base": "BANKNIFTY", "expiry": "25APR"},
    ]
    
    positions = []
    
    for test_pos in test_positions:
        try:
            position = fetcher.generate_straddle_symbols(
                test_pos["base"], 
                test_pos["expiry"]
            )
            positions.append(position)
            
        except Exception as e:
            logger.error(f"Failed to generate symbols for {test_pos['base']}: {e}")
    
    if positions:
        # Get all symbols for websocket subscription
        all_symbols = fetcher.get_all_symbols_for_subscription(positions)
        
        logger.info("\n" + "="*50)
        logger.info("SUMMARY")
        logger.info("="*50)
        logger.info(f"Generated {len(positions)} straddle positions")
        logger.info(f"Total symbols to subscribe: {len(all_symbols)}")
        
        for position in positions:
            logger.info(f"\n{position['base']} Straddle:")
            logger.info(f"  Strike: {position['atm_strike']}")
            logger.info(f"  CE: {position['ce_symbol']}")
            logger.info(f"  PE: {position['pe_symbol']}")
            logger.info(f"  FUT: {position['fut_symbol']}")


if __name__ == "__main__":
    main()
