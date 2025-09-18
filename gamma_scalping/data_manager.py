import logging
import threading
import time
from datetime import datetime, timedelta
from queue import Queue, Empty, Full
from typing import Dict, Any, List, Tuple
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.websocket import WebSocketClient
from utils.api_utils import get_option_chain
from utils.fyers_api import get_access_token
from utils.config import client_id
from fyers_apiv3 import fyersModel

class DataManager(threading.Thread):
    def __init__(self, price_queue: Queue, config, trading_state):
        super().__init__(name="DataManager")
        self.trading_state = trading_state
        self.price_queue = price_queue
        self.config = config
        self.ws = None
        self._running_lock = threading.Lock()
        self.running = False
        self.last_heartbeat = datetime.now() 
        self.ws_thread = None 
        self.all_symbols = self.get_all_symbols()
        
        # Initialize Fyers API for option chain fetching
        self.fyers = None
        self.init_fyers_api()
        
        # Option chain configuration
        self.strike_step_map = {
            "NIFTY": 50,
            "BANKNIFTY": 100,
        }
        
        self.lot_size_map = {
            "NIFTY": 50,
            "BANKNIFTY": 25,
        }
        
        self.chain_symbol_map = {
            "NIFTY": "NSE:NIFTY50",
            "BANKNIFTY": "NSE:NIFTYBANK",
        }
    
    @property
    def running(self):
        with self._running_lock:
            return self._running

    @running.setter 
    def running(self, value):
        with self._running_lock:
            self._running = value

    def start(self):
        """Start the data manager"""
        logging.info("Starting DataManager...")
        self.running = True
        super().start()
        
    def run(self):
        """Main thread loop"""
        try:
            self.connect_websocket()
            # Monitor connection
            while self.running:
                self.check_connection_health()
                time.sleep(60)  # Check every minute
                
        except Exception as e:
            logging.error(f"DataManager error: {e}")
        finally:
            self.cleanup()
            
    def connect_websocket(self):
        """Initialize and connect WebSocket"""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                if self.ws and hasattr(self.ws, 'data_socket') and self.ws.data_socket.is_connected():
                    return
                self.ws = WebSocketClient(
                    on_message=self.on_message,
                    on_connect=self.on_connect,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                # Get the data socket first
                data_socket = self.ws.fyers_data_socket()
                # Start WebSocket in separate thread
                self.ws_thread = threading.Thread(
                    target=data_socket.connect,
                    name="WebSocket"
                )
                self.ws_thread.daemon = True
                self.ws_thread.start()
                
                logging.info("WebSocket connection initiated")
                return
            except Exception as e:
                logging.error(f"WebSocket connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
    
    def on_connect(self, client):
        """Handle WebSocket connection"""
        logging.info("WebSocket connected!")
        try:
            client.data_socket.subscribe(
                data_type="SymbolUpdate",
                symbols=self.all_symbols
            )
            logging.info(f"Subscribed to {len(self.all_symbols)} symbols")
        except Exception as e:
            logging.error(f"Subscription error: {e}")
            
    def on_message(self, message):
        """Handle incoming WebSocket messages"""
        try:
            # logging.info(f"Message: {message}")
            if message.get("type") != 'sf':
                return

            self.last_heartbeat = datetime.now()

            self.price_queue.put_nowait({
                'symbol': message["symbol"],
                'bid': message["bid_price"], 
                'ask': message["ask_price"],
                'ltp': message["ltp"],
                'timestamp': self.last_heartbeat  #pytz.timezone('Asia/Kolkata')
            })
            # Check queue capacity before putting
            if self.price_queue.qsize() >= 850:  # 90% capacity
                logging.warning(f"Price queue at capacity: {self.price_queue.qsize()}")
                try:
                    while True:
                        self.price_queue.get_nowait()
                except Empty:
                    pass
                return  # Drop message instead of blocking

        except Full:
            logging.warning("Price queue full - dropping message")

        except KeyboardInterrupt:
            logging.info("Received interrupt signal")
            raise KeyboardInterrupt

        except Exception as e:
            logging.error(f"Error getting messages from websocket: {e}", exc_info=True)
            
    def get_all_symbols(self):
        """Get all symbols to subscribe to"""
        failed_symbols = []
        all_symbols = set()
        for pair_info in self.trading_state.pairs.values():
            try:
                all_symbols.add(pair_info['near_symbol'])
                all_symbols.add(pair_info['far_symbol'])
            except Exception as e:
                logging.error(f"Error getting symbols for {pair_info}: {e}")
                failed_symbols.append(pair_info)
        logging.info(f"Fetching symbols for : {all_symbols}")
        return list(all_symbols)
            
    def check_connection_health(self):
        """Check WebSocket connection health"""
        now = datetime.now()
        time_since_heartbeat = (now - self.last_heartbeat).total_seconds()

        if time_since_heartbeat > 300:  # 5 minutes
            logging.warning("No data received for {time_since_heartbeat:.0f} seconds!")
            
        try:
            if (self.ws and hasattr(self.ws, 'data_socket') and hasattr(self.ws.data_socket, 'is_connected')):
                if not self.ws.data_socket.is_connected():
                    logging.error("WebSocket disconnected! Attempting reconnect...")
                    self.connect_websocket()
        except Exception as e:
            logging.error(f"Connection health check failed: {e}")

    def stop_sockets(self):
        """Stop WebSocket connections - integrates with utils websocket"""
        try:
            if self.ws and hasattr(self.ws, 'stop_sockets'):
                self.ws.stop_sockets()
                logging.info("WebSocket connections stopped via utils method")
        except Exception as e:
            logging.error(f"Error stopping WebSocket connections: {e}")
            
    def stop(self):
        """Stop the data manager"""
        logging.info("Stopping DataManager...")
        self.running = False
        time.sleep(1)

    def fetch_option_chain(self, base: str, expiry: str, strike_count: int = 3):
        """
        Fetch option chain using existing utility function.
        
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
        
        logging.info(f"Fetching option chain for {base} ({chain_symbol}) with {strike_count} strikes")
        
        try:
            response = get_option_chain(chain_symbol, strike_count)
            
            if response.get("s") != "ok":
                logging.error(f"Option chain API failed for {base}: {response}")
                raise RuntimeError(f"Option chain failed for {base}: {response}")
            
            logging.info(f"Successfully fetched option chain for {base}")
            return response["data"]
            
        except Exception as e:
            logging.error(f"Error fetching option chain for {base}: {e}")
            raise
    
    def get_atm_strike(self, chain_data, base: str) -> int:
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
        
        logging.info(f"Underlying price: {underlying_price}, ATM strike: {atm_strike}")
        return atm_strike
    
    def find_atm_options(self, chain_data, atm_strike: int):
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
        
        logging.info(f"Found ATM options - CE: {ce_symbol}, PE: {pe_symbol}")
        return ce_symbol, pe_symbol
    
    def generate_straddle_position(self, base: str, expiry: str):
        """
        Generate all symbols needed for a straddle position.
        
        Args:
            base: Underlying symbol (NIFTY, BANKNIFTY)
            expiry: Expiry string (e.g., '25APR')
            
        Returns:
            Dictionary with position details and symbols
        """
        logging.info(f"Generating straddle position for {base} {expiry}")
        
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
            "underlying_price": None,
            "chain_data": chain_data
        }
        
        # Extract underlying price
        for record in chain_data.get("optionsChain", []):
            if (record.get("option_type") == "" and 
                record.get("strike_price") == -1):
                position["underlying_price"] = record.get("fp") or record.get("ltp")
                break
        
        logging.info(f"Generated straddle position for {base}:")
        logging.info(f"  ATM Strike: {atm_strike}")
        logging.info(f"  CE Symbol: {ce_symbol}")
        logging.info(f"  PE Symbol: {pe_symbol}")
        logging.info(f"  FUT Symbol: {fut_symbol}")
        logging.info(f"  Lot Size: {position['lot_size']}")
        logging.info(f"  Underlying Price: {position['underlying_price']}")
        
        return position
    
    def get_morning_positions(self):
        """
        Get morning positions for NIFTY and BANKNIFTY.
        This is the primary function to determine entry strikes and expiry.
        
        Returns:
            List of position dictionaries for morning setup
        """
        positions = []
        
        # Define morning positions (you can make this configurable)
        morning_configs = [
            {"base": "NIFTY", "expiry": "25APR"},
            {"base": "BANKNIFTY", "expiry": "25APR"},
        ]
        
        for config in morning_configs:
            try:
                position = self.generate_straddle_position(
                    config["base"], 
                    config["expiry"]
                )
                positions.append(position)
                logging.info(f"Successfully generated position for {config['base']}")
                
            except Exception as e:
                logging.error(f"Failed to generate position for {config['base']}: {e}")
        
        if positions:
            logging.info(f"Generated {len(positions)} morning positions")
            for pos in positions:
                logging.info(f"  {pos['base']}: Strike {pos['atm_strike']}, "
                           f"CE: {pos['ce_symbol']}, PE: {pos['pe_symbol']}")
        else:
            logging.warning("No positions generated for morning setup")
        
        return positions
        
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_sockets()

            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5)
                
        except Exception as e:
            logging.error(f"Error stopping WebSocket: {e}")
        logging.info("DataManager cleanup complete")