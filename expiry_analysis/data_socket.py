import logging
import threading
import time
from datetime import datetime, timedelta
import pytz
from queue import Queue, Empty, Full
import sys, os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from utils.websocket import WebSocketClient

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
                    on_connect=self.on_connect
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
            
    def stop(self):
        """Stop the data manager"""
        logging.info("Stopping DataManager...")
        self.running = False
        time.sleep(1)
        
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.ws:
                if hasattr(self.ws, 'stop_sockets'):
                    self.ws.stop_sockets()
                if self.ws_thread and self.ws_thread.is_alive():
                    self.ws_thread.join(timeout=5)
        except Exception as e:
            logging.error(f"Error stopping WebSocket: {e}")
        logging.info("DataManager cleanup complete")