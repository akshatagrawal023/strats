import time
import logging
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fyers_instance import FyersInstance
import queue
import threading
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class RateLimiter:
    def __init__(self, calls_per_second=10, calls_per_minute=200):
        self.calls_per_second = calls_per_second
        self.calls_per_minute = calls_per_minute
        
        # Deques to track timestamps of API calls
        self.second_queue = deque(maxlen=calls_per_second)
        self.minute_queue = deque(maxlen=calls_per_minute)
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
    def _clean_queues(self):
        """Remove expired timestamps from the queues"""
        current_time = time.time()
        
        # Clear calls older than 1 second
        while self.second_queue and current_time - self.second_queue[0] > 1.0:
            self.second_queue.popleft()
            
        # Clear calls older than 1 minute
        while self.minute_queue and current_time - self.minute_queue[0] > 60.0:
            self.minute_queue.popleft()
    
    def wait_if_needed(self):
        """
        Check if we need to wait before making another API call.
        If needed, block until a call slot becomes available.
        """
        with self.lock:
            self._clean_queues()
            
            # If we have capacity, return immediately
            if (len(self.second_queue) < self.calls_per_second and 
                len(self.minute_queue) < self.calls_per_minute):
                return 0
                
            # Calculate wait times needed
            second_wait = 0
            minute_wait = 0
            
            if len(self.second_queue) >= self.calls_per_second:
                # Wait until oldest call is more than 1 second old
                second_wait = max(0, 1.0 - (time.time() - self.second_queue[0]))
                
            if len(self.minute_queue) >= self.calls_per_minute:
                # Wait until oldest call is more than 1 minute old
                minute_wait = max(0, 60.0 - (time.time() - self.minute_queue[0]))
                
            # Return the longer wait time
            return max(second_wait, minute_wait)
    
    def add_call(self):
        """Record that a call was made at the current time"""
        with self.lock:
            current_time = time.time()
            self.second_queue.append(current_time)
            self.minute_queue.append(current_time)
    
    def execute(self, func, *args, **kwargs):
        """
        Execute a function with rate limiting.
        Will block if necessary to maintain rate limits.
        """
        wait_time = self.wait_if_needed()
        
        if wait_time > 0:
            time.sleep(wait_time)
            
        # Record this call
        self.add_call()
        
        # Execute the function
        return func(*args, **kwargs)

# Create a global rate limiter instance
rate_limiter = RateLimiter(calls_per_second=10, calls_per_minute=200)

def fyers_rate_limited_api_call(api_func, *args, **kwargs):
    """
    Make a rate-limited API call to Fyers
    """
    return rate_limiter.execute(api_func, *args, **kwargs)

def get_quotes(symbols):
    """
    Get quotes for one or more symbols.
    
    Args:
        symbols: String or list of symbols (e.g., "NSE:SBIN-EQ" or ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ"])
        
    Returns:
        Dictionary with quote data
    """
    # Get Fyers instance
    fyers = FyersInstance.get_instance()
    
    # Convert single symbol to string
    if isinstance(symbols, list):
        symbols = ",".join(symbols)
    
    # Prepare data
    data = {"symbols": symbols}
    
    # Make rate-limited API call
    return fyers_rate_limited_api_call(fyers.quotes, data=data)

def get_market_status():
    """
    Get current market status
    
    Returns:
        dict: Market status information
    """
    fyers = FyersInstance.get_instance()
    return fyers_rate_limited_api_call(fyers.market_status)

def get_profile():
    """
    Get user profile information
    
    Returns:
        dict: User profile data
    """
    fyers = FyersInstance.get_instance()
    return fyers_rate_limited_api_call(fyers.get_profile)

def get_funds():
    """
    Get user funds information
    
    Returns:
        dict: User funds data
    """
    fyers = FyersInstance.get_instance()
    return fyers_rate_limited_api_call(fyers.funds)

def get_positions():
    """
    Get user positions
    
    Returns:
        dict: User positions
    """
    fyers = FyersInstance.get_instance()
    return fyers_rate_limited_api_call(fyers.positions)

def get_option_chain(symbol, strikecount=1):
    data = {
    "symbol":symbol,
    "strikecount":strikecount,
    "timestamp": ""
    }
    fyers = FyersInstance.get_instance()
    response = fyers_rate_limited_api_call(fyers.optionchain, data=data)
    return response

def get_orderbook(order_id=None):
    """
    Get orderbook data
    
    Args:
        order_id (str, optional): Specific order ID to fetch
        
    Returns:
        dict: Orderbook data
    """
    fyers = FyersInstance.get_instance()
    data = {"id": order_id} if order_id else None
    return fyers_rate_limited_api_call(fyers.orderbook, data=data)

def place_order(order_list):
    fyers = FyersInstance.get_instance()
    response = fyers.place_basket_orders(data=order_list)
    return response