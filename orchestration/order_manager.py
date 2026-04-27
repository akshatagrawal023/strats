import asyncio
import logging
import threading
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.websocket import WebSocketClient

class OrderManager:
    """
    Listens to the Async Fyers Order Websocket in a background thread 
    and bridges state to our asyncio loop.
    This is a generalized class that can be securely imported into
    any live trading script (Iron Condor, Calendar, Straddle, etc).
    """
    def __init__(self):
        self.order_status = {}
        # Connect to order socket
        self.ws_client = WebSocketClient(
            on_connect=self._on_ws_connect,
            on_order=self._on_order_update
        )
        self.ws = self.ws_client.fyers_order_socket()
        
        # Start keep_running in a separate daemon thread so it doesn't block asyncio
        self.ws_thread = threading.Thread(target=self.ws.keep_running, daemon=True)
        self.ws_thread.start()
        
    def _on_ws_connect(self, ws):
        logging.info("[WS] Live Order WebSocket Authenticated and Connected.")
        
    def _on_order_update(self, message):
        """
        Receives instant JSON pushes from Fyers on any order status change.
        Status mapping: 1=Canceled, 2=Filled, 4=Transit, 5=Rejected, 6=Pending
        """
        if 'orders' in message:
            for ord in message['orders']:
                if 'id' in ord:
                    self.order_status[ord['id']] = ord['status']
                    logging.info(f"[WS] Leg Update -> OrdID: {ord['id']} | Status: {ord['status']}")
                    
    async def wait_for_fills(self, order_ids):
        """
        Yields control until the WebSocket explicitly confirms all leg IDs are Status 2 (Filled).
        Use this in combination with asyncio.wait_for(timeout=...) to build Chaser algorithms.
        """
        while True:
            all_filled = True
            for oid in order_ids:
                status = self.order_status.get(oid, 0)
                if status == 5: # Rejected
                    raise Exception(f"Order {oid} was rejected by Exchange/Broker!")
                if status != 2:
                    all_filled = False
                    
            if all_filled:
                return True
            await asyncio.sleep(0.5)
