import logging
from fyers_apiv3.FyersWebsocket import data_ws
from fyers_apiv3.FyersWebsocket import order_ws
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import client_id, secret_key, redirect_uri, state, grant_type, response_type
from utils.fyers_instance import FyersInstance

# Configure logging
logging.basicConfig(filename="logs/websocket.log",
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
)

class WebSocketClient:
    def __init__(self, on_connect, on_error=None, on_close=None, on_message=None, on_order=None):
        # Get the Fyers instance using singleton pattern
        fyers = FyersInstance.get_instance()
        self.access_token = fyers.token
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_error = on_error or self.default_on_error
        self.on_order = on_order or self.onOrder
        self.on_close = on_close or self.default_on_close
        self.data_socket = None  # Placeholder for the FyersDataSocket object
        self.order_socket = None  # Placeholder for the FyersOrderSocket object

    def fyers_data_socket(self):
        data_client = data_ws.FyersDataSocket(
            access_token=self.access_token,
            log_path="logs/",
            litemode=False,  #Only gives the last traded price and not -
            #vol traded today, bid size, ask_size, total_buy/sell_qty, low_price, high_price, prev_close
            write_to_file=False,
            reconnect=True,
            on_connect=lambda: self.on_connect(self),
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message,
            reconnect_retry=10,
        )
        self.data_socket = data_client
        return data_client

    def fyers_order_socket(self):
        order = order_ws.FyersDataSocket(
            access_token=self.access_token,
            log_path="logs/",
            write_to_file=False,
            on_connect=lambda: self.on_connect(self),
            on_close=self.on_close,
            on_error=self.on_error,
            on_orders=self.on_order,
            # on_general=onGeneral,  # Callback function to handle general events from the WebSocket.
            # on_positions = onPosition, #Callback function to handle position-related events from the WebSocket.
            # on_trades = onTrade  #Callback function to handle trade-related events from the WebSocket
        )
        self.order_socket = order
        return order

    def default_on_error(self, error):
        logging.error(f"WebSocket error: {error}")

    def onOrder(message):
        print("Order Response:", message)

    def default_on_close(self, message):
        logging.info("WebSocket connection closed. ", message)

    def stop_sockets(self):
        """Stop both data and order sockets if they are running."""
        if self.data_socket:
            self.data_socket.disconnect()
            logging.info("Data WebSocket disconnected.")
        if self.order_socket:
            self.order_socket.disconnect()
            logging.info("Order WebSocket disconnected.")