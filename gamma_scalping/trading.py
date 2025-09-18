# main.py
import logging
import time
import threading
import signal
import sys, os
import json
from queue import Queue
from data_manager import DataManager
from position_monitor import PositionMonitor
from config_live import Config
from datetime import datetime
from collections import defaultdict

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_system.log', encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
# Additional named logger for orders
order_logger = logging.getLogger("order_logger")
order_logger.setLevel(logging.INFO)

if not order_logger.handlers:
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler("logs/order.log", encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    order_logger.addHandler(fh)
    order_logger.propagate = False

class TradingSystem:
    def __init__(self):
        self.config = Config()
        self.running = True
        
        # Initialize queues for inter-component communication
        self.price_queue = Queue(maxsize=1000)
        self.signal_queue = Queue(maxsize=20)
        
        logging.info(f"Config")
        # Initialize components
        self.trading_state = TradingState(self.config)
        
        # Initialize percentile levels and trading pairs
        self.trading_state.initialize_percentile_levels()
        
        try:
            self.trading_state.ensure_open_positions_table()
            self.trading_state.ensure_closed_positions_table()
            self.trading_state.rehydrate_from_db()
            open_positions = self.trading_state.open_positions 
            order_logger.info(f"Rehydrated {len(open_positions)} open positions from local database")
        except Exception as e:
            order_logger.error(f"Failed to rehydrate open positions from DB: {e}")

        print("Rehydrated open positions")

        # logging.info(f"Trading state")
        self.data_manager = DataManager(self.price_queue, self.config, self.trading_state)
        logging.info(f"Data_manager state")
        self.data_processor = DataProcessor(self.price_queue, self.signal_queue, self.config, self.trading_state)
        self.trade_manager = TradeManager(self.signal_queue, self.config, order_logger, self.trading_state)
        self.position_monitor = PositionMonitor(self.trading_state, self.config.POSITION_MONITOR_INTERVAL)
        print("Position monitor initialized")

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def is_running(self):
        return self.running

    def start(self):
        """Start all components"""
        logging.info("Starting Trading System...")
        
        try:
            # Start components in order
            self.data_manager.start()
            self.data_processor.start()
            self.trade_manager.start()
            self.position_monitor.start()
            
            logging.info("All components started successfully")
                
        except Exception as e:
            logging.error(f"Error in main system: {e}")
            self.shutdown()

        # Main monitoring loop
        while self.running:
            try:
                self.monitor_system()
                time.sleep(30)
            except Exception as e:
                logging.error(f"Error in system monitoring: {e}")

            try:   
                time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                self.shutdown()
                break
            
    def monitor_system(self):
        """Monitor system health"""
        # Check if components are alive
        try:
            components = {
                'DataManager': self.data_manager,
                'DataProcessor': self.data_processor, 
                'TradeManager': self.trade_manager,
                'PositionMonitor': self.position_monitor
            }

            for name, component in components.items():
                if not component.is_alive():
                    logging.error(f"{name} is not running!")
                else:
                    logging.debug(f"{name} is healthy")

                price_queue_size = self.price_queue.qsize()
                signal_queue_size = self.signal_queue.qsize()
                    
            # Monitor queue health
            if price_queue_size > 800:  # 80% of maxsize
                logging.warning("Price queue near capacity: {price_queue_size}/1000")
            if signal_queue_size > 80:
                logging.warning("Signal queue near capacity: {signal_queue_size}/100")

            # Log current spread trades count (using cached value)
            if self.trading_state.total_spread_trades > 0:
                logging.info(f"Active spread trades: {self.trading_state.total_spread_trades}")

            if hasattr(self, '_monitor_count'):
                self._monitor_count += 1
            else:
                self._monitor_count = 1
                
            # if self._monitor_count % 10 == 0:  # Every 10th check (5 minutes)
            #     self._log_system_summary()

        except Exception as e:
            logging.error(f"Error in system monitoring: {e}")

    def shutdown(self, signum=None, frame=None):

        if not self.running:
            return

        logging.info("Shutting down Trading System...")
        self.running = False
        # Stop components
        self.data_manager.stop()
        self.data_processor.stop()
        self.trade_manager.stop()
        self.position_monitor.stop()
        
        # Shorter timeout for faster shutdown
        for component in [self.data_manager, self.data_processor, self.trade_manager, self.position_monitor]:
            try:
                component.join(timeout=5) 
                if component.is_alive():
                    logging.warning(f"{component.__class__.__name__} did not stop gracefully")
            except Exception as e:
                logging.error(f"Error waiting for {component.__class__.__name__}: {e}")
    
        logging.info("Trading System shutdown complete")
        sys.exit(0)

if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        system = TradingSystem()
        system.start()

        logging.info("Testing system components...")
        logging.info(f"DataManager alive: {system.data_manager.is_alive()}")
        logging.info(f"DataProcessor alive: {system.data_processor.is_alive()}")
        logging.info(f"TradeManager alive: {system.trade_manager.is_alive()}")
        logging.info(f"PositionMonitor alive: {system.position_monitor.is_alive()}")

    except KeyboardInterrupt:
        system.shutdown()

    except Exception as e:
        logging.error(f"Failed to start trading system: {e}")
        sys.exit(1)