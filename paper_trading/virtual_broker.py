import pandas as pd
import datetime
import os
import sys

# Append the root directory to access utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.trade_costs import calculate_total_trade_cost

class VirtualBroker:
    def __init__(self, start_capital=1000000.0, output_dir="paper_trading_logs"):
        self.capital = start_capital
        self.positions = {}
        self.trade_history = []
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.csv_path = os.path.join(output_dir, "mtm_pnl_tracking.csv")
        # Initialize CSV if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write("timestamp,sim_id,unrealized_pnl,margin_locked,net_credit\n")

    def virtual_place_basket(self, sim_id, margin_locked, legs):
        """
        Execute a virtual basket order.
        legs format: [{'symbol': 'NSE:NIFTY...', 'qty': 250, 'side': 1 (Buy)/-1 (Sell), 'price': 150.50}]
        """
        gross_credit = 0.0
        trades_for_costs = []
        
        for leg in legs:
            # If side=1 (Buy), we pay (negative credit). 
            # If side=-1 (Sell), we receive (positive credit).
            gross_credit -= (leg['qty'] * leg['side'] * leg['price'])
            
            # Format for trade_costs
            trades_for_costs.append({
                'type': 'options',
                'price': leg['price'],
                'lot_size': leg['qty'], # qty is total units (lot_size * lots)
                'is_buy': leg['side'] == 1,
                'is_sell': leg['side'] == -1
            })
            
        # Deduct entry execution costs (Brokerage, STT, NSE, etc.)
        exec_cost = calculate_total_trade_cost(trades_for_costs)
        net_credit = gross_credit - exec_cost
        
        self.positions[sim_id] = {
            'legs': legs,
            'margin_locked': margin_locked,
            'net_credit': net_credit,
            'entry_cost': exec_cost
        }
        
        # Log the entry event
        print(f"[{sim_id}] MOCKED ENTRY -> Margin: {margin_locked:.2f} | Gross Credit: {gross_credit:.2f} | Est Taxes: {exec_cost:.2f} | Net Credit: {net_credit:.2f}")

    def update_pnl(self, sim_id, current_prices):
        """
        Updates Mark-to-Market PnL by calculating the instant liquidation cost.
        current_prices: dict mapping symbol -> {'ask': price, 'bid': price}
        """
        if sim_id not in self.positions:
            return None
            
        pos = self.positions[sim_id]
        liquidation_cost = 0.0
        exit_trades_for_costs = []
        
        missing_data = False
        for leg in pos['legs']:
            sym = leg['symbol']
            if sym not in current_prices:
                missing_data = True
                continue
                
            # If we bought (side=1), to liquidate we MUST sell at Bid.
            # If we sold (side=-1), to liquidate we MUST buy at Ask.
            liquidation_price = current_prices[sym]['bid'] if leg['side'] == 1 else current_prices[sym]['ask']
            
            if liquidation_price is None or liquidation_price != liquidation_price:
                missing_data = True
                continue
                
            # liquidation cost: the credit derived from liquidating.
            # side is flipped: -leg['side']
            liquidation_cost -= (leg['qty'] * (-leg['side']) * liquidation_price)
            
            exit_trades_for_costs.append({
                'type': 'options',
                'price': liquidation_price,
                'lot_size': leg['qty'],
                'is_buy': leg['side'] == -1,
                'is_sell': leg['side'] == 1
            })
            
        if missing_data:
            return None
            
        # Calculate exactly what the taxes would be to liquidate right now
        exit_taxes = calculate_total_trade_cost(exit_trades_for_costs)
        
        # Total PnL = (Net Money received at start) + (Gross Money from liquidation) - (Taxes to liquidate)
        unrealized_pnl = pos['net_credit'] + liquidation_cost - exit_taxes
        
        # Log strictly to CSV
        ts = datetime.datetime.now().isoformat()
        with open(self.csv_path, 'a') as f:
            f.write(f"{ts},{sim_id},{unrealized_pnl:.2f},{pos['margin_locked']:.2f},{pos['net_credit']:.2f}\n")
            
        return unrealized_pnl
