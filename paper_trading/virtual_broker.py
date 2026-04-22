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

        # In-memory PnL buffer — flushed by HDF5Archiver, not per-tick file writes
        self._pnl_buffer = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Keep CSV as a lightweight fallback log (written only at flush intervals)
        self.csv_path = os.path.join(output_dir, "mtm_pnl_tracking.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write("timestamp,sim_id,unrealized_pnl,margin_locked,net_credit\n")

    def virtual_place_basket(self, sim_id, margin_locked, legs):
        """
        Execute a virtual basket order.
        legs format: [{'symbol': 'CE_24500', 'qty': 250, 'side': 1/-1, 'price': 150.50}]

        Pre-computes and caches exit_tax_estimate at entry time so MTM ticks
        can reuse it without re-invoking calculate_total_trade_cost each tick.
        """
        gross_credit = 0.0
        trades_for_costs = []

        for leg in legs:
            # If side=1 (Buy), we pay. If side=-1 (Sell), we receive.
            gross_credit -= (leg['qty'] * leg['side'] * leg['price'])

            trades_for_costs.append({
                'type': 'options',
                'price': leg['price'],
                'lot_size': leg['qty'],
                'is_buy': leg['side'] == 1,
                'is_sell': leg['side'] == -1
            })

        # Entry execution costs
        exec_cost = calculate_total_trade_cost(trades_for_costs)
        net_credit = gross_credit - exec_cost

        # --- Pre-compute exit tax estimate ---
        # Since the spread structure is symmetric and we will exit all legs,
        # exit taxes ≈ entry taxes. Cache once, reuse on every MTM tick.
        # This eliminates the per-tick calculate_total_trade_cost() call that was
        # causing the 8ms MTM spike.
        exit_trades_estimate = []
        for leg in legs:
            exit_trades_estimate.append({
                'type': 'options',
                'price': leg['price'],   # Use entry price as proxy for exit tax estimate
                'lot_size': leg['qty'],
                'is_buy': leg['side'] == -1,   # Flipped vs entry
                'is_sell': leg['side'] == 1
            })
        exit_tax_estimate = calculate_total_trade_cost(exit_trades_estimate)

        self.positions[sim_id] = {
            'legs': legs,
            'margin_locked': margin_locked,
            'net_credit': net_credit,
            'entry_cost': exec_cost,
            'exit_tax_estimate': exit_tax_estimate,  # Cached — no recompute per tick
        }

        print(f"[{sim_id}] MOCKED ENTRY -> Margin: {margin_locked:.2f} | "
              f"Gross Credit: {gross_credit:.2f} | Est Taxes: {exec_cost:.2f} | "
              f"Net Credit: {net_credit:.2f} | Exit Tax Est: {exit_tax_estimate:.2f}")

    def update_pnl(self, sim_id, current_prices):
        """
        Updates Mark-to-Market PnL — hot path, must be fast (<1ms target).

        Uses cached exit_tax_estimate to avoid per-tick cost recalculation.
        Appends PnL to in-memory buffer instead of opening file each call.

        current_prices: dict mapping symbol -> {'ask': price, 'bid': price}
        """
        if sim_id not in self.positions:
            return None

        pos = self.positions[sim_id]
        liquidation_cost = 0.0

        missing_data = False
        for leg in pos['legs']:
            sym = leg['symbol']
            if sym not in current_prices:
                missing_data = True
                continue

            # Liquidate at worst-case price: sold legs buy at Ask, bought legs sell at Bid
            liquidation_price = (
                current_prices[sym]['bid'] if leg['side'] == 1
                else current_prices[sym]['ask']
            )

            if liquidation_price is None or liquidation_price != liquidation_price:  # NaN check
                missing_data = True
                continue

            liquidation_cost -= (leg['qty'] * (-leg['side']) * liquidation_price)

        if missing_data:
            return None

        # Use cached exit tax — eliminates per-tick allocation and GC pressure
        unrealized_pnl = pos['net_credit'] + liquidation_cost - pos['exit_tax_estimate']

        # Buffer PnL row in memory — flushed by HDF5Archiver or shutdown handler
        ts = datetime.datetime.now().isoformat()
        self._pnl_buffer.append(
            f"{ts},{sim_id},{unrealized_pnl:.2f},{pos['margin_locked']:.2f},{pos['net_credit']:.2f}"
        )

        return unrealized_pnl

    def flush_pnl_csv(self):
        """
        Flush buffered PnL rows to CSV. Called by the archiver flush loop
        or on KeyboardInterrupt shutdown — never called per-tick.
        """
        if not self._pnl_buffer:
            return
        with open(self.csv_path, 'a') as f:
            f.write('\n'.join(self._pnl_buffer) + '\n')
        self._pnl_buffer.clear()

    def get_latest_pnl_snapshot(self) -> dict:
        """
        Returns a dict of {sim_id: latest_pnl} for the archiver to record.
        Only returns sim IDs where PnL was recorded this tick.
        """
        snapshot = {}
        for line in reversed(self._pnl_buffer):
            parts = line.split(',')
            if len(parts) >= 3:
                sim_id = parts[1]
                if sim_id not in snapshot:
                    try:
                        snapshot[sim_id] = float(parts[2])
                    except ValueError:
                        pass
        return snapshot
