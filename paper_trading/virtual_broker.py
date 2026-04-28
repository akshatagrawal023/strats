import pandas as pd
import datetime
import os
import sys
import csv
import os
import sys

# Append the root directory to access utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.trade_costs import calculate_total_trade_cost

class VirtualBroker:
    def __init__(self, start_capital=1000000.0):
        self.capital = start_capital
        self.positions = {}
        self.trade_history = []

        # In-memory PnL buffers per strategy
        self._pnl_buffers = {}

    def _get_strat_key(self, sim_id):
        if "IB_W" in sim_id: return "iron_butterfly"
        elif "RATIO_SPREAD" in sim_id: return "ratio_spread"
        elif "BACKSPREAD" in sim_id: return "backspread"
        elif "BULL_SPREAD" in sim_id or "BEAR_SPREAD" in sim_id: return "directional_spread"
        elif "CALENDAR" in sim_id: return "calendar_spread"
        elif "VT_IC" in sim_id: return "VolTrade"
        elif "SS_" in sim_id: return "ShortStraddle"
        return "unknown"

    def _log_trade_event(self, sim_id, event_type, legs, cost, pnl, features=None, trigger=""):
        """Logs detailed trade execution data into strategy-specific folder."""
        strat_key = self._get_strat_key(sim_id)
        
        base_dir = os.path.dirname(__file__)
        strat_dir = os.path.join(base_dir, "strategies", strat_key, "logs")
        os.makedirs(strat_dir, exist_ok=True)
        log_file = os.path.join(strat_dir, "trades.csv")
        
        file_exists = os.path.exists(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "sim_id", "event_type", "trigger", "spot", "iv_z", "skew_z", "cost", "pnl", "legs"])
            
            ts = datetime.datetime.now().isoformat()
            spot = legs[0].get('spot', '') if legs else ''
            # we format legs list into a readable string
            legs_str = " | ".join([f"{l['symbol']} Q:{l['qty']} S:{l['side']} P:{l['price']}" for l in legs])
            
            iv_z = features.get('iv_z_score', '') if features else ''
            skew_z = features.get('skew_z', '') if features else ''
            
            writer.writerow([ts, sim_id, event_type, trigger, spot, iv_z, skew_z, f"{cost:.2f}", f"{pnl:.2f}", legs_str])

    def virtual_place_basket(self, sim_id, margin_locked, legs, features=None, trigger=""):
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
            'unrealized_pnl': 0.0,
        }

        print(f"[{sim_id}] MOCKED ENTRY -> Margin: {margin_locked:.2f} | "
              f"Gross Credit: {gross_credit:.2f} | Est Taxes: {exec_cost:.2f} | "
              f"Net Credit: {net_credit:.2f} | Exit Tax Est: {exit_tax_estimate:.2f}")
              
        self._log_trade_event(sim_id, "ENTRY", legs, exec_cost, 0.0, features, trigger)

    def update_pnl(self, sim_id, current_prices, features=None):
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
        pos['unrealized_pnl'] = unrealized_pnl

        # Buffer PnL row in memory — flushed by HDF5Archiver or shutdown handler
        ts = datetime.datetime.now().isoformat()
        
        iv_z = features.get('iv_z_score', '') if features else ''
        skew_z = features.get('skew_z', '') if features else ''
        
        strat_key = self._get_strat_key(sim_id)
        if strat_key not in self._pnl_buffers:
            self._pnl_buffers[strat_key] = []
            
        self._pnl_buffers[strat_key].append(
            f"{ts},{sim_id},{unrealized_pnl:.2f},{pos['margin_locked']:.2f},{pos['net_credit']:.2f},{iv_z},{skew_z}"
        )

        return unrealized_pnl

    def virtual_close_all(self, sim_id, current_prices, features=None, trigger=""):
        """
        Settles all legs of a position and removes it from active tracking.
        """
        pnl = self.update_pnl(sim_id, current_prices, features)
        if pnl is not None:
            ts = datetime.datetime.now().isoformat()
            self.trade_history.append({
                'sim_id': sim_id,
                'final_pnl': pnl,
                'exit_time': ts,
                'margin': self.positions[sim_id]['margin_locked']
            })
            
            # format pseudo-legs for logging the exit prices
            exit_legs = []
            for leg in self.positions[sim_id]['legs']:
                sym = leg['symbol']
                price = current_prices[sym]['bid'] if leg['side'] == 1 else current_prices[sym]['ask']
                exit_legs.append({'symbol': sym, 'qty': leg['qty'], 'side': -leg['side'], 'price': price, 'spot': current_prices.get('spot', '')})
                
            self._log_trade_event(sim_id, "EXIT", exit_legs, self.positions[sim_id]['exit_tax_estimate'], pnl, features, trigger)
            
            del self.positions[sim_id]
            print(f"[{sim_id}] MOCKED EXIT -> Final PnL: ₹{pnl:.2f} | Time: {ts}")
        return pnl

    def flush_pnl_csv(self):
        """
        Flush buffered PnL rows to strategy-specific CSVs.
        """
        for strat_key, buffer in self._pnl_buffers.items():
            if not buffer:
                continue
                
            base_dir = os.path.dirname(__file__)
            strat_dir = os.path.join(base_dir, "strategies", strat_key, "logs")
            os.makedirs(strat_dir, exist_ok=True)
            csv_path = os.path.join(strat_dir, "mtm.csv")
            
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a') as f:
                if not file_exists:
                    f.write("timestamp,sim_id,unrealized_pnl,margin_locked,net_credit,iv_z,skew_z\n")
                f.write('\n'.join(buffer) + '\n')
            buffer.clear()

    def get_latest_pnl_snapshot(self) -> dict:
        """
        Returns a dict of {sim_id: latest_pnl} for the archiver to record.
        Only returns sim IDs where PnL was recorded this tick.
        """
        snapshot = {}
        for strat_key, buffer in self._pnl_buffers.items():
            for line in reversed(buffer):
                parts = line.split(',')
                if len(parts) >= 3:
                    sim_id = parts[1]
                    if sim_id not in snapshot:
                        try:
                            snapshot[sim_id] = float(parts[2])
                        except ValueError:
                            pass
        return snapshot
