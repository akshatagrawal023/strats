import sys, os
from datetime import datetime, time as dtime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from greeklib import VolatilityDashboard
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.historical_data import hist_data
from utils.symbol_utils import get_future_symbols
from utils.api_utils import get_option_chain

import time
from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()

def process_chain(data):
    dashboard = VolatilityDashboard()
    
    # Benchmark performance
    start_time = time.time()
    
    dashboard = VolatilityDashboard()
    dashboard.update_data(data)
    
    # Compute Greeks
    greeks = dashboard.compute_greeks()
    
    # Get aggregated exposure
    aggregated = dashboard.aggregate_greeks()
    
    print(f"Underlying: {dashboard.underlying_data['symbol']} @ {dashboard.underlying_data['price']}")
    print(f"Aggregated Greeks: Delta={aggregated['delta']:.3f}, Gamma={aggregated['gamma']:.6f}")
    print(f"Theta={aggregated['theta']:.3f}, Vega={aggregated['vega']:.3f}")
    
    return dashboard, greeks, aggregated

def display_greeks_table(aggregated, underlying_price):
    table = Table(title="Live Options Greeks Dashboard")
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Interpretation", style="green")
    
    table.add_row("Underlying", f"₹{underlying_price:.2f}", "SBIN Current Price")
    table.add_row("Delta", f"{aggregated['delta']:.3f}", "Directional exposure")
    table.add_row("Gamma", f"{aggregated['gamma']:.6f}", "Delta sensitivity")
    table.add_row("Theta", f"{aggregated['theta']:.3f}", "Daily time decay (₹)")
    table.add_row("Vega", f"{aggregated['vega']:.3f}", "Volatility exposure (per 1%)")
    table.add_row("Net Exposure", f"₹{aggregated['net_exposure']:.2f}", "Delta * Price")
    
    return table

if __name__ == "__main__":
    symbol = "NSE:SBIN-EQ"  
    with Live(console=console, refresh_per_second=4) as live:
        while True:
            data = get_option_chain(symbol, strikecount=1)
            dashboard, greeks, aggregated = process_chain(data)
            dashboard.update_data(data)
            aggregated = dashboard.aggregate_greeks()
            
            table = display_greeks_table(aggregated, dashboard.underlying_data['price'])
            live.update(table)
            
            time.sleep(0.25)





