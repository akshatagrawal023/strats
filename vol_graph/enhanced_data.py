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
from rich.panel import Panel
from rich.columns import Columns
from rich.layout import Layout
from rich.text import Text
from vol.front_end import run_web_dashboard, broadcast_greeks

console = Console()

def process_full_chain(data, strikecount=5):
    """Process full options chain with multiple strikes and maturities"""
    # Extract the actual data from the API response
    if 'data' in data:
        actual_data = data['data']
    else:
        actual_data = data
    
    # Get all expiries
    expiries = actual_data.get('expiryData', [])
    options_chain = actual_data.get('optionsChain', [])
    
    # Find underlying data (strike_price = -1)
    underlying = None
    for option in options_chain:
        if option.get('strike_price') == -1:
            underlying = option
            break
    
    if not underlying:
        underlying = {}
    
    # Filter out underlying data (strike_price = -1)
    options_only = [opt for opt in options_chain if opt.get('strike_price', 0) > 0]
    
    # Group by expiry
    expiry_groups = {}
    for option in options_only:
        # Extract expiry from symbol (e.g., "NSE:SBIN25SEP820CE" -> "25SEP")
        symbol = option.get('symbol', '')
        if '25SEP' in symbol:
            expiry = '25SEP'
        elif '25OCT' in symbol:
            expiry = '25OCT'
        elif '25NOV' in symbol:
            expiry = '25NOV'
        else:
            expiry = 'Unknown'
            
        if expiry not in expiry_groups:
            expiry_groups[expiry] = {'calls': [], 'puts': []}
        
        if option.get('option_type') == 'CE':
            expiry_groups[expiry]['calls'].append(option)
        elif option.get('option_type') == 'PE':
            expiry_groups[expiry]['puts'].append(option)
    
    # Calculate Greeks for each option
    all_greeks = []
    for expiry, options in expiry_groups.items():
        for option in options['calls'] + options['puts']:
            try:
                # Calculate Greeks using Black-Scholes
                greeks = calculate_option_greeks(
                    underlying.get('ltp', 0),
                    option.get('strike_price', 0),
                    option.get('ltp', 0),
                    time_to_expiry=30/365,  # Approximate
                    risk_free_rate=0.05,
                    option_type=option.get('option_type', 'CE')
                )
                
                all_greeks.append({
                    'symbol': option.get('symbol', ''),
                    'strike': option.get('strike_price', 0),
                    'price': option.get('ltp', 0),
                    'bid': option.get('bid', 0),
                    'ask': option.get('ask', 0),
                    'volume': option.get('volume', 0),
                    'oi': option.get('oi', 0),
                    'type': option.get('option_type', ''),
                    'expiry': expiry,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'vega': greeks['vega'],
                    'iv': greeks['iv']
                })
            except Exception as e:
                continue
    
    return {
        'underlying': underlying,
        'greeks': all_greeks,
        'expiry_groups': expiry_groups,
        'india_vix': actual_data.get('indiavixData', {})
    }

def calculate_option_greeks(S, K, price, time_to_expiry, risk_free_rate=0.05, volatility=0.2, option_type='CE'):
    """Calculate Black-Scholes Greeks"""
    try:
        if S <= 0 or K <= 0 or time_to_expiry <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'iv': 0}
            
        # Simple approximation - in production, use proper IV calculation
        d1 = (np.log(S/K) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        delta = norm_cdf(d1) if option_type == 'CE' else norm_cdf(d1) - 1
        gamma = norm_pdf(d1) / (S * volatility * np.sqrt(time_to_expiry))
        theta = -(S * norm_pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry)) - risk_free_rate * K * np.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2)
        vega = S * norm_pdf(d1) * np.sqrt(time_to_expiry)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'iv': volatility
        }
    except:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'iv': 0}

def norm_cdf(x):
    """Standard normal CDF"""
    return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))

def norm_pdf(x):
    """Standard normal PDF"""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def create_terminal_display(chain_data):
    """Create a proper terminal display"""
    underlying = chain_data['underlying']
    greeks = chain_data['greeks']
    india_vix = chain_data['india_vix']
    
    # Create summary text
    underlying_price = underlying.get('ltp', 0)
    underlying_change = underlying.get('ltpch', 0)
    vix_price = india_vix.get('ltp', 0)
    vix_change = india_vix.get('ltpch', 0)
    
    summary_text = f"[bold green]{underlying.get('description', 'SBIN')}[/bold green] " \
                  f"[bold white]₹{underlying_price:.2f}[/bold white] " \
                  f"([green]{underlying_change:+.2f}[/green]) | " \
                  f"[yellow]INDIA VIX: {vix_price:.2f}[/yellow] " \
                  f"([green]{vix_change:+.2f}[/green]) | " \
                  f"[cyan]Options: {len(greeks)}[/cyan]"
    
    # Create options table
    table = Table(title="Options Chain with Greeks", show_header=True, header_style="bold magenta")
    
    # Add columns
    table.add_column("Strike", style="cyan", width=6)
    table.add_column("Type", style="yellow", width=3)
    table.add_column("Bid", style="green", width=6)
    table.add_column("Ask", style="red", width=6)
    table.add_column("LTP", style="white", width=6)
    table.add_column("Vol", style="blue", width=8)
    table.add_column("OI", style="magenta", width=8)
    table.add_column("Delta", style="cyan", width=6)
    table.add_column("Gamma", style="yellow", width=6)
    table.add_column("Theta", style="red", width=6)
    table.add_column("Vega", style="green", width=6)
    
    # Sort by strike price and limit to first 12 options
    sorted_greeks = sorted(greeks, key=lambda x: x['strike'])[:12]
    
    # Add rows
    for option in sorted_greeks:
        table.add_row(
            f"{option['strike']}",
            option['type'],
            f"{option['bid']:.2f}",
            f"{option['ask']:.2f}",
            f"{option['price']:.2f}",
            f"{option['volume']:,}",
            f"{option['oi']:,}",
            f"{option['delta']:.3f}",
            f"{option['gamma']:.3f}",
            f"{option['theta']:.1f}",
            f"{option['vega']:.1f}"
        )
    
    return summary_text, table

if __name__ == "__main__":
    symbol = "NSE:SBIN-EQ"
    
    # Start web dashboard server in background
    run_web_dashboard()
    
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            try:
                # Get more strikes for comprehensive view
                data = get_option_chain(symbol, strikecount=5)
                chain_data = process_full_chain(data, strikecount=5)
                
                # Create terminal display
                summary_text, table = create_terminal_display(chain_data)
                
                # Clear screen and display
                console.clear()
                console.print(summary_text)
                console.print()
                console.print(table)
                
                # Prepare options data for web dashboard
                options_for_web = []
                for greek in chain_data['greeks']:
                    options_for_web.append({
                        'strike': greek['strike'],
                        'type': greek['type'],
                        'bid': f"{greek['bid']:.2f}",
                        'ask': f"{greek['ask']:.2f}",
                        'price': f"{greek['price']:.2f}",
                        'volume': f"{greek['volume']:,}",
                        'oi': f"{greek['oi']:,}",
                        'delta': f"{greek['delta']:.3f}",
                        'gamma': f"{greek['gamma']:.3f}",
                        'theta': f"{greek['theta']:.1f}",
                        'vega': f"{greek['vega']:.1f}"
                    })
                
                # Broadcast to web dashboard (enhanced payload with options data)
                enhanced_payload = {
                    'underlying': f"₹{chain_data['underlying'].get('ltp', 0):.2f}",
                    'underlying_change': f"{chain_data['underlying'].get('ltpch', 0):+.2f}",
                    'india_vix': f"{chain_data['india_vix'].get('ltp', 0):.2f}",
                    'vix_change': f"{chain_data['india_vix'].get('ltpch', 0):+.2f}",
                    'total_options': len(chain_data['greeks']),
                    'expiries': len(chain_data['expiry_groups']),
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'options_chain': options_for_web  # Add options data for the table
                }
                
                # Add aggregated Greeks if available
                if chain_data['greeks']:
                    total_delta = sum(g['delta'] for g in chain_data['greeks'])
                    total_gamma = sum(g['gamma'] for g in chain_data['greeks'])
                    total_theta = sum(g['theta'] for g in chain_data['greeks'])
                    total_vega = sum(g['vega'] for g in chain_data['greeks'])
                    
                    enhanced_payload.update({
                        'total_delta': f"{total_delta:.3f}",
                        'total_gamma': f"{total_gamma:.6f}",
                        'total_theta': f"{total_theta:.3f}",
                        'total_vega': f"{total_vega:.3f}"
                    })
                
                broadcast_greeks(enhanced_payload)
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            
            time.sleep(1)
