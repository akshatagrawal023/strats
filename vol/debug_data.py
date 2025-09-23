import sys, os
from datetime import datetime, time as dtime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_utils import get_option_chain
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON

console = Console()

def debug_data_structure():
    """Debug the actual data structure we're receiving"""
    symbol = "NSE:SBIN-EQ"
    
    console.print("[bold blue]Fetching options data...[/bold blue]")
    
    try:
        data = get_option_chain(symbol, strikecount=5)
        
        console.print("[green]✓ Data fetched successfully[/green]")
        console.print(f"[blue]Data type: {type(data)}[/blue]")
        
        if isinstance(data, dict):
            console.print(f"[blue]Data keys: {list(data.keys())}[/blue]")
            
            # Show the full structure
            console.print("\n[bold yellow]Full Data Structure:[/bold yellow]")
            console.print(JSON.from_data(data, indent=2))
            
            # Check optionsChain specifically
            if 'optionsChain' in data:
                options_chain = data['optionsChain']
                console.print(f"\n[blue]Options chain length: {len(options_chain)}[/blue]")
                
                # Show first few options
                for i, option in enumerate(options_chain[:5]):
                    console.print(f"\n[cyan]Option {i+1}:[/cyan]")
                    console.print(f"  Symbol: {option.get('symbol', 'N/A')}")
                    console.print(f"  Strike: {option.get('strike_price', 'N/A')}")
                    console.print(f"  Type: {option.get('option_type', 'N/A')}")
                    console.print(f"  LTP: {option.get('ltp', 'N/A')}")
                    console.print(f"  Bid: {option.get('bid', 'N/A')}")
                    console.print(f"  Ask: {option.get('ask', 'N/A')}")
            
            # Check underlying data
            if 'optionsChain' in data and data['optionsChain']:
                underlying = data['optionsChain'][0]
                console.print(f"\n[green]Underlying data:[/green]")
                console.print(f"  Symbol: {underlying.get('symbol', 'N/A')}")
                console.print(f"  Description: {underlying.get('description', 'N/A')}")
                console.print(f"  LTP: {underlying.get('ltp', 'N/A')}")
                console.print(f"  Change: {underlying.get('ltpch', 'N/A')}")
            
            # Check INDIA VIX
            if 'indiavixData' in data:
                vix = data['indiavixData']
                console.print(f"\n[yellow]INDIA VIX data:[/yellow]")
                console.print(f"  Symbol: {vix.get('symbol', 'N/A')}")
                console.print(f"  LTP: {vix.get('ltp', 'N/A')}")
                console.print(f"  Change: {vix.get('ltpch', 'N/A')}")
        
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    debug_data_structure()
