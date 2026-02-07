import asyncio
from typing import List, Dict, Tuple
from greeks import calculate_iv_vectorized
import numpy as np

class AsyncIVCalculator:
    def __init__(self, max_concurrent_symbols: int = 4):
        self.max_concurrent = max_concurrent_symbols
        self.semaphore = asyncio.Semaphore(max_concurrent_symbols)
    
    async def calculate_for_symbol(self, 
                                 symbol: str, 
                                 prices: np.ndarray,
                                 underlying_price: float,
                                 strikes: np.ndarray,
                                 days_to_expiry: int,
                                 risk_free_rate: float,
                                 is_call: np.ndarray) -> Tuple[str, np.ndarray]:
       async with self.semaphore:
            # Convert days to years
            T = days_to_expiry / 365.0
            
            # Create underlying array (same price for all strikes)
            S = np.full_like(strikes, underlying_price)
            
            # Run CPU-bound calculation in thread pool
            loop = asyncio.get_event_loop()
            ivs = await loop.run_in_executor(
                None,  # Default ThreadPoolExecutor
                calculate_iv_vectorized,
                prices, S, strikes, T, risk_free_rate, is_call
            )
            
            return symbol, ivs
    
    async def calculate_batch(self, symbol_data: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        # Create tasks for each symbol
        tasks = []
        for symbol, data in symbol_data.items():
            task = asyncio.create_task(
                self.calculate_for_symbol(
                    symbol=symbol,
                    prices=data["prices"],
                    underlying_price=data["underlying"],
                    strikes=data["strikes"],
                    days_to_expiry=data["days_to_expiry"],
                    risk_free_rate=data["risk_free_rate"],
                    is_call=data["is_call"]
                )
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        iv_results = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"Error: {result}")
            else:
                symbol, ivs = result
                iv_results[symbol] = ivs
        
        return iv_results