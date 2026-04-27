import asyncio
import asyncio
import time
import logging
from typing import Dict, List
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_utils import get_option_chain
from orchestration.chain_processor import OptionDataProcessor
from orchestration.option_chain_buffer import OptionChainBuffer
from paper_trading.market_features import compute_atm_iv_and_greeks, compute_T_from_expiry, RISK_FREE_RATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ProductionHFTPipeline:
    """Production pipeline to fetch option chains and maintain rolling buffers."""
    
    def __init__(self, symbols: List[str], strike_count: int = 25, buffer_seconds: int = 300, interval_seconds: int = 3):
        self.symbols = symbols
        self.interval = interval_seconds
        
        # Instantiate parser
        self.processor = OptionDataProcessor(strike_count=strike_count)
        
        # Instantiate a buffer for each symbol
        # max_strikes = 2 * strike_count + 1
        num_strikes = 2 * strike_count + 1
        self.buffers: Dict[str, OptionChainBuffer] = {
            sym: OptionChainBuffer(
                buffer_seconds=buffer_seconds,
                interval_seconds=interval_seconds,
                n_channels=11,  # Based on OptionDataProcessor output
                max_strikes=num_strikes
            ) for sym in symbols
        }
        
        self.running = False
        self.available_expiries = {}
        
        self.metrics = {
            'updates_received': 0,
            'errors': 0,
        }
        
    async def discover_expiries(self, symbol: str):
        """Warmup call to fetch and cache all available expiry timestamps."""
        try:
            # Minimal strike count to fetch the meta payload quickly
            resp = get_option_chain(symbol, strikecount=1)
            if resp and resp.get('s') == 'ok':
                expiry_data = resp.get('data', {}).get('expiryData', [])
                expiries = {}
                for exp in expiry_data:
                    flag = exp.get('expiry_flag', 'U') # W=Weekly, M=Monthly
                    date_str = exp.get('date', 'Unknown')
                    ts = exp.get('expiry', '')
                    if ts:
                        key = f"{flag}_{date_str}"
                        expiries[key] = str(ts)
                
                self.available_expiries[symbol] = expiries
                logging.info(f"[{symbol}] Discovered {len(expiries)} expiries: {list(expiries.keys())[:3]}...")
        except Exception as e:
            logging.error(f"Failed to discover expiries for {symbol}: {e}")
        
    async def _fetch_and_buffer(self, symbol: str):
        """Fetch chain for a single symbol and add to its buffer."""
        try:
            t0 = time.perf_counter()
            
            # 1. Network Fetch Phase
            resp = get_option_chain(symbol, self.processor.strike_count)
            t1 = time.perf_counter()
            
            # 2. Parsing Phase
            mat, spot, fut, vix, pcr, parsed_expiry_ts = self.processor.create_matrix_from_response(resp)
            # print(mat)
            t2 = time.perf_counter()
            
            if mat is not None and not np.isnan(spot) and not np.isnan(parsed_expiry_ts):
                # We compute T here so it can be added to the buffer metadata
                T = compute_T_from_expiry(parsed_expiry_ts)

                # 3. Memory Buffering Phase
                success = self.buffers[symbol].add_matrix(
                    raw_matrix=mat,
                    timestamp=time.time(),
                    underlying=spot,
                    expiry=parsed_expiry_ts,
                    future_price=fut,
                    vix=vix,
                    pcr=pcr,
                    T_value=T
                )
                t3 = time.perf_counter()
                
                if success:
                    # 4. Greeks Computation Phase — full 11-channel matrix for ALL strikes
                    # Computed ONCE here, shared by all consumers (iron_butterfly_sim,
                    # live_iron_butterfly, vol_trade, etc.). No strategy recomputes.
                    # 4. Greeks Computation Phase
                    greeks_mat, timings = compute_atm_iv_and_greeks(
                        ce_bid=mat[0], ce_ask=mat[1],
                        pe_bid=mat[2], pe_ask=mat[3],
                        spot=spot, strikes=mat[10],
                        T=T, r=RISK_FREE_RATE
                    )

                    # 5. IV Velocity Computation Phase (Channels 11, 12)
                    prev_greeks = self.buffers[symbol].get_greeks_slice(lookback=1)
                    if prev_greeks is not None and len(prev_greeks) > 0:
                        ce_iv_vel = greeks_mat[6] - prev_greeks[0][6]
                        pe_iv_vel = greeks_mat[7] - prev_greeks[0][7]
                    else:
                        ce_iv_vel = np.zeros_like(greeks_mat[6])
                        pe_iv_vel = np.zeros_like(greeks_mat[7])

                    # Combine into final 13-channel matrix
                    greeks_mat_ext = np.vstack([greeks_mat, ce_iv_vel, pe_iv_vel])

                    t4 = time.perf_counter()
                    self.buffers[symbol].add_greeks_matrix(greeks_mat_ext)

                    self.metrics['updates_received'] += 1
                    total_greeks = (t4 - t3) * 1000
                    logging.info(
                        f"[{symbol}] Network: {(t1-t0)*1000:.2f}ms | Parse: {(t2-t1)*1000:.2f}ms | "
                        f"Buffer: {(t3-t2)*1000:.2f}ms | "
                        f"Greeks: {total_greeks:.2f}ms (IV:{timings.get('iv', 0):.2f}, "
                        f"Core:{timings.get('core', 0):.2f}, Feat:{timings.get('features', 0):.2f})"
                    )
                else:
                    # Matrix dimensions mismatched (e.g. illiquid strikes dropped by API)
                    self.metrics['errors'] += 1
            else:
                self.metrics['errors'] += 1
                
        except Exception as e:
            logging.error(f"Error fetching {symbol}: {e}")
            self.metrics['errors'] += 1

    async def run(self):
        """Main loop to continuously fetch data for all symbols."""
        self.running = True
        logging.info(f"Starting pipeline for {self.symbols}...")
        
        # 1. Warmup / Discovery Phase
        for sym in self.symbols:
            await self.discover_expiries(sym)
        
        
        while self.running:
            start_time = time.time()
            
            # Fetch all symbols concurrently
            tasks = [self._fetch_and_buffer(sym) for sym in self.symbols]
            await asyncio.gather(*tasks)
            
            # Sleep remainder of interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self):
        self.running = False

if __name__ == "__main__":
    async def test_run():
        print("Starting isolated test of prod_pipeline...")
        pipeline = ProductionHFTPipeline(symbols=["NSE:NIFTY50-INDEX"], strike_count=5, interval_seconds=3)
        loop_task = asyncio.create_task(pipeline.run())
        
        # Test for 15 seconds
        for i in range(5):
            await asyncio.sleep(3)
            for sym, buf in pipeline.buffers.items():
                slices = len(buf.timestamps)
                latest_spot = list(buf.underlying_prices)[-1] if slices > 0 else "None"
                print(f"[{i+1}/5] {sym} -> Buffered snapshots: {slices}, Latest Spot: {latest_spot}")
                
        pipeline.stop()
        await loop_task
        print("Isolated test completed.")

    asyncio.run(test_run())