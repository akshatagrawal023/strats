import asyncio

class ProductionHFTPipeline(HFTDataPipeline):
    """Production-ready enhancements."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add monitoring
        self.metrics = {
            'updates_received': 0,
            'updates_processed': 0,
            'errors': 0,
            'avg_latency_ms': 0
        }
        
        # Add health check
        self.health_check_interval = 60  # seconds
        
    async def _monitor_health(self):
        """Periodic health check."""
        while self.running:
            await asyncio.sleep(self.health_check_interval)
            
            for symbol, buffer in self.buffers.items():
                if buffer.is_filled:
                    latest = buffer.get_time_slice(lookback=1)
                    if latest is not None:
                        # Check for stale data
                        time_diff = time.time() - buffer.timestamps[-1]
                        if time_diff > 10:  # No update for 10 seconds
                            print(f"WARNING: {symbol} data stale ({time_diff:.1f}s old)")
    
    async def run(self):
        """Run with health monitoring."""
        asyncio.create_task(self._monitor_health())
        await super().run()