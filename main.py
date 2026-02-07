import asyncio
from core.greeks_engine import AsyncGreeksEngine, GreeksRequest
from orchestrator import MLTradingOrchestrator

async def main():
    # Initialize components
    greeks_engine = AsyncGreeksEngine(max_concurrent=4)
    orchestrator = MLTradingOrchestrator(greeks_engine)
    
    # Define strategy
    strategy_config = {
        'name': 'iv_mean_reversion',
        'symbols': ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS'],
        'min_confidence': 0.7,
        'stop_loss_multiplier': 1.5,
        'take_profit_multiplier': 2.0,
        'max_position_size': 0.1  # 10% of capital
    }
    
    # Generate signals
    signals = await orchestrator.generate_signals(
        strategy_config['symbols'],
        strategy_config
    )
    
    # Execute trades
    for signal in signals:
        print(f"Signal: {signal.symbol} - {signal.signal} "
              f"(Confidence: {signal.confidence:.2%})")
        print(f"Features: IV Percentile: {signal.features.get('iv_percentile', 0):.1f}%")
        print(f"Stop Loss: {signal.stop_loss:.2%}, Take Profit: {signal.take_profit:.2%}")
        print()

if __name__ == "__main__":
    asyncio.run(main())