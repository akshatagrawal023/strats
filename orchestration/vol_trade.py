import asyncio

from orchestration.prod_pipeline import ProductionHFTPipeline

async def main():
    # Initialize pipeline
    pipeline = ProductionHFTPipeline(
        symbols=["NIFTY", "BANKNIFTY"],
        buffer_seconds=300  # 5 minutes
    )
    
    # Start pipeline in background
    asyncio.create_task(pipeline.run())
    
    # Feature extraction
    extractor = FeatureExtractor(pipeline)
    
    # Wait for buffer to fill
    await asyncio.sleep(60)
    
    # Extract features for ML
    while True:
        for symbol in pipeline.symbols:
            # Get features
            iv_percentile = extractor.calculate_iv_percentile(symbol)
            skew = extractor.calculate_skew_evolution(symbol)
            momentum = extractor.calculate_momentum_features(symbol)
            
            # Feed to ML model
            if iv_percentile is not None and skew is not None:
                # Prepare feature vector
                features = np.array([
                    iv_percentile,
                    skew[-1, 1],  # Latest ATM skew
                    momentum.get('momentum_1min', 0),
                    # ... more features
                ])
                
                # Model prediction
                # signal = model.predict(features.reshape(1, -1))
        
        await asyncio.sleep(3)  # Check every 3 seconds

asyncio.run(main())