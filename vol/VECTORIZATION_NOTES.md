# Vectorization & Algorithmic Differentiation Notes

## Why IV Calculation Can't Be "Truly" Vectorized

**Newton-Raphson is inherently iterative** - each strike needs its own convergence loop. 

Current implementation:
- `calculate_iv_vectorized()` uses `prange()` which **parallelizes** across strikes (multiple CPU cores)
- Each strike still runs sequentially (iterations: guess → compute → converge)
- This is the **fastest possible** for Newton-Raphson

**True vectorization** would mean processing all strikes in a single SIMD operation, but:
- Each strike converges at different rates
- Some strikes might not converge at all (different max_iter)
- Newton-Raphson requires sequential iterations per strike

**Bottom line**: The `prange()` parallelization IS the optimization - we're using all CPU cores. "True vectorization" isn't possible for iterative algorithms.

## Greeks Calculation

Same story - Black-Scholes formulas are vectorized (all strikes computed in parallel), but:
- Each strike's formula is independent → can parallelize
- We're using `prange()` for multi-core speedup
- This IS vectorized (parallel execution), just not SIMD vectorized

## Algorithmic Differentiation (AD)

**Would it help?** **Probably not for vanilla options.**

### AD Benefits:
- Automatically computes gradients (useful for complex payoffs)
- Can handle exotic structures without manual formulas

### Why Not Needed Here:
1. **We have closed-form Greeks** - Delta, Gamma, etc. from Black-Scholes are faster
2. **Vanilla options** - No complex payoffs requiring AD
3. **Performance** - Closed formulas are O(1), AD adds overhead

### When AD WOULD Help:
- **Exotic payoffs** (path-dependent, barrier, etc.)
- **Complex models** (stochastic vol, jumps) where closed-form doesn't exist
- **Higher-order Greeks** (speed, vomma) if you want them automatically

**For now**: Stick with Black-Scholes closed formulas - they're faster and simpler.

## Performance Notes

Current setup:
- **IV calculation**: ~0.1ms per strike (with prange parallelization)
- **Greeks calculation**: ~0.01ms per strike
- **Total for 7 strikes**: ~0.7ms (very fast!)

Bottleneck is likely:
- **API latency** (network) - ~100-500ms
- **Not the computation** - computation is <1ms

## Summary

✅ **Current implementation is optimal** - parallelization via `prange()` is the best we can do  
✅ **No need for AD** - closed-form formulas are faster  
✅ **Focus optimization on**: API calls, network latency, data pipeline

