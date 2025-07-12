# Development Plan: Multi-Indicator Optimizer

This document tracks the implementation path for expanding `demo_hybrid_optimization_vector.py` into a full-featured, multi-indicator optimisation framework suitable for machine-learning pipelines.

---
## 1. Milestone Overview
| Milestone | Deliverables | Status |
|-----------|--------------|--------|
| 1. Project Scaffold | repo, docs, requirements, `indicators/` package | ✅ |
| 2. Vectorised Indicator Library | MACD, ADX, Bollinger, Chaikin Vol, RSI, Stochastic, Volume Oscillator, Chaikin MF | ⬜ |
| 3. Generic Optimisation Core | grid iterators, back-test adapter, multiprocessing | ⬜ |
| 4. Parquet Export + Summary | `results.parquet`, Rich summary table | ⬜ |
| 5. Benchmark & Tune | runtime ≤ 30 min on M-series Mac | ⬜ |

## 2. Indicator Parameter Grids
```
Trend
  MACD: fast 10-14-18, slow 20-26-32, signal 6-9-12
  ADX : period 10-14-18

Volatility
  Bollinger: window 10-14-18, std 1.5-2-2.5
  Chaikin Vol: period 10-14-18

Momentum
  RSI:      period 10-24 step 2, oversold 25-35 step 2, overbought 65-80 step 3
  Stochastic: k 10-14-18, d 3-5-7, oversold/overbought 20/80, 30/70

Volume
  Volume Oscillator: fast 10-14-18, slow 20-26-32
  Chaikin MF: period 10-14-18
```

## 3. Task Granularity & Parallelism
* One worker task = (pair, timeframe).
* Worker iterates over **all** indicators and grids; avoids re-pickling functions.
* `multiprocessing.Pool(imap_unordered)` keeps cores saturated.

## 4. Data Pipeline
1. Collect per-run metrics → list[dict] → `pandas.DataFrame`.
2. Persist to `results.parquet` (column-ar, compressed).
3. Optionally shard by base currency for streaming via `tf.data`.

## 5. Next Actions
- [ ] Implement `indicators/` package with vectorised helpers.
- [ ] Refactor optimisation loop to call indicator registry.
- [ ] Add Parquet export + CLI flag.
- [ ] Unit-test each indicator calculation.
