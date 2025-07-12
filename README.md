# Multi-Indicator Optimizer

Vectorised, multiprocessing crypto-pair optimisation engine that sweeps across multiple technical indicators (RSI, MACD, ADX, Bollinger Bands, Stochastic, Chaikin Money Flow, Volume Oscillator, etc.) for all Kraken pairs and time-frames, producing a machine-learning-ready dataset of performance metrics.

## Features
* Pure-NumPy vectorised calculations — **10×** speed-up vs naive loops.
* Automatic grid search over indicator parameters.
* Fully parallel — one task per (pair, timeframe) keeps every CPU core busy.
* Rich progress bars & system metrics.
* Exports results to **Parquet** for direct consumption by TensorFlow / PyTorch pipelines.

## Quick Start
```bash
python demo_multi_indicator_optimization.py            # run full sweep
```

## Project Status
Initial scaffold committed. See `DEVELOPMENT_PLAN.md` for roadmap.
