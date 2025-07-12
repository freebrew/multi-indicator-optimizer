"""Demo script: optimise multiple technical indicators for a single pair
across predefined time-frames.

This is a *minimal* proof-of-concept implementing Milestone 2 of
`DEVELOPMENT_PLAN.md`.  It loads OHLCV data for one pair, calculates a grid
of parameters for several indicators using **pandas-ta**, back-tests simple
entry/exit rules, and prints a quick Rich summary.

Run:
    python demo_single_pair_multi_indicator.py --pair ETHUSD --timeframes 60 240

Dependencies: see requirements.txt (numpy, pandas, pandas_ta, rich).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
from rich.console import Console
from rich.table import Table

DATA_DIR = Path("Kraken_OHLCVT")
DEFAULT_TIMEFRAMES = [60]

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pair_entries_exits(entry_idx: np.ndarray, exit_idx: np.ndarray):
    if entry_idx.size == 0 or exit_idx.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pos = np.searchsorted(exit_idx, entry_idx, side="left")
    valid = pos < exit_idx.size
    entry_idx = entry_idx[valid]
    exit_idx = exit_idx[pos[valid]]
    valid = exit_idx > entry_idx
    return entry_idx[valid], exit_idx[valid]


def backtest_signals(prices: np.ndarray, entries: np.ndarray, exits: np.ndarray):
    entry_idx = np.where(entries)[0]
    exit_idx = np.where(exits)[0]
    entry_idx, exit_idx = _pair_entries_exits(entry_idx, exit_idx)
    if entry_idx.size == 0:
        return {"trades": 0, "win_rate": 0.0, "sharpe": 0.0, "total_return": 0.0}
    r = (prices[exit_idx] - prices[entry_idx]) / prices[entry_idx]
    win_rate = (r > 0).mean()
    sharpe = 0.0
    if r.size > 1 and (std := r.std(ddof=1)) > 1e-8:
        sharpe = (r.mean() / std) * np.sqrt(252)
    return {
        "trades": int(r.size),
        "win_rate": float(win_rate),
        "sharpe": float(sharpe),
        "total_return": float(r.sum()),
    }

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

def macd_grid():
    for fast in (10, 14, 18):
        for slow in (20, 26, 32):
            if slow <= fast:
                continue
            for sig in (6, 9, 12):
                yield {"fast": fast, "slow": slow, "signal": sig}

def adx_grid():
    for p in (10, 14, 18):
        yield {"length": p}

def bb_grid():
    for length in (10, 14, 18):
        for std in (1.5, 2.0, 2.5):
            yield {"length": length, "std": std}

def cv_grid():
    for p in (10, 14, 18):
        yield {"length": p}

def rsi_grid():
    for period in range(10, 26, 2):
        for os in range(25, 36, 2):
            for ob in range(65, 81, 3):
                if ob <= os + 10:
                    continue
                yield {"length": period, "oversold": os, "overbought": ob}

def stoch_grid():
    for k in (10, 14, 18):
        for d in (3, 5, 7):
            for os, ob in ((20, 80), (30, 70)):
                yield {"k": k, "d": d, "oversold": os, "overbought": ob}

def vo_grid():
    for fast in (10, 14, 18):
        for slow in (20, 26, 32):
            if slow <= fast:
                continue
            yield {"fast": fast, "slow": slow}

def cmf_grid():
    for p in (10, 14, 18):
        yield {"length": p}

INDICATORS = {
    "MACD": macd_grid,
    "ADX": adx_grid,
    "Bollinger": bb_grid,
    "ChaikinVol": cv_grid,
    "RSI": rsi_grid,
    "Stochastic": stoch_grid,
    "VolumeOsc": vo_grid,
    "ChaikinMF": cmf_grid,
}

# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame, name: str, p: dict):
    if name == "MACD":
        hist = ta.macd(df.close, fast=p["fast"], slow=p["slow"], signal=p["signal"]).iloc[:, 2].to_numpy()
        return (hist[:-1] <= 0) & (hist[1:] > 0), (hist[:-1] >= 0) & (hist[1:] < 0)
    if name == "ADX":
        plus = ta.adx(df.high, df.low, df.close, length=p["length"]).iloc[:, 1].to_numpy()
        minus = ta.adx(df.high, df.low, df.close, length=p["length"]).iloc[:, 2].to_numpy()
        return (plus[:-1] > minus[:-1]) & (plus[1:] <= minus[1:]), (plus[:-1] < minus[:-1]) & (plus[1:] >= minus[1:])
    if name == "Bollinger":
        bb = ta.bbands(df.close, length=p["length"], std=p["std"])
        lower = bb.iloc[:, 2].to_numpy(); mid = bb.iloc[:, 1].to_numpy(); price = df.close.to_numpy()
        return price[:-1] < lower[:-1], price[1:] > mid[1:]
    if name == "ChaikinVol":
        cv = ta.cv(df.high, df.low, length=p["length"]).to_numpy()
        return (cv[:-1] < 0) & (cv[1:] > 0), (cv[:-1] > 0) & (cv[1:] < 0)
    if name == "RSI":
        rsi = ta.rsi(df.close, length=p["length"]).to_numpy()
        return rsi[:-1] <= p["oversold"], rsi[1:] >= p["overbought"]
    if name == "Stochastic":
        k_line = ta.stoch(df.high, df.low, df.close, k=p["k"], d=p["d"]).iloc[:, 0].to_numpy()
        return k_line[:-1] <= p["oversold"], k_line[1:] >= p["overbought"]
    if name == "VolumeOsc":
        vo = ta.vo(df.volume, fast=p["fast"], slow=p["slow"]).to_numpy()
        return (vo[:-1] <= 0) & (vo[1:] > 0), (vo[:-1] >= 0) & (vo[1:] < 0)
    if name == "ChaikinMF":
        cmf = ta.cmf(df.high, df.low, df.close, df.volume, length=p["length"]).to_numpy()
        return (cmf[:-1] <= 0) & (cmf[1:] > 0), (cmf[:-1] >= 0) & (cmf[1:] < 0)
    raise ValueError(name)

# ---------------------------------------------------------------------------
# Optimise
# ---------------------------------------------------------------------------

def optimise_pair(pair: str, tfs: list[int]):
    recs: list[dict] = []
    for tf in tfs:
        csv = DATA_DIR / f"{pair}_{tf}.csv"
        if not csv.exists():
            console.print(f"[red]Missing {csv}")
            continue
        df = pd.read_csv(csv, names=["ts", "open", "high", "low", "close", "volume", "count"], header=None).dropna()
        prices = df.close.to_numpy()
        for ind, grid in INDICATORS.items():
            for params in grid():
                try:
                    ent, ex = generate_signals(df, ind, params)
                except Exception:
                    continue
                m = backtest_signals(prices, ent, ex)
                recs.append({"indicator": ind, "timeframe": tf, "pair": pair, "params": json.dumps(params), **m})
    return pd.DataFrame(recs)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="ETHUSD")
    parser.add_argument("--timeframes", nargs="*", type=int, default=DEFAULT_TIMEFRAMES)
    a = parser.parse_args()
    df = optimise_pair(a.pair, a.timeframes)
    if df.empty:
        console.print("[red]No results")
        return
    best = df.sort_values("sharpe", ascending=False).groupby("indicator").head(1)
    tab = Table(title="Best Sharpe per Indicator")
    for c in ["indicator", "timeframe", "params", "trades", "win_rate", "sharpe", "total_return"]:
        tab.add_column(c)
    for _, r in best.iterrows():
        tab.add_row(*[str(r[c]) for c in tab.columns])
    console.print(tab)
    out = Path("results_single_pair.parquet")
    df.to_parquet(out, index=False)
    console.print(f"[green]Saved full results -> {out}")

if __name__ == "__main__":
    main()
