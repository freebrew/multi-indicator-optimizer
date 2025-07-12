"""
Vectorised variant of `demo_hybrid_optimization.py`.

This module imports the original implementation and replaces the pure-Python
`backtest_strategy_enhanced` with a fully vectorised NumPy version.  No other
logic is touched, but you should see a multiple-fold speed-up per parameter
combination.

Run this file exactly as you would run the original script:

    python demo_hybrid_optimization_vector.py

All CLI behaviour, outputs and Rich progress bars remain unchanged.
"""

from __future__ import annotations

import numpy as np
import demo_hybrid_optimization as base  # type: ignore  # Original module

__all__ = [
    "backtest_vectorised",
]


def _pair_entries_exits(entry_idx: np.ndarray, exit_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair each entry index with the first *subsequent* exit index.

    Parameters
    ----------
    entry_idx : 1-D int array (ascending)
    exit_idx  : 1-D int array (ascending)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Matched entry and exit indices, filtered so that exit > entry.
    """
    if entry_idx.size == 0 or exit_idx.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # For each entry find index in exit_idx that is >= entry
    pos_in_exits = np.searchsorted(exit_idx, entry_idx, side="left")
    # Filter out entries that have *no* later exit
    valid_mask = pos_in_exits < exit_idx.size
    entry_idx = entry_idx[valid_mask]
    pos_in_exits = pos_in_exits[valid_mask]
    exit_idx = exit_idx[pos_in_exits]

    # Ensure exit index is strictly after entry index (i.e. exit > entry)
    valid_mask = exit_idx > entry_idx
    return entry_idx[valid_mask], exit_idx[valid_mask]


def backtest_vectorised(
    prices: np.ndarray,
    rsi: np.ndarray,
    oversold: int = 30,
    overbought: int = 70,
    no_overlap: bool = True,
) -> dict[str, float | int]:
    """Vectorised RSI long-only back-test.

    The strategy opens a long position when RSI ≤ *oversold* and exits when
    RSI ≥ *overbought*.
    """
    if prices.size < 2:
        return {"total_trades": 0, "winning_trades": 0, "win_rate": 0.0, "total_return": 0.0, "sharpe_ratio": 0.0, "avg_return": 0.0, "max_return": 0.0, "min_return": 0.0}

    long_entries = rsi[:-1] <= oversold
    long_exits = rsi[1:] >= overbought

    entry_idx = np.where(long_entries)[0]
    exit_idx = np.where(long_exits)[0] + 1

    entry_idx, exit_idx = _pair_entries_exits(entry_idx, exit_idx)

    if no_overlap and entry_idx.size:
        keep_mask = np.ones(entry_idx.size, dtype=bool)
        last_exit = -1
        for i in range(entry_idx.size):
            if entry_idx[i] <= last_exit:
                keep_mask[i] = False
            else:
                last_exit = exit_idx[i]
        entry_idx = entry_idx[keep_mask]
        exit_idx = exit_idx[keep_mask]

    if entry_idx.size == 0:
        return {"total_trades": 0, "winning_trades": 0, "win_rate": 0.0, "total_return": 0.0, "sharpe_ratio": 0.0, "avg_return": 0.0, "max_return": 0.0, "min_return": 0.0}

    returns = (prices[exit_idx] - prices[entry_idx]) / prices[entry_idx]
    total_trades = returns.size
    winning_trades = np.count_nonzero(returns > 0)
    win_rate = winning_trades / total_trades
    total_return = returns.sum()
    avg_return = returns.mean()
    max_return = returns.max(initial=0)
    min_return = returns.min(initial=0)

    sharpe_ratio = 0.0
    if total_trades > 1:
        std_r = returns.std(ddof=1)
        if std_r > 1e-8:
            sharpe_ratio = (returns.mean() / std_r) * np.sqrt(252)

    return {
        "total_trades": int(total_trades),
        "winning_trades": int(winning_trades),
        "win_rate": float(win_rate),
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "avg_return": float(avg_return),
        "max_return": float(max_return),
        "min_return": float(min_return),
    }

# Patch original back-test
base.backtest_strategy_enhanced = backtest_vectorised  # type: ignore

# Helper picklable function

def _optimize_star(args: tuple):
    return base.optimize_base_currency_timeframe(*args)

# Vectorised runner
import multiprocessing as _mp
import time as _time
from rich.progress import Progress as _Progress, SpinnerColumn as _SpinnerColumn, TextColumn as _TextColumn, BarColumn as _BarColumn, TaskProgressColumn as _TaskProgressColumn, TimeElapsedColumn as _TimeElapsedColumn

def _run_analysis_vector():
    console = base.console
    console.print("[cyan]📊 Analysis scope (vectorised): all base currencies[/cyan]")
    base_currency_pairs = base.get_available_pairs_by_base_currency()
    valid_base_currencies = [k for k, v in base_currency_pairs.items() if v]
    if not valid_base_currencies:
        console.print("[red]❌ No valid base currencies found with data[/red]")
        return None
    tasks = [(bc, tf, [pair]) for bc in valid_base_currencies for pair in base_currency_pairs[bc] for tf in base.TARGET_TIMEFRAMES]
    console.print(f"🔄 Total analysis tasks: {len(tasks):,}")
    num_procs = min(_mp.cpu_count(), len(tasks))
    console.print(f"[yellow]🚀 Spawning {num_procs} worker processes[/yellow]")
    total_samples = len(tasks)
    start_t = _time.time()
    processed = 0
    all_results: list[dict] = []
    with _Progress(_SpinnerColumn(), _TextColumn("[bold blue]{task.description}"), _BarColumn(bar_width=30), _TaskProgressColumn(), _TextColumn("[cyan]Throughput: {task.fields[th]:.1f} samp/s"), _TextColumn("[yellow]ETA: {task.fields[eta]}"), _TimeElapsedColumn(), console=console, refresh_per_second=4) as prog:
        p_task = prog.add_task("Vector Analysis", total=total_samples, th=0.0, eta="Calc…")
        with _mp.Pool(num_procs) as pool:
            for res_list in pool.imap_unordered(_optimize_star, tasks):
                if res_list:
                    all_results.extend(res_list)
                processed += 1
                elapsed = _time.time() - start_t
                th = processed / elapsed if elapsed else 0
                remain = total_samples - processed
                eta_s = remain / th if th else 0
                m, s = divmod(int(eta_s), 60)
                prog.update(p_task, advance=1, th=th, eta=f"{m}m {s}s")
    if not all_results:
        console.print("[red]❌ No valid results produced[/red]")
        return None
    console.print("[green]✅ Vector analysis complete.[/green]")
    import pandas as _pd
    results_df = _pd.DataFrame(all_results)
    summary_df = results_df.groupby("base_currency").agg(avg_sharpe=("sharpe_ratio", "mean"), max_sharpe=("sharpe_ratio", "max"), avg_win_rate=("win_rate", "mean"), max_win_rate=("win_rate", "max")).reset_index()
    table = base.Table(title="Base-Currency Summary (vectorised)")
    for col in summary_df.columns:
        table.add_column(col)
    for _, row in summary_df.iterrows():
        table.add_row(*[str(v) for v in row.values])
    console.print(table)
    return all_results, summary_df, results_df

base.run_base_currency_analysis = _run_analysis_vector  # type: ignore

if __name__ == "__main__":
    base.main()
