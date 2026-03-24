"""
Backtesting engine for the Japan value quant model.
Simulates quarterly rebalancing against historical data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2026-03-01"
    initial_capital: float = 10_000_000  # 10M JPY
    rebalance_frequency: str = "quarterly"  # quarterly | monthly
    top_n: int = 20
    equal_weight: bool = True
    transaction_cost_bps: int = 10  # 10 bps per trade
    benchmark: str = "^N225"


@dataclass
class BacktestResult:
    portfolio_value: pd.Series = None
    benchmark_value: pd.Series = None
    trades: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    holdings_history: list = field(default_factory=list)


def _get_rebalance_dates(start: str, end: str, freq: str) -> list[datetime]:
    """Generate rebalance dates."""
    if freq == "quarterly":
        dates = pd.date_range(start=start, end=end, freq="QS")
    elif freq == "monthly":
        dates = pd.date_range(start=start, end=end, freq="MS")
    else:
        dates = pd.date_range(start=start, end=end, freq="QS")
    return dates.tolist()


def _fetch_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch daily returns for a list of tickers."""
    logger.info(f"Fetching returns for {len(tickers)} tickers")
    prices = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if isinstance(prices.columns, pd.MultiIndex):
        close = prices["Close"]
    else:
        close = prices[["Close"]]
        close.columns = tickers[:1]

    returns = close.pct_change().fillna(0)
    return returns, close


def run_backtest(
    scored_snapshots: dict[str, list],
    config: BacktestConfig = None,
) -> BacktestResult:
    """
    Run a backtest given scored snapshots at each rebalance date.

    scored_snapshots: dict mapping date string -> list of (ticker, score) tuples,
                      sorted by score descending. Each entry represents the model's
                      ranking at that point in time.

    For a simple first pass, you can pass a single snapshot and the engine
    will hold those positions for the full period.
    """
    if config is None:
        config = BacktestConfig()

    result = BacktestResult()

    # Collect all tickers we'll ever hold
    all_tickers = set()
    for date_str, rankings in scored_snapshots.items():
        for ticker, score in rankings[:config.top_n]:
            all_tickers.add(ticker)

    all_tickers = list(all_tickers)
    if not all_tickers:
        logger.error("No tickers to backtest")
        return result

    # Fetch all returns upfront
    returns, prices = _fetch_returns(
        all_tickers + [config.benchmark],
        config.start_date,
        config.end_date,
    )

    # Initialize portfolio
    capital = config.initial_capital
    portfolio_values = []
    holdings = {}

    rebalance_dates = sorted(scored_snapshots.keys())
    rebalance_idx = 0

    for date in returns.index:
        date_str = date.strftime("%Y-%m-%d")

        # Check for rebalance
        if rebalance_idx < len(rebalance_dates) and date_str >= rebalance_dates[rebalance_idx]:
            rankings = scored_snapshots[rebalance_dates[rebalance_idx]]
            new_holdings = [t for t, s in rankings[:config.top_n] if t in returns.columns]

            # Transaction costs
            n_trades = len(set(new_holdings) ^ set(holdings.keys()))
            cost = capital * (config.transaction_cost_bps / 10000) * n_trades / max(len(new_holdings), 1)
            capital -= cost

            # Equal weight allocation
            if new_holdings:
                weight = 1.0 / len(new_holdings)
                holdings = {t: weight for t in new_holdings}

            result.trades.append({
                "date": date_str,
                "holdings": list(holdings.keys()),
                "n_trades": n_trades,
                "cost": cost,
            })
            result.holdings_history.append({
                "date": date_str,
                "tickers": list(holdings.keys()),
            })

            rebalance_idx += 1

        # Compute daily portfolio return
        if holdings:
            daily_return = sum(
                holdings[t] * returns.loc[date, t]
                for t in holdings
                if t in returns.columns
            )
            capital *= (1 + daily_return)

        portfolio_values.append({"date": date, "value": capital})

    # Build series
    pv = pd.DataFrame(portfolio_values).set_index("date")["value"]
    result.portfolio_value = pv

    # Benchmark
    if config.benchmark in prices.columns:
        bench = prices[config.benchmark].dropna()
        bench_normalized = bench / bench.iloc[0] * config.initial_capital
        result.benchmark_value = bench_normalized

    # Compute performance metrics
    result.metrics = _compute_metrics(pv, result.benchmark_value, config)

    return result


def _compute_metrics(
    portfolio: pd.Series,
    benchmark: pd.Series | None,
    config: BacktestConfig,
) -> dict:
    """Compute standard performance metrics."""
    returns = portfolio.pct_change().dropna()
    trading_days = 252

    total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
    n_years = len(returns) / trading_days
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    volatility = returns.std() * np.sqrt(trading_days)
    sharpe = (cagr - 0.005) / volatility if volatility > 0 else 0  # 0.5% risk-free (Japan)

    # Sortino (downside deviation only)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(trading_days) if len(downside) > 0 else 0.001
    sortino = (cagr - 0.005) / downside_std

    # Max drawdown
    cummax = portfolio.cummax()
    drawdown = (portfolio - cummax) / cummax
    max_drawdown = drawdown.min()

    metrics = {
        "total_return": round(total_return * 100, 2),
        "cagr": round(cagr * 100, 2),
        "volatility": round(volatility * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown": round(max_drawdown * 100, 2),
        "initial_capital": config.initial_capital,
        "final_value": round(portfolio.iloc[-1], 0),
        "n_trading_days": len(returns),
    }

    # Benchmark comparison
    if benchmark is not None and len(benchmark) > 1:
        bench_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
        metrics["benchmark_return"] = round(bench_return * 100, 2)
        metrics["alpha"] = round((total_return - bench_return) * 100, 2)

    return metrics
