"""
Walk-forward backtesting engine for the Japanese equities factor model.

Performs rolling out-of-sample testing:
1. At each rebalance date, estimate the factor model on trailing data
2. Generate alpha signal from the fitted model
3. Construct portfolio using the optimizer
4. Measure out-of-sample performance until next rebalance

This prevents look-ahead bias — the model only sees data available
at the time of each decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .factor_model import (
    fama_macbeth_regression,
    generate_alpha_signal,
    prepare_factor_matrix,
    zscore,
    winsorize,
)
from .portfolio import (
    PortfolioConstraints,
    PortfolioResult,
    optimize_portfolio,
)

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Walk-forward backtest parameters."""
    estimation_window: int = 12      # months of trailing data for model fit
    rebalance_freq: str = "M"        # M=monthly, Q=quarterly
    min_estimation_periods: int = 6  # min periods to fit model
    transaction_cost_bps: int = 30   # one-way cost in bps
    slippage_bps: int = 10           # market impact estimate
    benchmark: str = "equal_weight"  # benchmark type
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)


@dataclass
class BacktestPeriod:
    """Single rebalance period result."""
    rebalance_date: str
    portfolio: PortfolioResult
    period_return: float
    benchmark_return: float
    excess_return: float
    turnover: float
    transaction_costs: float
    n_stocks_in_model: int
    r_squared: float            # cross-sectional R² from model fit
    top_factors: list           # highest-weight factors this period


@dataclass
class WalkForwardResult:
    """Full walk-forward backtest results."""
    periods: list[BacktestPeriod]
    cumulative_return: pd.Series      # portfolio cumulative return
    cumulative_benchmark: pd.Series   # benchmark cumulative return
    cumulative_excess: pd.Series      # excess return
    performance_stats: dict           # CAGR, Sharpe, max DD, etc.
    factor_stability: pd.DataFrame    # factor weight changes over time
    trade_log: list[dict]             # all trades executed


def walk_forward_backtest(
    fundamentals_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    config: WalkForwardConfig = None,
    sectors: pd.Series = None,
) -> WalkForwardResult:
    """
    Run walk-forward backtest.

    fundamentals_panel: DataFrame with [ticker, date, ...factor columns]
    returns_panel: DataFrame with columns=tickers, rows=dates (for covariance)
    forward_returns: DataFrame with [ticker, date, fwd_return]
    config: WalkForwardConfig
    sectors: Series ticker -> sector
    """
    if config is None:
        config = WalkForwardConfig()

    # Get rebalance dates
    all_dates = sorted(fundamentals_panel["date"].unique())
    if len(all_dates) < config.min_estimation_periods + 1:
        logger.warning(f"Insufficient data: {len(all_dates)} periods, need {config.min_estimation_periods + 1}")
        return _empty_wf_result()

    # Determine rebalance schedule
    rebalance_dates = _get_rebalance_dates(all_dates, config.rebalance_freq)

    periods = []
    trade_log = []
    prev_weights = None
    weight_history = []
    cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10000

    for i, reb_date in enumerate(rebalance_dates):
        # Estimation window: trailing dates up to (not including) rebalance
        est_dates = [d for d in all_dates if d < reb_date]
        est_dates = est_dates[-config.estimation_window:]

        if len(est_dates) < config.min_estimation_periods:
            continue

        # Get next period return date (out-of-sample)
        future_dates = [d for d in all_dates if d > reb_date]
        if not future_dates:
            continue
        oos_date = future_dates[0]

        # --- Pass 1: Fit factor model on estimation window ---
        est_fundamentals = fundamentals_panel[
            fundamentals_panel["date"].isin(est_dates)
        ]
        est_returns = forward_returns[
            forward_returns["date"].isin(est_dates)
        ]

        if est_fundamentals.empty or est_returns.empty:
            continue

        factor_cols = [c for c in est_fundamentals.columns
                       if c not in ("ticker", "date")]

        model_result = fama_macbeth_regression(
            est_returns, est_fundamentals, min_obs=10
        )

        if not model_result.optimal_weights:
            continue

        # --- Pass 2: Generate alpha for rebalance date ---
        reb_data = fundamentals_panel[
            fundamentals_panel["date"] == reb_date
        ]
        if reb_data.empty:
            # Use latest available
            reb_data = fundamentals_panel[
                fundamentals_panel["date"] == est_dates[-1]
            ]

        if reb_data.empty:
            continue

        current_factors = reb_data.set_index("ticker")[factor_cols]
        alpha = generate_alpha_signal(current_factors, model_result.optimal_weights)

        # --- Pass 3: Optimize portfolio ---
        portfolio = optimize_portfolio(
            alpha=alpha,
            returns_panel=returns_panel,
            sectors=sectors,
            prev_weights=prev_weights,
            constraints=config.constraints,
        )

        # Record weight history
        for t, w in portfolio.weights.items():
            weight_history.append({
                "date": reb_date,
                "ticker": t,
                "weight": w,
            })

        # --- Pass 4: Measure out-of-sample return ---
        oos_returns = forward_returns[forward_returns["date"] == oos_date]
        oos_ret_map = dict(zip(oos_returns["ticker"], oos_returns["fwd_return"]))

        port_return = sum(
            w * oos_ret_map.get(t, 0)
            for t, w in portfolio.weights.items()
        )

        # Benchmark: equal weight across same universe
        bench_tickers = list(oos_ret_map.keys())
        bench_return = (
            np.mean([oos_ret_map[t] for t in bench_tickers])
            if bench_tickers else 0
        )

        # Transaction costs
        turnover = portfolio.turnover
        tc = turnover * cost_rate

        # Log trades
        if prev_weights is not None:
            all_tickers = set(list(portfolio.weights.index) +
                              list(prev_weights.index))
            for t in all_tickers:
                new_w = portfolio.weights.get(t, 0)
                old_w = prev_weights.get(t, 0)
                if abs(new_w - old_w) > 0.001:
                    trade_log.append({
                        "date": reb_date,
                        "ticker": t,
                        "old_weight": round(old_w, 4),
                        "new_weight": round(new_w, 4),
                        "trade": round(new_w - old_w, 4),
                        "direction": "BUY" if new_w > old_w else "SELL",
                    })

        # R² from model
        avg_r2 = (
            model_result.r_squared.mean()
            if len(model_result.r_squared) > 0 else 0
        )

        # Top factors
        sorted_weights = sorted(
            model_result.optimal_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top_factors = [
            {"factor": f, "weight": round(w, 4)}
            for f, w in sorted_weights[:5]
        ]

        period = BacktestPeriod(
            rebalance_date=str(reb_date),
            portfolio=portfolio,
            period_return=round(port_return - tc, 6),
            benchmark_return=round(bench_return, 6),
            excess_return=round(port_return - tc - bench_return, 6),
            turnover=round(turnover, 4),
            transaction_costs=round(tc, 6),
            n_stocks_in_model=portfolio.n_holdings,
            r_squared=round(avg_r2, 4),
            top_factors=top_factors,
        )
        periods.append(period)
        prev_weights = portfolio.weights

    if not periods:
        return _empty_wf_result()

    # Build cumulative return series
    cum_port = _cumulative_returns([p.period_return for p in periods],
                                    [p.rebalance_date for p in periods])
    cum_bench = _cumulative_returns([p.benchmark_return for p in periods],
                                     [p.rebalance_date for p in periods])
    cum_excess = _cumulative_returns([p.excess_return for p in periods],
                                      [p.rebalance_date for p in periods])

    # Performance statistics
    perf = _compute_performance_stats(
        [p.period_return for p in periods],
        [p.benchmark_return for p in periods],
        config.rebalance_freq,
    )

    # Factor stability
    factor_stability = pd.DataFrame(weight_history) if weight_history else pd.DataFrame()

    return WalkForwardResult(
        periods=periods,
        cumulative_return=cum_port,
        cumulative_benchmark=cum_bench,
        cumulative_excess=cum_excess,
        performance_stats=perf,
        factor_stability=factor_stability,
        trade_log=trade_log,
    )


def _get_rebalance_dates(dates: list, freq: str) -> list:
    """Select rebalance dates from available dates."""
    if freq == "Q":
        step = 3
    elif freq == "M":
        step = 1
    elif freq == "W":
        step = 1  # every date if weekly
    else:
        step = 1

    # Skip first estimation_window dates
    if len(dates) <= step:
        return dates

    if freq == "Q":
        return dates[::3]
    return dates


def _cumulative_returns(returns: list, dates: list) -> pd.Series:
    """Compute cumulative return series from periodic returns."""
    cum = [1.0]
    for r in returns:
        cum.append(cum[-1] * (1 + r))
    # Index starts at 1 (skip the initial 1.0)
    return pd.Series(cum[1:], index=dates)


def _compute_performance_stats(
    port_returns: list,
    bench_returns: list,
    freq: str,
) -> dict:
    """Compute comprehensive performance statistics."""
    pr = np.array(port_returns)
    br = np.array(bench_returns)
    excess = pr - br

    n = len(pr)
    if n == 0:
        return {}

    # Annualization factor
    if freq == "M":
        ann = 12
    elif freq == "Q":
        ann = 4
    elif freq == "W":
        ann = 52
    else:
        ann = 12

    # Portfolio stats
    mean_ret = pr.mean()
    std_ret = pr.std() if n > 1 else 0
    cagr = (1 + mean_ret) ** ann - 1
    vol = std_ret * np.sqrt(ann)
    sharpe = (mean_ret * ann) / vol if vol > 0 else 0

    # Drawdown
    cum = np.cumprod(1 + pr)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Hit rate
    hit_rate = (pr > 0).mean()

    # Benchmark stats
    bench_cagr = (1 + br.mean()) ** ann - 1
    bench_vol = br.std() * np.sqrt(ann) if n > 1 else 0

    # Excess stats
    excess_mean = excess.mean()
    tracking_error = excess.std() * np.sqrt(ann) if n > 1 else 0
    info_ratio = (excess_mean * ann) / tracking_error if tracking_error > 0 else 0

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win/loss ratio
    wins = pr[pr > 0]
    losses = pr[pr < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    win_loss = avg_win / avg_loss if avg_loss > 0 else 0

    # Sortino ratio (downside deviation)
    downside = pr[pr < 0]
    downside_dev = downside.std() * np.sqrt(ann) if len(downside) > 1 else vol
    sortino = (mean_ret * ann) / downside_dev if downside_dev > 0 else 0

    return {
        "cagr": round(cagr * 100, 2),
        "annual_vol": round(vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "hit_rate": round(hit_rate * 100, 1),
        "win_loss_ratio": round(win_loss, 2),
        "n_periods": n,
        "benchmark_cagr": round(bench_cagr * 100, 2),
        "benchmark_vol": round(bench_vol * 100, 2),
        "excess_return": round(excess_mean * ann * 100, 2),
        "tracking_error": round(tracking_error * 100, 2),
        "information_ratio": round(info_ratio, 3),
        "avg_period_return": round(mean_ret * 100, 4),
        "best_period": round(pr.max() * 100, 2),
        "worst_period": round(pr.min() * 100, 2),
    }


def _empty_wf_result() -> WalkForwardResult:
    return WalkForwardResult(
        periods=[],
        cumulative_return=pd.Series(dtype=float),
        cumulative_benchmark=pd.Series(dtype=float),
        cumulative_excess=pd.Series(dtype=float),
        performance_stats={},
        factor_stability=pd.DataFrame(),
        trade_log=[],
    )
