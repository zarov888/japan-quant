"""
Cross-sectional factor model for Japanese equities.
Estimates factor returns via Fama-MacBeth regression, optimizes weights
from historical data, and generates alpha signals.

The model runs cross-sectional regressions at each period:
    R_i,t = alpha_t + sum(beta_k * F_i,k,t-1) + epsilon_i,t

Where R_i,t is stock i's forward return and F_i,k,t-1 is stock i's
lagged factor exposure k. Factor returns (beta_k) are estimated each
period, and the time-series of factor returns tells us which factors
actually predict returns in this market.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ── Factor definitions ────────────────────────────────────────

FACTOR_DEFS = {
    # Value
    "ev_to_ebitda_inv": {"raw": "ev_to_ebitda", "transform": "invert", "group": "value"},
    "ebitda_to_ev": {"raw": "ebitda_to_ev", "transform": "identity", "group": "value"},
    "pb_inv": {"raw": "pb_ratio", "transform": "invert", "group": "value"},
    "fcf_yield": {"raw": "fcf_yield", "transform": "identity", "group": "value"},
    "dividend_yield": {"raw": "dividend_yield", "transform": "identity", "group": "value"},
    # Leverage
    "lt_debt_to_ev": {"raw": "lt_debt_to_ev", "transform": "identity", "group": "leverage"},
    "debt_to_equity_inv": {"raw": "debt_to_equity", "transform": "invert", "group": "leverage"},
    # Quality
    "roe": {"raw": "roe", "transform": "identity", "group": "quality"},
    "operating_margin": {"raw": "operating_margin", "transform": "identity", "group": "quality"},
    "gross_profit_to_assets": {"raw": "gross_profit_to_assets", "transform": "identity", "group": "quality"},
    "asset_turnover": {"raw": "asset_turnover", "transform": "identity", "group": "quality"},
    # Momentum
    "52w_pos": {"raw": "52w_pos", "transform": "identity", "group": "momentum"},
    "sma_cross": {"raw": "sma_cross", "transform": "identity", "group": "momentum"},
    # Size (negative — small premium)
    "log_mcap_inv": {"raw": "market_cap", "transform": "log_invert", "group": "size"},
    # Governance
    "governance_composite": {"raw": "governance_composite", "transform": "identity", "group": "governance"},
}


@dataclass
class FactorModelResult:
    """Results from factor model estimation."""
    factor_returns: pd.DataFrame  # time-series of estimated factor returns
    factor_stats: pd.DataFrame    # t-stats, mean returns, IC, etc.
    optimal_weights: dict         # learned factor weights
    alpha_signal: pd.Series       # composite alpha for latest period
    ic_series: dict               # information coefficient time-series per factor
    r_squared: pd.Series          # cross-sectional R² per period


def winsorize(s: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    """Winsorize at given percentiles."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score normalization."""
    mu = s.mean()
    sigma = s.std()
    if sigma == 0 or pd.isna(sigma):
        return s * 0.0
    return (s - mu) / sigma


def _transform(series: pd.Series, method: str) -> pd.Series:
    """Apply factor transformation."""
    if method == "identity":
        return series
    elif method == "invert":
        # Higher raw = lower score (e.g., P/E → 1/P/E = earnings yield)
        return series.apply(lambda x: 1.0 / x if x and x != 0 else np.nan)
    elif method == "log_invert":
        # Log then negate (small = high score)
        return -series.apply(lambda x: np.log(x) if x and x > 0 else np.nan)
    return series


def prepare_factor_matrix(
    snapshots: list[dict],
    factor_defs: dict = None,
) -> pd.DataFrame:
    """
    Build a panel of z-scored, winsorized factor exposures from fundamental snapshots.

    snapshots: list of dicts, each containing {ticker, date, ...factor_values}
    Returns: DataFrame with MultiIndex (date, ticker) and factor columns.
    """
    if factor_defs is None:
        factor_defs = FACTOR_DEFS

    rows = []
    for snap in snapshots:
        ticker = snap.get("ticker")
        date = snap.get("date")
        row = {"ticker": ticker, "date": date}
        for fname, fdef in factor_defs.items():
            raw_val = snap.get(fdef["raw"])
            if raw_val is not None and not (isinstance(raw_val, float) and np.isnan(raw_val)):
                row[fname] = raw_val
            else:
                row[fname] = np.nan
        rows.append(row)

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel

    # Apply transforms
    for fname, fdef in factor_defs.items():
        if fname in panel.columns:
            panel[fname] = _transform(panel[fname], fdef["transform"])

    # Cross-sectional winsorize and z-score per date
    factor_cols = [f for f in factor_defs.keys() if f in panel.columns]
    dates = panel["date"].unique()

    normalized = []
    for dt in dates:
        mask = panel["date"] == dt
        sub = panel[mask].copy()
        for fc in factor_cols:
            sub[fc] = winsorize(sub[fc].astype(float))
            sub[fc] = zscore(sub[fc])
        normalized.append(sub)

    return pd.concat(normalized, ignore_index=True)


def fama_macbeth_regression(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    min_obs: int = 20,
) -> FactorModelResult:
    """
    Fama-MacBeth two-pass regression.

    Pass 1: For each period t, run cross-sectional regression of forward
            returns on lagged factor exposures.
    Pass 2: Average the time-series of factor return estimates.
            t-statistics test whether average factor return differs from zero.

    returns: DataFrame with columns [ticker, date, fwd_return]
    factors: DataFrame with columns [ticker, date, factor1, factor2, ...]
    """
    factor_cols = [c for c in factors.columns if c not in ("ticker", "date")]

    # Merge returns with lagged factors
    merged = returns.merge(factors, on=["ticker", "date"], how="inner")
    dates = sorted(merged["date"].unique())

    period_results = []
    ic_data = {fc: [] for fc in factor_cols}

    for dt in dates:
        cross = merged[merged["date"] == dt].dropna(subset=["fwd_return"] + factor_cols)

        if len(cross) < min_obs:
            continue

        y = cross["fwd_return"].values
        X = cross[factor_cols].values

        # Add intercept
        X_int = np.column_stack([np.ones(len(y)), X])

        try:
            # OLS: beta = (X'X)^-1 X'y
            betas = np.linalg.lstsq(X_int, y, rcond=None)[0]
            y_hat = X_int @ betas
            resid = y - y_hat
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            result = {"date": dt, "intercept": betas[0], "r_squared": r2, "n_obs": len(y)}
            for i, fc in enumerate(factor_cols):
                result[fc] = betas[i + 1]
                # Rank IC: Spearman correlation between factor and forward return
                ic, _ = stats.spearmanr(cross[fc].values, y)
                ic_data[fc].append({"date": dt, "ic": ic})

            period_results.append(result)

        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug(f"Regression failed for {dt}: {e}")
            continue

    if not period_results:
        logger.warning("No valid regression periods")
        return FactorModelResult(
            factor_returns=pd.DataFrame(),
            factor_stats=pd.DataFrame(),
            optimal_weights={},
            alpha_signal=pd.Series(dtype=float),
            ic_series={},
            r_squared=pd.Series(dtype=float),
        )

    # Factor returns time-series
    fr_df = pd.DataFrame(period_results).set_index("date")

    # Pass 2: factor statistics
    stats_rows = []
    for fc in factor_cols:
        if fc not in fr_df.columns:
            continue
        series = fr_df[fc].dropna()
        if len(series) < 3:
            continue
        mean_ret = series.mean()
        std_ret = series.std()
        t_stat = mean_ret / (std_ret / np.sqrt(len(series))) if std_ret > 0 else 0

        # IC statistics
        ic_vals = pd.DataFrame(ic_data[fc])
        mean_ic = ic_vals["ic"].mean() if len(ic_vals) > 0 else 0
        ic_ir = mean_ic / ic_vals["ic"].std() if len(ic_vals) > 0 and ic_vals["ic"].std() > 0 else 0

        stats_rows.append({
            "factor": fc,
            "group": FACTOR_DEFS.get(fc, {}).get("group", "other"),
            "mean_return": round(mean_ret * 100, 4),
            "std_return": round(std_ret * 100, 4),
            "t_stat": round(t_stat, 3),
            "significant": abs(t_stat) > 1.96,
            "mean_ic": round(mean_ic, 4),
            "ic_ir": round(ic_ir, 3),
            "n_periods": len(series),
            "pct_positive": round((series > 0).mean() * 100, 1),
        })

    factor_stats = pd.DataFrame(stats_rows)

    # Optimal weights: use IC-weighted or t-stat-weighted approach
    optimal_weights = _compute_optimal_weights(factor_stats, fr_df, factor_cols)

    # R² series
    r2_series = fr_df["r_squared"] if "r_squared" in fr_df.columns else pd.Series(dtype=float)

    # IC series
    ic_series = {fc: pd.DataFrame(ic_data[fc]) for fc in factor_cols if ic_data[fc]}

    return FactorModelResult(
        factor_returns=fr_df[factor_cols] if factor_cols else fr_df,
        factor_stats=factor_stats,
        optimal_weights=optimal_weights,
        alpha_signal=pd.Series(dtype=float),
        ic_series=ic_series,
        r_squared=r2_series,
    )


def _compute_optimal_weights(
    factor_stats: pd.DataFrame,
    factor_returns: pd.DataFrame,
    factor_cols: list[str],
) -> dict:
    """
    Compute optimal factor weights using IC-weighted approach.
    Factors with higher information coefficients and consistency get more weight.
    Negative IC factors get negative weight (short signal).
    """
    weights = {}
    total = 0

    for _, row in factor_stats.iterrows():
        fc = row["factor"]
        # Weight = |mean_IC| * IC_IR (penalize inconsistent factors)
        ic = row.get("mean_ic", 0)
        ir = row.get("ic_ir", 0)
        w = abs(ic) * max(abs(ir), 0.1)  # floor IR to avoid zeroing out

        # Sign: positive IC means factor predicts positive returns
        w = w * np.sign(ic) if ic != 0 else 0

        weights[fc] = w
        total += abs(w)

    # Normalize to sum of absolute weights = 1
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


def generate_alpha_signal(
    current_factors: pd.DataFrame,
    optimal_weights: dict,
) -> pd.Series:
    """
    Generate composite alpha signal for current period using learned weights.

    current_factors: DataFrame with ticker as index, factor columns
    optimal_weights: dict of factor -> weight from the model
    """
    alpha = pd.Series(0.0, index=current_factors.index)

    for factor, weight in optimal_weights.items():
        if factor in current_factors.columns:
            # Z-score the current cross-section
            z = zscore(winsorize(current_factors[factor].astype(float)))
            alpha += weight * z.fillna(0)

    return alpha


def build_factor_snapshots(
    fundamentals_by_date: dict[str, list[dict]],
    returns_by_date: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function: build factor matrix and return matrix from
    historical fundamental snapshots and price returns.

    fundamentals_by_date: {date_str: [list of fundamental dicts]}
    returns_by_date: {date_str: DataFrame with ticker, fwd_return}
    """
    all_snaps = []
    for date_str, funds in fundamentals_by_date.items():
        for f in funds:
            snap = dict(f)
            snap["date"] = date_str
            all_snaps.append(snap)

    factors = prepare_factor_matrix(all_snaps)

    all_returns = []
    for date_str, ret_df in returns_by_date.items():
        ret_df = ret_df.copy()
        ret_df["date"] = date_str
        all_returns.append(ret_df)

    returns = pd.concat(all_returns, ignore_index=True) if all_returns else pd.DataFrame()

    return factors, returns
