"""
Multi-model signal generation for Japanese equities.

Provides three independent alpha models that can be blended:

1. FACTOR RANK MODEL — Cross-sectional percentile ranking on fundamental factors.
   Works on a single snapshot. No estimation history needed.

2. MOMENTUM MODEL — Price-derived time-series signals: 12-1 month momentum,
   52-week position, moving average crossovers, idiosyncratic volatility.
   Built from yfinance price history.

3. ML ENSEMBLE — Gradient-boosted tree trained on factor exposures predicting
   cross-sectional return ranks. Uses scikit-learn with leave-one-out
   sector-neutral training to prevent overfitting.

Each model outputs a z-scored alpha per ticker. The blender combines them
with configurable weights and outputs a final composite alpha that feeds
into the portfolio optimizer and the SCREEN tab.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelSignal:
    """Output from a single alpha model."""
    name: str
    alpha: pd.Series          # ticker -> z-scored alpha
    coverage: float           # fraction of universe with signal
    metadata: dict = field(default_factory=dict)


@dataclass
class BlendedSignal:
    """Combined signal from multiple models."""
    alpha: pd.Series
    model_signals: list[ModelSignal]
    weights_used: dict
    correlation_matrix: pd.DataFrame


# ── Model 1: Factor Rank ──────────────────────────────────────

def factor_rank_model(
    df: pd.DataFrame,
    factor_weights: dict = None,
) -> ModelSignal:
    """
    Pure cross-sectional ranking model on fundamental factors.

    Ranks each factor cross-sectionally, normalizes to [0,1], then
    takes a weighted sum. Works on a single snapshot — no history needed.

    df: DataFrame with Ticker column and fundamental columns
    factor_weights: dict of column_name -> (weight, direction)
                    direction: 1 = higher is better, -1 = lower is better
    """
    if factor_weights is None:
        factor_weights = {
            "ebitda_to_ev":          (0.15, 1),   # earnings yield — higher better
            "pb_ratio":              (0.10, -1),  # P/B — lower better
            "ev_to_ebitda":          (0.10, -1),  # EV/EBITDA — lower better
            "fcf_yield":             (0.10, 1),   # FCF yield — higher better
            "dividend_yield":        (0.05, 1),   # div yield
            "lt_debt_to_ev":         (0.08, 1),   # leverage intensity — for LSV
            "roe":                   (0.08, 1),   # profitability
            "operating_margin":      (0.05, 1),   # efficiency
            "gross_profit_to_assets":(0.07, 1),   # capital efficiency
            "asset_turnover":        (0.05, 1),   # asset productivity
            "quality_flags":         (0.07, 1),   # Piotroski-lite
            "52w_pos":               (0.05, 1),   # momentum
            "sma_cross":             (0.05, 1),   # trend
        }

    tickers = df["Ticker"].values if "Ticker" in df.columns else df.index.values
    n = len(df)
    alpha = np.zeros(n)
    total_weight = 0

    for col, (weight, direction) in factor_weights.items():
        if col not in df.columns:
            continue
        vals = df[col].astype(float)
        valid = vals.notna()
        if valid.sum() < 5:
            continue

        # Percentile rank (0 to 1)
        ranked = vals.rank(pct=True, na_option="keep")
        if direction < 0:
            ranked = 1.0 - ranked

        alpha += (ranked.fillna(0.5).values * weight)
        total_weight += weight

    if total_weight > 0:
        alpha = alpha / total_weight

    # Z-score the final alpha
    mu = np.nanmean(alpha)
    sigma = np.nanstd(alpha)
    if sigma > 0:
        alpha = (alpha - mu) / sigma

    alpha_series = pd.Series(alpha, index=tickers)
    coverage = (pd.notna(alpha_series) & (alpha_series != 0)).mean()

    return ModelSignal(
        name="FACTOR_RANK",
        alpha=alpha_series,
        coverage=round(coverage, 3),
        metadata={
            "n_factors": len([c for c in factor_weights if c in df.columns]),
            "total_weight": round(total_weight, 3),
            "method": "cross_sectional_percentile_rank",
        },
    )


# ── Model 2: Momentum ─────────────────────────────────────────

def momentum_model(
    df: pd.DataFrame,
    price_history: dict[str, pd.DataFrame] = None,
) -> ModelSignal:
    """
    Price-derived momentum and mean-reversion signals.

    Uses existing derived columns (52w_pos, sma_cross) plus additional
    momentum factors if price history is available.

    df: DataFrame with Ticker, 52w_pos, sma_cross, beta, etc.
    price_history: optional dict of ticker -> price DataFrame for richer signals
    """
    tickers = df["Ticker"].values if "Ticker" in df.columns else df.index.values
    n = len(df)

    components = {}

    # 52-week position (already computed)
    if "52w_pos" in df.columns:
        vals = df["52w_pos"].astype(float).rank(pct=True, na_option="keep").fillna(0.5)
        components["52w_position"] = vals.values

    # SMA cross (trend strength)
    if "sma_cross" in df.columns:
        vals = df["sma_cross"].astype(float).rank(pct=True, na_option="keep").fillna(0.5)
        components["sma_cross"] = vals.values

    # Low beta premium (defensive momentum)
    if "beta" in df.columns:
        beta_rank = df["beta"].astype(float).rank(pct=True, na_option="keep")
        components["low_beta"] = (1.0 - beta_rank.fillna(0.5)).values

    # Price-derived momentum from history
    if price_history:
        mom_12_1 = []
        vol_scores = []
        for t in tickers:
            ph = price_history.get(t)
            if ph is not None and not ph.empty:
                close = _get_close(ph)
                if close is not None and len(close) > 21:
                    # 12-1 month momentum (skip last month for reversal)
                    if len(close) > 252:
                        ret_12m = float(close.iloc[-21] / close.iloc[-252] - 1)
                    elif len(close) > 63:
                        ret_12m = float(close.iloc[-21] / close.iloc[0] - 1)
                    else:
                        ret_12m = np.nan
                    mom_12_1.append(ret_12m)

                    # Idiosyncratic volatility (lower = better, lottery effect)
                    daily_ret = close.pct_change().dropna().tail(63)
                    vol_scores.append(daily_ret.std() * np.sqrt(252))
                else:
                    mom_12_1.append(np.nan)
                    vol_scores.append(np.nan)
            else:
                mom_12_1.append(np.nan)
                vol_scores.append(np.nan)

        mom_s = pd.Series(mom_12_1).rank(pct=True, na_option="keep").fillna(0.5)
        components["momentum_12_1"] = mom_s.values

        vol_s = pd.Series(vol_scores).rank(pct=True, na_option="keep")
        components["low_vol"] = (1.0 - vol_s.fillna(0.5)).values

    if not components:
        return ModelSignal(
            name="MOMENTUM",
            alpha=pd.Series(0.0, index=tickers),
            coverage=0.0,
            metadata={"error": "no_momentum_data"},
        )

    # Weight momentum components
    weights = {
        "momentum_12_1": 0.35,
        "52w_position": 0.25,
        "sma_cross": 0.15,
        "low_vol": 0.15,
        "low_beta": 0.10,
    }

    alpha = np.zeros(n)
    tw = 0
    for comp, w in weights.items():
        if comp in components:
            alpha += components[comp] * w
            tw += w
    if tw > 0:
        alpha /= tw

    mu = np.nanmean(alpha)
    sigma = np.nanstd(alpha)
    if sigma > 0:
        alpha = (alpha - mu) / sigma

    alpha_series = pd.Series(alpha, index=tickers)
    coverage = (pd.notna(alpha_series) & (alpha_series != 0)).mean()

    return ModelSignal(
        name="MOMENTUM",
        alpha=alpha_series,
        coverage=round(coverage, 3),
        metadata={
            "n_components": len(components),
            "components": list(components.keys()),
            "method": "price_derived_momentum",
        },
    )


# ── Model 3: ML Ensemble ──────────────────────────────────────

def ml_ensemble_model(
    df: pd.DataFrame,
    target_col: str = "Composite",
) -> ModelSignal:
    """
    Gradient-boosted tree predicting cross-sectional alpha rank.

    Uses all available fundamental factors as features. Trains on
    the current cross-section using leave-one-out or K-fold to
    generate out-of-sample predictions for each stock.

    This captures nonlinear interactions between factors that
    the linear ranking model misses (e.g., cheap + high leverage
    + improving margins = strong signal).

    df: DataFrame with Ticker, fundamental columns, and target_col
    target_col: column to predict (typically Composite or forward return)
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_predict
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("scikit-learn not installed. ML model unavailable.")
        tickers = df["Ticker"].values if "Ticker" in df.columns else df.index.values
        return ModelSignal(
            name="ML_ENSEMBLE",
            alpha=pd.Series(0.0, index=tickers),
            coverage=0.0,
            metadata={"error": "sklearn_not_installed"},
        )

    tickers = df["Ticker"].values if "Ticker" in df.columns else df.index.values

    # Feature columns
    feature_cols = [
        "ebitda_to_ev", "pb_ratio", "ev_to_ebitda", "fcf_yield",
        "dividend_yield", "lt_debt_to_ev", "roe", "operating_margin",
        "gross_profit_to_assets", "asset_turnover", "quality_flags",
        "52w_pos", "sma_cross", "beta", "mcap_b",
        "debt_to_equity", "current_ratio", "revenue_growth",
        "earnings_growth", "net_debt_to_ebitda",
    ]
    available = [c for c in feature_cols if c in df.columns]

    if len(available) < 5 or target_col not in df.columns:
        return ModelSignal(
            name="ML_ENSEMBLE",
            alpha=pd.Series(0.0, index=tickers),
            coverage=0.0,
            metadata={"error": "insufficient_features"},
        )

    X = df[available].copy()
    y = df[target_col].copy()

    # Drop rows where target is NaN
    valid = y.notna() & X.notna().all(axis=1)
    if valid.sum() < 20:
        return ModelSignal(
            name="ML_ENSEMBLE",
            alpha=pd.Series(0.0, index=tickers),
            coverage=0.0,
            metadata={"error": "insufficient_valid_rows"},
        )

    X_valid = X[valid].values
    y_valid = y[valid].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # Gradient boosting with conservative hyperparameters (avoid overfit)
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )

    # Cross-validated predictions (out-of-sample for each row)
    try:
        n_folds = min(5, valid.sum())
        predictions = cross_val_predict(model, X_scaled, y_valid, cv=n_folds)
    except Exception as e:
        logger.warning(f"ML cross-val failed: {e}")
        return ModelSignal(
            name="ML_ENSEMBLE",
            alpha=pd.Series(0.0, index=tickers),
            coverage=0.0,
            metadata={"error": str(e)},
        )

    # Also fit on full data for feature importances
    model.fit(X_scaled, y_valid)
    importances = dict(zip(available, model.feature_importances_))

    # Map predictions back to all tickers
    alpha = np.full(len(tickers), np.nan)
    valid_idx = np.where(valid.values)[0]
    for i, idx in enumerate(valid_idx):
        alpha[idx] = predictions[i]

    # Z-score
    mu = np.nanmean(alpha)
    sigma = np.nanstd(alpha)
    if sigma > 0:
        alpha = (alpha - mu) / sigma

    alpha = np.nan_to_num(alpha, nan=0.0)
    alpha_series = pd.Series(alpha, index=tickers)
    coverage = valid.mean()

    # Sort importances
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    return ModelSignal(
        name="ML_ENSEMBLE",
        alpha=alpha_series,
        coverage=round(coverage, 3),
        metadata={
            "n_features": len(available),
            "n_samples": int(valid.sum()),
            "feature_importances": {k: round(v, 4) for k, v in sorted_imp[:10]},
            "method": "gradient_boosted_cross_val_predict",
            "cv_folds": min(5, int(valid.sum())),
        },
    )


# ── Model 4: Mean Reversion ───────────────────────────────────

def mean_reversion_model(
    df: pd.DataFrame,
    price_history: dict[str, pd.DataFrame] = None,
) -> ModelSignal:
    """
    Short-term mean reversion signals — Japan equities exhibit strong
    reversal at 1-month and 1-week horizons (Asness, 2013).

    Components:
    - 1-month reversal: stocks that dropped last month tend to bounce
    - Distance from 200-day MA: oversold names revert up
    - RSI contrarian: low RSI = oversold = buy signal
    - Volume spike: high recent volume on a down-move = capitulation
    """
    tickers = df["Ticker"].values if "Ticker" in df.columns else df.index.values
    n = len(df)
    components = {}

    # Distance from 200-day MA (from existing columns)
    if "current_price" in df.columns and "two_hundred_day_avg" in df.columns:
        price = df["current_price"].astype(float)
        ma200 = df["two_hundred_day_avg"].astype(float)
        valid = price.notna() & ma200.notna() & (ma200 > 0)
        dist = pd.Series(np.nan, index=df.index)
        dist[valid] = (price[valid] / ma200[valid]) - 1
        # Invert: more oversold (negative distance) = higher score
        dist_rank = (-dist).rank(pct=True, na_option="keep").fillna(0.5)
        components["ma200_reversion"] = dist_rank.values

    # 52w position inverted: stocks near 52w lows tend to mean-revert
    if "52w_pos" in df.columns:
        pos = df["52w_pos"].astype(float)
        inv_rank = (1 - pos.rank(pct=True, na_option="keep")).fillna(0.5)
        components["52w_contrarian"] = inv_rank.values

    # Price-derived short-term reversal from history
    if price_history:
        rev_1m = []
        rsi_vals = []
        vol_spike = []
        for t in tickers:
            ph = price_history.get(t)
            if ph is not None and not ph.empty:
                close = _get_close(ph)
                if close is not None and len(close) > 21:
                    # 1-month reversal (last 21 trading days return, inverted)
                    ret_1m = float(close.iloc[-1] / close.iloc[-21] - 1)
                    rev_1m.append(-ret_1m)  # losers become winners

                    # RSI (14-day)
                    delta = close.diff().tail(14)
                    gain = delta.clip(lower=0).mean()
                    loss = (-delta.clip(upper=0)).mean()
                    rs = gain / loss if loss > 0 else 100
                    rsi = 100 - (100 / (1 + rs))
                    rsi_vals.append(100 - rsi)  # Invert: low RSI = high score

                    # Volume spike on down-move (capitulation indicator)
                    vol_col = _get_volume(ph)
                    if vol_col is not None and len(vol_col) > 21:
                        vol_recent = float(vol_col.tail(5).mean())
                        vol_avg = float(vol_col.tail(63).mean())
                        is_down = float(ret_1m) < 0
                        spike = (vol_recent / vol_avg - 1) if vol_avg > 0 else 0.0
                        vol_spike.append(spike if is_down else 0.0)
                    else:
                        vol_spike.append(np.nan)
                else:
                    rev_1m.append(np.nan)
                    rsi_vals.append(np.nan)
                    vol_spike.append(np.nan)
            else:
                rev_1m.append(np.nan)
                rsi_vals.append(np.nan)
                vol_spike.append(np.nan)

        rev_s = pd.Series(rev_1m).rank(pct=True, na_option="keep").fillna(0.5)
        components["reversal_1m"] = rev_s.values

        rsi_s = pd.Series(rsi_vals).rank(pct=True, na_option="keep").fillna(0.5)
        components["rsi_contrarian"] = rsi_s.values

        vol_s = pd.Series(vol_spike).rank(pct=True, na_option="keep").fillna(0.5)
        components["capitulation"] = vol_s.values

    if not components:
        return ModelSignal(
            name="MEAN_REVERSION",
            alpha=pd.Series(0.0, index=tickers),
            coverage=0.0,
            metadata={"error": "no_reversion_data"},
        )

    weights = {
        "reversal_1m": 0.30,
        "rsi_contrarian": 0.20,
        "ma200_reversion": 0.20,
        "52w_contrarian": 0.15,
        "capitulation": 0.15,
    }

    alpha = np.zeros(n)
    tw = 0
    for comp, w in weights.items():
        if comp in components:
            alpha += components[comp] * w
            tw += w
    if tw > 0:
        alpha /= tw

    mu = np.nanmean(alpha)
    sigma = np.nanstd(alpha)
    if sigma > 0:
        alpha = (alpha - mu) / sigma

    alpha_series = pd.Series(alpha, index=tickers)
    coverage = (pd.notna(alpha_series) & (alpha_series != 0)).mean()

    return ModelSignal(
        name="MEAN_REVERSION",
        alpha=alpha_series,
        coverage=round(coverage, 3),
        metadata={
            "n_components": len(components),
            "components": list(components.keys()),
            "method": "short_term_mean_reversion",
        },
    )


# ── Signal Quality Diagnostics ────────────────────────────────

def quintile_analysis(
    alpha: pd.Series,
    df: pd.DataFrame,
    metric_col: str = "Composite",
) -> dict:
    """
    Analyze alpha signal quality by quintile.

    Splits universe into 5 quintiles by alpha, then reports average
    score/metric per quintile. A good signal should show monotonic
    increase from Q1 (worst) to Q5 (best).
    """
    merged = pd.DataFrame({
        "alpha": alpha,
        "metric": df.set_index("Ticker")[metric_col] if "Ticker" in df.columns else df[metric_col],
    }).dropna()

    if len(merged) < 10:
        return {"error": "insufficient_data"}

    merged["quintile"] = pd.qcut(merged["alpha"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    result = {}
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        qdata = merged[merged["quintile"] == q]
        result[q] = {
            "count": len(qdata),
            "avg_alpha": round(qdata["alpha"].mean(), 4),
            "avg_metric": round(qdata["metric"].mean(), 4),
            "min_alpha": round(qdata["alpha"].min(), 4),
            "max_alpha": round(qdata["alpha"].max(), 4),
        }

    # Monotonicity score: how well does alpha predict the metric?
    q_means = [result[q]["avg_metric"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]]
    monotonic = sum(1 for i in range(4) if q_means[i] < q_means[i + 1]) / 4
    result["monotonicity"] = round(monotonic, 2)
    result["q5_q1_spread"] = round(q_means[4] - q_means[0], 4)

    return result


def signal_diagnostics(
    alpha: pd.Series,
    df: pd.DataFrame,
) -> dict:
    """
    Comprehensive signal quality diagnostics.
    """
    tickers = df["Ticker"].values if "Ticker" in df.columns else df.index.values
    aligned = alpha.reindex(tickers).fillna(0)

    diag = {
        "n_stocks": len(aligned),
        "coverage": round((aligned != 0).mean(), 3),
        "mean": round(aligned.mean(), 4),
        "std": round(aligned.std(), 4),
        "skew": round(aligned.skew(), 3),
        "kurtosis": round(aligned.kurtosis(), 3),
        "pct_positive": round((aligned > 0).mean() * 100, 1),
        "q10": round(aligned.quantile(0.1), 4),
        "q25": round(aligned.quantile(0.25), 4),
        "median": round(aligned.median(), 4),
        "q75": round(aligned.quantile(0.75), 4),
        "q90": round(aligned.quantile(0.9), 4),
        "spread_90_10": round(aligned.quantile(0.9) - aligned.quantile(0.1), 4),
    }

    # IC with composite score if available
    if "Composite" in df.columns:
        from scipy import stats as sp_stats
        comp = df.set_index("Ticker")["Composite"] if "Ticker" in df.columns else df["Composite"]
        common = aligned.index.intersection(comp.index)
        if len(common) > 10:
            ic, p_val = sp_stats.spearmanr(aligned[common], comp[common])
            diag["ic_vs_composite"] = round(ic, 4)
            diag["ic_pval"] = round(p_val, 4)

    return diag


def build_returns_panel(
    tickers: list[str],
    price_loader,
    months: int = 12,
) -> pd.DataFrame:
    """
    Build a returns panel from price history for covariance estimation.

    tickers: list of ticker symbols
    price_loader: callable(ticker) -> price DataFrame
    months: how many months of daily returns to include
    """
    returns_dict = {}
    for t in tickers:
        try:
            ph = price_loader(t)
            if ph is not None and not ph.empty:
                close = _get_close(ph)
                if close is not None and len(close) > 21:
                    daily_ret = close.pct_change().dropna().tail(months * 21)
                    returns_dict[t] = daily_ret
        except Exception:
            continue

    if not returns_dict:
        return pd.DataFrame()

    panel = pd.DataFrame(returns_dict)
    panel = panel.fillna(0)
    return panel


# ── Signal Blender ─────────────────────────────────────────────

def blend_signals(
    signals: list[ModelSignal],
    weights: dict[str, float] = None,
) -> BlendedSignal:
    """
    Blend multiple model signals into a single composite alpha.

    Default weights: FACTOR_RANK 0.40, MOMENTUM 0.20, ML_ENSEMBLE 0.20, MEAN_REVERSION 0.20
    """
    if weights is None:
        weights = {
            "FACTOR_RANK": 0.40,
            "MOMENTUM": 0.20,
            "ML_ENSEMBLE": 0.20,
            "MEAN_REVERSION": 0.20,
        }

    # Normalize weights to sum to 1
    active_signals = [s for s in signals if s.coverage > 0.1]
    if not active_signals:
        active_signals = signals

    active_weights = {}
    for s in active_signals:
        active_weights[s.name] = weights.get(s.name, 1.0 / len(active_signals))

    total = sum(active_weights.values())
    if total > 0:
        active_weights = {k: v / total for k, v in active_weights.items()}

    # Get common index
    all_tickers = set()
    for s in active_signals:
        all_tickers.update(s.alpha.index)
    all_tickers = sorted(all_tickers)

    blended = pd.Series(0.0, index=all_tickers)
    for s in active_signals:
        w = active_weights.get(s.name, 0)
        aligned = s.alpha.reindex(all_tickers, fill_value=0)
        blended += w * aligned

    # Final z-score
    mu = blended.mean()
    sigma = blended.std()
    if sigma > 0:
        blended = (blended - mu) / sigma

    # Correlation matrix between signals
    signal_df = pd.DataFrame({
        s.name: s.alpha.reindex(all_tickers, fill_value=0)
        for s in active_signals
    })
    corr = signal_df.corr()

    return BlendedSignal(
        alpha=blended,
        model_signals=active_signals,
        weights_used=active_weights,
        correlation_matrix=corr,
    )


# ── Helpers ────────────────────────────────────────────────────

def _get_close(price_df: pd.DataFrame) -> pd.Series | None:
    """Extract close price from potentially MultiIndex DataFrame."""
    if price_df.empty:
        return None
    if isinstance(price_df.columns, pd.MultiIndex):
        for col in price_df.columns:
            if "close" in str(col).lower():
                return price_df[col]
        return price_df.iloc[:, 0]
    if "Close" in price_df.columns:
        return price_df["Close"]
    return price_df.iloc[:, 0]


def _get_volume(price_df: pd.DataFrame) -> pd.Series | None:
    """Extract volume from potentially MultiIndex DataFrame."""
    if price_df.empty:
        return None
    if isinstance(price_df.columns, pd.MultiIndex):
        for col in price_df.columns:
            if "volume" in str(col).lower():
                return price_df[col]
        return None
    if "Volume" in price_df.columns:
        return price_df["Volume"]
    return None
