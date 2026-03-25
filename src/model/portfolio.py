"""
Mean-variance portfolio optimizer for Japanese equities.

Constructs optimal portfolios from alpha signals subject to:
- Long-only or long/short constraints
- Position size limits (single name, sector)
- Turnover penalties to control transaction costs
- Risk budget via covariance shrinkage (Ledoit-Wolf)

The optimizer solves:
    max  w' * alpha  -  (lambda/2) * w' * Sigma * w  -  kappa * |w - w_prev|
    s.t. sum(w) = 1, w >= 0 (long-only), w_i <= max_weight
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConstraints:
    """Portfolio construction constraints."""
    long_only: bool = True
    max_position: float = 0.05        # 5% single name cap
    min_position: float = 0.002       # 20bps floor (if held)
    max_sector_weight: float = 0.25   # 25% sector cap
    max_names: int = 50               # max holdings
    min_names: int = 15               # min holdings
    turnover_penalty: float = 0.005   # 50bps turnover cost
    risk_aversion: float = 1.0        # lambda in objective


@dataclass
class PortfolioResult:
    """Output of portfolio optimization."""
    weights: pd.Series              # ticker -> weight
    expected_return: float          # portfolio expected alpha
    expected_risk: float            # portfolio volatility
    sharpe_ratio: float             # ex-ante Sharpe
    n_holdings: int
    sector_weights: dict            # sector -> aggregate weight
    active_risk: float              # tracking error vs equal-weight
    turnover: float                 # vs previous portfolio


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf constant-correlation shrinkage estimator.

    Shrinks sample covariance toward structured target (constant correlation).
    More stable than sample covariance when T/N is small.
    """
    X = returns.values
    T, N = X.shape

    if T < 2 or N < 2:
        return np.eye(N) * returns.var().mean()

    # Demean
    X = X - X.mean(axis=0)

    # Sample covariance
    S = (X.T @ X) / T

    # Shrinkage target: constant correlation matrix
    var = np.diag(S)
    std = np.sqrt(var)
    # Average correlation
    std_outer = np.outer(std, std)
    std_outer[std_outer == 0] = 1.0
    corr = S / std_outer
    np.fill_diagonal(corr, 0)
    rho_bar = corr.sum() / (N * (N - 1))

    F = rho_bar * std_outer
    np.fill_diagonal(F, var)

    # Optimal shrinkage intensity (analytical formula)
    # Frobenius norm components
    X2 = X ** 2
    sample2 = (X.T @ X) / T
    pi_mat = (X2.T @ X2) / T - sample2 ** 2
    pi_hat = pi_mat.sum()

    gamma_hat = np.linalg.norm(S - F, 'fro') ** 2

    rho_diag = 0
    rho_off = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                rho_diag += pi_mat[i, i]
            else:
                theta_ii = ((X[:, i] ** 2) * X[:, i] * X[:, j]).mean() - S[i, i] * S[i, j]
                theta_jj = ((X[:, j] ** 2) * X[:, i] * X[:, j]).mean() - S[j, j] * S[i, j]
                rho_off += (
                    rho_bar * 0.5
                    * (std[j] / std[i] * theta_ii + std[i] / std[j] * theta_jj)
                    if std[i] > 0 and std[j] > 0 else 0
                )

    rho_hat = rho_diag + rho_off
    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0
    delta = max(0.0, min(1.0, kappa / T))

    logger.debug(f"Ledoit-Wolf shrinkage intensity: {delta:.4f}")

    return delta * F + (1 - delta) * S


def _estimate_covariance(
    returns_panel: pd.DataFrame,
    tickers: list[str],
    method: str = "ledoit_wolf",
) -> np.ndarray:
    """
    Estimate covariance matrix for given tickers from return panel.

    returns_panel: DataFrame with columns = tickers, rows = dates
    """
    # Align to available tickers
    available = [t for t in tickers if t in returns_panel.columns]
    if len(available) < 2:
        return np.eye(len(tickers)) * 0.04  # 20% vol assumption

    ret = returns_panel[available].dropna(how="all")
    ret = ret.fillna(0)

    if method == "ledoit_wolf":
        cov = ledoit_wolf_shrinkage(ret)
    else:
        cov = ret.cov().values

    # Map back to full ticker list (missing tickers get diagonal assumption)
    n = len(tickers)
    full_cov = np.eye(n) * 0.04
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    for i, t1 in enumerate(available):
        for j, t2 in enumerate(available):
            idx1 = ticker_idx.get(t1)
            idx2 = ticker_idx.get(t2)
            if idx1 is not None and idx2 is not None:
                full_cov[idx1, idx2] = cov[i, j]

    return full_cov


def optimize_portfolio(
    alpha: pd.Series,
    returns_panel: pd.DataFrame = None,
    sectors: pd.Series = None,
    prev_weights: pd.Series = None,
    constraints: PortfolioConstraints = None,
) -> PortfolioResult:
    """
    Construct optimal portfolio from alpha signal.

    Uses quadratic optimization via iterative refinement when scipy
    is not available, otherwise delegates to scipy.optimize.minimize.

    alpha: Series indexed by ticker, higher = more attractive
    returns_panel: historical return DataFrame for covariance estimation
    sectors: Series indexed by ticker -> sector name
    prev_weights: previous period weights for turnover calc
    constraints: PortfolioConstraints
    """
    if constraints is None:
        constraints = PortfolioConstraints()

    tickers = alpha.dropna().index.tolist()
    n = len(tickers)

    if n == 0:
        return _empty_result()

    # Rank and select top N names by alpha
    alpha_sorted = alpha.loc[tickers].sort_values(ascending=False)
    selected = alpha_sorted.head(constraints.max_names).index.tolist()
    n_sel = len(selected)

    if n_sel < constraints.min_names:
        # Pad with next-best names
        remaining = [t for t in alpha_sorted.index if t not in selected]
        selected += remaining[: constraints.min_names - n_sel]
        n_sel = len(selected)

    alpha_vec = alpha.loc[selected].values.astype(float)

    # Estimate covariance
    if returns_panel is not None and not returns_panel.empty:
        cov = _estimate_covariance(returns_panel, selected)
    else:
        cov = np.eye(n_sel) * 0.04

    # Previous weights vector
    if prev_weights is not None:
        w_prev = np.array([prev_weights.get(t, 0.0) for t in selected])
    else:
        w_prev = np.zeros(n_sel)

    # Solve via scipy if available, else analytical approximation
    try:
        from scipy.optimize import minimize

        w = _scipy_optimize(
            alpha_vec, cov, w_prev, n_sel, constraints
        )
    except ImportError:
        w = _analytical_optimize(
            alpha_vec, cov, w_prev, n_sel, constraints
        )

    # Apply sector constraints
    if sectors is not None:
        w = _apply_sector_constraints(w, selected, sectors, constraints)

    # Ensure min position (zero out tiny positions, re-normalize)
    w[w < constraints.min_position] = 0
    if w.sum() > 0:
        w = w / w.sum()

    weights = pd.Series(w, index=selected)
    weights = weights[weights > 0].sort_values(ascending=False)

    # Portfolio statistics
    sel_idx = [selected.index(t) for t in weights.index]
    w_final = weights.values
    alpha_final = alpha.loc[weights.index].values
    cov_final = cov[np.ix_(sel_idx, sel_idx)]

    exp_ret = float(w_final @ alpha_final)
    exp_risk = float(np.sqrt(w_final @ cov_final @ w_final))
    sharpe = exp_ret / exp_risk if exp_risk > 0 else 0

    # Sector weights
    sect_wt = {}
    if sectors is not None:
        for t, wt in weights.items():
            s = sectors.get(t, "Other")
            sect_wt[s] = sect_wt.get(s, 0) + wt

    # Turnover
    if prev_weights is not None:
        turnover = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0))
                       for t in set(list(weights.index) + list(prev_weights.index)))
    else:
        turnover = 2.0  # Full initial investment

    # Tracking error vs equal weight
    w_eq = np.ones(len(w_final)) / len(w_final)
    active = w_final - w_eq
    active_risk = float(np.sqrt(active @ cov_final @ active))

    return PortfolioResult(
        weights=weights,
        expected_return=round(exp_ret, 6),
        expected_risk=round(exp_risk, 6),
        sharpe_ratio=round(sharpe, 3),
        n_holdings=len(weights),
        sector_weights=sect_wt,
        active_risk=round(active_risk, 6),
        turnover=round(turnover, 4),
    )


def _scipy_optimize(
    alpha: np.ndarray,
    cov: np.ndarray,
    w_prev: np.ndarray,
    n: int,
    constraints: PortfolioConstraints,
) -> np.ndarray:
    """Quadratic optimization via scipy."""
    from scipy.optimize import minimize

    lam = constraints.risk_aversion
    kappa = constraints.turnover_penalty

    def objective(w):
        ret = w @ alpha
        risk = w @ cov @ w
        turnover = np.sum(np.abs(w - w_prev))
        return -(ret - (lam / 2) * risk - kappa * turnover)

    # Constraints
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds
    if constraints.long_only:
        bounds = [(0, constraints.max_position)] * n
    else:
        bounds = [(-constraints.max_position, constraints.max_position)] * n

    # Initial guess: equal weight
    w0 = np.ones(n) / n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    if result.success:
        return result.x
    else:
        logger.warning(f"Optimizer did not converge: {result.message}")
        return result.x


def _analytical_optimize(
    alpha: np.ndarray,
    cov: np.ndarray,
    w_prev: np.ndarray,
    n: int,
    constraints: PortfolioConstraints,
) -> np.ndarray:
    """
    Fallback: alpha-weighted portfolio with risk scaling.
    Used when scipy is not available.
    """
    # Shift alpha to positive
    a = alpha - alpha.min() + 1e-6

    # Inverse-variance scaling
    var = np.diag(cov)
    var[var <= 0] = 0.04
    inv_var = 1.0 / var

    # Combined score: alpha * inverse_variance
    score = a * inv_var
    score = np.maximum(score, 0)

    # Normalize
    total = score.sum()
    if total > 0:
        w = score / total
    else:
        w = np.ones(n) / n

    # Cap positions
    w = np.minimum(w, constraints.max_position)
    if w.sum() > 0:
        w = w / w.sum()

    return w


def _apply_sector_constraints(
    w: np.ndarray,
    tickers: list[str],
    sectors: pd.Series,
    constraints: PortfolioConstraints,
) -> np.ndarray:
    """Iteratively reduce weights in over-concentrated sectors."""
    max_iter = 20
    for _ in range(max_iter):
        # Calculate sector weights
        sect_wt = {}
        for i, t in enumerate(tickers):
            s = sectors.get(t, "Other")
            sect_wt[s] = sect_wt.get(s, 0) + w[i]

        # Find breaching sectors
        breach = {s: wt for s, wt in sect_wt.items()
                  if wt > constraints.max_sector_weight}

        if not breach:
            break

        # Scale down breaching sector weights
        for s, wt in breach.items():
            scale = constraints.max_sector_weight / wt
            for i, t in enumerate(tickers):
                if sectors.get(t, "Other") == s:
                    w[i] *= scale

        # Re-normalize
        if w.sum() > 0:
            w = w / w.sum()

    return w


def _empty_result() -> PortfolioResult:
    return PortfolioResult(
        weights=pd.Series(dtype=float),
        expected_return=0.0,
        expected_risk=0.0,
        sharpe_ratio=0.0,
        n_holdings=0,
        sector_weights={},
        active_risk=0.0,
        turnover=0.0,
    )


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """
    Risk parity: each asset contributes equally to portfolio variance.
    Analytical approximation: w_i proportional to 1/sigma_i.
    """
    vol = np.sqrt(np.diag(cov))
    vol[vol <= 0] = 1.0
    w = 1.0 / vol
    return w / w.sum()


def equal_weight(n: int) -> np.ndarray:
    """Equal-weight benchmark."""
    return np.ones(n) / n
