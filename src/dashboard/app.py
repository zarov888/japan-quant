"""
JVQ Terminal — Bloomberg-style Japan equity screener.
"""
from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.universe import load_universe, INDEX_REGISTRY
from src.data.fetcher import fetch_universe, fetch_price_history
from src.model.scorer import score_universe, results_to_dataframe, load_scoring_config
from src.data.governance import add_governance_to_df
from src.data.history import save_snapshot, load_all_snapshots, get_score_history
from src.model.factor_model import (
    FACTOR_DEFS, prepare_factor_matrix, fama_macbeth_regression,
    generate_alpha_signal, FactorModelResult,
)
from src.model.portfolio import optimize_portfolio, PortfolioConstraints, PortfolioResult
from src.model.walkforward import walk_forward_backtest, WalkForwardConfig
from src.model.signals import (
    factor_rank_model, momentum_model, ml_ensemble_model,
    mean_reversion_model, blend_signals, ModelSignal, BlendedSignal,
    quintile_analysis, signal_diagnostics, build_returns_panel,
)

# ── Page ───────────────────────────────────────────────────────
st.set_page_config(page_title="JVQ", layout="wide", initial_sidebar_state="expanded")

# ── Bloomberg palette ──────────────────────────────────────────
BG = "#000000"
BG1 = "#0a0a0a"
BG2 = "#111111"
BORDER = "#1c1c1c"
ORANGE = "#ff8c00"
ORANGE_DIM = "#cc7000"
GREEN = "#00cc66"
RED = "#ff4444"
YELLOW = "#cccc00"
WHITE = "#dddddd"
GRAY = "#666666"
GRAY_DIM = "#333333"
FONT = "'Consolas', 'Menlo', 'Monaco', monospace"

def chart_layout(height=300, **kw):
    base = dict(
        template="plotly_dark",
        paper_bgcolor=BG1,
        plot_bgcolor=BG1,
        font=dict(family=FONT, size=10, color=GRAY),
        margin=dict(l=45, r=10, t=28, b=30),
        height=height,
        title_font=dict(size=11, color=ORANGE),
        showlegend=False,
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    )
    # Merge xaxis/yaxis defaults with any overrides
    xa = dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=9))
    ya = dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=9))
    if "xaxis" in kw:
        xa.update(kw.pop("xaxis"))
    if "yaxis" in kw:
        ya.update(kw.pop("yaxis"))
    if "legend" in kw:
        base["legend"] = kw.pop("legend")
    base["xaxis"] = xa
    base["yaxis"] = ya
    base.update(kw)
    return base

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

    /* base */
    .stApp {{ background-color: {BG}; color: {WHITE}; }}
    * {{ font-family: {FONT} !important; }}

    /* global spacing fixes */
    .block-container {{
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }}
    .stMarkdown {{ margin-bottom: 0 !important; }}
    div[data-testid="stVerticalBlock"] > div {{
        gap: 0.3rem !important;
    }}
    div[data-testid="stHorizontalBlock"] > div {{
        gap: 0.4rem !important;
    }}
    .element-container {{ margin-bottom: 0.2rem !important; }}

    /* top bar */
    .bb-top {{
        background: {BG1};
        border-bottom: 1px solid {BORDER};
        padding: 6px 12px;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        font-size: 11px;
        margin: -0.5rem -1rem 0.5rem -1rem;
    }}
    .bb-logo {{ color: {ORANGE}; font-weight: 600; font-size: 14px; letter-spacing: 3px; }}
    .bb-tag {{ color: {GRAY}; font-size: 10px; white-space: nowrap; }}
    .bb-live {{ color: {GREEN}; font-size: 10px; }}

    /* panels */
    .panel {{
        background: {BG1};
        border: 1px solid {BORDER};
        padding: 8px 10px;
        margin-bottom: 6px;
        font-size: 11px;
    }}
    .panel-title {{
        color: {ORANGE};
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
        padding-bottom: 4px;
        border-bottom: 1px solid {BORDER};
    }}

    /* data cells */
    .dc {{ display: inline-block; min-width: 70px; text-align: right; padding: 2px 6px; }}
    .dc-label {{ color: {GRAY}; font-size: 9px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .dc-val {{ color: {WHITE}; font-size: 12px; font-weight: 500; line-height: 1.4; }}
    .dc-green {{ color: {GREEN}; }}
    .dc-red {{ color: {RED}; }}
    .dc-orange {{ color: {ORANGE}; }}
    .dc-dim {{ color: {GRAY}; }}

    /* strip */
    .strip {{
        background: {BG2};
        border: 1px solid {BORDER};
        padding: 6px 10px;
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        row-gap: 4px;
        font-size: 10px;
        margin-bottom: 8px;
        line-height: 1.6;
    }}
    .strip-item {{ display: flex; gap: 5px; align-items: baseline; white-space: nowrap; }}
    .strip-label {{ color: {GRAY_DIM}; font-size: 9px; }}
    .strip-val {{ color: {WHITE}; font-size: 10px; }}

    /* kill streamlit chrome */
    header, footer, #MainMenu {{ visibility: hidden; }}
    h1,h2,h3 {{
        font-size: 11px !important;
        color: {ORANGE} !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 10px 0 8px 0 !important;
        padding-bottom: 4px;
        border-bottom: 1px solid {BORDER};
        line-height: 1.6 !important;
    }}

    /* metrics */
    [data-testid="stMetric"] {{
        background: {BG1};
        border: 1px solid {BORDER};
        padding: 8px 10px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {GRAY} !important;
        font-size: 9px !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        line-height: 1.4 !important;
        margin-bottom: 2px !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {ORANGE} !important;
        font-size: 15px !important;
        line-height: 1.3 !important;
    }}

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: {BG1};
        border-bottom: 1px solid {BORDER};
        gap: 0;
        overflow-x: auto;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {GRAY};
        font-size: 10px;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 8px 14px;
        border-radius: 0;
        white-space: nowrap;
    }}
    .stTabs [aria-selected="true"] {{
        color: {ORANGE} !important;
        border-bottom: 2px solid {ORANGE};
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 8px !important;
    }}

    /* hide ALL streamlit default sidebar widgets (keyboard, deploy, etc) */
    [data-testid="stSidebarKeyboardShortcuts"],
    [data-testid="stKeyboardShortcuts"],
    [data-testid="stStatusWidget"],
    button[title="Keyboard shortcuts"],
    button[kind="header"],
    [data-testid="stSidebar"] button[kind="headerNoPadding"],
    [data-testid="stSidebar"] [data-testid="stDecoration"],
    [data-testid="stSidebar"] > div > div > div > button,
    [data-testid="stToolbar"],
    .stDeployButton,
    #MainMenu,
    footer {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
    }}

    /* sidebar — always visible, never collapses */
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}
    [data-testid="stSidebar"] {{
        background: {BG1};
        border-right: 1px solid {BORDER};
        padding-top: 0.5rem;
        min-width: 280px !important;
        width: 280px !important;
    }}
    [data-testid="stSidebar"][aria-expanded="false"] {{
        min-width: 280px !important;
        width: 280px !important;
        margin-left: 0 !important;
        transform: none !important;
        visibility: visible !important;
    }}
    [data-testid="stSidebar"] .block-container {{
        padding-top: 0 !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 0 !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {{
        gap: 0.2rem !important;
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stCheckbox label {{
        font-size: 10px !important;
        color: {GRAY} !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    [data-testid="stSidebar"] .stMarkdown p {{
        font-size: 10px;
        margin-bottom: 2px;
    }}

    /* selectbox / inputs */
    .stSelectbox, .stMultiSelect, .stSlider, .stNumberInput {{
        margin-bottom: 2px !important;
    }}
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input {{
        font-size: 11px !important;
        background: {BG2} !important;
        border-color: {BORDER} !important;
        color: {WHITE} !important;
    }}

    /* dataframe */
    .stDataFrame {{
        border: 1px solid {BORDER};
        font-size: 10px;
    }}
    /* fix three-dot column menu text overlap */
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] [role="columnheader"] {{
        overflow: visible !important;
    }}
    [data-testid="stDataFrame"] div[data-testid="column-header-menu"],
    [data-testid="stDataFrame"] .glideDataEditor .gdg-header-menu {{
        background: {BG2} !important;
        color: {WHITE} !important;
        font-size: 11px !important;
        line-height: 1.6 !important;
        z-index: 999 !important;
    }}
    [data-testid="stDataFrame"] .gdg-style {{
        --gdg-bg-cell: {BG1} !important;
        --gdg-bg-header: {BG2} !important;
        --gdg-text-dark: {WHITE} !important;
        --gdg-text-header: {GRAY} !important;
    }}

    /* plotly charts — reduce bottom gap */
    .stPlotlyChart {{
        margin-bottom: 0 !important;
    }}

    /* equity header override */
    .eq-header {{
        background: {BG1};
        border: 1px solid {BORDER};
        padding: 8px 14px;
        display: flex;
        align-items: baseline;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 8px;
    }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────
def f_pct(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "--"
    return f"{v*100:.{d}f}%"

def f_num(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "--"
    return f"{v:.{d}f}"

def f_jpy(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "--"
    if abs(v) >= 1e12: return f"Y{v/1e12:.1f}T"
    if abs(v) >= 1e9: return f"Y{v/1e9:.0f}B"
    return f"Y{v/1e6:.0f}M"

def f_price(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "--"
    return f"{v:,.0f}"

def color_val(v, threshold=0, invert=False):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "dc-dim"
    if invert: return "dc-red" if v > threshold else "dc-green"
    return "dc-green" if v > threshold else "dc-red"

def pct_rank(series, val):
    """Percentile rank of val within series."""
    if pd.isna(val): return "--"
    rank = (series.dropna() < val).sum() / max(series.dropna().count(), 1) * 100
    return f"{rank:.0f}th"

def strip_html(items):
    """Render a data strip. Items: (label, val, css) or (label, val, css, tooltip)."""
    parts = []
    for item in items:
        label, val, css = item[0], item[1], item[2]
        tip = item[3] if len(item) > 3 else label
        parts.append(f'<div class="strip-item" title="{tip}"><span class="strip-label">{label}</span><span class="strip-val {css}">{val}</span></div>')
    return '<div class="strip">' + ''.join(parts) + '</div>'

def metric_cell(label, val, css="", tip=""):
    t = f' title="{tip}"' if tip else ""
    return f'<div class="dc"{t}><div class="dc-label">{label}</div><div class="dc-val {css}">{val}</div></div>'


# ── Data ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(index_name: str = "TOPIX 500"):
    config = load_universe("config/universe.yaml", index_override=index_name)
    fundamentals = fetch_universe(config)
    results = score_universe(fundamentals)
    df = results_to_dataframe(results)

    fund_map = {f["ticker"]: f for f in fundamentals}
    cols = [
        "pe_trailing", "pe_forward", "pb_ratio", "dividend_yield", "roe", "roa",
        "market_cap", "debt_to_equity", "cash_to_mcap", "ev_to_ebitda",
        "current_price", "fifty_two_week_high", "fifty_two_week_low",
        "fifty_day_avg", "two_hundred_day_avg", "beta", "avg_volume",
        "operating_margin", "profit_margin", "revenue_growth", "earnings_growth",
        "free_cashflow", "total_cash", "total_debt", "enterprise_value",
        "price_to_sales", "current_ratio", "industry",
        # Verdad-specific
        "ebitda", "total_revenue", "total_assets", "gross_profits",
        "long_term_debt", "lt_debt_to_ev", "net_debt_to_ebitda",
        "ebitda_to_ev", "gross_profit_to_assets", "lt_debt_to_assets",
        "asset_turnover", "net_income", "operating_cashflow",
        "short_ratio", "short_pct_float",
    ]
    for c in cols:
        df[c] = df["Ticker"].map(lambda t, c=c: fund_map.get(t, {}).get(c))

    df["mcap_b"] = df["market_cap"].fillna(0) / 1e9
    df["net_cash"] = df["total_cash"].fillna(0) - df["total_debt"].fillna(0)
    df["net_cash_b"] = df["net_cash"] / 1e9
    df["fcf_yield"] = np.where(
        (df["free_cashflow"].notna()) & (df["market_cap"].notna()) & (df["market_cap"] != 0),
        df["free_cashflow"] / df["market_cap"], np.nan)
    df["52w_pos"] = np.where(
        (df["fifty_two_week_high"].notna()) & (df["fifty_two_week_low"].notna()) & (df["fifty_two_week_high"] != df["fifty_two_week_low"]),
        (df["current_price"] - df["fifty_two_week_low"]) / (df["fifty_two_week_high"] - df["fifty_two_week_low"]), np.nan)
    df["sma_cross"] = np.where(
        (df["fifty_day_avg"].notna()) & (df["two_hundred_day_avg"].notna()) & (df["two_hundred_day_avg"] != 0),
        df["fifty_day_avg"] / df["two_hundred_day_avg"] - 1, np.nan)
    df["ev_to_fcf"] = np.where(
        (df["enterprise_value"].notna()) & (df["free_cashflow"].notna()) & (df["free_cashflow"] > 0),
        df["enterprise_value"] / df["free_cashflow"], np.nan)
    df["net_debt_to_ebitda"] = np.where(
        (df["ev_to_ebitda"].notna()) & (df["enterprise_value"].notna()) & (df["market_cap"].notna()),
        (df["total_debt"].fillna(0) - df["total_cash"].fillna(0)) / np.where(df["ev_to_ebitda"] != 0, df["enterprise_value"] / df["ev_to_ebitda"], np.nan),
        np.nan)
    # Piotroski-lite: simple quality flag count
    df["quality_flags"] = (
        (df["roe"].fillna(0) > 0.08).astype(int) +
        (df["operating_margin"].fillna(0) > 0.05).astype(int) +
        (df["revenue_growth"].fillna(0) > 0).astype(int) +
        (df["earnings_growth"].fillna(0) > 0).astype(int) +
        (df["current_ratio"].fillna(0) > 1.0).astype(int) +
        (df["net_cash"] > 0).astype(int) +
        (df["fcf_yield"].fillna(0) > 0.03).astype(int)
    )

    # Add governance reform signals
    df = add_governance_to_df(df, fund_map)

    # Save snapshot for historical tracking
    try:
        save_snapshot(results, tag=index_name.lower().replace(" ", "_"))
    except Exception:
        pass  # non-critical

    return df, fundamentals, results

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker, years=3):
    return fetch_price_history(ticker, years=years, cache_dir="data/cache")


# ── Loading Screen ─────────────────────────────────────────────
loading = st.empty()
loading.markdown(f"""
<div style="
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 80vh; background: {BG}; font-family: {FONT};
">
    <div style="color: {ORANGE}; font-size: 32px; font-weight: 600; letter-spacing: 6px; margin-bottom: 24px;">
        JVQ
    </div>
    <div style="color: {GRAY}; font-size: 11px; letter-spacing: 3px; margin-bottom: 32px;">
        EQUITY RESEARCH TERMINAL
    </div>
    <div style="display: flex; gap: 6px; margin-bottom: 24px;">
        <div style="width: 8px; height: 8px; background: {ORANGE}; animation: pulse 1.2s ease-in-out infinite;"></div>
        <div style="width: 8px; height: 8px; background: {ORANGE}; animation: pulse 1.2s ease-in-out 0.2s infinite;"></div>
        <div style="width: 8px; height: 8px; background: {ORANGE}; animation: pulse 1.2s ease-in-out 0.4s infinite;"></div>
        <div style="width: 8px; height: 8px; background: {ORANGE}; animation: pulse 1.2s ease-in-out 0.6s infinite;"></div>
        <div style="width: 8px; height: 8px; background: {ORANGE}; animation: pulse 1.2s ease-in-out 0.8s infinite;"></div>
    </div>
    <div style="color: {GRAY_DIM}; font-size: 10px; letter-spacing: 1px;">
        CONNECTING TO DATA FEED...
    </div>
    <div style="color: {GRAY_DIM}; font-size: 9px; margin-top: 8px;">
        SCORING TOPIX UNIVERSE
    </div>
    <div style="margin-top: 40px; color: {GRAY_DIM}; font-size: 8px;">
        BUILT BY NOAH | v2.0
    </div>
</div>
<style>
    @keyframes pulse {{
        0%, 100% {{ opacity: 0.2; transform: scale(0.8); }}
        50% {{ opacity: 1; transform: scale(1.2); }}
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar logo
st.sidebar.markdown(f"""
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:16px 0 16px 0; border-bottom:1px solid {BORDER}; margin:0 0 12px 0;">
    <div style="border:2px solid {ORANGE}; padding:8px 18px;">
        <span style="color:{ORANGE}; font-size:22px; font-weight:700; letter-spacing:6px; font-family:{FONT};">JVQ</span>
    </div>
    <div style="color:{GRAY}; font-size:8px; letter-spacing:2px; margin-top:6px;">JAPAN VALUE QUANT</div>
    <div style="color:{GRAY_DIM}; font-size:7px; letter-spacing:1px; margin-top:2px;">EQUITY RESEARCH TERMINAL</div>
</div>
""", unsafe_allow_html=True)

# Index selector in sidebar (before data load)
sel_index = st.sidebar.selectbox(
    "INDEX",
    list(INDEX_REGISTRY.keys()),
    index=2,  # default TOPIX 500
    key="idx_sel",
    help="Select which index universe to screen",
)

df, fundamentals, results = load_data(sel_index)
all_sectors = sorted(df["Sector"].dropna().unique().tolist())
loading.empty()

# ── Run multi-model alpha ─────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _compute_alpha(df_hash: str, df_json: str, w_rank: float = 0.40,
                   w_mom: float = 0.20, w_ml: float = 0.20, w_rev: float = 0.20) -> dict:
    """Run all four alpha models and blend."""
    import warnings
    warnings.filterwarnings("ignore")
    _df = pd.read_json(df_json)

    # Build price history dict for momentum and mean-reversion models
    tickers = _df["Ticker"].values if "Ticker" in _df.columns else _df.index.values
    price_hist = {}
    for t in tickers:
        try:
            ph = fetch_price_history(t, years=2, cache_dir="data/cache")
            if ph is not None and not ph.empty:
                price_hist[t] = ph
        except Exception:
            pass

    sig_rank = factor_rank_model(_df)
    sig_mom = momentum_model(_df, price_history=price_hist if price_hist else None)
    sig_ml = ml_ensemble_model(_df, target_col="Composite")
    sig_rev = mean_reversion_model(_df, price_history=price_hist if price_hist else None)

    custom_weights = {
        "FACTOR_RANK": w_rank,
        "MOMENTUM": w_mom,
        "ML_ENSEMBLE": w_ml,
        "MEAN_REVERSION": w_rev,
    }
    blended = blend_signals([sig_rank, sig_mom, sig_ml, sig_rev], weights=custom_weights)

    all_sigs = [sig_rank, sig_mom, sig_ml, sig_rev]
    return {
        "blended_alpha": blended.alpha.to_dict(),
        "signals": {
            s.name: {"alpha": s.alpha.to_dict(), "coverage": s.coverage, "metadata": s.metadata}
            for s in all_sigs
        },
        "weights": blended.weights_used,
        "corr": blended.correlation_matrix.to_dict(),
        "price_hist_count": len(price_hist),
    }

# Hash for caching
_df_hash = str(len(df)) + "_" + sel_index
_alpha_out = _compute_alpha(_df_hash, df.to_json())
df["Alpha"] = df["Ticker"].map(_alpha_out["blended_alpha"]).fillna(0).round(4)
df["AlphaRank"] = df["Alpha"].rank(ascending=False).astype(int)
blended_alpha = pd.Series(_alpha_out["blended_alpha"])
model_sigs = _alpha_out["signals"]

# ── Top Bar ────────────────────────────────────────────────────
st.markdown(f"""
<div class="bb-top">
    <span class="bb-logo">JVQ</span>
    <span class="bb-tag">JAPAN EQUITIES</span>
    <span class="bb-tag">|</span>
    <span class="bb-tag">{sel_index.upper()}</span>
    <span class="bb-tag">|</span>
    <span class="bb-tag">UNIVERSE {len(df)}</span>
    <span class="bb-tag">|</span>
    <span class="bb-tag">MODEL japan_lsv_v2</span>
    <span class="bb-tag">|</span>
    <span class="bb-tag">REBAL Q</span>
    <span class="bb-tag">|</span>
    <span class="bb-tag">BENCH ^N225</span>
    <span style="margin-left:auto;"></span>
    <span class="bb-live">LIVE</span>
    <span class="bb-tag">{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</span>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="color:{ORANGE}; font-size:12px; letter-spacing:2px; font-weight:600; margin-bottom:12px;">SCREENER</div>', unsafe_allow_html=True)

    preset_memos = {
        "Custom": "Configure manual screening parameters below.",
        "LSV Core": "Leveraged small value: targets small-cap equities with above-median leverage and EV/EBITDA below 8x. Replicates private equity factor exposure in public markets.",
        "PE Replication": "Strict private equity deal profile: P/B below 1.2, P/E below 12, significant leverage. Mirrors the valuation characteristics of top-quartile buyout transactions.",
        "Deleveraging Play": "Identifies companies actively reducing debt loads. Screens for improving balance sheets with ROE above 5%, where debt paydown drives equity value accrual.",
        "Quality Leverage": "High-quality leveraged equities: ROE above 8%, strong operating margins, manageable debt levels. Optimizes for risk-adjusted returns within the leveraged universe.",
        "Deep Value": "Statistically cheap equities trading below book value (P/B below 1.0) with P/E below 10. Targets deep discount valuations with asymmetric return profiles.",
        "High Dividend": "Income-focused screen: minimum 3% dividend yield. Filters for sustainable distributions backed by positive earnings and free cash flow generation.",
    }

    preset = st.selectbox("PRESET", list(preset_memos.keys()))

    st.markdown(f'<div style="color:{GRAY}; font-size:9px; line-height:1.4; margin:-4px 0 8px 0; padding:6px 8px; background:{BG2}; border-left:2px solid {ORANGE};">{preset_memos[preset]}</div>', unsafe_allow_html=True)

    if preset == "LSV Core":
        min_score, max_pe, max_pb, min_div, min_roe = 0.40, 15.0, 1.5, 0.0, 0.0
    elif preset == "PE Replication":
        min_score, max_pe, max_pb, min_div, min_roe = 0.35, 12.0, 1.2, 0.0, 0.0
    elif preset == "Deleveraging Play":
        min_score, max_pe, max_pb, min_div, min_roe = 0.30, 20.0, 2.0, 0.0, 5.0
    elif preset == "Quality Leverage":
        min_score, max_pe, max_pb, min_div, min_roe = 0.45, 20.0, 3.0, 0.0, 8.0
    elif preset == "Deep Value":
        min_score, max_pe, max_pb, min_div, min_roe = 0.35, 10.0, 1.0, 0.0, 0.0
    elif preset == "High Dividend":
        min_score, max_pe, max_pb, min_div, min_roe = 0.30, 30.0, 5.0, 3.0, 0.0
    else:
        min_score = st.slider("MIN COMPOSITE", 0.0, 1.0, 0.30, 0.01, format="%.2f")
        max_pe = st.number_input("MAX P/E", value=50.0, step=5.0)
        max_pb = st.number_input("MAX P/B", value=5.0, step=0.5)
        min_div = st.number_input("MIN DIV YIELD %", value=0.0, step=0.5)
        min_roe = st.number_input("MIN ROE %", value=0.0, step=1.0)

    top_n = st.slider("LIMIT", 5, max(len(df), 10), min(50, len(df)))
    sector_filter = st.multiselect("SECTORS", all_sectors, default=[])
    screen_only = st.checkbox("SCREEN PASS ONLY")
    max_de = st.number_input("MAX D/E", value=500.0, step=50.0)
    net_cash_only = st.checkbox("NET CASH POSITIVE ONLY")
    min_quality_flags = st.slider("MIN QUALITY FLAGS (0-7)", 0, 7, 0)

# ── Filter ─────────────────────────────────────────────────────
flt = df.copy()
flt = flt[flt["Composite"] >= min_score]
if screen_only:
    flt = flt[flt["Screen"] == "PASS"]
if sector_filter:
    flt = flt[flt["Sector"].isin(sector_filter)]
if max_pe < 50:
    flt = flt[(flt["pe_trailing"].isna()) | (flt["pe_trailing"] <= max_pe)]
if max_pb < 5:
    flt = flt[(flt["pb_ratio"].isna()) | (flt["pb_ratio"] <= max_pb)]
if min_div > 0:
    flt = flt[(flt["dividend_yield"].notna()) & (flt["dividend_yield"] >= min_div / 100)]
if min_roe > 0:
    flt = flt[(flt["roe"].notna()) & (flt["roe"] >= min_roe / 100)]
if max_de < 500:
    flt = flt[(flt["debt_to_equity"].isna()) | (flt["debt_to_equity"] <= max_de)]
if net_cash_only:
    flt = flt[flt["net_cash"] > 0]
if min_quality_flags > 0:
    flt = flt[flt["quality_flags"] >= min_quality_flags]
flt = flt.head(top_n)


# ── Summary Strip ──────────────────────────────────────────────
n_screened = len(flt[flt["Screen"] == "PASS"]) if len(flt) else 0
st.markdown(strip_html([
    ("PASSING", f"{len(flt)}/{len(df)}", "dc-orange", "Stocks passing current filter / total universe"),
    ("SCREENED", f"{n_screened}", "dc-green" if n_screened > 0 else "dc-red", "Stocks passing primary + bankruptcy hard screens"),
    ("TOP", f"{flt['Composite'].max():.3f}" if len(flt) else "--", "dc-orange", "Highest composite score in filtered set"),
    ("MED EV/EBITDA", f_num(flt["ev_to_ebitda"].median()), "", "Median Enterprise Value / EBITDA — lower = cheaper"),
    ("MED LTD/EV", f_pct(flt["lt_debt_to_ev"].median()), "", "Median Long-Term Debt / Enterprise Value — leverage intensity"),
    ("MED P/B", f_num(flt["pb_ratio"].median()), "", "Median Price-to-Book ratio — below 1.0 = trading below liquidation value"),
    ("MED ROE", f_pct(flt["roe"].median()), "", "Median Return on Equity — profitability per unit of shareholder equity"),
    ("TOT MCAP", f_jpy(flt["market_cap"].sum()), "", "Total market capitalization of filtered universe"),
    ("SECTORS", str(flt["Sector"].nunique()) if len(flt) else "0", "", "Number of distinct sectors represented"),
    ("AVG QLTY", f_num(flt["quality_flags"].median(), 0), "", "Median quality flags (0-7): profitability, margins, growth, liquidity, FCF"),
]), unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────
t_scr, t_pre, t_idx, t_sec, t_stk, t_val, t_qual, t_risk, t_tech, t_gov, t_port, t_opt, t_bt, t_wl, t_exp, t_comp, t_dash, t_mdl = st.tabs([
    "SCREEN", "PRESETS", "INDEX", "SECTOR", "EQUITY", "VALUATION", "QUALITY", "RISK", "TECHNICALS",
    "GOVERNANCE", "PORTFOLIO", "OPTIMIZE", "BACKTEST", "WATCHLIST", "EXPORT", "COMPARE", "DASHBOARD", "MODEL",
])


# ════════════════════════════════════════════════════════════════
# SCREEN
# ════════════════════════════════════════════════════════════════
with t_scr:
    # Main table
    tcols = ["Ticker", "Name", "Sector", "current_price", "Alpha", "AlphaRank", "Composite", "Screen", "LevValue", "Delever", "Quality", "Momentum",
             "pe_trailing", "pb_ratio", "ev_to_ebitda", "ebitda_to_ev", "lt_debt_to_ev",
             "dividend_yield", "roe", "debt_to_equity", "fcf_yield", "beta", "mcap_b", "quality_flags"]
    tdf = flt[tcols].copy()
    tdf.columns = ["TICKER", "NAME", "SECTOR", "PRICE", "ALPHA", "A#", "COMP", "SCR", "LEV", "DLV", "QUAL", "MOM",
                    "P/E", "P/B", "EV/EBITDA", "EBITDA/EV", "LTD/EV",
                    "DIV%", "ROE%", "D/E", "FCF_Y", "BETA", "MCAP_B", "QF"]
    tdf["PRICE"] = tdf["PRICE"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "--")
    for c in ["EBITDA/EV", "LTD/EV", "DIV%", "ROE%", "FCF_Y"]:
        tdf[c] = tdf[c].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
    tdf["MCAP_B"] = tdf["MCAP_B"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "--")

    col_tooltips = {
        "TICKER": "Tokyo Stock Exchange ticker symbol",
        "NAME": "Company name",
        "SECTOR": "GICS sector classification",
        "PRICE": "Last traded price (JPY)",
        "ALPHA": "Model alpha — blended signal from Factor Rank (50%), Momentum (25%), ML Ensemble (25%). Z-scored, higher = more attractive",
        "A#": "Alpha rank in universe (1 = highest alpha)",
        "COMP": "Composite score — weighted blend of all factor groups (0-1)",
        "SCR": "Screen status — PASS means stock meets all primary + bankruptcy hard filters",
        "LEV": "Leverage-Value score — EBITDA/EV, EV/EBITDA, LT Debt/EV, P/B weighted",
        "DLV": "Deleveraging score — debt paydown, asset turnover, asset growth, capital efficiency",
        "QUAL": "Quality score — gross profit/assets, operating margin, ROE, balance sheet",
        "MOM": "Momentum score — 52-week relative strength, distance from high, SMA cross",
        "P/E": "Price-to-Earnings (trailing 12 months) — lower = cheaper",
        "P/B": "Price-to-Book — below 1.0 means trading below liquidation value",
        "EV/EBITDA": "Enterprise Value / EBITDA — core valuation metric, lower = cheaper",
        "EBITDA/EV": "EBITDA / Enterprise Value (%) — earnings yield on the whole firm",
        "LTD/EV": "Long-Term Debt / Enterprise Value (%) — leverage intensity",
        "DIV%": "Dividend yield (%)",
        "ROE%": "Return on Equity (%)",
        "D/E": "Debt-to-Equity ratio",
        "FCF_Y": "Free Cash Flow Yield (%) — FCF / Market Cap",
        "BETA": "Beta vs market — volatility relative to benchmark",
        "MCAP_B": "Market capitalization in billions JPY",
        "QF": "Quality flags (0-7): ROE>8%, margin>5%, rev growth, earn growth, CR>1, net cash, FCF>3%",
    }
    col_cfg = {col: st.column_config.Column(help=tip) for col, tip in col_tooltips.items()}
    st.dataframe(tdf, use_container_width=True, height=700, column_config=col_cfg)

    # Distribution + stacked bars side by side
    dl, dr = st.columns(2)
    with dl:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["Composite"], nbinsx=30, marker_color=GRAY_DIM, marker_line_color=BORDER, name="ALL"))
        fig.add_trace(go.Histogram(x=flt["Composite"], nbinsx=30, marker_color=ORANGE_DIM, marker_line_color=ORANGE, name="PASS"))
        fig.update_layout(**chart_layout(250, barmode="overlay", showlegend=True), title="COMPOSITE DISTRIBUTION")
        fig.update_layout(legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    with dr:
        fig = go.Figure()
        for col, clr, lbl in [("LevValue", ORANGE, "LEV"), ("Delever", GREEN, "DLV"), ("Quality", YELLOW, "QUAL"), ("Momentum", "#06b6d4", "MOM")]:
            fig.add_trace(go.Bar(x=flt["Ticker"].head(20), y=flt[col].head(20), name=lbl, marker_color=clr, marker_line_width=0))
        fig.update_layout(**chart_layout(250, barmode="stack", showlegend=True), title="SCORE DECOMPOSITION (TOP 20)")
        fig.update_layout(legend=dict(orientation="h", y=1.12), xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PRESETS — strategy documentation
# ════════════════════════════════════════════════════════════════
with t_pre:
    def preset_card(name, params, thesis, factors, ideal, risks, color=ORANGE):
        min_s, max_pe, max_pb, min_d, min_r = params
        param_str = f"MIN COMP: {min_s} | MAX P/E: {max_pe} | MAX P/B: {max_pb} | MIN DIV: {min_d}% | MIN ROE: {min_r}%"
        # Count how many stocks match this preset
        p = df.copy()
        p = p[p["Composite"] >= min_s]
        if max_pe < 50: p = p[(p["pe_trailing"].isna()) | (p["pe_trailing"] <= max_pe)]
        if max_pb < 5: p = p[(p["pb_ratio"].isna()) | (p["pb_ratio"] <= max_pb)]
        if min_d > 0: p = p[(p["dividend_yield"].notna()) & (p["dividend_yield"] >= min_d / 100)]
        if min_r > 0: p = p[(p["roe"].notna()) & (p["roe"] >= min_r / 100)]
        n_match = len(p)
        avg_comp = f"{p['Composite'].mean():.3f}" if n_match > 0 else "--"
        return f"""
        <div style="background:{BG2}; border:1px solid {BORDER}; border-left:3px solid {color}; padding:16px 20px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px;">
                <span style="color:{color}; font-size:13px; font-weight:600; letter-spacing:2px;">{name}</span>
                <span style="color:{GRAY}; font-size:9px;">{n_match} MATCHES | AVG COMP: {avg_comp}</span>
            </div>
            <div style="color:{GRAY_DIM}; font-size:8px; letter-spacing:0.5px; margin-bottom:10px; font-family:{FONT};">{param_str}</div>
            <div style="color:{WHITE}; font-size:10px; line-height:1.6; margin-bottom:12px;">{thesis}</div>
            <div style="margin-bottom:10px;">
                <div style="color:{GRAY}; font-size:8px; letter-spacing:1px; margin-bottom:4px;">KEY FACTORS</div>
                <div style="color:{GRAY}; font-size:9px; line-height:1.5;">{factors}</div>
            </div>
            <div style="margin-bottom:10px;">
                <div style="color:{GRAY}; font-size:8px; letter-spacing:1px; margin-bottom:4px;">IDEAL CANDIDATE</div>
                <div style="color:{GRAY}; font-size:9px; line-height:1.5;">{ideal}</div>
            </div>
            <div>
                <div style="color:{GRAY}; font-size:8px; letter-spacing:1px; margin-bottom:4px;">RISK CONSIDERATIONS</div>
                <div style="color:{GRAY}; font-size:9px; line-height:1.5;">{risks}</div>
            </div>
        </div>
        """

    st.markdown(f'<div style="color:{GRAY}; font-size:9px; margin-bottom:12px;">Each preset applies a distinct investment thesis to the Japan equity universe. Parameters are applied as hard filters on the sidebar. Select a preset in the sidebar to activate it, or use Custom mode for manual screening.</div>', unsafe_allow_html=True)

    st.markdown(preset_card(
        "LSV CORE",
        (0.40, 15.0, 1.5, 0.0, 0.0),
        "The foundational strategy. Targets small-capitalization Japanese equities with meaningful financial leverage trading at low enterprise valuations. The thesis is rooted in academic research showing that the combination of size, leverage, and value explains the majority of private equity's historical outperformance — and that these same factors can be captured in public markets without the fee drag, illiquidity, or J-curve of PE fund structures.",
        "EV/EBITDA below 8x (enterprise cheapness) | LT Debt/EV above 10% (leverage intensity) | Market cap 30B-300B JPY (small-cap sweet spot) | EBITDA/EV as primary earnings yield | P/B emphasis for Japan market context",
        "Small-cap industrial or manufacturing company. Trades at 5-7x EV/EBITDA with 20-30% LT Debt/EV. Generating positive free cash flow with stable margins. Below the radar of large institutional coverage. Benefits from yen weakness or domestic demand recovery.",
        "Small-cap liquidity constraints — wider bid-ask spreads and potential difficulty exiting positions during stress. Leverage amplifies both upside and downside. Sector concentration risk if the screen over-indexes on industrials or materials.",
        ORANGE,
    ), unsafe_allow_html=True)

    st.markdown(preset_card(
        "PE REPLICATION",
        (0.35, 12.0, 1.2, 0.0, 0.0),
        "Strict mimicry of private equity deal characteristics. The average PE buyout targets companies at approximately $180M market cap, with net debt/EBITDA of 3-4x, purchased at 7-8x EV/EBITDA. This preset identifies public equities that match that exact profile — effectively replicating the PE portfolio without the 2-and-20 fee structure. Historical backtests of this approach in U.S. markets have shown returns comparable to top-quartile PE funds.",
        "P/B below 1.2 (deep asset discount) | P/E below 12 (earnings cheapness) | Composite score above 0.35 (baseline quality) | High leverage factor weight | Emphasis on EBITDA/EV and LT Debt/EV as the primary sorting metrics",
        "Leveraged manufacturer or capital-intensive business. Trades at or below book value with single-digit P/E. Carries meaningful but serviceable debt. Generates enough cash flow to cover interest expense with margin for debt reduction. Exactly the kind of company a PE firm would buy, strip costs, and relist at 12-15x.",
        "Value trap risk is elevated — extremely cheap stocks can stay cheap. Bankruptcy risk is real if debt service becomes unmanageable during cyclical downturns. The strategy requires discipline to hold through periods of underperformance and rebalance quarterly.",
        "#06b6d4",
    ), unsafe_allow_html=True)

    st.markdown(preset_card(
        "DELEVERAGING PLAY",
        (0.30, 20.0, 2.0, 0.0, 5.0),
        "Targets the deleveraging flywheel — the single most statistically significant predictor of forward returns within leveraged equities. When a company pays down debt, interest expense falls, free cash flow rises, credit risk decreases, and the equity absorbs a disproportionate share of enterprise value growth. This is the core mechanism through which leverage creates equity alpha: not through risk-taking, but through the systematic reduction of that risk over time.",
        "Debt paydown signal (declining LT debt YoY) | ROE above 5% (minimum profitability threshold) | Improving asset turnover | Asset growth as secondary confirmation | Gross profit/assets for capital efficiency",
        "Mid-cap company that took on debt 2-3 years ago for an acquisition or capex cycle and is now actively paying it down. ROE is recovering as interest burden falls. Free cash flow is inflecting upward. The balance sheet is visibly improving quarter over quarter. Management has stated a commitment to deleveraging in earnings calls.",
        "Timing risk — the stock may have already re-rated if the market has priced in the deleveraging. Requires monitoring of quarterly balance sheet data to confirm the thesis is playing out. If revenue declines, the deleveraging flywheel can reverse (rising debt/EBITDA even with constant debt).",
        GREEN,
    ), unsafe_allow_html=True)

    st.markdown(preset_card(
        "QUALITY LEVERAGE",
        (0.45, 20.0, 3.0, 0.0, 8.0),
        "Combines leverage exposure with quality filters to optimize risk-adjusted returns. Pure leveraged value can be volatile — this preset adds profitability and margin requirements to screen out the weakest balance sheets while retaining the leverage premium. The ROE floor of 8% aligns with the TSE's governance reform target, meaning these companies are already meeting regulatory expectations for capital efficiency.",
        "ROE above 8% (TSE governance threshold) | Operating margin stability | Gross profit/assets as quality factor | Moderate leverage (not extreme) | Current ratio for liquidity confirmation",
        "Well-managed industrial company with above-average profitability. Uses leverage efficiently to enhance equity returns. Generates consistent free cash flow. Has a clear capital allocation strategy (dividends, buybacks, or reinvestment). Sector leadership or niche market position that protects margins.",
        "Higher composite threshold (0.45) reduces the investable universe significantly. May miss the deepest value opportunities where the asymmetry is greatest. Quality screens can inadvertently select for mature, low-growth businesses that have limited re-rating potential.",
        YELLOW,
    ), unsafe_allow_html=True)

    st.markdown(preset_card(
        "DEEP VALUE",
        (0.35, 10.0, 1.0, 0.0, 0.0),
        "Targets statistically cheap equities trading below book value with single-digit earnings multiples. In the Japanese market, where approximately 40% of listed companies still trade below P/B 1.0, this screen identifies the most extreme valuation dislocations. These companies often have significant hidden asset value — real estate carried at historical cost, cross-shareholdings not reflected in market price, or excess cash that substantially exceeds the equity market capitalization.",
        "P/B below 1.0 (below liquidation value) | P/E below 10 (extreme earnings cheapness) | No leverage or profitability requirement | Pure valuation focus | Cash-to-market-cap ratio as supplementary factor",
        "Regional bank or mature industrial trading at 0.5x book with 7x P/E. Holds real estate on the balance sheet at 1970s acquisition cost. Net cash positive. Minimal analyst coverage. The gap between intrinsic asset value and market price exceeds 50%. A catalyst — governance reform, activist involvement, or management buyout — could close the discount.",
        "Value traps are the primary risk. Without quality or momentum filters, the screen will include companies that are cheap for structural reasons (declining industries, poor management, shrinking end markets). Requires position-level due diligence to separate genuine deep value from permanent capital impairment.",
        RED,
    ), unsafe_allow_html=True)

    st.markdown(preset_card(
        "HIGH DIVIDEND",
        (0.30, 30.0, 5.0, 3.0, 0.0),
        "Income-focused strategy targeting sustainable above-market dividend yields. In Japan's near-zero interest rate environment, equities yielding 3%+ represent a significant income premium. This preset filters for companies where the dividend is backed by real earnings and cash flow generation, reducing the risk of yield traps where unsustainable payouts mask deteriorating fundamentals. Japan's ongoing governance reform is driving increased payout ratios across the market.",
        "Dividend yield above 3% (income floor) | Positive free cash flow (payout sustainability) | Profitability requirement via bankruptcy screen | Broad P/E allowance (up to 30) | No leverage requirement — can include both leveraged and unleveraged names",
        "Mature utility, telecom, or financial with stable earnings and a long dividend track record. Payout ratio between 30-60% — high enough to be attractive, low enough to be sustainable. Minimal capex requirements. Predictable revenue streams. Management committed to shareholder returns under TSE governance pressure.",
        "Dividend cuts are the primary risk — a yield of 5% means nothing if the company reduces or eliminates the payout. Sector concentration in utilities, telecoms, and financials can introduce correlation risk. Rising interest rates (if they occur in Japan) could reduce the relative attractiveness of dividend equities.",
        "#a855f7",
    ), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# INDEX
# ════════════════════════════════════════════════════════════════
with t_idx:
    il, ir = st.columns([1.3, 0.7])

    with il:
        # Valuation map
        fig = go.Figure()
        for sec in all_sectors:
            s = df[df["Sector"] == sec]
            fig.add_trace(go.Scatter(
                x=s["pe_trailing"], y=s["pb_ratio"], mode="markers",
                name=sec, marker=dict(size=7, line=dict(width=0.3, color=BG)),
                text=s["Ticker"], hovertemplate="%{text}<br>P/E: %{x:.1f}<br>P/B: %{y:.2f}",
            ))
        fig.add_hline(y=1.0, line_dash="dot", line_color=GRAY_DIM, line_width=1)
        fig.add_vline(x=15.0, line_dash="dot", line_color=GRAY_DIM, line_width=1)
        fig.update_layout(**chart_layout(450, showlegend=True,
                          legend=dict(font=dict(size=8), orientation="v", x=1.02, bgcolor="rgba(0,0,0,0)")),
                          title="VALUATION MAP", xaxis_title="P/E", yaxis_title="P/B")
        st.plotly_chart(fig, use_container_width=True)

    with ir:
        # Universe stats
        stats = []
        for metric, col, fmt, dec in [
            ("Composite", "Composite", "n", 3), ("P/E", "pe_trailing", "n", 1),
            ("P/B", "pb_ratio", "n", 2), ("EV/EBITDA", "ev_to_ebitda", "n", 1),
            ("Div Yield", "dividend_yield", "p", 1), ("ROE", "roe", "p", 1),
            ("Op Margin", "operating_margin", "p", 1), ("FCF Yield", "fcf_yield", "p", 1),
            ("D/E", "debt_to_equity", "n", 0), ("Beta", "beta", "n", 2),
            ("MCap (B)", "mcap_b", "n", 0), ("Cash/MCap", "cash_to_mcap", "p", 1),
        ]:
            s = df[col].dropna()
            fn = f_pct if fmt == "p" else lambda v, d=dec: f_num(v, d)
            stats.append({
                "": metric,
                "MEAN": fn(s.mean()),
                "MED": fn(s.median()),
                "P10": fn(s.quantile(0.1)),
                "P90": fn(s.quantile(0.9)),
            })
        st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True, height=450)

    # Sector table
    sec_agg = df.groupby("Sector").agg(
        N=("Ticker", "size"),
        Comp=("Composite", "mean"),
        PB=("pb_ratio", "median"),
        PE=("pe_trailing", "median"),
        Div=("dividend_yield", "median"),
        ROE=("roe", "median"),
        MCap=("mcap_b", "sum"),
        Beta=("beta", "median"),
        QF=("quality_flags", "median"),
    ).reset_index().sort_values("MCap", ascending=False)
    sec_agg["Div"] = sec_agg["Div"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "--")
    sec_agg["ROE"] = sec_agg["ROE"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "--")
    sec_agg["MCap"] = sec_agg["MCap"].apply(lambda x: f"{x:,.0f}")
    sec_agg.columns = ["SECTOR", "N", "AVG COMP", "MED P/B", "MED P/E", "MED DIV", "MED ROE", "TOT MCAP(B)", "MED BETA", "MED QF"]
    st.dataframe(sec_agg, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# SECTOR
# ════════════════════════════════════════════════════════════════
with t_sec:
    sel_sec = st.selectbox("SECTOR", ["ALL"] + all_sectors, key="sec_sel")
    sdf = flt if sel_sec == "ALL" else flt[flt["Sector"] == sel_sec]

    if len(sdf) == 0:
        st.warning("No stocks match.")
    else:
        st.markdown(strip_html([
            ("N", str(len(sdf)), "dc-orange", "Number of stocks in this sector"),
            ("AVG COMP", f_num(sdf["Composite"].mean(), 3), "", "Average composite score for this sector"),
            ("MED P/B", f_num(sdf["pb_ratio"].median()), "", "Median Price-to-Book ratio"),
            ("MED P/E", f_num(sdf["pe_trailing"].median()), "", "Median Price-to-Earnings (trailing)"),
            ("MED DIV", f_pct(sdf["dividend_yield"].median()), "", "Median dividend yield"),
            ("MED ROE", f_pct(sdf["roe"].median()), "", "Median Return on Equity"),
            ("TOT MCAP", f_jpy(sdf["market_cap"].sum()), "", "Total market cap of sector"),
        ]), unsafe_allow_html=True)

        sl, sr = st.columns(2)
        with sl:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sdf["Ticker"], y=sdf["Composite"], marker_color=ORANGE, marker_line_width=0))
            fig.update_layout(**chart_layout(300, xaxis=dict(tickangle=-45, tickfont=dict(size=8))),
                              title=f"COMPOSITE: {sel_sec}")
            st.plotly_chart(fig, use_container_width=True)

        with sr:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sdf["pb_ratio"], y=sdf["dividend_yield"].fillna(0) * 100,
                mode="markers+text", text=sdf["Ticker"], textposition="top center",
                textfont=dict(size=8, color=GRAY),
                marker=dict(size=np.clip(sdf["mcap_b"].fillna(1).apply(np.log) * 4, 4, 25),
                            color=ORANGE, line=dict(width=0.5, color=BG)),
            ))
            fig.update_layout(**chart_layout(300), title=f"P/B vs DIV YIELD: {sel_sec}",
                              xaxis_title="P/B", yaxis_title="DIV %")
            st.plotly_chart(fig, use_container_width=True)

        # Sector detail table
        stbl = sdf[["Ticker", "Name", "Composite", "Screen", "LevValue", "Delever", "Quality", "Momentum",
                     "pe_trailing", "pb_ratio", "ev_to_ebitda", "ebitda_to_ev", "lt_debt_to_ev",
                     "dividend_yield", "roe", "debt_to_equity", "fcf_yield", "beta", "mcap_b", "quality_flags"]].copy()
        stbl.columns = ["TICKER", "NAME", "COMP", "SCR", "LEV", "DLV", "QUAL", "MOM",
                        "P/E", "P/B", "EV/EB", "EBITDA/EV", "LTD/EV",
                        "DIV%", "ROE%", "D/E", "FCF_Y", "BETA", "MCAP_B", "QF"]
        for c in ["EBITDA/EV", "LTD/EV", "DIV%", "ROE%", "FCF_Y"]:
            stbl[c] = stbl[c].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
        st.dataframe(stbl, use_container_width=True, height=400)


# ════════════════════════════════════════════════════════════════
# EQUITY (single stock deep dive)
# ════════════════════════════════════════════════════════════════
with t_stk:
    stk_list = flt["Ticker"].tolist()
    if not stk_list:
        st.warning("No stocks match.")
    else:
        sel_stk = st.selectbox("EQUITY", stk_list,
                               format_func=lambda t: f"{t}  {flt[flt['Ticker']==t]['Name'].values[0]}",
                               key="stk_sel")
        r = flt[flt["Ticker"] == sel_stk].iloc[0]

        # Header strip
        st.markdown(f"""
        <div class="eq-header">
            <span style="color:{ORANGE}; font-size:16px; font-weight:600;">{sel_stk}</span>
            <span style="color:{WHITE}; font-size:13px;">{r['Name']}</span>
            <span style="color:{GRAY}; font-size:10px;">{r['Sector']} / {r.get('industry') or '--'}</span>
            <span style="margin-left:auto;"></span>
            <span style="color:{WHITE}; font-size:16px; font-weight:600;">{f_price(r.get('current_price'))}</span>
            <span style="color:{GRAY}; font-size:10px;">JPY</span>
        </div>
        """, unsafe_allow_html=True)

        # Score strip
        scr_flag = r.get("Screen", "FAIL")
        st.markdown(strip_html([
            ("COMPOSITE", f_num(r["Composite"], 3), "dc-orange", "Weighted composite of all factor scores"),
            ("SCREEN", scr_flag, "dc-green" if scr_flag == "PASS" else "dc-red", "PASS = meets size, leverage, valuation, and bankruptcy screens"),
            ("LEV/VALUE", f_num(r["LevValue"], 3), "", "Leverage-Value score: EBITDA/EV, EV/EBITDA, LT Debt/EV, P/B"),
            ("DELEVER", f_num(r["Delever"], 3), "", "Deleveraging score: debt paydown, asset turnover, gross profit efficiency"),
            ("QUALITY", f_num(r["Quality"], 3), "", "Quality score: gross profit/assets, operating margin, ROE"),
            ("MOMENTUM", f_num(r["Momentum"], 3), "", "Momentum score: 52-week relative strength, SMA cross signal"),
            ("QUALITY FLAGS", f_num(r["quality_flags"], 0), "dc-orange" if r["quality_flags"] >= 5 else "", "Count of 7 quality checks: ROE>8%, margin>5%, rev growth, earnings growth, CR>1, net cash, FCF yield>3%"),
            ("RANK", f"#{flt.index.get_loc(r.name)+1}/{len(flt)}", "dc-orange", "Position in filtered universe by composite score"),
        ]), unsafe_allow_html=True)

        eq_l, eq_r = st.columns([1.4, 0.6])

        with eq_l:
            # Price chart
            prices = load_prices(sel_stk, years=3)
            if not prices.empty:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.02)

                if isinstance(prices.columns, pd.MultiIndex):
                    close = prices[("Close", sel_stk)] if ("Close", sel_stk) in prices.columns else prices.iloc[:, 0]
                    vol_col = ("Volume", sel_stk) if ("Volume", sel_stk) in prices.columns else None
                else:
                    close = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
                    vol_col = "Volume" if "Volume" in prices.columns else None

                sma50 = close.rolling(50).mean()
                sma200 = close.rolling(200).mean()

                fig.add_trace(go.Scatter(x=prices.index, y=close, mode="lines",
                    line=dict(color=WHITE, width=1), name="CLOSE"), row=1, col=1)
                fig.add_trace(go.Scatter(x=prices.index, y=sma50, mode="lines",
                    line=dict(color=ORANGE, width=1, dash="dot"), name="SMA50"), row=1, col=1)
                fig.add_trace(go.Scatter(x=prices.index, y=sma200, mode="lines",
                    line=dict(color=GREEN, width=1, dash="dot"), name="SMA200"), row=1, col=1)

                if vol_col and vol_col in prices.columns:
                    fig.add_trace(go.Bar(x=prices.index, y=prices[vol_col],
                        marker_color=GRAY_DIM, marker_line_width=0, name="VOL"), row=2, col=1)

                fig.update_layout(**chart_layout(380, showlegend=True), title=f"{sel_stk} 3Y")
                fig.update_layout(legend=dict(orientation="h", y=1.06))
                fig.update_yaxes(title_text="JPY", row=1, col=1, gridcolor=BORDER)
                fig.update_yaxes(title_text="VOL", row=2, col=1, gridcolor=BORDER)
                fig.update_xaxes(gridcolor=BORDER)
                st.plotly_chart(fig, use_container_width=True)

        with eq_r:
            # Radar
            vals = [r["LevValue"], r["Delever"], r["Quality"], r["Momentum"], r["LevValue"]]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=["LEV/VAL", "DELEVER", "QUAL", "MOM", "LEV/VAL"],
                fill="toself", fillcolor=f"rgba(255,140,0,0.1)", line=dict(color=ORANGE, width=2),
            ))
            fig.update_layout(**chart_layout(220),
                polar=dict(radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=8), gridcolor=BORDER),
                           angularaxis=dict(tickfont=dict(size=9, color=WHITE), gridcolor=BORDER),
                           bgcolor=BG1),
                title="FACTOR PROFILE")
            st.plotly_chart(fig, use_container_width=True)

        # Fundamentals grid
        fl, fm, fr = st.columns(3)

        with fl:
            st.markdown("### VALUATION")
            vdata = [
                ("P/E Trailing", f_num(r.get("pe_trailing")), pct_rank(df["pe_trailing"], r.get("pe_trailing"))),
                ("P/E Forward", f_num(r.get("pe_forward")), pct_rank(df["pe_forward"], r.get("pe_forward"))),
                ("P/B", f_num(r.get("pb_ratio")), pct_rank(df["pb_ratio"], r.get("pb_ratio"))),
                ("P/S", f_num(r.get("price_to_sales")), pct_rank(df["price_to_sales"], r.get("price_to_sales"))),
                ("EV/EBITDA", f_num(r.get("ev_to_ebitda")), pct_rank(df["ev_to_ebitda"], r.get("ev_to_ebitda"))),
                ("EV/FCF", f_num(r.get("ev_to_fcf")), "--"),
                ("Div Yield", f_pct(r.get("dividend_yield")), pct_rank(df["dividend_yield"], r.get("dividend_yield"))),
                ("FCF Yield", f_pct(r.get("fcf_yield")), pct_rank(df["fcf_yield"], r.get("fcf_yield"))),
                ("Cash/MCap", f_pct(r.get("cash_to_mcap")), pct_rank(df["cash_to_mcap"], r.get("cash_to_mcap"))),
            ]
            st.dataframe(pd.DataFrame(vdata, columns=["METRIC", "VALUE", "PCTLE"]),
                         use_container_width=True, hide_index=True, height=350)

        with fm:
            st.markdown("### QUALITY")
            qdata = [
                ("ROE", f_pct(r.get("roe")), pct_rank(df["roe"], r.get("roe"))),
                ("ROA", f_pct(r.get("roa")), pct_rank(df["roa"], r.get("roa"))),
                ("Op Margin", f_pct(r.get("operating_margin")), pct_rank(df["operating_margin"], r.get("operating_margin"))),
                ("Net Margin", f_pct(r.get("profit_margin")), pct_rank(df["profit_margin"], r.get("profit_margin"))),
                ("Rev Growth", f_pct(r.get("revenue_growth")), pct_rank(df["revenue_growth"], r.get("revenue_growth"))),
                ("Earn Growth", f_pct(r.get("earnings_growth")), pct_rank(df["earnings_growth"], r.get("earnings_growth"))),
                ("Quality Flags", f_num(r.get("quality_flags"), 0), f"{r['quality_flags']}/7"),
            ]
            st.dataframe(pd.DataFrame(qdata, columns=["METRIC", "VALUE", "PCTLE"]),
                         use_container_width=True, hide_index=True, height=350)

        with fr:
            st.markdown("### LEVERAGE & BALANCE SHEET")
            bdata = [
                ("Market Cap", f_jpy(r.get("market_cap")), pct_rank(df["market_cap"], r.get("market_cap"))),
                ("Enterprise Val", f_jpy(r.get("enterprise_value")), "--"),
                ("LT Debt/EV", f_pct(r.get("lt_debt_to_ev")), pct_rank(df["lt_debt_to_ev"], r.get("lt_debt_to_ev"))),
                ("Net Debt/EBITDA", f_num(r.get("net_debt_to_ebitda")), pct_rank(df["net_debt_to_ebitda"], r.get("net_debt_to_ebitda"))),
                ("EBITDA/EV", f_pct(r.get("ebitda_to_ev")), pct_rank(df["ebitda_to_ev"], r.get("ebitda_to_ev"))),
                ("GrProfit/Assets", f_pct(r.get("gross_profit_to_assets")), pct_rank(df["gross_profit_to_assets"], r.get("gross_profit_to_assets"))),
                ("Asset Turnover", f_num(r.get("asset_turnover")), pct_rank(df["asset_turnover"], r.get("asset_turnover"))),
                ("D/E", f_num(r.get("debt_to_equity"), 0), pct_rank(df["debt_to_equity"], r.get("debt_to_equity"))),
                ("Current Ratio", f_num(r.get("current_ratio")), pct_rank(df["current_ratio"], r.get("current_ratio"))),
                ("Net Cash", f_jpy(r.get("net_cash")), "dc-green" if r.get("net_cash", 0) > 0 else "dc-red"),
                ("Beta", f_num(r.get("beta")), pct_rank(df["beta"], r.get("beta"))),
            ]
            st.dataframe(pd.DataFrame(bdata, columns=["METRIC", "VALUE", "PCTLE"]),
                         use_container_width=True, hide_index=True, height=420)

        # 52W range
        if pd.notna(r.get("fifty_two_week_low")) and pd.notna(r.get("fifty_two_week_high")):
            lo, hi, pr = r["fifty_two_week_low"], r["fifty_two_week_high"], r.get("current_price", 0)
            pct = (pr - lo) / (hi - lo) * 100 if hi != lo else 50
            st.markdown(strip_html([
                ("52W LOW", f_price(lo), "dc-red", "52-week low price"),
                ("CURRENT", f_price(pr), "dc-orange", "Last traded price"),
                ("52W HIGH", f_price(hi), "dc-green", "52-week high price"),
                ("POSITION", f"{pct:.0f}%", "dc-orange" if pct > 30 else "dc-red", "Current price position within 52-week range (0%=low, 100%=high)"),
                ("SMA50", f_price(r.get("fifty_day_avg")), "", "50-day simple moving average"),
                ("SMA200", f_price(r.get("two_hundred_day_avg")), "", "200-day simple moving average"),
                ("SMA CROSS", f"{r.get('sma_cross', 0)*100:.1f}%" if pd.notna(r.get("sma_cross")) else "--",
                 "dc-green" if r.get("sma_cross", 0) > 0 else "dc-red", "SMA50/SMA200 ratio minus 1 — positive = golden cross, negative = death cross"),
                ("AVG VOL", f"{r.get('avg_volume', 0):,.0f}" if pd.notna(r.get("avg_volume")) else "--", "", "Average daily trading volume (shares)"),
            ]), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# VALUATION
# ════════════════════════════════════════════════════════════════
with t_val:
    vl, vr = st.columns(2)

    with vl:
        # P/B ranking
        pb_sorted = flt[["Ticker", "pb_ratio"]].dropna(subset=["pb_ratio"]).sort_values("pb_ratio")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pb_sorted["Ticker"], y=pb_sorted["pb_ratio"],
            marker_color=np.where(pb_sorted["pb_ratio"] < 1.0, GREEN, np.where(pb_sorted["pb_ratio"] < 1.5, ORANGE, RED)),
            marker_line_width=0,
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color=GRAY, annotation_text="BOOK VALUE", annotation_font=dict(size=9, color=GRAY))
        fig.update_layout(**chart_layout(350, xaxis=dict(tickangle=-45, tickfont=dict(size=8))), title="P/B RATIO (SORTED)")
        st.plotly_chart(fig, use_container_width=True)

    with vr:
        # Dividend yield ranking
        div_sorted = flt[["Ticker", "dividend_yield"]].dropna(subset=["dividend_yield"]).sort_values("dividend_yield", ascending=False)
        div_sorted["div_pct"] = div_sorted["dividend_yield"] * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=div_sorted["Ticker"], y=div_sorted["div_pct"],
            marker_color=np.where(div_sorted["div_pct"] >= 3.0, GREEN, np.where(div_sorted["div_pct"] >= 2.0, ORANGE, GRAY)),
            marker_line_width=0,
        ))
        fig.add_hline(y=2.0, line_dash="dash", line_color=GRAY, annotation_text="2%", annotation_font=dict(size=9, color=GRAY))
        fig.update_layout(**chart_layout(350, xaxis=dict(tickangle=-45, tickfont=dict(size=8))), title="DIVIDEND YIELD % (SORTED)")
        st.plotly_chart(fig, use_container_width=True)

    # EV/EBITDA vs FCF Yield
    fig = go.Figure()
    ev_data = flt.dropna(subset=["ev_to_ebitda", "fcf_yield"])
    fig.add_trace(go.Scatter(
        x=ev_data["ev_to_ebitda"], y=ev_data["fcf_yield"] * 100,
        mode="markers+text", text=ev_data["Ticker"], textposition="top center",
        textfont=dict(size=8, color=GRAY),
        marker=dict(size=9, color=ORANGE, line=dict(width=0.5, color=BG)),
    ))
    fig.add_hline(y=5.0, line_dash="dot", line_color=GRAY_DIM)
    fig.add_vline(x=10.0, line_dash="dot", line_color=GRAY_DIM)
    fig.update_layout(**chart_layout(380), title="EV/EBITDA vs FCF YIELD",
                      xaxis_title="EV/EBITDA", yaxis_title="FCF YIELD %")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# QUALITY
# ════════════════════════════════════════════════════════════════
with t_qual:
    ql, qr = st.columns(2)

    with ql:
        # ROE ranking
        roe_sorted = flt[["Ticker", "roe"]].dropna(subset=["roe"]).sort_values("roe", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=roe_sorted["Ticker"], y=roe_sorted["roe"] * 100,
            marker_color=np.where(roe_sorted["roe"] >= 0.10, GREEN, np.where(roe_sorted["roe"] >= 0.05, ORANGE, RED)),
            marker_line_width=0,
        ))
        fig.add_hline(y=8.0, line_dash="dash", line_color=GRAY, annotation_text="8% MIN", annotation_font=dict(size=9, color=GRAY))
        fig.update_layout(**chart_layout(300, xaxis=dict(tickangle=-45, tickfont=dict(size=8))), title="ROE % (SORTED)")
        st.plotly_chart(fig, use_container_width=True)

    with qr:
        # Quality flags distribution
        qf_counts = flt["quality_flags"].value_counts().sort_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=qf_counts.index, y=qf_counts.values,
            marker_color=[RED if x <= 2 else YELLOW if x <= 4 else GREEN for x in qf_counts.index],
            marker_line_width=0,
        ))
        fig.update_layout(**chart_layout(300), title="QUALITY FLAGS DISTRIBUTION (0-7)",
                          xaxis_title="FLAGS PASSED", yaxis_title="COUNT")
        st.plotly_chart(fig, use_container_width=True)

    # ROE vs Operating Margin
    fig = go.Figure()
    rm_data = flt.dropna(subset=["roe", "operating_margin"])
    fig.add_trace(go.Scatter(
        x=rm_data["roe"] * 100, y=rm_data["operating_margin"] * 100,
        mode="markers+text", text=rm_data["Ticker"], textposition="top center",
        textfont=dict(size=8, color=GRAY),
        marker=dict(size=np.clip(rm_data["mcap_b"].fillna(1).apply(np.log) * 4, 4, 20),
                    color=ORANGE, line=dict(width=0.5, color=BG)),
    ))
    fig.add_hline(y=5.0, line_dash="dot", line_color=GRAY_DIM)
    fig.add_vline(x=8.0, line_dash="dot", line_color=GRAY_DIM)
    fig.update_layout(**chart_layout(350), title="ROE vs OPERATING MARGIN",
                      xaxis_title="ROE %", yaxis_title="OP MARGIN %")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# RISK
# ════════════════════════════════════════════════════════════════
with t_risk:
    rl, rr = st.columns(2)

    with rl:
        # Beta
        beta_sorted = flt[["Ticker", "beta"]].dropna(subset=["beta"]).sort_values("beta")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=beta_sorted["Ticker"], y=beta_sorted["beta"],
            marker_color=np.where(beta_sorted["beta"] > 1.0, RED, GREEN),
            marker_line_width=0,
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color=GRAY)
        fig.update_layout(**chart_layout(300, xaxis=dict(tickangle=-45, tickfont=dict(size=8))), title="BETA (SORTED)")
        st.plotly_chart(fig, use_container_width=True)

    with rr:
        # Net cash
        nc_sorted = flt[["Ticker", "net_cash_b"]].sort_values("net_cash_b", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=nc_sorted["Ticker"], y=nc_sorted["net_cash_b"],
            marker_color=np.where(nc_sorted["net_cash_b"] >= 0, GREEN, RED),
            marker_line_width=0,
        ))
        fig.update_layout(**chart_layout(300, xaxis=dict(tickangle=-45, tickfont=dict(size=8))), title="NET CASH (JPY B)")
        st.plotly_chart(fig, use_container_width=True)

    # D/E vs Current Ratio
    fig = go.Figure()
    lev_data = flt.dropna(subset=["debt_to_equity", "current_ratio"])
    fig.add_trace(go.Scatter(
        x=lev_data["debt_to_equity"], y=lev_data["current_ratio"],
        mode="markers+text", text=lev_data["Ticker"], textposition="top center",
        textfont=dict(size=8, color=GRAY),
        marker=dict(size=9, color=ORANGE, line=dict(width=0.5, color=BG)),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color=YELLOW, annotation_text="CR=1.0",
                  annotation_font=dict(size=9, color=GRAY))
    fig.add_vline(x=100, line_dash="dot", line_color=RED, annotation_text="D/E=100",
                  annotation_font=dict(size=9, color=GRAY))
    fig.update_layout(**chart_layout(350), title="LEVERAGE MAP", xaxis_title="D/E", yaxis_title="CURRENT RATIO")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    corr_cols = ["Composite", "LevValue", "Delever", "Quality", "Momentum",
                 "pe_trailing", "pb_ratio", "ebitda_to_ev", "lt_debt_to_ev", "roe", "beta", "fcf_yield"]
    corr_labels = ["COMP", "LEV", "DLV", "QUAL", "MOM", "P/E", "P/B", "EBITDA/EV", "LTD/EV", "ROE", "BETA", "FCF"]
    cm = flt[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
    fig = go.Figure(data=go.Heatmap(
        z=cm.values, x=corr_labels, y=corr_labels,
        colorscale=[[0, RED], [0.5, BG1], [1, GREEN]],
        zmin=-1, zmax=1, text=cm.round(2).values, texttemplate="%{text}", textfont=dict(size=9),
    ))
    fig.update_layout(**chart_layout(400), title="FACTOR CORRELATION")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TECHNICALS
# ════════════════════════════════════════════════════════════════
with t_tech:
    # Technical signals table
    tech_data = flt[["Ticker", "Name", "current_price", "fifty_day_avg", "two_hundred_day_avg",
                     "fifty_two_week_high", "fifty_two_week_low", "52w_pos", "sma_cross", "beta", "avg_volume"]].copy()
    tech_data["SMA50"] = tech_data["fifty_day_avg"].apply(lambda x: f_price(x))
    tech_data["SMA200"] = tech_data["two_hundred_day_avg"].apply(lambda x: f_price(x))
    tech_data["PRICE"] = tech_data["current_price"].apply(lambda x: f_price(x))
    tech_data["52W_HI"] = tech_data["fifty_two_week_high"].apply(lambda x: f_price(x))
    tech_data["52W_LO"] = tech_data["fifty_two_week_low"].apply(lambda x: f_price(x))
    tech_data["52W_%"] = tech_data["52w_pos"].apply(lambda x: f"{x*100:.0f}%" if pd.notna(x) else "--")
    tech_data["SMA_X%"] = tech_data["sma_cross"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "--")
    tech_data["SIGNAL"] = tech_data["sma_cross"].apply(
        lambda x: "GOLDEN" if pd.notna(x) and x > 0.02 else ("DEATH" if pd.notna(x) and x < -0.02 else "NEUTRAL"))
    tech_data["VOL"] = tech_data["avg_volume"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "--")
    tech_data["BETA"] = tech_data["beta"].apply(lambda x: f_num(x))

    display_tech = tech_data[["Ticker", "Name", "PRICE", "SMA50", "SMA200", "SMA_X%", "SIGNAL",
                               "52W_LO", "52W_HI", "52W_%", "VOL", "BETA"]]
    display_tech.columns = ["TICKER", "NAME", "PRICE", "SMA50", "SMA200", "SMA_X", "SIGNAL",
                            "52W_LO", "52W_HI", "52W_%", "AVG_VOL", "BETA"]
    st.dataframe(display_tech, use_container_width=True, height=500)

    tl, tr = st.columns(2)
    with tl:
        # 52-week position ranking
        pos_sorted = flt[["Ticker", "52w_pos"]].dropna(subset=["52w_pos"]).sort_values("52w_pos", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pos_sorted["Ticker"], y=pos_sorted["52w_pos"] * 100,
            marker_color=np.where(pos_sorted["52w_pos"] > 0.7, GREEN, np.where(pos_sorted["52w_pos"] > 0.3, ORANGE, RED)),
            marker_line_width=0,
        ))
        fig.add_hline(y=50, line_dash="dash", line_color=GRAY)
        fig.update_layout(**chart_layout(300, xaxis=dict(tickangle=-45, tickfont=dict(size=8))), title="52W RANGE POSITION %")
        st.plotly_chart(fig, use_container_width=True)

    with tr:
        # SMA cross
        sma_sorted = flt[["Ticker", "sma_cross"]].dropna(subset=["sma_cross"]).sort_values("sma_cross", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sma_sorted["Ticker"], y=sma_sorted["sma_cross"] * 100,
            marker_color=np.where(sma_sorted["sma_cross"] > 0, GREEN, RED),
            marker_line_width=0,
        ))
        fig.add_hline(y=0, line_dash="dash", line_color=GRAY)
        fig.update_layout(**chart_layout(300, xaxis=dict(tickangle=-45, tickfont=dict(size=8))),
                          title="SMA50/SMA200 CROSS %", yaxis_title="% ABOVE/BELOW")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# GOVERNANCE — Japan corporate reform signals
# ════════════════════════════════════════════════════════════════
with t_gov:
    st.markdown(f'<div style="color:{GRAY}; font-size:9px; margin-bottom:8px;">TSE governance reform signals: P/B compliance, cross-shareholding unwind potential, activist targeting, shareholder return gaps.</div>', unsafe_allow_html=True)

    # Governance overview strip
    n_tse = len(flt[flt["tse_pressure"] == True]) if "tse_pressure" in flt.columns else 0
    n_activist = len(flt[flt["activist_risk"] == "HIGH"]) if "activist_risk" in flt.columns else 0
    avg_gov = flt["governance_composite"].mean() if "governance_composite" in flt.columns else 0
    st.markdown(strip_html([
        ("TSE PRESSURE", str(n_tse), "dc-red" if n_tse > 0 else "", "Stocks with P/B < 1.0 under TSE Prime compliance pressure"),
        ("ACTIVIST TARGETS", str(n_activist), "dc-red" if n_activist > 0 else "", "Stocks with HIGH activist target score"),
        ("AVG GOV SCORE", f"{avg_gov:.3f}" if avg_gov else "--", "dc-orange", "Average governance reform composite score"),
        ("UNIVERSE", str(len(flt)), "", "Total filtered stocks"),
    ]), unsafe_allow_html=True)

    gl, gr = st.columns(2)

    with gl:
        # Governance composite ranking
        if "governance_composite" in flt.columns:
            gov_sorted = flt[["Ticker", "governance_composite"]].dropna().sort_values("governance_composite", ascending=False).head(25)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=gov_sorted["Ticker"], y=gov_sorted["governance_composite"],
                marker_color=np.where(gov_sorted["governance_composite"] > 0.5, RED, np.where(gov_sorted["governance_composite"] > 0.3, ORANGE, GREEN)),
                marker_line_width=0,
            ))
            fig.update_layout(**chart_layout(320, xaxis=dict(tickangle=-45, tickfont=dict(size=8))),
                              title="GOVERNANCE REFORM SCORE (HIGHER = MORE REFORM UPSIDE)")
            st.plotly_chart(fig, use_container_width=True)

    with gr:
        # Activist target score
        if "activist_target_score" in flt.columns:
            act_sorted = flt[["Ticker", "activist_target_score"]].dropna().sort_values("activist_target_score", ascending=False).head(25)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=act_sorted["Ticker"], y=act_sorted["activist_target_score"],
                marker_color=np.where(act_sorted["activist_target_score"] >= 0.70, RED,
                    np.where(act_sorted["activist_target_score"] >= 0.45, ORANGE, GREEN)),
                marker_line_width=0,
            ))
            fig.update_layout(**chart_layout(320, xaxis=dict(tickangle=-45, tickfont=dict(size=8))),
                              title="ACTIVIST TARGET SCORE")
            st.plotly_chart(fig, use_container_width=True)

    # Cross-shareholding unwind vs P/B scatter
    if "cross_shareholding_unwind_score" in flt.columns:
        fig = go.Figure()
        scatter_data = flt.dropna(subset=["pb_ratio", "cross_shareholding_unwind_score"])
        fig.add_trace(go.Scatter(
            x=scatter_data["pb_ratio"], y=scatter_data["cross_shareholding_unwind_score"],
            mode="markers+text", text=scatter_data["Ticker"], textposition="top center",
            textfont=dict(size=7, color=GRAY),
            marker=dict(size=np.clip(scatter_data["mcap_b"].fillna(1).apply(np.log) * 3, 4, 20),
                        color=scatter_data["governance_composite"], colorscale=[[0, GREEN], [0.5, ORANGE], [1, RED]],
                        colorbar=dict(title=dict(text="GOV", font=dict(size=9)), thickness=10),
                        line=dict(width=0.5, color=BG)),
        ))
        fig.add_vline(x=1.0, line_dash="dot", line_color=YELLOW, annotation_text="P/B=1.0",
                      annotation_font=dict(size=9, color=GRAY))
        fig.update_layout(**chart_layout(350), title="CROSS-SHAREHOLDING UNWIND vs P/B",
                          xaxis_title="P/B RATIO", yaxis_title="UNWIND SCORE")
        st.plotly_chart(fig, use_container_width=True)

    # Governance detail table
    gov_tcols = ["Ticker", "Name", "pb_ratio", "roe", "governance_composite",
                 "activist_target_score", "cross_shareholding_unwind_score",
                 "pb_improvement_urgency", "shareholder_return_potential",
                 "activist_risk", "est_payout_ratio", "cash_to_mcap"]
    gov_available = [c for c in gov_tcols if c in flt.columns]
    gtbl = flt[gov_available].sort_values("governance_composite", ascending=False).head(30).copy()
    rename_map = {
        "pb_ratio": "P/B", "roe": "ROE", "governance_composite": "GOV",
        "activist_target_score": "ACTIVIST", "cross_shareholding_unwind_score": "UNWIND",
        "pb_improvement_urgency": "PB_URG", "shareholder_return_potential": "SH_RET",
        "activist_risk": "ACT_RISK", "est_payout_ratio": "PAYOUT", "cash_to_mcap": "CASH/MC",
    }
    gtbl = gtbl.rename(columns={k: v for k, v in rename_map.items() if k in gtbl.columns})
    for c in ["ROE", "PAYOUT", "CASH/MC"]:
        if c in gtbl.columns:
            gtbl[c] = gtbl[c].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
    st.dataframe(gtbl, use_container_width=True, height=400, hide_index=True)


# ════════════════════════════════════════════════════════════════
# PORTFOLIO — custom portfolio builder
# ════════════════════════════════════════════════════════════════
with t_port:
    st.markdown(f'<div style="color:{GRAY}; font-size:9px; margin-bottom:8px;">Build a custom portfolio by adding and removing stocks. Weights are auto-calculated. Use this portfolio in BACKTEST and OPTIMIZE tabs.</div>', unsafe_allow_html=True)

    # Portfolio state
    if "custom_portfolio" not in st.session_state:
        st.session_state.custom_portfolio = {}

    # ── Add stocks ────────────────────────────────────────────
    st.markdown("### ADD STOCKS")
    pa_cols = st.columns([3, 1, 1])
    with pa_cols[0]:
        port_add_ticker = st.selectbox(
            "TICKER",
            [""] + df["Ticker"].tolist(),
            key="port_add_tkr",
            help="Select a stock to add to your custom portfolio",
        )
    with pa_cols[1]:
        port_add_weight = st.number_input(
            "WEIGHT %", value=0.0, min_value=0.0, max_value=100.0, step=1.0,
            key="port_add_wt",
            help="Target weight (%). Leave at 0 for equal-weight auto-calculation.",
        )
    with pa_cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("ADD TO PORTFOLIO", key="port_add_btn") and port_add_ticker:
            st.session_state.custom_portfolio[port_add_ticker] = {
                "manual_weight": port_add_weight / 100.0 if port_add_weight > 0 else None,
                "added": datetime.now().strftime("%Y-%m-%d"),
            }
            st.rerun()

    # Quick add: top N from screen
    st.markdown(f'<span style="color:{GRAY};font-size:9px;">QUICK ADD</span>', unsafe_allow_html=True)
    qa_cols = st.columns(4)
    with qa_cols[0]:
        if st.button("TOP 10 BY ALPHA", key="qa_alpha"):
            top = df.nlargest(10, "Alpha")["Ticker"].tolist()
            for t in top:
                if t not in st.session_state.custom_portfolio:
                    st.session_state.custom_portfolio[t] = {"manual_weight": None, "added": datetime.now().strftime("%Y-%m-%d")}
            st.rerun()
    with qa_cols[1]:
        if st.button("TOP 10 BY SCORE", key="qa_comp"):
            top = df.nlargest(10, "Composite")["Ticker"].tolist()
            for t in top:
                if t not in st.session_state.custom_portfolio:
                    st.session_state.custom_portfolio[t] = {"manual_weight": None, "added": datetime.now().strftime("%Y-%m-%d")}
            st.rerun()
    with qa_cols[2]:
        if st.button("SCREEN PASS ONLY", key="qa_pass"):
            passed = df[df["Screen"] == "PASS"]["Ticker"].tolist()[:20]
            for t in passed:
                if t not in st.session_state.custom_portfolio:
                    st.session_state.custom_portfolio[t] = {"manual_weight": None, "added": datetime.now().strftime("%Y-%m-%d")}
            st.rerun()
    with qa_cols[3]:
        if st.button("CLEAR ALL", key="qa_clear", type="primary"):
            st.session_state.custom_portfolio = {}
            st.rerun()

    # ── Current portfolio table ───────────────────────────────
    cp = st.session_state.custom_portfolio
    if cp:
        st.markdown("### CURRENT PORTFOLIO")

        # Calculate weights
        n_stocks = len(cp)
        manual_total = sum(v["manual_weight"] for v in cp.values() if v["manual_weight"] is not None)
        auto_count = sum(1 for v in cp.values() if v["manual_weight"] is None)
        remaining = max(0, 1.0 - manual_total)
        auto_weight = remaining / auto_count if auto_count > 0 else 0

        port_rows = []
        for tkr, info in cp.items():
            row_data = df[df["Ticker"] == tkr]
            w = info["manual_weight"] if info["manual_weight"] is not None else auto_weight
            price = row_data["current_price"].values[0] if len(row_data) > 0 and pd.notna(row_data["current_price"].values[0]) else 0
            alpha_val = row_data["Alpha"].values[0] if len(row_data) > 0 else 0
            comp_val = row_data["Composite"].values[0] if len(row_data) > 0 else 0
            sector = row_data["Sector"].values[0] if len(row_data) > 0 else "--"
            name = row_data["Name"].values[0] if len(row_data) > 0 else "--"
            port_rows.append({
                "TICKER": tkr,
                "NAME": str(name)[:20],
                "SECTOR": sector,
                "WEIGHT": w,
                "PRICE": price,
                "ALPHA": alpha_val,
                "COMP": comp_val,
                "TYPE": "MANUAL" if info["manual_weight"] is not None else "AUTO",
            })

        port_df = pd.DataFrame(port_rows)
        total_weight = port_df["WEIGHT"].sum()

        # Summary strip
        sectors_in_port = port_df["SECTOR"].nunique()
        avg_alpha = port_df["ALPHA"].mean()
        avg_comp = port_df["COMP"].mean()
        st.markdown(strip_html([
            ("HOLDINGS", str(n_stocks), "dc-orange", "Number of stocks in portfolio"),
            ("TOT WEIGHT", f"{total_weight*100:.1f}%", "dc-green" if abs(total_weight - 1.0) < 0.01 else "dc-red", "Total portfolio weight (should be ~100%)"),
            ("SECTORS", str(sectors_in_port), "", "Number of distinct sectors"),
            ("AVG ALPHA", f"{avg_alpha:.3f}", "dc-green" if avg_alpha > 0 else "dc-red", "Average model alpha of portfolio"),
            ("AVG SCORE", f"{avg_comp:.3f}", "dc-orange", "Average composite score"),
        ]), unsafe_allow_html=True)

        # Display table with remove buttons
        display_df = port_df.copy()
        display_df["WEIGHT"] = display_df["WEIGHT"].apply(lambda x: f"{x*100:.1f}%")
        display_df["PRICE"] = display_df["PRICE"].apply(lambda x: f"{x:,.0f}" if x else "--")
        display_df["ALPHA"] = display_df["ALPHA"].apply(lambda x: f"{x:.4f}")
        display_df["COMP"] = display_df["COMP"].apply(lambda x: f"{x:.3f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True,
                      column_config={
                          "TICKER": st.column_config.TextColumn("TICKER", help="Stock ticker"),
                          "WEIGHT": st.column_config.TextColumn("WEIGHT", help="Portfolio weight (manual or auto equal-weight)"),
                          "ALPHA": st.column_config.TextColumn("ALPHA", help="Blended model alpha signal"),
                          "COMP": st.column_config.TextColumn("COMP", help="Composite screening score"),
                          "TYPE": st.column_config.TextColumn("TYPE", help="MANUAL = user-set weight, AUTO = equal-weight residual"),
                      })

        # Remove individual stocks
        st.markdown(f'<span style="color:{GRAY};font-size:9px;">REMOVE STOCKS</span>', unsafe_allow_html=True)
        rm_cols = st.columns(min(6, n_stocks))
        for i, tkr in enumerate(list(cp.keys())):
            col_idx = i % min(6, n_stocks)
            with rm_cols[col_idx]:
                if st.button(f"X {tkr}", key=f"rm_{tkr}"):
                    del st.session_state.custom_portfolio[tkr]
                    st.rerun()

        # Sector allocation chart
        sect_wts = port_df.groupby("SECTOR")["WEIGHT"].sum().sort_values(ascending=False)
        if len(sect_wts) > 1:
            st.markdown("### SECTOR ALLOCATION")
            sc1, sc2 = st.columns(2)
            with sc1:
                fig_sw = go.Figure(data=[go.Pie(
                    labels=sect_wts.index.tolist(),
                    values=(sect_wts.values * 100).round(1).tolist(),
                    hole=0.4, textinfo="label+percent",
                    textfont=dict(size=9, color=WHITE),
                    marker=dict(colors=[ORANGE, GREEN, YELLOW, "#6699ff", "#cc66ff",
                                        RED, GRAY, "#ff6699", ORANGE_DIM, "#66cccc"]),
                )])
                fig_sw.update_layout(**chart_layout(300, title="SECTOR WEIGHTS"))
                st.plotly_chart(fig_sw, use_container_width=True)

            with sc2:
                fig_ab = go.Figure()
                fig_ab.add_trace(go.Bar(
                    x=port_df["TICKER"], y=(port_df["WEIGHT"].astype(float) * 100).tolist(),
                    marker_color=ORANGE,
                    text=display_df["WEIGHT"].tolist(),
                    textposition="outside", textfont=dict(size=8, color=WHITE),
                ))
                fig_ab.update_layout(**chart_layout(300, title="POSITION WEIGHTS",
                    xaxis=dict(tickangle=-45, tickfont=dict(size=8)), yaxis=dict(title="WEIGHT %")))
                st.plotly_chart(fig_ab, use_container_width=True)

    else:
        st.markdown(f"""<div style="text-align:center; padding:60px; color:{GRAY};">
        <div style="font-size:12px; color:{ORANGE}; letter-spacing:2px; margin-bottom:12px;">NO PORTFOLIO DEFINED</div>
        <div style="font-size:10px;">Add stocks above or use QUICK ADD buttons to build a custom portfolio.<br>
        Your portfolio will be available in BACKTEST and OPTIMIZE tabs.</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# OPTIMIZE — portfolio optimizer
# ════════════════════════════════════════════════════════════════
with t_opt:
    st.markdown(f'<div style="color:{GRAY}; font-size:9px; margin-bottom:8px;">Run mean-variance optimization on your custom portfolio or the full screened universe. Adjustable risk aversion, position limits, and sector constraints.</div>', unsafe_allow_html=True)

    st.markdown("### OPTIMIZER SETTINGS")

    oc1, oc2, oc3, oc4 = st.columns(4)
    with oc1:
        opt_source = st.selectbox("SOURCE", ["CUSTOM PORTFOLIO", "FULL SCREEN", "TOP 20 ALPHA", "TOP 30 ALPHA"], key="opt_src",
                                   help="Which stocks to optimize over")
    with oc2:
        opt_risk = st.slider("RISK AVERSION", 0.1, 5.0, 1.0, 0.1, key="opt_risk",
                              help="Lambda in mean-variance objective. Higher = more conservative, lower = more aggressive")
    with oc3:
        opt_max_pos = st.slider("MAX POSITION %", 2, 20, 5, 1, key="opt_maxpos",
                                 help="Maximum weight for any single stock")
    with oc4:
        opt_max_sect = st.slider("MAX SECTOR %", 10, 50, 25, 5, key="opt_maxsect",
                                  help="Maximum aggregate weight for any single sector")

    oc5, oc6, oc7, oc8 = st.columns(4)
    with oc5:
        opt_max_names = st.number_input("MAX HOLDINGS", 5, 50, 25, key="opt_maxn")
    with oc6:
        opt_min_names = st.number_input("MIN HOLDINGS", 3, 30, 8, key="opt_minn")
    with oc7:
        opt_turnover = st.slider("TURNOVER PENALTY (BPS)", 0, 100, 50, 5, key="opt_turn",
                                  help="Cost charged per unit of turnover to penalize excessive trading")
    with oc8:
        opt_signal = st.selectbox("ALPHA SOURCE", ["BLENDED ALPHA", "COMPOSITE SCORE", "FACTOR RANK ONLY", "MOMENTUM ONLY"], key="opt_sig")

    if st.button("RUN OPTIMIZER", key="opt_run", type="primary"):
        with st.spinner("Optimizing portfolio..."):
            # Determine universe
            if opt_source == "CUSTOM PORTFOLIO" and st.session_state.get("custom_portfolio"):
                opt_tickers = list(st.session_state.custom_portfolio.keys())
            elif opt_source == "TOP 20 ALPHA":
                opt_tickers = df.nlargest(20, "Alpha")["Ticker"].tolist()
            elif opt_source == "TOP 30 ALPHA":
                opt_tickers = df.nlargest(30, "Alpha")["Ticker"].tolist()
            else:
                opt_tickers = flt["Ticker"].tolist()[:50]

            if len(opt_tickers) < 3:
                st.warning("Need at least 3 stocks. Add stocks in PORTFOLIO tab or select a different source.")
            else:
                # Build alpha vector
                if opt_signal == "COMPOSITE SCORE":
                    opt_alpha = df.set_index("Ticker")["Composite"].reindex(opt_tickers).fillna(0)
                elif opt_signal == "FACTOR RANK ONLY":
                    fr_alpha = pd.Series(model_sigs.get("FACTOR_RANK", {}).get("alpha", {}))
                    opt_alpha = fr_alpha.reindex(opt_tickers).fillna(0)
                elif opt_signal == "MOMENTUM ONLY":
                    mom_alpha = pd.Series(model_sigs.get("MOMENTUM", {}).get("alpha", {}))
                    opt_alpha = mom_alpha.reindex(opt_tickers).fillna(0)
                else:
                    opt_alpha = blended_alpha.reindex(opt_tickers).fillna(0)

                # Z-score alpha
                if opt_alpha.std() > 0:
                    opt_alpha = (opt_alpha - opt_alpha.mean()) / opt_alpha.std()

                sectors_map = pd.Series(dict(zip(df["Ticker"], df["Sector"]))) if "Sector" in df.columns else None

                opt_result = optimize_portfolio(
                    alpha=opt_alpha,
                    sectors=sectors_map,
                    constraints=PortfolioConstraints(
                        max_position=opt_max_pos / 100.0,
                        min_position=0.002,
                        max_sector_weight=opt_max_sect / 100.0,
                        max_names=opt_max_names,
                        min_names=opt_min_names,
                        turnover_penalty=opt_turnover / 10000.0,
                        risk_aversion=opt_risk,
                    ),
                )

                if opt_result.n_holdings > 0:
                    st.markdown("### OPTIMIZED PORTFOLIO")

                    # Metrics
                    om1, om2, om3, om4, om5, om6 = st.columns(6)
                    om1.metric("HOLDINGS", opt_result.n_holdings)
                    om2.metric("EX-ANTE SHARPE", f"{opt_result.sharpe_ratio:.3f}")
                    om3.metric("EXPECTED ALPHA", f"{opt_result.expected_return:.4f}")
                    om4.metric("EXPECTED RISK", f"{opt_result.expected_risk*100:.2f}%")
                    om5.metric("ACTIVE RISK", f"{opt_result.active_risk*100:.2f}%")
                    om6.metric("TOP WEIGHT", f"{opt_result.weights.max()*100:.1f}%")

                    # Weights table
                    st.markdown("### POSITION WEIGHTS")
                    ow_df = pd.DataFrame({
                        "#": range(1, opt_result.n_holdings + 1),
                        "TICKER": opt_result.weights.index,
                        "WEIGHT %": (opt_result.weights.values * 100).round(2),
                    })
                    # Add fundamentals
                    for col_name, df_col in [("NAME", "Name"), ("SECTOR", "Sector"), ("PRICE", "current_price"),
                                              ("ALPHA", "Alpha"), ("COMP", "Composite"), ("P/E", "pe_trailing"),
                                              ("P/B", "pb_ratio"), ("ROE", "roe")]:
                        if df_col in df.columns:
                            lookup = dict(zip(df["Ticker"], df[df_col]))
                            ow_df[col_name] = ow_df["TICKER"].map(lookup)

                    if "PRICE" in ow_df.columns:
                        ow_df["PRICE"] = ow_df["PRICE"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "--")
                    if "ALPHA" in ow_df.columns:
                        ow_df["ALPHA"] = ow_df["ALPHA"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "--")
                    if "ROE" in ow_df.columns:
                        ow_df["ROE"] = ow_df["ROE"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "--")

                    st.dataframe(ow_df, use_container_width=True, hide_index=True,
                                  column_config={
                                      "WEIGHT %": st.column_config.NumberColumn("WEIGHT %", help="Optimized portfolio weight", format="%.2f"),
                                      "ALPHA": st.column_config.TextColumn("ALPHA", help="Blended model alpha"),
                                      "COMP": st.column_config.NumberColumn("COMP", help="Composite score", format="%.3f"),
                                  })

                    # Charts
                    opt_l, opt_r = st.columns(2)
                    with opt_l:
                        pw = opt_result.weights.head(20)
                        fig_pw = go.Figure()
                        fig_pw.add_trace(go.Bar(
                            x=pw.index, y=pw.values * 100,
                            marker_color=ORANGE,
                            text=[f"{v*100:.1f}%" for v in pw.values],
                            textposition="outside", textfont=dict(size=8, color=WHITE),
                        ))
                        fig_pw.update_layout(**chart_layout(320, title="OPTIMIZED WEIGHTS",
                            xaxis=dict(tickangle=-45, tickfont=dict(size=8)), yaxis=dict(title="WEIGHT %")))
                        st.plotly_chart(fig_pw, use_container_width=True)

                    with opt_r:
                        if opt_result.sector_weights:
                            sects = opt_result.sector_weights
                            fig_sect = go.Figure(data=[go.Pie(
                                labels=list(sects.keys()),
                                values=[v * 100 for v in sects.values()],
                                hole=0.4, textinfo="label+percent",
                                textfont=dict(size=9, color=WHITE),
                                marker=dict(colors=[ORANGE, GREEN, YELLOW, "#6699ff", "#cc66ff",
                                                    RED, GRAY, "#ff6699", ORANGE_DIM, "#66cccc"]),
                            )])
                            fig_sect.update_layout(**chart_layout(320, title="SECTOR ALLOCATION"))
                            st.plotly_chart(fig_sect, use_container_width=True)

                    # Risk contribution
                    st.markdown("### RISK ANALYSIS")
                    st.markdown(f"""<div class="strip">
                        <div class="strip-item"><span class="strip-label">PORTFOLIO VOL</span><span class="strip-val">{opt_result.expected_risk*100:.2f}%</span></div>
                        <div class="strip-item"><span class="strip-label">ACTIVE RISK</span><span class="strip-val">{opt_result.active_risk*100:.2f}%</span></div>
                        <div class="strip-item"><span class="strip-label">RISK AVERSION</span><span class="strip-val">{opt_risk:.1f}</span></div>
                        <div class="strip-item"><span class="strip-label">TURN PENALTY</span><span class="strip-val">{opt_turnover}bps</span></div>
                        <div class="strip-item"><span class="strip-label">MAX POS</span><span class="strip-val">{opt_max_pos}%</span></div>
                        <div class="strip-item"><span class="strip-label">MAX SECT</span><span class="strip-val">{opt_max_sect}%</span></div>
                    </div>""", unsafe_allow_html=True)

                    # Compare to equal weight
                    st.markdown("### EQUAL WEIGHT COMPARISON")
                    ew_return = opt_alpha.mean()
                    opt_exp_return = opt_result.expected_return
                    st.markdown(f"""<div class="panel">
                    <span style="color:{GRAY};font-size:10px;">
                    <b style="color:{ORANGE};">OPTIMIZED</b>: Expected alpha = {opt_exp_return:.4f}, Sharpe = {opt_result.sharpe_ratio:.3f}, Holdings = {opt_result.n_holdings}<br>
                    <b style="color:{GRAY};">EQUAL WEIGHT</b>: Expected alpha = {ew_return:.4f}, Holdings = {len(opt_tickers)}<br><br>
                    The optimizer tilts toward high-alpha names while respecting risk and concentration constraints.
                    Higher risk aversion produces more equal-weight-like portfolios. Lower risk aversion concentrates into top picks.
                    </span></div>""", unsafe_allow_html=True)

                    # Store optimized portfolio for backtest
                    st.session_state["optimized_portfolio"] = {
                        t: w for t, w in opt_result.weights.items()
                    }
                    st.markdown(f'<div style="color:{GREEN};font-size:9px;margin-top:8px;">Optimized portfolio saved. Available in BACKTEST tab.</div>', unsafe_allow_html=True)

                else:
                    st.warning("Optimizer returned no holdings. Try relaxing constraints.")

    # Show saved optimized portfolio
    elif "optimized_portfolio" in st.session_state and st.session_state["optimized_portfolio"]:
        saved = st.session_state["optimized_portfolio"]
        st.markdown(f'<div style="color:{GRAY};font-size:10px;">Last optimized portfolio: {len(saved)} holdings. Click RUN OPTIMIZER to update.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# BACKTEST — historical simulation
# ════════════════════════════════════════════════════════════════
with t_bt:
    st.markdown(f'<div style="color:{GRAY}; font-size:9px; margin-bottom:8px;">Simulated backtest using quarterly rebalancing. Uses historical price data from yfinance. Select a portfolio source below.</div>', unsafe_allow_html=True)

    bt_row1 = st.columns(5)
    with bt_row1[0]:
        bt_source = st.selectbox("PORTFOLIO SOURCE", [
            "TOP N BY SCORE",
            "CUSTOM PORTFOLIO",
            "OPTIMIZED PORTFOLIO",
            "TOP N BY ALPHA",
        ], key="bt_source", help="Which portfolio to backtest")
    with bt_row1[1]:
        bt_start = st.text_input("START DATE", "2022-01-01", key="bt_start")
    with bt_row1[2]:
        bt_end = st.text_input("END DATE", "2026-03-01", key="bt_end")
    with bt_row1[3]:
        bt_top_n = st.number_input("TOP N HOLDINGS", value=15, min_value=5, max_value=30, key="bt_topn")
    with bt_row1[4]:
        bt_capital = st.number_input("INITIAL CAPITAL (JPY)", value=10_000_000, step=1_000_000, key="bt_cap")

    # Show which stocks will be backtested
    if bt_source == "CUSTOM PORTFOLIO" and st.session_state.get("custom_portfolio"):
        cp_tickers = list(st.session_state.custom_portfolio.keys())
        st.markdown(f'<div style="color:{GRAY};font-size:9px;">CUSTOM: {len(cp_tickers)} stocks — {", ".join(cp_tickers[:10])}{"..." if len(cp_tickers) > 10 else ""}</div>', unsafe_allow_html=True)
    elif bt_source == "OPTIMIZED PORTFOLIO" and st.session_state.get("optimized_portfolio"):
        op_tickers = list(st.session_state["optimized_portfolio"].keys())
        st.markdown(f'<div style="color:{GRAY};font-size:9px;">OPTIMIZED: {len(op_tickers)} stocks — {", ".join(op_tickers[:10])}{"..." if len(op_tickers) > 10 else ""}</div>', unsafe_allow_html=True)
    elif bt_source == "TOP N BY ALPHA":
        st.markdown(f'<div style="color:{GRAY};font-size:9px;">TOP {bt_top_n} stocks ranked by blended model alpha signal</div>', unsafe_allow_html=True)
    elif bt_source == "CUSTOM PORTFOLIO" and not st.session_state.get("custom_portfolio"):
        st.warning("No custom portfolio defined. Add stocks in PORTFOLIO tab.")
    elif bt_source == "OPTIMIZED PORTFOLIO" and not st.session_state.get("optimized_portfolio"):
        st.warning("No optimized portfolio saved. Run optimizer in OPTIMIZE tab first.")

    if st.button("RUN BACKTEST", key="bt_run"):
        with st.spinner("Running backtest..."):
            from src.backtest.engine import run_backtest, BacktestConfig

            # Build snapshot based on source
            if bt_source == "CUSTOM PORTFOLIO" and st.session_state.get("custom_portfolio"):
                cp_tkrs = list(st.session_state.custom_portfolio.keys())
                scored = [(t, 1.0 / (i + 1)) for i, t in enumerate(cp_tkrs)]
                bt_top_n = len(cp_tkrs)
            elif bt_source == "OPTIMIZED PORTFOLIO" and st.session_state.get("optimized_portfolio"):
                opt_p = st.session_state["optimized_portfolio"]
                scored = sorted(opt_p.items(), key=lambda x: x[1], reverse=True)
                bt_top_n = len(scored)
            elif bt_source == "TOP N BY ALPHA":
                alpha_ranked = df.nlargest(bt_top_n, "Alpha")
                scored = list(zip(alpha_ranked["Ticker"], alpha_ranked["Alpha"]))
            else:
                scored = [(r.ticker, r.composite) for r in results]
                scored.sort(key=lambda x: x[1], reverse=True)

            snapshot = {bt_start: scored}

            bt_cfg = BacktestConfig(
                start_date=bt_start,
                end_date=bt_end,
                initial_capital=bt_capital,
                top_n=bt_top_n,
                equal_weight=True,
                transaction_cost_bps=10,
            )

            try:
                bt_result = run_backtest(snapshot, bt_cfg)

                if bt_result.portfolio_value is not None and len(bt_result.portfolio_value) > 0:
                    # Performance metrics strip
                    m = bt_result.metrics
                    st.markdown(strip_html([
                        ("TOTAL RETURN", f"{m.get('total_return', 0):.1f}%", "dc-green" if m.get("total_return", 0) > 0 else "dc-red", "Total portfolio return over period"),
                        ("CAGR", f"{m.get('cagr', 0):.1f}%", "dc-green" if m.get("cagr", 0) > 0 else "dc-red", "Compound annual growth rate"),
                        ("SHARPE", f"{m.get('sharpe_ratio', 0):.3f}", "dc-orange", "Sharpe ratio (risk-free = 0.5% Japan)"),
                        ("SORTINO", f"{m.get('sortino_ratio', 0):.3f}", "", "Sortino ratio (downside risk only)"),
                        ("MAX DD", f"{m.get('max_drawdown', 0):.1f}%", "dc-red", "Maximum drawdown from peak"),
                        ("VOL", f"{m.get('volatility', 0):.1f}%", "", "Annualized volatility"),
                        ("FINAL", f"Y{m.get('final_value', 0):,.0f}", "dc-orange", "Final portfolio value"),
                        ("ALPHA", f"{m.get('alpha', 0):.1f}%", "dc-green" if m.get("alpha", 0) > 0 else "dc-red", "Excess return vs Nikkei 225"),
                    ]), unsafe_allow_html=True)

                    # Portfolio value chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=bt_result.portfolio_value.index, y=bt_result.portfolio_value.values,
                        mode="lines", line=dict(color=ORANGE, width=2), name="PORTFOLIO",
                    ))
                    if bt_result.benchmark_value is not None:
                        fig.add_trace(go.Scatter(
                            x=bt_result.benchmark_value.index, y=bt_result.benchmark_value.values,
                            mode="lines", line=dict(color=GRAY, width=1, dash="dot"), name="NIKKEI 225",
                        ))
                    fig.update_layout(**chart_layout(400, showlegend=True),
                                      title="PORTFOLIO vs BENCHMARK", yaxis_title="JPY")
                    fig.update_layout(legend=dict(orientation="h", y=1.06))
                    st.plotly_chart(fig, use_container_width=True)

                    # Drawdown chart
                    pv = bt_result.portfolio_value
                    cummax = pv.cummax()
                    drawdown = (pv - cummax) / cummax * 100
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=drawdown.index, y=drawdown.values,
                        mode="lines", fill="tozeroy", line=dict(color=RED, width=1),
                        fillcolor="rgba(255,50,50,0.15)",
                    ))
                    fig.update_layout(**chart_layout(250), title="DRAWDOWN %", yaxis_title="DD %")
                    st.plotly_chart(fig, use_container_width=True)

                    # Trade log
                    if bt_result.trades:
                        st.markdown(f'<div style="color:{ORANGE}; font-size:10px; letter-spacing:1px; margin:8px 0 4px;">TRADE LOG</div>', unsafe_allow_html=True)
                        trade_df = pd.DataFrame(bt_result.trades)
                        trade_df["cost"] = trade_df["cost"].apply(lambda x: f"Y{x:,.0f}")
                        st.dataframe(trade_df, use_container_width=True, hide_index=True, height=200)
                else:
                    st.warning("Backtest produced no data. Check date range and tickers.")
            except Exception as e:
                st.error(f"Backtest failed: {e}")
    else:
        st.markdown(f'<div style="color:{GRAY}; font-size:10px; text-align:center; padding:40px;">Configure parameters above and click RUN BACKTEST to simulate quarterly rebalancing of the current screen.</div>', unsafe_allow_html=True)

        # Show score history if available
        hist = load_all_snapshots()
        if not hist.empty:
            st.markdown(f'<div style="color:{ORANGE}; font-size:10px; letter-spacing:1px; margin:12px 0 4px;">SCORE HISTORY ({len(hist["timestamp"].dt.date.unique())} snapshots)</div>', unsafe_allow_html=True)

            hist_ticker = st.selectbox("TRACK TICKER", flt["Ticker"].tolist()[:20], key="hist_tkr")
            tkr_hist = hist[hist["ticker"] == hist_ticker]
            if not tkr_hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tkr_hist["timestamp"], y=tkr_hist["composite"],
                    mode="lines+markers", line=dict(color=ORANGE, width=2),
                    marker=dict(size=4, color=ORANGE)))
                fig.update_layout(**chart_layout(250), title=f"COMPOSITE SCORE HISTORY: {hist_ticker}",
                                  yaxis_title="SCORE")
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# WATCHLIST — saved picks and portfolio tracker
# ════════════════════════════════════════════════════════════════
with t_wl:
    st.markdown(f'<div style="color:{GRAY}; font-size:9px; margin-bottom:8px;">Track selected positions. Add tickers to your watchlist and monitor score changes, prices, and P&L.</div>', unsafe_allow_html=True)

    # Watchlist state
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = {}

    wl_cols = st.columns([3, 1, 1, 1])
    with wl_cols[0]:
        wl_add = st.selectbox("ADD TO WATCHLIST", [""] + flt["Ticker"].tolist(), key="wl_add")
    with wl_cols[1]:
        wl_shares = st.number_input("SHARES", value=100, min_value=1, step=100, key="wl_shares")
    with wl_cols[2]:
        wl_cost = st.number_input("AVG COST", value=0.0, step=100.0, key="wl_cost")
    with wl_cols[3]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("ADD", key="wl_add_btn") and wl_add:
            st.session_state.watchlist[wl_add] = {
                "shares": wl_shares,
                "avg_cost": wl_cost,
                "added": datetime.now().strftime("%Y-%m-%d"),
            }
            st.rerun()

    if st.session_state.watchlist:
        wl_rows = []
        total_value = 0
        total_cost = 0
        for ticker, pos in st.session_state.watchlist.items():
            row_data = flt[flt["Ticker"] == ticker]
            if row_data.empty:
                row_data = df[df["Ticker"] == ticker]
            if not row_data.empty:
                r = row_data.iloc[0]
                price = r.get("current_price", 0) or 0
                mkt_val = price * pos["shares"]
                cost_basis = pos["avg_cost"] * pos["shares"] if pos["avg_cost"] > 0 else 0
                pnl = mkt_val - cost_basis if cost_basis > 0 else 0
                pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
                total_value += mkt_val
                total_cost += cost_basis
                wl_rows.append({
                    "TICKER": ticker,
                    "NAME": r.get("Name", ""),
                    "PRICE": f"{price:,.0f}",
                    "SHARES": pos["shares"],
                    "MKT VAL": f"Y{mkt_val:,.0f}",
                    "AVG COST": f"{pos['avg_cost']:,.0f}" if pos["avg_cost"] > 0 else "--",
                    "P&L": f"Y{pnl:,.0f}" if cost_basis > 0 else "--",
                    "P&L%": f"{pnl_pct:.1f}%" if cost_basis > 0 else "--",
                    "COMP": f"{r.get('Composite', 0):.3f}",
                    "SCR": r.get("Screen", "--"),
                    "ADDED": pos["added"],
                })

        if wl_rows:
            total_pnl = total_value - total_cost if total_cost > 0 else 0
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
            st.markdown(strip_html([
                ("POSITIONS", str(len(wl_rows)), "dc-orange", "Number of watchlist positions"),
                ("TOTAL VALUE", f"Y{total_value:,.0f}", "", "Total market value of all positions"),
                ("TOTAL COST", f"Y{total_cost:,.0f}" if total_cost > 0 else "--", "", "Total cost basis"),
                ("TOTAL P&L", f"Y{total_pnl:,.0f}" if total_cost > 0 else "--",
                 "dc-green" if total_pnl > 0 else "dc-red", "Unrealized profit/loss"),
                ("P&L %", f"{total_pnl_pct:.1f}%" if total_cost > 0 else "--",
                 "dc-green" if total_pnl_pct > 0 else "dc-red", "Unrealized return %"),
            ]), unsafe_allow_html=True)

            st.dataframe(pd.DataFrame(wl_rows), use_container_width=True, hide_index=True)

            # Remove button
            wl_remove = st.selectbox("REMOVE FROM WATCHLIST", [""] + list(st.session_state.watchlist.keys()), key="wl_rem")
            if st.button("REMOVE", key="wl_rem_btn") and wl_remove:
                del st.session_state.watchlist[wl_remove]
                st.rerun()

            # Watchlist price chart
            st.markdown(f'<div style="color:{ORANGE}; font-size:10px; letter-spacing:1px; margin:8px 0 4px;">WATCHLIST PERFORMANCE</div>', unsafe_allow_html=True)
            fig = go.Figure()
            colors_wl = [ORANGE, GREEN, "#06b6d4", YELLOW, "#a855f7", RED, WHITE, GRAY]
            for i, ticker in enumerate(list(st.session_state.watchlist.keys())[:8]):
                prices = load_prices(ticker, years=1)
                if not prices.empty:
                    if isinstance(prices.columns, pd.MultiIndex):
                        close = prices[("Close", ticker)] if ("Close", ticker) in prices.columns else prices.iloc[:, 0]
                    else:
                        close = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
                    norm = close / close.iloc[0] * 100
                    fig.add_trace(go.Scatter(x=prices.index, y=norm, mode="lines",
                        line=dict(color=colors_wl[i % len(colors_wl)], width=1.5), name=ticker))
            fig.add_hline(y=100, line_dash="dot", line_color=GRAY_DIM)
            fig.update_layout(**chart_layout(300, showlegend=True), title="RELATIVE PERFORMANCE (1Y, INDEXED)")
            fig.update_layout(legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(f'<div style="color:{GRAY}; font-size:10px; text-align:center; padding:40px;">No positions in watchlist. Use the selector above to add tickers.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# EXPORT — data export and tear sheets
# ════════════════════════════════════════════════════════════════
with t_exp:
    st.markdown(f'<div style="color:{GRAY}; font-size:9px; margin-bottom:8px;">Export screen results, fundamentals, and tear sheets.</div>', unsafe_allow_html=True)

    exp_cols = st.columns(3)

    with exp_cols[0]:
        st.markdown(f'<div style="color:{ORANGE}; font-size:10px; letter-spacing:1px; margin-bottom:4px;">SCREEN EXPORT</div>', unsafe_allow_html=True)
        # CSV download of current screen
        csv_buf = io.StringIO()
        flt.to_csv(csv_buf, index=False)
        st.download_button(
            "DOWNLOAD SCREEN CSV",
            csv_buf.getvalue(),
            file_name=f"jvq_screen_{sel_index.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="dl_screen",
        )
        st.markdown(f'<div style="color:{GRAY}; font-size:9px;">{len(flt)} stocks, {len(flt.columns)} columns</div>', unsafe_allow_html=True)

    with exp_cols[1]:
        st.markdown(f'<div style="color:{ORANGE}; font-size:10px; letter-spacing:1px; margin-bottom:4px;">FULL UNIVERSE</div>', unsafe_allow_html=True)
        csv_full = io.StringIO()
        df.to_csv(csv_full, index=False)
        st.download_button(
            "DOWNLOAD FULL CSV",
            csv_full.getvalue(),
            file_name=f"jvq_universe_{sel_index.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="dl_full",
        )
        st.markdown(f'<div style="color:{GRAY}; font-size:9px;">{len(df)} stocks, all columns</div>', unsafe_allow_html=True)

    with exp_cols[2]:
        st.markdown(f'<div style="color:{ORANGE}; font-size:10px; letter-spacing:1px; margin-bottom:4px;">GOVERNANCE DATA</div>', unsafe_allow_html=True)
        gov_cols_export = ["Ticker", "Name", "Sector", "pb_ratio", "roe", "governance_composite",
                           "activist_target_score", "cross_shareholding_unwind_score",
                           "pb_improvement_urgency", "shareholder_return_potential",
                           "activist_risk", "est_payout_ratio", "cash_to_mcap"]
        gov_available_exp = [c for c in gov_cols_export if c in df.columns]
        gov_df = df[gov_available_exp].sort_values("governance_composite", ascending=False) if "governance_composite" in df.columns else df[["Ticker", "Name"]]
        csv_gov = io.StringIO()
        gov_df.to_csv(csv_gov, index=False)
        st.download_button(
            "DOWNLOAD GOVERNANCE CSV",
            csv_gov.getvalue(),
            file_name=f"jvq_governance_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="dl_gov",
        )
        st.markdown(f'<div style="color:{GRAY}; font-size:9px;">{len(gov_df)} stocks, governance signals</div>', unsafe_allow_html=True)

    # Tear sheet section
    st.markdown(f'<div style="color:{ORANGE}; font-size:10px; letter-spacing:1px; margin:16px 0 4px;">EQUITY TEAR SHEET</div>', unsafe_allow_html=True)
    ts_ticker = st.selectbox("SELECT EQUITY", flt["Ticker"].tolist(), key="ts_sel")
    if ts_ticker:
        ts_row = df[df["Ticker"] == ts_ticker]
        if not ts_row.empty:
            r = ts_row.iloc[0]
            ts_buf = io.StringIO()
            ts_buf.write(f"{'='*60}\n")
            ts_buf.write(f"JVQ TERMINAL — EQUITY TEAR SHEET\n")
            ts_buf.write(f"{'='*60}\n\n")
            ts_buf.write(f"Ticker:     {r['Ticker']}\n")
            ts_buf.write(f"Name:       {r['Name']}\n")
            ts_buf.write(f"Sector:     {r['Sector']}\n")
            ts_buf.write(f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

            ts_buf.write(f"{'─'*40}\n")
            ts_buf.write(f"SCORES\n")
            ts_buf.write(f"{'─'*40}\n")
            ts_buf.write(f"Composite:      {r.get('Composite', '--')}\n")
            ts_buf.write(f"Screen:         {r.get('Screen', '--')}\n")
            ts_buf.write(f"Lev/Value:      {r.get('LevValue', '--')}\n")
            ts_buf.write(f"Deleveraging:   {r.get('Delever', '--')}\n")
            ts_buf.write(f"Quality:        {r.get('Quality', '--')}\n")
            ts_buf.write(f"Momentum:       {r.get('Momentum', '--')}\n\n")

            ts_buf.write(f"{'─'*40}\n")
            ts_buf.write(f"VALUATION\n")
            ts_buf.write(f"{'─'*40}\n")
            ts_buf.write(f"Price:          {f_price(r.get('current_price'))}\n")
            ts_buf.write(f"Market Cap:     {f_jpy(r.get('market_cap'))}\n")
            ts_buf.write(f"P/E:            {f_num(r.get('pe_trailing'))}\n")
            ts_buf.write(f"P/B:            {f_num(r.get('pb_ratio'))}\n")
            ts_buf.write(f"EV/EBITDA:      {f_num(r.get('ev_to_ebitda'))}\n")
            ts_buf.write(f"EBITDA/EV:      {f_pct(r.get('ebitda_to_ev'))}\n")
            ts_buf.write(f"Div Yield:      {f_pct(r.get('dividend_yield'))}\n")
            ts_buf.write(f"FCF Yield:      {f_pct(r.get('fcf_yield'))}\n\n")

            ts_buf.write(f"{'─'*40}\n")
            ts_buf.write(f"LEVERAGE\n")
            ts_buf.write(f"{'─'*40}\n")
            ts_buf.write(f"LT Debt/EV:     {f_pct(r.get('lt_debt_to_ev'))}\n")
            ts_buf.write(f"Net Debt/EBITDA:{f_num(r.get('net_debt_to_ebitda'))}\n")
            ts_buf.write(f"D/E:            {f_num(r.get('debt_to_equity'), 0)}\n")
            ts_buf.write(f"Current Ratio:  {f_num(r.get('current_ratio'))}\n\n")

            ts_buf.write(f"{'─'*40}\n")
            ts_buf.write(f"QUALITY\n")
            ts_buf.write(f"{'─'*40}\n")
            ts_buf.write(f"ROE:            {f_pct(r.get('roe'))}\n")
            ts_buf.write(f"Op Margin:      {f_pct(r.get('operating_margin'))}\n")
            ts_buf.write(f"GrProfit/Assets:{f_pct(r.get('gross_profit_to_assets'))}\n")
            ts_buf.write(f"Asset Turnover: {f_num(r.get('asset_turnover'))}\n\n")

            if "governance_composite" in r.index:
                ts_buf.write(f"{'─'*40}\n")
                ts_buf.write(f"GOVERNANCE\n")
                ts_buf.write(f"{'─'*40}\n")
                ts_buf.write(f"Gov Composite:  {r.get('governance_composite', '--')}\n")
                ts_buf.write(f"Activist Risk:  {r.get('activist_risk', '--')}\n")
                ts_buf.write(f"TSE Pressure:   {r.get('tse_pressure', '--')}\n")
                ts_buf.write(f"Unwind Score:   {r.get('cross_shareholding_unwind_score', '--')}\n\n")

            ts_buf.write(f"{'='*60}\n")
            ts_buf.write(f"BUILT BY NOAH | NOT FINANCIAL ADVICE\n")
            ts_buf.write(f"{'='*60}\n")

            st.download_button(
                f"DOWNLOAD TEAR SHEET: {ts_ticker}",
                ts_buf.getvalue(),
                file_name=f"tearsheet_{ts_ticker.replace('.', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                key="dl_ts",
            )

            st.code(ts_buf.getvalue(), language="text")


# ════════════════════════════════════════════════════════════════
# COMPARE
# ════════════════════════════════════════════════════════════════
with t_comp:
    compare_tickers = st.multiselect("SELECT (2-6)", flt["Ticker"].tolist(),
                                     default=flt["Ticker"].tolist()[:3], max_selections=6, key="comp_sel")

    if len(compare_tickers) >= 2:
        cdf = flt[flt["Ticker"].isin(compare_tickers)]
        colors = [ORANGE, GREEN, "#06b6d4", YELLOW, "#a855f7", RED]

        cl, cr = st.columns(2)

        with cl:
            # Radar overlay
            fig = go.Figure()
            for i, (_, row) in enumerate(cdf.iterrows()):
                vals = [row["LevValue"], row["Delever"], row["Quality"], row["Momentum"], row["LevValue"]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=["LEV/VAL", "DELEVER", "QUAL", "MOM", "LEV/VAL"],
                    fill="toself",
                    fillcolor=f"rgba({int(colors[i][1:3],16)},{int(colors[i][3:5],16)},{int(colors[i][5:7],16)},0.08)",
                    line=dict(color=colors[i], width=2), name=row["Ticker"],
                ))
            fig.update_layout(**chart_layout(380, showlegend=True),
                polar=dict(radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=8), gridcolor=BORDER),
                           angularaxis=dict(tickfont=dict(size=9, color=WHITE), gridcolor=BORDER),
                           bgcolor=BG1),
                title="FACTOR OVERLAY")
            fig.update_layout(legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            # Normalized price
            fig = go.Figure()
            for i, ticker in enumerate(compare_tickers):
                prices = load_prices(ticker, years=3)
                if not prices.empty:
                    if isinstance(prices.columns, pd.MultiIndex):
                        close = prices[("Close", ticker)] if ("Close", ticker) in prices.columns else prices.iloc[:, 0]
                    else:
                        close = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
                    norm = close / close.iloc[0] * 100
                    fig.add_trace(go.Scatter(x=prices.index, y=norm, mode="lines",
                        line=dict(color=colors[i], width=1.5), name=ticker))
            fig.add_hline(y=100, line_dash="dot", line_color=GRAY_DIM)
            fig.update_layout(**chart_layout(380, showlegend=True), title="RELATIVE PERFORMANCE (3Y, INDEXED)")
            fig.update_layout(legend=dict(orientation="h", y=1.08), yaxis_title="INDEXED")
            st.plotly_chart(fig, use_container_width=True)

        # Comparison table
        ctbl = cdf[["Ticker", "Name", "Sector", "Composite", "Screen", "LevValue", "Delever", "Quality", "Momentum",
                     "pe_trailing", "pb_ratio", "ev_to_ebitda", "ebitda_to_ev", "lt_debt_to_ev",
                     "dividend_yield", "roe", "debt_to_equity", "fcf_yield", "beta", "mcap_b", "quality_flags"]].copy()
        ctbl.columns = ["TICKER", "NAME", "SECTOR", "COMP", "SCR", "LEV", "DLV", "QUAL", "MOM",
                        "P/E", "P/B", "EV/EB", "EBITDA/EV", "LTD/EV",
                        "DIV%", "ROE%", "D/E", "FCF_Y", "BETA", "MCAP_B", "QF"]
        for c in ["EBITDA/EV", "LTD/EV", "DIV%", "ROE%", "FCF_Y"]:
            ctbl[c] = ctbl[c].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
        st.dataframe(ctbl, use_container_width=True, hide_index=True)
    else:
        st.info("Select at least 2 stocks.")


# ════════════════════════════════════════════════════════════════
# DASHBOARD — custom chart builder
# ════════════════════════════════════════════════════════════════
with t_dash:
    st.markdown(f'<div style="color:{GRAY}; font-size:9px; margin-bottom:8px;">Select a preset dashboard or build your own. Drag charts around by changing the layout. All charts use the current filtered universe.</div>', unsafe_allow_html=True)

    # Available metrics for charting
    CHART_METRICS = {
        "Alpha": {"col": "Alpha", "label": "MODEL ALPHA", "fmt": ".4f", "group": "Model"},
        "Composite": {"col": "Composite", "label": "COMPOSITE SCORE", "fmt": ".3f", "group": "Model"},
        "LevValue": {"col": "LevValue", "label": "LEV-VALUE SCORE", "fmt": ".3f", "group": "Model"},
        "Delever": {"col": "Delever", "label": "DELEVER SCORE", "fmt": ".3f", "group": "Model"},
        "Quality": {"col": "Quality", "label": "QUALITY SCORE", "fmt": ".3f", "group": "Model"},
        "Momentum": {"col": "Momentum", "label": "MOMENTUM SCORE", "fmt": ".3f", "group": "Model"},
        "P/E": {"col": "pe_trailing", "label": "P/E RATIO", "fmt": ".1f", "group": "Valuation"},
        "P/B": {"col": "pb_ratio", "label": "P/B RATIO", "fmt": ".2f", "group": "Valuation"},
        "EV/EBITDA": {"col": "ev_to_ebitda", "label": "EV/EBITDA", "fmt": ".1f", "group": "Valuation"},
        "EBITDA/EV": {"col": "ebitda_to_ev", "label": "EBITDA/EV", "fmt": ".3f", "group": "Valuation"},
        "Div Yield": {"col": "dividend_yield", "label": "DIV YIELD", "fmt": ".3f", "group": "Valuation"},
        "FCF Yield": {"col": "fcf_yield", "label": "FCF YIELD", "fmt": ".3f", "group": "Valuation"},
        "ROE": {"col": "roe", "label": "ROE", "fmt": ".3f", "group": "Quality"},
        "Op Margin": {"col": "operating_margin", "label": "OP MARGIN", "fmt": ".3f", "group": "Quality"},
        "Revenue Growth": {"col": "revenue_growth", "label": "REV GROWTH", "fmt": ".3f", "group": "Quality"},
        "D/E": {"col": "debt_to_equity", "label": "DEBT/EQUITY", "fmt": ".1f", "group": "Risk"},
        "Beta": {"col": "beta", "label": "BETA", "fmt": ".2f", "group": "Risk"},
        "LTD/EV": {"col": "lt_debt_to_ev", "label": "LT DEBT/EV", "fmt": ".3f", "group": "Risk"},
        "MCap (B)": {"col": "mcap_b", "label": "MCAP (B JPY)", "fmt": ".0f", "group": "Size"},
        "52w Pos": {"col": "52w_pos", "label": "52W POSITION", "fmt": ".2f", "group": "Momentum"},
        "SMA Cross": {"col": "sma_cross", "label": "SMA CROSS", "fmt": ".3f", "group": "Momentum"},
        "Qual Flags": {"col": "quality_flags", "label": "QUALITY FLAGS", "fmt": ".0f", "group": "Quality"},
    }
    metric_names = list(CHART_METRICS.keys())

    CHART_TYPES = ["Scatter", "Histogram", "Bar (Top 20)", "Box by Sector", "Heatmap (Sector Avg)"]

    # ── Preset dashboards ─────────────────────────────────────
    PRESET_DASHBOARDS = {
        "VALUE HUNTER": {
            "desc": "Identify undervalued stocks with strong earnings yield and low multiples",
            "charts": [
                {"type": "Scatter", "x": "EV/EBITDA", "y": "EBITDA/EV", "color": "Alpha", "title": "VALUATION vs EARNINGS YIELD"},
                {"type": "Scatter", "x": "P/B", "y": "ROE", "color": "Composite", "title": "P/B vs ROE (VALUE TRAP FILTER)"},
                {"type": "Histogram", "x": "EV/EBITDA", "title": "EV/EBITDA DISTRIBUTION"},
                {"type": "Bar (Top 20)", "x": "EBITDA/EV", "title": "TOP 20 EARNINGS YIELD"},
            ],
        },
        "RISK MONITOR": {
            "desc": "Track portfolio risk exposures across leverage, volatility, and concentration",
            "charts": [
                {"type": "Scatter", "x": "Beta", "y": "D/E", "color": "Alpha", "title": "BETA vs LEVERAGE"},
                {"type": "Box by Sector", "x": "Beta", "title": "BETA BY SECTOR"},
                {"type": "Scatter", "x": "LTD/EV", "y": "FCF Yield", "color": "Composite", "title": "LEVERAGE vs FCF YIELD"},
                {"type": "Heatmap (Sector Avg)", "x": "Beta", "title": "SECTOR RISK HEATMAP"},
            ],
        },
        "QUALITY SCREEN": {
            "desc": "Find high-quality compounders with strong margins, ROE, and growth",
            "charts": [
                {"type": "Scatter", "x": "ROE", "y": "Op Margin", "color": "Quality", "title": "ROE vs OPERATING MARGIN"},
                {"type": "Bar (Top 20)", "x": "Qual Flags", "title": "TOP 20 QUALITY FLAGS"},
                {"type": "Histogram", "x": "ROE", "title": "ROE DISTRIBUTION"},
                {"type": "Box by Sector", "x": "Op Margin", "title": "MARGIN BY SECTOR"},
            ],
        },
        "MOMENTUM RADAR": {
            "desc": "Spot price momentum and trend strength across the universe",
            "charts": [
                {"type": "Scatter", "x": "52w Pos", "y": "SMA Cross", "color": "Momentum", "title": "52W POSITION vs SMA CROSS"},
                {"type": "Scatter", "x": "52w Pos", "y": "Alpha", "color": "Composite", "title": "MOMENTUM vs ALPHA"},
                {"type": "Bar (Top 20)", "x": "Momentum", "title": "TOP 20 MOMENTUM SCORES"},
                {"type": "Histogram", "x": "52w Pos", "title": "52W POSITION DISTRIBUTION"},
            ],
        },
        "ALPHA DECOMPOSITION": {
            "desc": "Understand what drives alpha — factor exposures, model agreement, conviction",
            "charts": [
                {"type": "Scatter", "x": "Composite", "y": "Alpha", "color": "Momentum", "title": "SCORE vs MODEL ALPHA"},
                {"type": "Scatter", "x": "LevValue", "y": "Quality", "color": "Alpha", "title": "VALUE-QUALITY MAP"},
                {"type": "Bar (Top 20)", "x": "Alpha", "title": "TOP 20 ALPHA"},
                {"type": "Histogram", "x": "Alpha", "title": "ALPHA DISTRIBUTION"},
            ],
        },
    }

    # Dashboard selector
    dash_mode = st.radio("MODE", ["PRESET", "CUSTOM"], horizontal=True, key="dash_mode",
                          help="PRESET: pre-built dashboards. CUSTOM: build your own chart layout.")

    if dash_mode == "PRESET":
        preset_name = st.selectbox("DASHBOARD", list(PRESET_DASHBOARDS.keys()), key="dash_preset")
        preset = PRESET_DASHBOARDS[preset_name]
        st.markdown(f'<div style="color:{GRAY};font-size:10px;margin-bottom:10px;">{preset["desc"]}</div>', unsafe_allow_html=True)
        charts_to_render = preset["charts"]
    else:
        # ── Custom chart builder ──────────────────────────────
        st.markdown("### BUILD YOUR DASHBOARD")

        if "custom_charts" not in st.session_state:
            st.session_state.custom_charts = []

        # Add chart form
        with st.expander("ADD CHART", expanded=len(st.session_state.custom_charts) == 0):
            cc1, cc2 = st.columns(2)
            with cc1:
                new_chart_type = st.selectbox("CHART TYPE", CHART_TYPES, key="new_chart_type")
                new_chart_x = st.selectbox("X-AXIS / METRIC", metric_names, key="new_chart_x")
            with cc2:
                new_chart_title = st.text_input("TITLE (optional)", "", key="new_chart_title")
                if new_chart_type == "Scatter":
                    new_chart_y = st.selectbox("Y-AXIS", metric_names, index=min(1, len(metric_names)-1), key="new_chart_y")
                    new_chart_color = st.selectbox("COLOR BY", ["None"] + metric_names, key="new_chart_color")
                else:
                    new_chart_y = None
                    new_chart_color = None

            if st.button("ADD CHART", key="dash_add_chart"):
                chart_def = {
                    "type": new_chart_type,
                    "x": new_chart_x,
                    "y": new_chart_y,
                    "color": new_chart_color if new_chart_color != "None" else None,
                    "title": new_chart_title or f"{new_chart_type.upper()}: {new_chart_x}",
                }
                st.session_state.custom_charts.append(chart_def)
                st.rerun()

        # Management row
        if st.session_state.custom_charts:
            mgmt = st.columns([1, 1, 4])
            with mgmt[0]:
                if st.button("CLEAR ALL CHARTS", key="dash_clear"):
                    st.session_state.custom_charts = []
                    st.rerun()
            with mgmt[1]:
                if st.button("REMOVE LAST", key="dash_rm_last"):
                    st.session_state.custom_charts.pop()
                    st.rerun()

        charts_to_render = st.session_state.custom_charts

    # ── Render charts ─────────────────────────────────────────
    if charts_to_render:
        # 2-column grid
        for row_start in range(0, len(charts_to_render), 2):
            row_charts = charts_to_render[row_start:row_start + 2]
            cols = st.columns(len(row_charts))
            for ci, chart in enumerate(row_charts):
                with cols[ci]:
                    ctype = chart["type"]
                    cx_name = chart.get("x", "Alpha")
                    cy_name = chart.get("y")
                    ccolor_name = chart.get("color")
                    ctitle = chart.get("title", ctype)

                    cx = CHART_METRICS.get(cx_name, {})
                    cy = CHART_METRICS.get(cy_name, {}) if cy_name else None
                    cc = CHART_METRICS.get(ccolor_name, {}) if ccolor_name else None

                    cx_col = cx.get("col", "Alpha")
                    fig = go.Figure()

                    if ctype == "Scatter" and cy:
                        cy_col = cy.get("col", "Composite")
                        x_vals = flt[cx_col].astype(float) if cx_col in flt.columns else pd.Series(dtype=float)
                        y_vals = flt[cy_col].astype(float) if cy_col in flt.columns else pd.Series(dtype=float)

                        marker_args = dict(size=6, opacity=0.7, line=dict(width=0.5, color=BORDER))
                        if cc and cc.get("col") in flt.columns:
                            color_vals = flt[cc["col"]].astype(float)
                            marker_args["color"] = color_vals
                            marker_args["colorscale"] = [[0, RED], [0.5, GRAY], [1, GREEN]]
                            marker_args["showscale"] = True
                            marker_args["colorbar"] = dict(thickness=10, len=0.5, title=dict(text=cc.get("label", ""), font=dict(size=8)))
                        else:
                            marker_args["color"] = ORANGE

                        fig.add_trace(go.Scatter(
                            x=x_vals, y=y_vals,
                            mode="markers",
                            marker=marker_args,
                            text=flt["Ticker"] if "Ticker" in flt.columns else None,
                            hovertemplate="%{text}<br>" + cx.get("label","X") + ": %{x}<br>" + cy.get("label","Y") + ": %{y}<extra></extra>",
                        ))
                        fig.update_layout(**chart_layout(320, title=ctitle,
                            xaxis=dict(title=cx.get("label", cx_name)),
                            yaxis=dict(title=cy.get("label", cy_name or ""))))

                    elif ctype == "Histogram":
                        vals = flt[cx_col].dropna().astype(float) if cx_col in flt.columns else pd.Series(dtype=float)
                        fig.add_trace(go.Histogram(x=vals, nbinsx=30, marker_color=ORANGE, opacity=0.8))
                        median_val = vals.median() if len(vals) > 0 else 0
                        fig.add_vline(x=median_val, line_dash="dot", line_color=YELLOW, annotation_text=f"MED: {median_val:.2f}",
                                       annotation_font=dict(size=8, color=YELLOW))
                        fig.update_layout(**chart_layout(320, title=ctitle, xaxis=dict(title=cx.get("label", cx_name))))

                    elif ctype == "Bar (Top 20)":
                        if cx_col in flt.columns:
                            top = flt.nlargest(20, cx_col)
                            fig.add_trace(go.Bar(
                                x=top["Ticker"], y=top[cx_col].astype(float),
                                marker_color=ORANGE,
                                text=top[cx_col].apply(lambda v: f"{v:{cx.get('fmt', '.2f')}}" if pd.notna(v) else ""),
                                textposition="outside", textfont=dict(size=7, color=WHITE),
                            ))
                        fig.update_layout(**chart_layout(320, title=ctitle,
                            xaxis=dict(tickangle=-45, tickfont=dict(size=7)), yaxis=dict(title=cx.get("label", cx_name))))

                    elif ctype == "Box by Sector":
                        if cx_col in flt.columns and "Sector" in flt.columns:
                            sectors_sorted = flt.groupby("Sector")[cx_col].median().sort_values(ascending=False).index
                            sector_colors = [ORANGE, GREEN, YELLOW, "#6699ff", "#cc66ff", RED, GRAY, "#ff6699", ORANGE_DIM, "#66cccc"]
                            for si, sect in enumerate(sectors_sorted):
                                sdata = flt[flt["Sector"] == sect][cx_col].dropna().astype(float)
                                if len(sdata) > 2:
                                    fig.add_trace(go.Box(
                                        y=sdata, name=sect[:12],
                                        marker_color=sector_colors[si % len(sector_colors)],
                                        boxmean=True, line=dict(width=1),
                                    ))
                        fig.update_layout(**chart_layout(320, title=ctitle, showlegend=False,
                            xaxis=dict(tickangle=-45, tickfont=dict(size=7))))

                    elif ctype == "Heatmap (Sector Avg)":
                        if cx_col in flt.columns and "Sector" in flt.columns:
                            # Sector x multiple metrics heatmap
                            heat_metrics = ["Alpha", "Composite", cx_name]
                            heat_cols = []
                            for hm in heat_metrics:
                                hc = CHART_METRICS.get(hm, {}).get("col")
                                if hc and hc in flt.columns:
                                    heat_cols.append((hm, hc))

                            if heat_cols:
                                sectors_list = sorted(flt["Sector"].dropna().unique())
                                z_data = []
                                for hm_name, hm_col in heat_cols:
                                    row = []
                                    for sect in sectors_list:
                                        val = flt[flt["Sector"] == sect][hm_col].astype(float).mean()
                                        row.append(round(val, 3) if pd.notna(val) else 0)
                                    z_data.append(row)

                                fig = go.Figure(data=go.Heatmap(
                                    z=z_data,
                                    x=[s[:12] for s in sectors_list],
                                    y=[h[0] for h in heat_cols],
                                    colorscale=[[0, RED], [0.5, BG1], [1, GREEN]],
                                    text=[[f"{v:.3f}" for v in row] for row in z_data],
                                    texttemplate="%{text}",
                                    textfont=dict(size=8),
                                ))
                        fig.update_layout(**chart_layout(320, title=ctitle,
                            xaxis=dict(tickangle=-45, tickfont=dict(size=7))))

                    st.plotly_chart(fig, use_container_width=True)

    elif dash_mode == "CUSTOM":
        st.markdown(f"""<div style="text-align:center; padding:50px; color:{GRAY};">
        <div style="font-size:12px; color:{ORANGE}; letter-spacing:2px; margin-bottom:12px;">NO CHARTS</div>
        <div style="font-size:10px;">Click ADD CHART above to build your custom dashboard.<br>
        Choose chart types, metrics, and colors to visualize the data you care about.</div>
        </div>""", unsafe_allow_html=True)


# ── MODEL tab ──────────────────────────────────────────────────
with t_mdl:
    st.markdown("### MULTI-MODEL ALPHA SYSTEM")
    st.markdown(f"""<div class="panel"><div class="panel-title">SIGNAL GENERATION ENGINE</div>
    <span style="color:{GRAY};font-size:10px;">
    Four independent alpha models blended into a single composite signal. Each model captures
    different return drivers: fundamental value (rank model), price dynamics (momentum),
    nonlinear factor interactions (ML ensemble), and short-term reversal (mean reversion).
    The ALPHA column in SCREEN reflects this blended output.
    </span></div>""", unsafe_allow_html=True)

    alpha_out = _alpha_out
    _mdl_blended = pd.Series(alpha_out["blended_alpha"])
    model_weights = alpha_out["weights"]
    model_sigs_mdl = alpha_out["signals"]
    ph_count = alpha_out.get("price_hist_count", 0)

    # ── Configurable Weights ──────────────────────────────────
    st.markdown("### BLEND WEIGHTS")
    st.markdown(f'<span style="color:{GRAY};font-size:9px;">Adjust model weights and click RECOMPUTE to re-blend. Weights auto-normalize to 100%.</span>', unsafe_allow_html=True)

    wt_cols = st.columns(5)
    with wt_cols[0]:
        w_rank = st.slider("FACTOR RANK", 0, 100, 40, 5, key="mdl_w_rank")
    with wt_cols[1]:
        w_mom = st.slider("MOMENTUM", 0, 100, 20, 5, key="mdl_w_mom")
    with wt_cols[2]:
        w_ml = st.slider("ML ENSEMBLE", 0, 100, 20, 5, key="mdl_w_ml")
    with wt_cols[3]:
        w_rev = st.slider("MEAN REVERSION", 0, 100, 20, 5, key="mdl_w_rev")
    with wt_cols[4]:
        st.markdown("<br>", unsafe_allow_html=True)
        recompute = st.button("RECOMPUTE", key="mdl_recompute")

    if recompute:
        total_w = w_rank + w_mom + w_ml + w_rev
        if total_w > 0:
            alpha_out = _compute_alpha(
                _df_hash, df.to_json(),
                w_rank=w_rank / total_w, w_mom=w_mom / total_w,
                w_ml=w_ml / total_w, w_rev=w_rev / total_w,
            )
            _mdl_blended = pd.Series(alpha_out["blended_alpha"])
            model_weights = alpha_out["weights"]
            model_sigs_mdl = alpha_out["signals"]
            st.success(f"Recomputed with weights: {w_rank}/{w_mom}/{w_ml}/{w_rev} (normalized)")

    # Summary metrics
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("UNIVERSE", len(df))
    mc2.metric("MODELS", len(model_sigs_mdl))
    n_active = sum(1 for s in model_sigs_mdl.values() if s["coverage"] > 0.1)
    mc3.metric("ACTIVE", n_active)
    mc4.metric("BLEND", " / ".join(f"{int(w*100)}%" for w in model_weights.values()))
    mc5.metric("ALPHA SPREAD", f"{_mdl_blended.quantile(0.9) - _mdl_blended.quantile(0.1):.2f}")
    mc6.metric("PRICE DATA", f"{ph_count} stocks")

    # ── Per-model breakdown ───────────────────────────────────
    st.markdown("### MODEL BREAKDOWN")

    model_colors = {"FACTOR_RANK": ORANGE, "MOMENTUM": "#6699ff", "ML_ENSEMBLE": GREEN, "MEAN_REVERSION": "#cc66ff"}

    for mname, mdata in model_sigs_mdl.items():
        color = model_colors.get(mname, GRAY)
        weight_pct = model_weights.get(mname, 0) * 100
        cov = mdata["coverage"] * 100
        meta = mdata["metadata"]
        alpha_s = pd.Series(mdata["alpha"])

        st.markdown(f"""<div class="panel">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <span style="color:{color};font-size:11px;font-weight:600;letter-spacing:1.5px;">{mname.replace("_"," ")}</span>
            <span style="color:{GRAY};font-size:9px;">WEIGHT: {weight_pct:.0f}%</span>
            <span style="color:{GRAY};font-size:9px;">|</span>
            <span style="color:{GRAY};font-size:9px;">COVERAGE: {cov:.0f}%</span>
            <span style="color:{GRAY};font-size:9px;">|</span>
            <span style="color:{GRAY};font-size:9px;">METHOD: {meta.get("method","--")}</span>
            {"" if "components" not in meta else f'<span style="color:{GRAY};font-size:9px;">| COMPONENTS: {", ".join(meta["components"])}</span>'}
        </div>
        </div>""", unsafe_allow_html=True)

        m_left, m_right = st.columns([1, 1])

        with m_left:
            fig_m = go.Figure()
            fig_m.add_trace(go.Histogram(
                x=alpha_s.values, nbinsx=35,
                marker_color=color, opacity=0.8,
            ))
            fig_m.add_vline(x=0, line_dash="dot", line_color=GRAY_DIM)
            fig_m.update_layout(**chart_layout(220, title=f"{mname.replace('_',' ')} ALPHA DISTRIBUTION"))
            st.plotly_chart(fig_m, use_container_width=True)

        with m_right:
            top_picks = alpha_s.sort_values(ascending=False).head(10)
            tp_df = pd.DataFrame({"TICKER": top_picks.index, "ALPHA": top_picks.values.round(4), "#": range(1, len(top_picks)+1)})
            st.dataframe(tp_df, use_container_width=True, hide_index=True, height=220)

        # Model-specific metadata
        if mname == "ML_ENSEMBLE" and "feature_importances" in meta:
            imp = meta["feature_importances"]
            if imp:
                fig_imp = go.Figure()
                sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
                fig_imp.add_trace(go.Bar(
                    x=[v for _, v in sorted_imp],
                    y=[k.upper().replace("_"," ")[:16] for k, _ in sorted_imp],
                    orientation="h",
                    marker_color=GREEN,
                ))
                fig_imp.update_layout(**chart_layout(250, title="ML FEATURE IMPORTANCE (TOP 10)"))
                st.plotly_chart(fig_imp, use_container_width=True)

    # ── Signal Correlation ────────────────────────────────────
    st.markdown("### SIGNAL CORRELATION")
    st.markdown(f'<span style="color:{GRAY};font-size:10px;">Low inter-model correlation means diversified alpha sources. '
                 f'High correlation means models agree, which can strengthen conviction but reduces diversification.</span>',
                 unsafe_allow_html=True)

    corr_dict = alpha_out["corr"]
    if corr_dict:
        corr_df = pd.DataFrame(corr_dict)
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=[c.replace("_", " ") for c in corr_df.columns],
            y=[c.replace("_", " ") for c in corr_df.index],
            colorscale=[[0, RED], [0.5, BG1], [1, GREEN]],
            zmid=0, zmin=-1, zmax=1,
            text=corr_df.round(3).values,
            texttemplate="%{text}",
            textfont=dict(size=11),
        ))
        fig_corr.update_layout(**chart_layout(300, title="INTER-MODEL CORRELATIONS"))
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Quintile Analysis ─────────────────────────────────────
    st.markdown("### QUINTILE ANALYSIS")
    st.markdown(f'<span style="color:{GRAY};font-size:10px;">A good alpha signal shows monotonic increase in composite score from Q1 (worst alpha) to Q5 (best alpha). '
                 f'High Q5-Q1 spread indicates strong discriminative power.</span>', unsafe_allow_html=True)

    qa_tabs = st.columns(len(model_sigs_mdl) + 1)
    qa_names = list(model_sigs_mdl.keys()) + ["BLENDED"]
    qa_alphas = [pd.Series(model_sigs_mdl[n]["alpha"]) for n in model_sigs_mdl] + [_mdl_blended]
    qa_colors_list = [model_colors.get(n, GRAY) for n in model_sigs_mdl] + [ORANGE]

    for qi, (qa_col, qa_name, qa_alpha, qa_color) in enumerate(zip(qa_tabs, qa_names, qa_alphas, qa_colors_list)):
        with qa_col:
            qa_result = quintile_analysis(qa_alpha, df)
            if "error" not in qa_result:
                q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
                q_scores = [qa_result[q]["avg_metric"] for q in q_labels]
                q_counts = [qa_result[q]["count"] for q in q_labels]
                fig_qa = go.Figure()
                fig_qa.add_trace(go.Bar(
                    x=q_labels, y=q_scores,
                    marker_color=[RED, ORANGE_DIM, YELLOW, GREEN, qa_color],
                    text=[f"{v:.1f}" for v in q_scores],
                    textposition="outside", textfont=dict(size=8, color=WHITE),
                ))
                mono_str = f"MONO: {qa_result['monotonicity']:.0%}"
                spread_str = f"SPREAD: {qa_result['q5_q1_spread']:.2f}"
                fig_qa.update_layout(**chart_layout(200, title=f"{qa_name[:10]} ({mono_str} | {spread_str})",
                    yaxis=dict(title="AVG SCORE")))
                st.plotly_chart(fig_qa, use_container_width=True)

    # ── Signal Diagnostics ────────────────────────────────────
    st.markdown("### SIGNAL DIAGNOSTICS")

    diag = signal_diagnostics(_mdl_blended, df)
    diag_cols = st.columns(6)
    diag_items = [
        ("COVERAGE", f"{diag.get('coverage', 0)*100:.0f}%"),
        ("SKEW", f"{diag.get('skew', 0):.2f}"),
        ("KURTOSIS", f"{diag.get('kurtosis', 0):.2f}"),
        ("% POSITIVE", f"{diag.get('pct_positive', 0):.0f}%"),
        ("90-10 SPREAD", f"{diag.get('spread_90_10', 0):.2f}"),
        ("IC vs SCORE", f"{diag.get('ic_vs_composite', 0):.3f}"),
    ]
    for di, (dlabel, dval) in enumerate(diag_items):
        diag_cols[di].metric(dlabel, dval)

    # Distribution statistics table
    diag_left, diag_right = st.columns(2)
    with diag_left:
        st.markdown(f'<div style="color:{ORANGE};font-size:10px;letter-spacing:1px;margin:8px 0 4px;">BLENDED ALPHA DISTRIBUTION</div>', unsafe_allow_html=True)
        dist_data = {
            "STATISTIC": ["Mean", "Std Dev", "Skewness", "Kurtosis", "Q10", "Q25", "Median", "Q75", "Q90"],
            "VALUE": [diag.get(k, 0) for k in ["mean", "std", "skew", "kurtosis", "q10", "q25", "median", "q75", "q90"]],
        }
        st.dataframe(pd.DataFrame(dist_data), use_container_width=True, hide_index=True, height=300)

    with diag_right:
        # Scatter: blended alpha vs composite score
        if "Composite" in df.columns and "Ticker" in df.columns:
            scatter_df = pd.DataFrame({
                "Alpha": _mdl_blended.reindex(df["Ticker"]).values,
                "Score": df["Composite"].values,
                "Ticker": df["Ticker"].values,
            }).dropna()
            if len(scatter_df) > 5:
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=scatter_df["Alpha"], y=scatter_df["Score"],
                    mode="markers", marker=dict(size=4, color=ORANGE, opacity=0.6),
                    text=scatter_df["Ticker"], hovertemplate="%{text}<br>Alpha: %{x:.3f}<br>Score: %{y:.1f}",
                ))
                # Trend line
                z = np.polyfit(scatter_df["Alpha"], scatter_df["Score"], 1)
                p = np.poly1d(z)
                x_range = np.linspace(scatter_df["Alpha"].min(), scatter_df["Alpha"].max(), 50)
                fig_sc.add_trace(go.Scatter(
                    x=x_range, y=p(x_range),
                    mode="lines", line=dict(color=GREEN, width=1, dash="dash"), name="FIT",
                ))
                fig_sc.update_layout(**chart_layout(300, title="ALPHA vs COMPOSITE SCORE",
                    xaxis=dict(title="BLENDED ALPHA"), yaxis=dict(title="COMPOSITE SCORE")))
                st.plotly_chart(fig_sc, use_container_width=True)

    # ── Blended Alpha Output ──────────────────────────────────
    st.markdown("### BLENDED ALPHA — TOP 25")

    alpha_sorted = _mdl_blended.sort_values(ascending=False)
    top_n = min(25, len(alpha_sorted))
    blend_df = pd.DataFrame({
        "#": range(1, top_n + 1),
        "TICKER": alpha_sorted.head(top_n).index,
        "ALPHA": alpha_sorted.head(top_n).values.round(4),
    })
    for mname, mdata in model_sigs_mdl.items():
        ms = pd.Series(mdata["alpha"])
        blend_df[mname.replace("_"," ")[:8]] = blend_df["TICKER"].map(ms).round(3)

    col_cfg = {
        "#": st.column_config.NumberColumn("RANK", help="Blended alpha rank"),
        "ALPHA": st.column_config.NumberColumn("BLENDED", help="Final blended z-scored alpha", format="%.4f"),
    }
    st.dataframe(blend_df, use_container_width=True, hide_index=True, column_config=col_cfg)

    # ── Portfolio Construction ────────────────────────────────
    st.markdown("### OPTIMIZED PORTFOLIO")
    st.markdown(f'<span style="color:{GRAY};font-size:10px;">Mean-variance optimization on blended alpha signal. '
                 f'Ledoit-Wolf shrinkage covariance. Long-only, 5% max position, 25% max sector.</span>',
                 unsafe_allow_html=True)

    # Build returns panel for covariance estimation
    _ret_panel = build_returns_panel(
        _mdl_blended.nlargest(50).index.tolist(),
        lambda t: fetch_price_history(t, years=2, cache_dir="data/cache"),
        months=12,
    )

    sectors_map = pd.Series(dict(zip(df["Ticker"], df["Sector"]))) if "Sector" in df.columns else None
    port_result = optimize_portfolio(
        alpha=_mdl_blended,
        returns_panel=_ret_panel if not _ret_panel.empty else None,
        sectors=sectors_map,
        constraints=PortfolioConstraints(max_position=0.05, max_names=30, min_names=10, max_sector_weight=0.25),
    )

    if port_result.n_holdings > 0:
        pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
        pc1.metric("HOLDINGS", port_result.n_holdings)
        pc2.metric("EX-ANTE SHARPE", f"{port_result.sharpe_ratio:.3f}")
        pc3.metric("ACTIVE RISK", f"{port_result.active_risk*100:.2f}%")
        pc4.metric("TOP WEIGHT", f"{port_result.weights.max()*100:.1f}%")
        pc5.metric("TURNOVER", f"{port_result.turnover*100:.0f}%")
        pc6.metric("COV MATRIX", "REAL" if not _ret_panel.empty else "IDENTITY")

        pl, pr_ = st.columns(2)
        with pl:
            pw = port_result.weights.head(20)
            fig_pw = go.Figure()
            fig_pw.add_trace(go.Bar(
                x=pw.index, y=pw.values * 100,
                marker_color=ORANGE,
                text=[f"{v*100:.1f}%" for v in pw.values],
                textposition="outside", textfont=dict(size=8, color=WHITE),
            ))
            fig_pw.update_layout(**chart_layout(320, title="PORTFOLIO WEIGHTS (TOP 20)",
                xaxis=dict(tickangle=-45, tickfont=dict(size=8)), yaxis=dict(title="WEIGHT %")))
            st.plotly_chart(fig_pw, use_container_width=True)

        with pr_:
            if port_result.sector_weights:
                sects = port_result.sector_weights
                fig_sect = go.Figure(data=[go.Pie(
                    labels=list(sects.keys()),
                    values=[v * 100 for v in sects.values()],
                    hole=0.4, textinfo="label+percent",
                    textfont=dict(size=9, color=WHITE),
                    marker=dict(colors=[ORANGE, GREEN, YELLOW, "#6699ff", "#cc66ff",
                                        RED, GRAY, "#ff6699", ORANGE_DIM, "#66cccc"]),
                )])
                fig_sect.update_layout(**chart_layout(320, title="SECTOR ALLOCATION"))
                st.plotly_chart(fig_sect, use_container_width=True)

    # ── Methodology ───────────────────────────────────────────
    st.markdown("### METHODOLOGY")
    st.markdown(f"""<div class="panel">
    <div class="panel-title">MULTI-MODEL ARCHITECTURE</div>
    <span style="color:{GRAY};font-size:10px;">
    <b style="color:{ORANGE};">MODEL 1: FACTOR RANK (40%)</b> — Cross-sectional percentile ranking on 13 fundamental factors
    across value, leverage, quality, and momentum groups. Each factor is rank-normalized to [0,1],
    directionally adjusted (lower P/E = higher rank), and combined via weighted sum. Works on a single
    snapshot with no estimation history required.<br><br>
    <b style="color:{ORANGE};">MODEL 2: MOMENTUM (20%)</b> — Price-derived signals: 12-1 month momentum (skip last month
    for short-term reversal), 52-week relative position, 50/200 SMA crossover, low beta premium,
    and idiosyncratic volatility (low vol = higher signal). Now uses actual yfinance price history
    for richer signals.<br><br>
    <b style="color:{ORANGE};">MODEL 3: ML ENSEMBLE (20%)</b> — Gradient-boosted decision tree (100 trees, depth 3, lr 0.05)
    trained on all 20 fundamental features predicting composite score rank. 5-fold cross-validated
    predictions ensure each stock's alpha is out-of-sample. Captures nonlinear factor interactions.<br><br>
    <b style="color:{ORANGE};">MODEL 4: MEAN REVERSION (20%)</b> — Short-term reversal signals tuned for Japan equities:
    1-month price reversal, RSI contrarian (buy oversold), distance from 200-day MA, 52-week
    contrarian (buy near lows), and capitulation detection (high volume on down-moves). Japan shows
    strong mean-reversion at short horizons (Asness et al., 2013).<br><br>
    <b style="color:{ORANGE};">BLENDING</b> — Individual model alphas are z-scored and combined with configurable weights.
    Default: 40/20/20/20. Final blend is re-z-scored. Low inter-model correlation provides
    diversified alpha sources. Weights can be adjusted in the MODEL tab above.<br><br>
    <b style="color:{ORANGE};">PORTFOLIO</b> — Mean-variance optimization with Ledoit-Wolf constant-correlation shrinkage
    covariance estimated from actual daily returns (12-month window). SLSQP solver maximizes
    alpha exposure minus risk penalty minus turnover cost. Constraints: 5% single name, 25% sector,
    30 max holdings.
    </span></div>""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; color:{GRAY_DIM}; font-size:8px; padding:8px; border-top:1px solid {BORDER}; margin-top:12px;">
JVQ TERMINAL v2.0 | BUILT BY NOAH | {len(df)} EQUITIES | {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")} | DATA: YFINANCE | NOT FINANCIAL ADVICE
</div>
""", unsafe_allow_html=True)
