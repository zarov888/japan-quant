"""
JVQ Terminal — Bloomberg-style Japan equity screener.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.universe import load_universe
from src.data.fetcher import fetch_universe, fetch_price_history
from src.model.scorer import score_universe, results_to_dataframe, load_scoring_config

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
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=9)),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=9)),
        title_font=dict(size=11, color=ORANGE),
        showlegend=False,
    )
    base.update(kw)
    # ensure legend defaults if not overridden
    if "legend" not in base:
        base["legend"] = dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)")
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

    /* sidebar */
    [data-testid="stSidebar"] {{
        background: {BG1};
        border-right: 1px solid {BORDER};
        padding-top: 0.5rem;
    }}
    [data-testid="stSidebar"] .block-container {{
        padding-top: 0.5rem !important;
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
    """Render a data strip."""
    parts = []
    for label, val, css in items:
        parts.append(f'<div class="strip-item"><span class="strip-label">{label}</span><span class="strip-val {css}">{val}</span></div>')
    return '<div class="strip">' + ''.join(parts) + '</div>'

def metric_cell(label, val, css=""):
    return f'<div class="dc"><div class="dc-label">{label}</div><div class="dc-val {css}">{val}</div></div>'


# ── Data ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    config = load_universe("config/universe.yaml")
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
        JAPAN VALUE QUANT TERMINAL
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
        SCORING 74 EQUITIES ACROSS TOPIX CORE UNIVERSE
    </div>
    <div style="margin-top: 40px; color: {GRAY_DIM}; font-size: 8px;">
        BUILT BY NOAH | v1.0
    </div>
</div>
<style>
    @keyframes pulse {{
        0%, 100% {{ opacity: 0.2; transform: scale(0.8); }}
        50% {{ opacity: 1; transform: scale(1.2); }}
    }}
</style>
""", unsafe_allow_html=True)

df, fundamentals, results = load_data()
all_sectors = sorted(df["Sector"].dropna().unique().tolist())
loading.empty()

# ── Top Bar ────────────────────────────────────────────────────
st.markdown(f"""
<div class="bb-top">
    <span class="bb-logo">JVQ</span>
    <span class="bb-tag">JAPAN VALUE QUANT</span>
    <span class="bb-tag">|</span>
    <span class="bb-tag">UNIVERSE {len(df)}</span>
    <span class="bb-tag">|</span>
    <span class="bb-tag">MODEL japan_deep_value_v1</span>
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

    preset = st.selectbox("PRESET", ["Custom", "Deep Value", "Quality Value", "High Dividend", "Momentum Value", "Net Cash"])
    if preset == "Deep Value":
        min_score, max_pe, max_pb, min_div, min_roe = 0.40, 12.0, 1.2, 0.0, 0.0
    elif preset == "Quality Value":
        min_score, max_pe, max_pb, min_div, min_roe = 0.50, 20.0, 3.0, 0.0, 10.0
    elif preset == "High Dividend":
        min_score, max_pe, max_pb, min_div, min_roe = 0.35, 30.0, 5.0, 3.0, 0.0
    elif preset == "Momentum Value":
        min_score, max_pe, max_pb, min_div, min_roe = 0.45, 25.0, 3.0, 0.0, 0.0
    elif preset == "Net Cash":
        min_score, max_pe, max_pb, min_div, min_roe = 0.35, 50.0, 1.5, 0.0, 0.0
    else:
        min_score = st.slider("MIN COMPOSITE", 0.0, 1.0, 0.35, 0.01, format="%.2f")
        max_pe = st.number_input("MAX P/E", value=50.0, step=5.0)
        max_pb = st.number_input("MAX P/B", value=5.0, step=0.5)
        min_div = st.number_input("MIN DIV YIELD %", value=0.0, step=0.5)
        min_roe = st.number_input("MIN ROE %", value=0.0, step=1.0)

    top_n = st.slider("LIMIT", 5, 74, 50)
    sector_filter = st.multiselect("SECTORS", all_sectors, default=[])
    max_de = st.number_input("MAX D/E", value=500.0, step=50.0)
    net_cash_only = st.checkbox("NET CASH POSITIVE ONLY")
    min_quality_flags = st.slider("MIN QUALITY FLAGS (0-7)", 0, 7, 0)

# ── Filter ─────────────────────────────────────────────────────
flt = df.copy()
flt = flt[flt["Composite"] >= min_score]
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
st.markdown(strip_html([
    ("PASSING", f"{len(flt)}/{len(df)}", "dc-orange"),
    ("TOP", f"{flt['Composite'].max():.3f}" if len(flt) else "--", "dc-orange"),
    ("MED COMP", f"{flt['Composite'].median():.3f}" if len(flt) else "--", ""),
    ("MED P/B", f_num(flt["pb_ratio"].median()), ""),
    ("MED P/E", f_num(flt["pe_trailing"].median()), ""),
    ("MED DIV", f_pct(flt["dividend_yield"].median()), ""),
    ("MED ROE", f_pct(flt["roe"].median()), ""),
    ("TOT MCAP", f_jpy(flt["market_cap"].sum()), ""),
    ("SECTORS", str(flt["Sector"].nunique()) if len(flt) else "0", ""),
    ("AVG BETA", f_num(flt["beta"].median()), ""),
    ("AVG QLTY", f_num(flt["quality_flags"].median(), 0), ""),
]), unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────
t_scr, t_idx, t_sec, t_stk, t_val, t_qual, t_risk, t_tech, t_comp = st.tabs([
    "SCREEN", "INDEX", "SECTOR", "EQUITY", "VALUATION", "QUALITY", "RISK", "TECHNICALS", "COMPARE",
])


# ════════════════════════════════════════════════════════════════
# SCREEN
# ════════════════════════════════════════════════════════════════
with t_scr:
    # Main table
    tcols = ["Ticker", "Name", "Sector", "Composite", "Value", "Quality", "Strength", "Momentum",
             "pe_trailing", "pb_ratio", "ev_to_ebitda", "dividend_yield", "roe",
             "debt_to_equity", "cash_to_mcap", "fcf_yield", "beta", "mcap_b", "quality_flags"]
    tdf = flt[tcols].copy()
    tdf.columns = ["TICKER", "NAME", "SECTOR", "COMP", "VAL", "QUAL", "STR", "MOM",
                    "P/E", "P/B", "EV/EBITDA", "DIV%", "ROE%", "D/E", "CASH/MC", "FCF_Y", "BETA", "MCAP_B", "QF"]
    for c in ["DIV%", "ROE%", "CASH/MC", "FCF_Y"]:
        tdf[c] = tdf[c].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
    tdf["MCAP_B"] = tdf["MCAP_B"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "--")
    st.dataframe(tdf, use_container_width=True, height=700)

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
        for col, clr in [("Value", ORANGE), ("Quality", GREEN), ("Strength", YELLOW), ("Momentum", "#06b6d4")]:
            fig.add_trace(go.Bar(x=flt["Ticker"].head(20), y=flt[col].head(20), name=col.upper()[:3], marker_color=clr, marker_line_width=0))
        fig.update_layout(**chart_layout(250, barmode="stack", showlegend=True), title="SCORE DECOMPOSITION (TOP 20)")
        fig.update_layout(legend=dict(orientation="h", y=1.12), xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
        st.plotly_chart(fig, use_container_width=True)


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
        fig.update_layout(**chart_layout(450, showlegend=True), title="VALUATION MAP",
                          xaxis_title="P/E", yaxis_title="P/B",
                          legend=dict(font=dict(size=8), orientation="v", x=1.02))
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
            ("N", str(len(sdf)), "dc-orange"),
            ("AVG COMP", f_num(sdf["Composite"].mean(), 3), ""),
            ("MED P/B", f_num(sdf["pb_ratio"].median()), ""),
            ("MED P/E", f_num(sdf["pe_trailing"].median()), ""),
            ("MED DIV", f_pct(sdf["dividend_yield"].median()), ""),
            ("MED ROE", f_pct(sdf["roe"].median()), ""),
            ("TOT MCAP", f_jpy(sdf["market_cap"].sum()), ""),
        ]), unsafe_allow_html=True)

        sl, sr = st.columns(2)
        with sl:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sdf["Ticker"], y=sdf["Composite"], marker_color=ORANGE, marker_line_width=0))
            fig.update_layout(**chart_layout(300), title=f"COMPOSITE: {sel_sec}",
                              xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
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
        stbl = sdf[["Ticker", "Name", "Composite", "Value", "Quality", "Strength", "Momentum",
                     "pe_trailing", "pb_ratio", "dividend_yield", "roe", "ev_to_ebitda",
                     "debt_to_equity", "cash_to_mcap", "fcf_yield", "beta", "mcap_b", "quality_flags"]].copy()
        stbl.columns = ["TICKER", "NAME", "COMP", "VAL", "QUAL", "STR", "MOM",
                        "P/E", "P/B", "DIV%", "ROE%", "EV/EB", "D/E", "CASH/MC", "FCF_Y", "BETA", "MCAP_B", "QF"]
        for c in ["DIV%", "ROE%", "CASH/MC", "FCF_Y"]:
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
        st.markdown(strip_html([
            ("COMPOSITE", f_num(r["Composite"], 3), "dc-orange"),
            ("VALUE", f_num(r["Value"], 3), ""),
            ("QUALITY", f_num(r["Quality"], 3), ""),
            ("STRENGTH", f_num(r["Strength"], 3), ""),
            ("MOMENTUM", f_num(r["Momentum"], 3), ""),
            ("QUALITY FLAGS", f_num(r["quality_flags"], 0), "dc-orange" if r["quality_flags"] >= 5 else ""),
            ("RANK", f"#{flt.index.get_loc(r.name)+1}/{len(flt)}", "dc-orange"),
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
            vals = [r["Value"], r["Quality"], r["Strength"], r["Momentum"], r["Value"]]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=["VAL", "QUAL", "STR", "MOM", "VAL"],
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
            st.markdown("### BALANCE SHEET")
            bdata = [
                ("Market Cap", f_jpy(r.get("market_cap")), pct_rank(df["market_cap"], r.get("market_cap"))),
                ("Enterprise Val", f_jpy(r.get("enterprise_value")), "--"),
                ("Total Cash", f_jpy(r.get("total_cash")), "--"),
                ("Total Debt", f_jpy(r.get("total_debt")), "--"),
                ("Net Cash", f_jpy(r.get("net_cash")), "dc-green" if r.get("net_cash", 0) > 0 else "dc-red"),
                ("D/E", f_num(r.get("debt_to_equity"), 0), pct_rank(df["debt_to_equity"], r.get("debt_to_equity"))),
                ("Current Ratio", f_num(r.get("current_ratio")), pct_rank(df["current_ratio"], r.get("current_ratio"))),
                ("Beta", f_num(r.get("beta")), pct_rank(df["beta"], r.get("beta"))),
            ]
            st.dataframe(pd.DataFrame(bdata, columns=["METRIC", "VALUE", "PCTLE"]),
                         use_container_width=True, hide_index=True, height=350)

        # 52W range
        if pd.notna(r.get("fifty_two_week_low")) and pd.notna(r.get("fifty_two_week_high")):
            lo, hi, pr = r["fifty_two_week_low"], r["fifty_two_week_high"], r.get("current_price", 0)
            pct = (pr - lo) / (hi - lo) * 100 if hi != lo else 50
            st.markdown(strip_html([
                ("52W LOW", f_price(lo), "dc-red"),
                ("CURRENT", f_price(pr), "dc-orange"),
                ("52W HIGH", f_price(hi), "dc-green"),
                ("POSITION", f"{pct:.0f}%", "dc-orange" if pct > 30 else "dc-red"),
                ("SMA50", f_price(r.get("fifty_day_avg")), ""),
                ("SMA200", f_price(r.get("two_hundred_day_avg")), ""),
                ("SMA CROSS", f"{r.get('sma_cross', 0)*100:.1f}%" if pd.notna(r.get("sma_cross")) else "--",
                 "dc-green" if r.get("sma_cross", 0) > 0 else "dc-red"),
                ("AVG VOL", f"{r.get('avg_volume', 0):,.0f}" if pd.notna(r.get("avg_volume")) else "--", ""),
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
        fig.update_layout(**chart_layout(350), title="P/B RATIO (SORTED)",
                          xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
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
        fig.update_layout(**chart_layout(350), title="DIVIDEND YIELD % (SORTED)",
                          xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
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
        fig.update_layout(**chart_layout(300), title="ROE % (SORTED)",
                          xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
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
        fig.update_layout(**chart_layout(300), title="BETA (SORTED)",
                          xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
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
        fig.update_layout(**chart_layout(300), title="NET CASH (JPY B)",
                          xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
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
    corr_cols = ["Composite", "Value", "Quality", "Strength", "Momentum",
                 "pe_trailing", "pb_ratio", "dividend_yield", "roe", "beta", "fcf_yield"]
    corr_labels = ["COMP", "VAL", "QUAL", "STR", "MOM", "P/E", "P/B", "DIV", "ROE", "BETA", "FCF"]
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
        fig.update_layout(**chart_layout(300), title="52W RANGE POSITION %",
                          xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
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
        fig.update_layout(**chart_layout(300), title="SMA50/SMA200 CROSS %",
                          xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                          yaxis_title="% ABOVE/BELOW")
        st.plotly_chart(fig, use_container_width=True)


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
                vals = [row["Value"], row["Quality"], row["Strength"], row["Momentum"], row["Value"]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=["VAL", "QUAL", "STR", "MOM", "VAL"],
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
        ctbl = cdf[["Ticker", "Name", "Sector", "Composite", "Value", "Quality", "Strength", "Momentum",
                     "pe_trailing", "pb_ratio", "dividend_yield", "roe", "ev_to_ebitda",
                     "debt_to_equity", "cash_to_mcap", "fcf_yield", "beta", "mcap_b", "quality_flags"]].copy()
        ctbl.columns = ["TICKER", "NAME", "SECTOR", "COMP", "VAL", "QUAL", "STR", "MOM",
                        "P/E", "P/B", "DIV%", "ROE%", "EV/EB", "D/E", "CASH/MC", "FCF_Y", "BETA", "MCAP_B", "QF"]
        for c in ["DIV%", "ROE%", "CASH/MC", "FCF_Y"]:
            ctbl[c] = ctbl[c].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
        st.dataframe(ctbl, use_container_width=True, hide_index=True)
    else:
        st.info("Select at least 2 stocks.")


# ── Footer ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; color:{GRAY_DIM}; font-size:8px; padding:8px; border-top:1px solid {BORDER}; margin-top:12px;">
JVQ TERMINAL v1.0 | BUILT BY NOAH | {len(df)} EQUITIES | {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")} | DATA: YFINANCE | NOT FINANCIAL ADVICE
</div>
""", unsafe_allow_html=True)
