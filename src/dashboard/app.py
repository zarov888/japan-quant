"""
Japan Value Quant Terminal
Professional multi-factor equity screening and analysis platform.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.universe import load_universe
from src.data.fetcher import fetch_universe, fetch_price_history
from src.model.scorer import score_universe, results_to_dataframe, load_scoring_config

# ── Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="JVQ Terminal",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Terminal Theme ─────────────────────────────────────────────
CHART_TEMPLATE = dict(
    layout=dict(
        template="plotly_dark",
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font=dict(family="JetBrains Mono, Fira Code, Consolas, monospace", size=11, color="#c8c8c8"),
        title_font=dict(size=13, color="#8a8a8a"),
        xaxis=dict(gridcolor="#1a1a2a", zerolinecolor="#1a1a2a"),
        yaxis=dict(gridcolor="#1a1a2a", zerolinecolor="#1a1a2a"),
        margin=dict(l=50, r=20, t=40, b=40),
        colorway=["#00d4aa", "#ff6b6b", "#4ecdc4", "#ffe66d", "#a855f7",
                   "#06b6d4", "#f97316", "#84cc16", "#ec4899", "#6366f1"],
    )
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: #0a0a0f;
        color: #c8c8c8;
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    }

    /* Header bar */
    .terminal-header {
        background: linear-gradient(90deg, #0a0a0f 0%, #0f1420 50%, #0a0a0f 100%);
        border-bottom: 1px solid #1a1a2a;
        padding: 8px 16px;
        margin: -1rem -1rem 1rem -1rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .terminal-header .title {
        color: #00d4aa;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 2px;
    }
    .terminal-header .subtitle {
        color: #555;
        font-size: 11px;
    }

    /* Kill default streamlit styling */
    h1, h2, h3 {
        font-family: 'JetBrains Mono', monospace !important;
        color: #8a8a8a !important;
        font-weight: 500 !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 13px !important;
        border-bottom: 1px solid #1a1a2a;
        padding-bottom: 6px;
        margin-bottom: 12px !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #0f1118;
        border: 1px solid #1a1a2a;
        border-radius: 4px;
        padding: 10px 14px;
    }
    [data-testid="stMetricLabel"] {
        color: #555 !important;
        font-size: 10px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stMetricValue"] {
        color: #00d4aa !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 18px !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #0f1118;
        border-bottom: 1px solid #1a1a2a;
        gap: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #555;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 8px 20px;
        border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4aa !important;
        border-bottom: 2px solid #00d4aa;
        background: transparent;
    }

    /* Dataframes */
    .stDataFrame {
        border: 1px solid #1a1a2a;
        border-radius: 4px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0c0c14;
        border-right: 1px solid #1a1a2a;
    }
    [data-testid="stSidebar"] .stMarkdown {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }

    /* Selectbox, slider, multiselect */
    .stSelectbox, .stMultiSelect, .stSlider {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Status indicators */
    .status-green { color: #00d4aa; }
    .status-red { color: #ff6b6b; }
    .status-yellow { color: #ffe66d; }
    .status-dim { color: #555; }

    /* Dense table style */
    .dense-table {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        line-height: 1.4;
    }

    /* Remove streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Data Loading ───────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    config = load_universe("config/universe.yaml")
    fundamentals = fetch_universe(config)
    results = score_universe(fundamentals)
    df = results_to_dataframe(results)

    fund_map = {f["ticker"]: f for f in fundamentals}
    enrichment_cols = [
        "pe_trailing", "pe_forward", "pb_ratio", "dividend_yield", "roe", "roa",
        "market_cap", "debt_to_equity", "cash_to_mcap", "ev_to_ebitda",
        "current_price", "fifty_two_week_high", "fifty_two_week_low",
        "fifty_day_avg", "two_hundred_day_avg", "beta", "avg_volume",
        "operating_margin", "profit_margin", "revenue_growth", "earnings_growth",
        "free_cashflow", "total_cash", "total_debt", "enterprise_value",
        "price_to_sales", "current_ratio", "industry",
    ]
    for col in enrichment_cols:
        df[col] = df["Ticker"].map(lambda t, c=col: fund_map.get(t, {}).get(c))

    # Computed columns
    df["mcap_b"] = df["market_cap"].fillna(0) / 1e9
    df["52w_range_pct"] = np.where(
        (df["fifty_two_week_high"].notna()) & (df["fifty_two_week_low"].notna()) & (df["fifty_two_week_high"] != df["fifty_two_week_low"]),
        (df["current_price"] - df["fifty_two_week_low"]) / (df["fifty_two_week_high"] - df["fifty_two_week_low"]),
        np.nan,
    )
    df["sma_cross"] = np.where(
        (df["fifty_day_avg"].notna()) & (df["two_hundred_day_avg"].notna()) & (df["two_hundred_day_avg"] != 0),
        df["fifty_day_avg"] / df["two_hundred_day_avg"] - 1,
        np.nan,
    )
    df["fcf_yield"] = np.where(
        (df["free_cashflow"].notna()) & (df["market_cap"].notna()) & (df["market_cap"] != 0),
        df["free_cashflow"] / df["market_cap"],
        np.nan,
    )
    df["net_cash"] = df["total_cash"].fillna(0) - df["total_debt"].fillna(0)
    df["net_cash_b"] = df["net_cash"] / 1e9

    return df, fundamentals, results


@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(ticker: str, years: int = 3):
    return fetch_price_history(ticker, years=years, cache_dir="data/cache")


def fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v * 100:.{decimals}f}%"

def fmt_num(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{decimals}f}"

def fmt_jpy(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if abs(v) >= 1e12:
        return f"{v/1e12:.1f}T"
    if abs(v) >= 1e9:
        return f"{v/1e9:.1f}B"
    return f"{v/1e6:.0f}M"

def apply_chart_theme(fig, height=400):
    fig.update_layout(**CHART_TEMPLATE["layout"], height=height)
    return fig


# ── Load ───────────────────────────────────────────────────────
with st.spinner("Loading universe..."):
    df, fundamentals, results = load_data()

all_sectors = sorted(df["Sector"].dropna().unique().tolist())

# ── Sidebar Controls ───────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="color:#00d4aa; font-size:14px; letter-spacing:2px; font-weight:600;">JVQ TERMINAL</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#555; font-size:10px; margin-bottom:16px;">JAPAN VALUE QUANT v1.0</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="color:#555; font-size:10px; letter-spacing:1px; margin-bottom:4px;">SCREENING</div>', unsafe_allow_html=True)
    min_score = st.slider("Min composite", 0.0, 1.0, 0.40, 0.01, format="%.2f")
    top_n = st.slider("Display limit", 5, 74, 50)

    sector_filter = st.multiselect("Sectors", options=all_sectors, default=[])

    st.markdown("---")
    st.markdown('<div style="color:#555; font-size:10px; letter-spacing:1px; margin-bottom:4px;">VALUE FILTERS</div>', unsafe_allow_html=True)
    max_pe = st.number_input("Max P/E", value=50.0, step=5.0)
    max_pb = st.number_input("Max P/B", value=5.0, step=0.5)
    min_div = st.number_input("Min Div Yield %", value=0.0, step=0.5)

    st.markdown("---")
    st.markdown('<div style="color:#555; font-size:10px; letter-spacing:1px; margin-bottom:4px;">QUALITY FILTERS</div>', unsafe_allow_html=True)
    min_roe = st.number_input("Min ROE %", value=0.0, step=1.0)
    max_de = st.number_input("Max D/E", value=500.0, step=50.0)

    st.markdown("---")
    st.markdown(f'<div style="color:#333; font-size:9px;">Universe: {len(df)} stocks | TOPIX Core</div>', unsafe_allow_html=True)

# ── Apply Filters ──────────────────────────────────────────────
filtered = df.copy()
filtered = filtered[filtered["Composite"] >= min_score]

if sector_filter:
    filtered = filtered[filtered["Sector"].isin(sector_filter)]
if max_pe < 50:
    filtered = filtered[(filtered["pe_trailing"].isna()) | (filtered["pe_trailing"] <= max_pe)]
if max_pb < 5:
    filtered = filtered[(filtered["pb_ratio"].isna()) | (filtered["pb_ratio"] <= max_pb)]
if min_div > 0:
    filtered = filtered[(filtered["dividend_yield"].notna()) & (filtered["dividend_yield"] >= min_div / 100)]
if min_roe > 0:
    filtered = filtered[(filtered["roe"].notna()) & (filtered["roe"] >= min_roe / 100)]
if max_de < 500:
    filtered = filtered[(filtered["debt_to_equity"].isna()) | (filtered["debt_to_equity"] <= max_de)]

filtered = filtered.head(top_n)

# ── Header ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="terminal-header">
    <span class="title">JVQ TERMINAL</span>
    <span class="subtitle" style="margin-left:16px;">
        {len(filtered)}/{len(df)} PASSING | MODEL: japan_deep_value_v1 | REBAL: QUARTERLY | BENCH: NIKKEI 225
    </span>
</div>
""", unsafe_allow_html=True)

# ── Summary Strip ──────────────────────────────────────────────
c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
c1.metric("UNIVERSE", len(df))
c2.metric("PASSING", len(filtered))
c3.metric("TOP SCORE", f"{filtered['Composite'].max():.3f}" if len(filtered) else "--")
c4.metric("MEDIAN", f"{filtered['Composite'].median():.3f}" if len(filtered) else "--")
c5.metric("AVG P/B", fmt_num(filtered["pb_ratio"].median()))
c6.metric("AVG P/E", fmt_num(filtered["pe_trailing"].median()))
c7.metric("AVG DIV", fmt_pct(filtered["dividend_yield"].median()))
c8.metric("SECTORS", filtered["Sector"].nunique() if len(filtered) else 0)


# ── Tabs ───────────────────────────────────────────────────────
tab_screen, tab_index, tab_sector, tab_stock, tab_factors, tab_risk, tab_compare = st.tabs([
    "SCREENER", "INDEX", "SECTORS", "STOCK", "FACTORS", "RISK", "COMPARE",
])


# ════════════════════════════════════════════════════════════════
# TAB: SCREENER
# ════════════════════════════════════════════════════════════════
with tab_screen:
    st.markdown("### SCREENING RESULTS")

    # Composite score distribution
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df["Composite"], nbinsx=30,
        marker_color="#1a1a2a", marker_line_color="#00d4aa", marker_line_width=1,
        name="All",
    ))
    if len(filtered) > 0:
        fig_dist.add_trace(go.Histogram(
            x=filtered["Composite"], nbinsx=30,
            marker_color="rgba(0,212,170,0.4)", marker_line_color="#00d4aa", marker_line_width=1,
            name="Passing",
        ))
    fig_dist.update_layout(barmode="overlay", xaxis_title="Composite Score", yaxis_title="Count")
    apply_chart_theme(fig_dist, 250)
    fig_dist.update_layout(title="SCORE DISTRIBUTION", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_dist, use_container_width=True)

    # Main data table
    display_cols = [
        "Ticker", "Name", "Sector", "Composite", "Value", "Quality", "Strength", "Momentum",
        "pe_trailing", "pb_ratio", "dividend_yield", "roe", "mcap_b", "beta",
    ]
    display_df = filtered[display_cols].copy()
    display_df.columns = [
        "TICKER", "NAME", "SECTOR", "COMP", "VAL", "QUAL", "STR", "MOM",
        "P/E", "P/B", "DIV%", "ROE%", "MCAP(B)", "BETA",
    ]
    display_df["DIV%"] = display_df["DIV%"].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
    display_df["ROE%"] = display_df["ROE%"].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
    display_df["MCAP(B)"] = display_df["MCAP(B)"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "--")

    st.dataframe(display_df, use_container_width=True, height=600)

    # Stacked score breakdown
    fig_stack = go.Figure()
    for col, color in [("Value", "#00d4aa"), ("Quality", "#4ecdc4"), ("Strength", "#a855f7"), ("Momentum", "#ffe66d")]:
        fig_stack.add_trace(go.Bar(
            x=filtered["Ticker"], y=filtered[col], name=col.upper(),
            marker_color=color, marker_line_width=0,
        ))
    fig_stack.update_layout(barmode="stack", xaxis_title="", yaxis_title="SCORE")
    apply_chart_theme(fig_stack, 350)
    fig_stack.update_layout(title="FACTOR DECOMPOSITION", legend=dict(orientation="h", y=1.1),
                            xaxis=dict(tickangle=-45, tickfont=dict(size=9)))
    st.plotly_chart(fig_stack, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB: INDEX
# ════════════════════════════════════════════════════════════════
with tab_index:
    st.markdown("### UNIVERSE OVERVIEW")

    idx_c1, idx_c2 = st.columns(2)

    with idx_c1:
        # Market cap distribution
        fig_mcap = go.Figure()
        fig_mcap.add_trace(go.Bar(
            x=df.sort_values("mcap_b", ascending=False)["Ticker"],
            y=df.sort_values("mcap_b", ascending=False)["mcap_b"],
            marker_color=np.where(
                df.sort_values("mcap_b", ascending=False)["Ticker"].isin(filtered["Ticker"]),
                "#00d4aa", "#1a1a2a"
            ),
            marker_line_width=0,
        ))
        fig_mcap.update_layout(xaxis_title="", yaxis_title="MARKET CAP (JPY B)",
                               xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
        apply_chart_theme(fig_mcap, 350)
        fig_mcap.update_layout(title="MARKET CAP DISTRIBUTION")
        st.plotly_chart(fig_mcap, use_container_width=True)

    with idx_c2:
        # Sector breakdown
        sector_agg = df.groupby("Sector").agg(
            count=("Ticker", "size"),
            avg_composite=("Composite", "mean"),
            total_mcap=("mcap_b", "sum"),
            avg_pb=("pb_ratio", "mean"),
            avg_pe=("pe_trailing", "mean"),
            avg_div=("dividend_yield", "mean"),
        ).reset_index().sort_values("total_mcap", ascending=True)

        fig_sector_bar = go.Figure()
        fig_sector_bar.add_trace(go.Barh(
            y=sector_agg["Sector"],
            x=sector_agg["total_mcap"],
            marker_color="#00d4aa",
            marker_line_width=0,
            text=sector_agg["count"].apply(lambda x: f"n={x}"),
            textposition="auto",
            textfont=dict(size=9),
        ))
        fig_sector_bar.update_layout(xaxis_title="TOTAL MCAP (JPY B)", yaxis_title="")
        apply_chart_theme(fig_sector_bar, 350)
        fig_sector_bar.update_layout(title="SECTOR MARKET CAP")
        st.plotly_chart(fig_sector_bar, use_container_width=True)

    # Full universe stats
    st.markdown("### UNIVERSE STATISTICS")
    stats_data = {
        "Metric": ["Composite", "P/E", "P/B", "EV/EBITDA", "Div Yield %", "ROE %",
                    "D/E", "Beta", "MCap (B JPY)", "Cash/MCap %", "FCF Yield %"],
        "Mean": [
            fmt_num(df["Composite"].mean(), 3),
            fmt_num(df["pe_trailing"].mean()),
            fmt_num(df["pb_ratio"].mean()),
            fmt_num(df["ev_to_ebitda"].mean()),
            fmt_pct(df["dividend_yield"].mean()),
            fmt_pct(df["roe"].mean()),
            fmt_num(df["debt_to_equity"].mean(), 0),
            fmt_num(df["beta"].mean()),
            fmt_num(df["mcap_b"].mean(), 0),
            fmt_pct(df["cash_to_mcap"].mean()),
            fmt_pct(df["fcf_yield"].mean()),
        ],
        "Median": [
            fmt_num(df["Composite"].median(), 3),
            fmt_num(df["pe_trailing"].median()),
            fmt_num(df["pb_ratio"].median()),
            fmt_num(df["ev_to_ebitda"].median()),
            fmt_pct(df["dividend_yield"].median()),
            fmt_pct(df["roe"].median()),
            fmt_num(df["debt_to_equity"].median(), 0),
            fmt_num(df["beta"].median()),
            fmt_num(df["mcap_b"].median(), 0),
            fmt_pct(df["cash_to_mcap"].median()),
            fmt_pct(df["fcf_yield"].median()),
        ],
        "Min": [
            fmt_num(df["Composite"].min(), 3),
            fmt_num(df["pe_trailing"].min()),
            fmt_num(df["pb_ratio"].min()),
            fmt_num(df["ev_to_ebitda"].min()),
            fmt_pct(df["dividend_yield"].min()),
            fmt_pct(df["roe"].min()),
            fmt_num(df["debt_to_equity"].min(), 0),
            fmt_num(df["beta"].min()),
            fmt_num(df["mcap_b"].min(), 0),
            fmt_pct(df["cash_to_mcap"].min()),
            fmt_pct(df["fcf_yield"].min()),
        ],
        "Max": [
            fmt_num(df["Composite"].max(), 3),
            fmt_num(df["pe_trailing"].max()),
            fmt_num(df["pb_ratio"].max()),
            fmt_num(df["ev_to_ebitda"].max()),
            fmt_pct(df["dividend_yield"].max()),
            fmt_pct(df["roe"].max()),
            fmt_num(df["debt_to_equity"].max(), 0),
            fmt_num(df["beta"].max()),
            fmt_num(df["mcap_b"].max(), 0),
            fmt_pct(df["cash_to_mcap"].max()),
            fmt_pct(df["fcf_yield"].max()),
        ],
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, height=420)

    # P/B vs P/E scatter for entire universe
    fig_pbpe = go.Figure()
    for sector in all_sectors:
        sdf = df[df["Sector"] == sector]
        fig_pbpe.add_trace(go.Scatter(
            x=sdf["pe_trailing"], y=sdf["pb_ratio"],
            mode="markers+text",
            text=sdf["Ticker"],
            textposition="top center",
            textfont=dict(size=8),
            name=sector,
            marker=dict(size=8, line=dict(width=0.5, color="#0a0a0f")),
        ))
    fig_pbpe.update_layout(xaxis_title="P/E TRAILING", yaxis_title="P/B RATIO")
    apply_chart_theme(fig_pbpe, 500)
    fig_pbpe.update_layout(title="VALUATION MAP: P/E vs P/B", legend=dict(font=dict(size=9)))
    st.plotly_chart(fig_pbpe, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB: SECTORS
# ════════════════════════════════════════════════════════════════
with tab_sector:
    st.markdown("### SECTOR ANALYSIS")

    selected_sector = st.selectbox("Select sector", ["ALL"] + all_sectors)
    sector_df = filtered if selected_sector == "ALL" else filtered[filtered["Sector"] == selected_sector]

    if len(sector_df) == 0:
        st.warning("No stocks match current filters for this sector.")
    else:
        sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
        sc1.metric("STOCKS", len(sector_df))
        sc2.metric("AVG COMP", fmt_num(sector_df["Composite"].mean(), 3))
        sc3.metric("MED P/B", fmt_num(sector_df["pb_ratio"].median()))
        sc4.metric("MED P/E", fmt_num(sector_df["pe_trailing"].median()))
        sc5.metric("MED DIV", fmt_pct(sector_df["dividend_yield"].median()))
        sc6.metric("MED ROE", fmt_pct(sector_df["roe"].median()))

        sec_l, sec_r = st.columns(2)

        with sec_l:
            # Score comparison within sector
            fig_sec_score = go.Figure()
            for col, color in [("Value", "#00d4aa"), ("Quality", "#4ecdc4"), ("Strength", "#a855f7"), ("Momentum", "#ffe66d")]:
                fig_sec_score.add_trace(go.Bar(
                    x=sector_df["Ticker"], y=sector_df[col], name=col.upper(),
                    marker_color=color, marker_line_width=0,
                ))
            fig_sec_score.update_layout(barmode="group", xaxis_title="", yaxis_title="SCORE",
                                         xaxis=dict(tickangle=-45, tickfont=dict(size=9)))
            apply_chart_theme(fig_sec_score, 400)
            fig_sec_score.update_layout(title=f"FACTOR SCORES: {selected_sector}", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_sec_score, use_container_width=True)

        with sec_r:
            # Valuation scatter
            fig_sec_val = go.Figure()
            fig_sec_val.add_trace(go.Scatter(
                x=sector_df["pb_ratio"], y=sector_df["dividend_yield"].fillna(0) * 100,
                mode="markers+text",
                text=sector_df["Ticker"],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    size=sector_df["mcap_b"].clip(1, None).apply(np.log) * 5,
                    color=sector_df["Composite"],
                    colorscale=[[0, "#1a1a2a"], [0.5, "#4ecdc4"], [1, "#00d4aa"]],
                    showscale=True,
                    colorbar=dict(title="COMP", titlefont=dict(size=9)),
                    line=dict(width=0.5, color="#0a0a0f"),
                ),
            ))
            fig_sec_val.update_layout(xaxis_title="P/B", yaxis_title="DIV YIELD %")
            apply_chart_theme(fig_sec_val, 400)
            fig_sec_val.update_layout(title=f"VALUATION: {selected_sector}")
            st.plotly_chart(fig_sec_val, use_container_width=True)

        # Detailed sector table
        sec_display = sector_df[["Ticker", "Name", "Composite", "Value", "Quality", "Strength", "Momentum",
                                  "pe_trailing", "pb_ratio", "dividend_yield", "roe", "ev_to_ebitda",
                                  "debt_to_equity", "cash_to_mcap", "mcap_b", "beta"]].copy()
        sec_display.columns = ["TICKER", "NAME", "COMP", "VAL", "QUAL", "STR", "MOM",
                               "P/E", "P/B", "DIV%", "ROE%", "EV/EBITDA", "D/E", "CASH/MC%", "MCAP(B)", "BETA"]
        sec_display["DIV%"] = sec_display["DIV%"].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
        sec_display["ROE%"] = sec_display["ROE%"].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
        sec_display["CASH/MC%"] = sec_display["CASH/MC%"].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
        st.dataframe(sec_display, use_container_width=True, height=400)


# ════════════════════════════════════════════════════════════════
# TAB: STOCK
# ════════════════════════════════════════════════════════════════
with tab_stock:
    st.markdown("### STOCK ANALYSIS")

    stock_list = filtered["Ticker"].tolist()
    if stock_list:
        selected_stock = st.selectbox(
            "Select stock",
            stock_list,
            format_func=lambda t: f"{t}  {filtered[filtered['Ticker']==t]['Name'].values[0]}",
        )
    else:
        selected_stock = None
        st.warning("No stocks match current filters.")

    if selected_stock:
        row = filtered[filtered["Ticker"] == selected_stock].iloc[0]

        # Header
        st.markdown(f"""
        <div style="display:flex; align-items:baseline; gap:16px; margin-bottom:12px;">
            <span style="color:#00d4aa; font-size:20px; font-weight:600; font-family:monospace;">{selected_stock}</span>
            <span style="color:#c8c8c8; font-size:14px;">{row['Name']}</span>
            <span style="color:#555; font-size:11px;">{row['Sector']} / {row.get('industry', '--')}</span>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics strip
        km1, km2, km3, km4, km5, km6, km7, km8 = st.columns(8)
        km1.metric("COMPOSITE", fmt_num(row["Composite"], 3))
        km2.metric("PRICE", f"{row['current_price']:,.0f}" if pd.notna(row.get("current_price")) else "--")
        km3.metric("MCAP", fmt_jpy(row.get("market_cap")))
        km4.metric("P/E", fmt_num(row.get("pe_trailing")))
        km5.metric("P/B", fmt_num(row.get("pb_ratio")))
        km6.metric("DIV", fmt_pct(row.get("dividend_yield")))
        km7.metric("ROE", fmt_pct(row.get("roe")))
        km8.metric("BETA", fmt_num(row.get("beta")))

        stk_l, stk_r = st.columns([1.2, 0.8])

        with stk_l:
            # Price chart
            prices = load_price_data(selected_stock, years=3)
            if not prices.empty:
                fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                          row_heights=[0.75, 0.25], vertical_spacing=0.03)

                close_col = "Close"
                if isinstance(prices.columns, pd.MultiIndex):
                    close_col = ("Close", selected_stock) if ("Close", selected_stock) in prices.columns else prices.columns[0]

                close = prices[close_col] if close_col in prices.columns else prices.iloc[:, 0]
                sma50 = close.rolling(50).mean()
                sma200 = close.rolling(200).mean()

                fig_price.add_trace(go.Scatter(
                    x=prices.index, y=close, mode="lines",
                    line=dict(color="#c8c8c8", width=1), name="CLOSE",
                ), row=1, col=1)
                fig_price.add_trace(go.Scatter(
                    x=prices.index, y=sma50, mode="lines",
                    line=dict(color="#00d4aa", width=1, dash="dot"), name="SMA50",
                ), row=1, col=1)
                fig_price.add_trace(go.Scatter(
                    x=prices.index, y=sma200, mode="lines",
                    line=dict(color="#a855f7", width=1, dash="dot"), name="SMA200",
                ), row=1, col=1)

                vol_col = "Volume"
                if isinstance(prices.columns, pd.MultiIndex):
                    vol_col = ("Volume", selected_stock) if ("Volume", selected_stock) in prices.columns else None
                if vol_col and vol_col in prices.columns:
                    fig_price.add_trace(go.Bar(
                        x=prices.index, y=prices[vol_col],
                        marker_color="rgba(0,212,170,0.3)", marker_line_width=0, name="VOL",
                    ), row=2, col=1)

                apply_chart_theme(fig_price, 420)
                fig_price.update_layout(title=f"{selected_stock} PRICE (3Y)",
                                        legend=dict(orientation="h", y=1.08),
                                        showlegend=True)
                fig_price.update_yaxes(title_text="JPY", row=1, col=1)
                fig_price.update_yaxes(title_text="VOL", row=2, col=1)
                st.plotly_chart(fig_price, use_container_width=True)

        with stk_r:
            # Factor radar
            categories = ["Value", "Quality", "Strength", "Momentum"]
            values = [row[c] for c in categories]
            values.append(values[0])

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=["VAL", "QUAL", "STR", "MOM", "VAL"],
                fill="toself",
                fillcolor="rgba(0,212,170,0.15)",
                line_color="#00d4aa",
                line_width=2,
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=8), gridcolor="#1a1a2a"),
                    angularaxis=dict(tickfont=dict(size=10, color="#c8c8c8"), gridcolor="#1a1a2a"),
                    bgcolor="#0a0a0f",
                ),
            )
            apply_chart_theme(fig_radar, 300)
            fig_radar.update_layout(title="FACTOR PROFILE", showlegend=False)
            st.plotly_chart(fig_radar, use_container_width=True)

        # Full fundamentals table
        st.markdown("### FUNDAMENTALS")
        fund_l, fund_r = st.columns(2)

        with fund_l:
            val_data = {
                "Metric": ["P/E Trailing", "P/E Forward", "P/B", "P/S", "EV/EBITDA",
                           "FCF Yield", "Div Yield", "Cash/MCap"],
                "Value": [
                    fmt_num(row.get("pe_trailing")),
                    fmt_num(row.get("pe_forward")),
                    fmt_num(row.get("pb_ratio")),
                    fmt_num(row.get("price_to_sales")),
                    fmt_num(row.get("ev_to_ebitda")),
                    fmt_pct(row.get("fcf_yield")),
                    fmt_pct(row.get("dividend_yield")),
                    fmt_pct(row.get("cash_to_mcap")),
                ],
            }
            st.markdown("**VALUATION**")
            st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True, height=320)

        with fund_r:
            qual_data = {
                "Metric": ["ROE", "ROA", "Operating Margin", "Profit Margin",
                           "Revenue Growth", "Earnings Growth", "D/E", "Current Ratio"],
                "Value": [
                    fmt_pct(row.get("roe")),
                    fmt_pct(row.get("roa")),
                    fmt_pct(row.get("operating_margin")),
                    fmt_pct(row.get("profit_margin")),
                    fmt_pct(row.get("revenue_growth")),
                    fmt_pct(row.get("earnings_growth")),
                    fmt_num(row.get("debt_to_equity"), 0),
                    fmt_num(row.get("current_ratio")),
                ],
            }
            st.markdown("**QUALITY / BALANCE SHEET**")
            st.dataframe(pd.DataFrame(qual_data), use_container_width=True, hide_index=True, height=320)

        # 52-week range visual
        if pd.notna(row.get("fifty_two_week_low")) and pd.notna(row.get("fifty_two_week_high")):
            low = row["fifty_two_week_low"]
            high = row["fifty_two_week_high"]
            price = row.get("current_price", low)

            fig_range = go.Figure()
            fig_range.add_trace(go.Scatter(
                x=[low, high], y=[0, 0], mode="lines",
                line=dict(color="#1a1a2a", width=8),
            ))
            fig_range.add_trace(go.Scatter(
                x=[price], y=[0], mode="markers",
                marker=dict(color="#00d4aa", size=14, symbol="diamond"),
                name=f"Current: {price:,.0f}",
            ))
            fig_range.add_annotation(x=low, y=0.1, text=f"52W LOW: {low:,.0f}", showarrow=False, font=dict(size=9, color="#ff6b6b"))
            fig_range.add_annotation(x=high, y=0.1, text=f"52W HIGH: {high:,.0f}", showarrow=False, font=dict(size=9, color="#00d4aa"))
            fig_range.update_yaxes(visible=False, range=[-0.5, 0.5])
            apply_chart_theme(fig_range, 100)
            fig_range.update_layout(title="52-WEEK RANGE", showlegend=True, margin=dict(t=30, b=10))
            st.plotly_chart(fig_range, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB: FACTORS
# ════════════════════════════════════════════════════════════════
with tab_factors:
    st.markdown("### FACTOR ANALYSIS")

    fac_l, fac_r = st.columns(2)

    with fac_l:
        # Value vs Quality scatter
        fig_vq = go.Figure()
        fig_vq.add_trace(go.Scatter(
            x=filtered["Value"], y=filtered["Quality"],
            mode="markers+text",
            text=filtered["Ticker"],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                size=10,
                color=filtered["Composite"],
                colorscale=[[0, "#1a1a2a"], [0.5, "#4ecdc4"], [1, "#00d4aa"]],
                showscale=True,
                colorbar=dict(title="COMP", titlefont=dict(size=9)),
                line=dict(width=0.5, color="#0a0a0f"),
            ),
        ))
        fig_vq.add_hline(y=filtered["Quality"].median(), line_dash="dot", line_color="#333", line_width=1)
        fig_vq.add_vline(x=filtered["Value"].median(), line_dash="dot", line_color="#333", line_width=1)
        fig_vq.update_layout(xaxis_title="VALUE", yaxis_title="QUALITY")
        apply_chart_theme(fig_vq, 450)
        fig_vq.update_layout(title="VALUE vs QUALITY")
        st.plotly_chart(fig_vq, use_container_width=True)

    with fac_r:
        # Strength vs Momentum scatter
        fig_sm = go.Figure()
        fig_sm.add_trace(go.Scatter(
            x=filtered["Strength"], y=filtered["Momentum"],
            mode="markers+text",
            text=filtered["Ticker"],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                size=10,
                color=filtered["Composite"],
                colorscale=[[0, "#1a1a2a"], [0.5, "#4ecdc4"], [1, "#00d4aa"]],
                showscale=True,
                colorbar=dict(title="COMP", titlefont=dict(size=9)),
                line=dict(width=0.5, color="#0a0a0f"),
            ),
        ))
        fig_sm.add_hline(y=filtered["Momentum"].median(), line_dash="dot", line_color="#333", line_width=1)
        fig_sm.add_vline(x=filtered["Strength"].median(), line_dash="dot", line_color="#333", line_width=1)
        fig_sm.update_layout(xaxis_title="STRENGTH", yaxis_title="MOMENTUM")
        apply_chart_theme(fig_sm, 450)
        fig_sm.update_layout(title="STRENGTH vs MOMENTUM")
        st.plotly_chart(fig_sm, use_container_width=True)

    # Correlation heatmap
    st.markdown("### FACTOR CORRELATIONS")
    corr_cols = ["Composite", "Value", "Quality", "Strength", "Momentum",
                 "pe_trailing", "pb_ratio", "dividend_yield", "roe", "cash_to_mcap",
                 "ev_to_ebitda", "debt_to_equity", "beta", "fcf_yield"]
    corr_labels = ["COMP", "VAL", "QUAL", "STR", "MOM",
                   "P/E", "P/B", "DIV", "ROE", "CASH/MC",
                   "EV/EBITDA", "D/E", "BETA", "FCF_Y"]
    corr_data = filtered[corr_cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = corr_data.corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_labels,
        y=corr_labels,
        colorscale=[[0, "#ff6b6b"], [0.5, "#0a0a0f"], [1, "#00d4aa"]],
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    apply_chart_theme(fig_corr, 500)
    fig_corr.update_layout(title="CROSS-FACTOR CORRELATION MATRIX")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Factor distributions
    st.markdown("### FACTOR DISTRIBUTIONS")
    fd1, fd2, fd3, fd4 = st.columns(4)
    for col_widget, factor, color in [
        (fd1, "Value", "#00d4aa"),
        (fd2, "Quality", "#4ecdc4"),
        (fd3, "Strength", "#a855f7"),
        (fd4, "Momentum", "#ffe66d"),
    ]:
        with col_widget:
            fig_fd = go.Figure()
            fig_fd.add_trace(go.Histogram(
                x=filtered[factor], nbinsx=20,
                marker_color=color, marker_line_width=0, opacity=0.8,
            ))
            apply_chart_theme(fig_fd, 250)
            fig_fd.update_layout(title=factor.upper(), xaxis_title="", yaxis_title="",
                                 margin=dict(l=30, r=10, t=35, b=30))
            st.plotly_chart(fig_fd, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB: RISK
# ════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown("### RISK ANALYSIS")

    risk_l, risk_r = st.columns(2)

    with risk_l:
        # Beta distribution
        fig_beta = go.Figure()
        fig_beta.add_trace(go.Scatter(
            x=filtered["Ticker"],
            y=filtered["beta"].fillna(1),
            mode="markers+lines",
            marker=dict(
                color=np.where(filtered["beta"].fillna(1) > 1, "#ff6b6b", "#00d4aa"),
                size=8,
            ),
            line=dict(color="#1a1a2a", width=1),
        ))
        fig_beta.add_hline(y=1.0, line_dash="dash", line_color="#555", line_width=1,
                           annotation_text="BETA=1.0", annotation_font_color="#555")
        fig_beta.update_layout(xaxis_title="", yaxis_title="BETA",
                               xaxis=dict(tickangle=-45, tickfont=dict(size=9)))
        apply_chart_theme(fig_beta, 400)
        fig_beta.update_layout(title="BETA PROFILE")
        st.plotly_chart(fig_beta, use_container_width=True)

    with risk_r:
        # Leverage: D/E vs Current Ratio
        fig_lev = go.Figure()
        de_clean = filtered["debt_to_equity"].fillna(0)
        cr_clean = filtered["current_ratio"].fillna(1)
        fig_lev.add_trace(go.Scatter(
            x=de_clean, y=cr_clean,
            mode="markers+text",
            text=filtered["Ticker"],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                size=10,
                color=filtered["Composite"],
                colorscale=[[0, "#1a1a2a"], [0.5, "#4ecdc4"], [1, "#00d4aa"]],
                showscale=True,
                colorbar=dict(title="COMP", titlefont=dict(size=9)),
                line=dict(width=0.5, color="#0a0a0f"),
            ),
        ))
        fig_lev.add_vline(x=100, line_dash="dot", line_color="#ff6b6b", line_width=1,
                          annotation_text="D/E=100", annotation_font_color="#555")
        fig_lev.add_hline(y=1.0, line_dash="dot", line_color="#ffe66d", line_width=1,
                          annotation_text="CR=1.0", annotation_font_color="#555")
        fig_lev.update_layout(xaxis_title="DEBT/EQUITY", yaxis_title="CURRENT RATIO")
        apply_chart_theme(fig_lev, 400)
        fig_lev.update_layout(title="LEVERAGE MAP")
        st.plotly_chart(fig_lev, use_container_width=True)

    # Net cash analysis
    st.markdown("### NET CASH POSITION")
    net_cash_df = filtered[["Ticker", "Name", "net_cash_b", "cash_to_mcap", "mcap_b"]].copy()
    net_cash_df = net_cash_df.sort_values("net_cash_b", ascending=False)

    fig_nc = go.Figure()
    fig_nc.add_trace(go.Bar(
        x=net_cash_df["Ticker"],
        y=net_cash_df["net_cash_b"],
        marker_color=np.where(net_cash_df["net_cash_b"] >= 0, "#00d4aa", "#ff6b6b"),
        marker_line_width=0,
    ))
    fig_nc.update_layout(xaxis_title="", yaxis_title="NET CASH (JPY B)",
                         xaxis=dict(tickangle=-45, tickfont=dict(size=9)))
    apply_chart_theme(fig_nc, 350)
    fig_nc.update_layout(title="NET CASH: TOTAL_CASH - TOTAL_DEBT")
    st.plotly_chart(fig_nc, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB: COMPARE
# ════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### STOCK COMPARISON")

    compare_stocks = st.multiselect(
        "Select stocks to compare (2-6)",
        filtered["Ticker"].tolist(),
        default=filtered["Ticker"].tolist()[:3],
        max_selections=6,
    )

    if len(compare_stocks) >= 2:
        comp_df = filtered[filtered["Ticker"].isin(compare_stocks)]

        # Overlaid radar
        fig_comp_radar = go.Figure()
        colors = ["#00d4aa", "#ff6b6b", "#4ecdc4", "#ffe66d", "#a855f7", "#06b6d4"]
        for i, (_, r) in enumerate(comp_df.iterrows()):
            vals = [r["Value"], r["Quality"], r["Strength"], r["Momentum"], r["Value"]]
            fig_comp_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=["VAL", "QUAL", "STR", "MOM", "VAL"],
                fill="toself",
                fillcolor=f"rgba({int(colors[i][1:3],16)},{int(colors[i][3:5],16)},{int(colors[i][5:7],16)},0.1)",
                line=dict(color=colors[i], width=2),
                name=r["Ticker"],
            ))
        fig_comp_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=8), gridcolor="#1a1a2a"),
                angularaxis=dict(tickfont=dict(size=10, color="#c8c8c8"), gridcolor="#1a1a2a"),
                bgcolor="#0a0a0f",
            ),
        )
        apply_chart_theme(fig_comp_radar, 450)
        fig_comp_radar.update_layout(title="FACTOR OVERLAY")
        st.plotly_chart(fig_comp_radar, use_container_width=True)

        # Comparison table
        comp_table = comp_df[["Ticker", "Name", "Sector", "Composite", "Value", "Quality",
                               "Strength", "Momentum", "pe_trailing", "pb_ratio",
                               "dividend_yield", "roe", "ev_to_ebitda", "debt_to_equity",
                               "cash_to_mcap", "beta", "mcap_b"]].copy()
        comp_table.columns = ["TICKER", "NAME", "SECTOR", "COMP", "VAL", "QUAL", "STR", "MOM",
                              "P/E", "P/B", "DIV%", "ROE%", "EV/EBITDA", "D/E", "CASH/MC%", "BETA", "MCAP(B)"]
        comp_table["DIV%"] = comp_table["DIV%"].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
        comp_table["ROE%"] = comp_table["ROE%"].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
        comp_table["CASH/MC%"] = comp_table["CASH/MC%"].apply(lambda x: f"{x*100:.1f}" if pd.notna(x) else "--")
        st.dataframe(comp_table, use_container_width=True, hide_index=True)

        # Price overlay
        st.markdown("### NORMALIZED PRICE COMPARISON (3Y)")
        fig_price_comp = go.Figure()
        for i, ticker in enumerate(compare_stocks):
            prices = load_price_data(ticker, years=3)
            if not prices.empty:
                close_col = "Close"
                if isinstance(prices.columns, pd.MultiIndex):
                    close_col = ("Close", ticker) if ("Close", ticker) in prices.columns else prices.columns[0]
                close = prices[close_col] if close_col in prices.columns else prices.iloc[:, 0]
                normalized = close / close.iloc[0] * 100
                fig_price_comp.add_trace(go.Scatter(
                    x=prices.index, y=normalized,
                    mode="lines",
                    line=dict(color=colors[i], width=1.5),
                    name=ticker,
                ))
        fig_price_comp.add_hline(y=100, line_dash="dot", line_color="#333", line_width=1)
        fig_price_comp.update_layout(xaxis_title="", yaxis_title="INDEXED (100 = START)")
        apply_chart_theme(fig_price_comp, 400)
        fig_price_comp.update_layout(title="RELATIVE PERFORMANCE", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_price_comp, use_container_width=True)
    else:
        st.info("Select at least 2 stocks to compare.")


# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="color:#333; font-size:9px; text-align:center; font-family:monospace;">'
    'JVQ TERMINAL v1.0 | MODEL: japan_deep_value_v1 | DATA: yfinance | '
    f'UNIVERSE: {len(df)} STOCKS | LAST UPDATE: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}'
    '</div>',
    unsafe_allow_html=True,
)
