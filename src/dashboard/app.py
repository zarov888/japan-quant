"""
Japan Value Quant Dashboard — Streamlit app with interactive 3D visualizations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.universe import load_universe
from src.data.fetcher import fetch_universe
from src.model.scorer import score_universe, results_to_dataframe, load_scoring_config

# ─── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Japan Value Quant",
    page_icon="🗾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for dark theme ─────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px;
        margin: 4px 0;
    }
    .stock-ticker {
        font-size: 1.4em;
        font-weight: bold;
        color: #e94560;
    }
    h1 { color: #e94560 !important; }
    h2, h3 { color: #0f3460 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Data loading (cached) ─────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    config = load_universe("config/universe.yaml")
    fundamentals = fetch_universe(config)
    results = score_universe(fundamentals)
    df = results_to_dataframe(results)

    # Enrich with raw fundamentals
    fund_map = {f["ticker"]: f for f in fundamentals}
    for col in ["pe_trailing", "pb_ratio", "dividend_yield", "roe", "market_cap",
                 "debt_to_equity", "cash_to_mcap", "ev_to_ebitda", "current_price",
                 "fifty_two_week_high", "fifty_two_week_low", "beta",
                 "operating_margin", "revenue_growth", "free_cashflow"]:
        df[col] = df["Ticker"].map(lambda t, c=col: fund_map.get(t, {}).get(c))

    return df, fundamentals, results


# ─── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🗾 Japan Value Quant")
st.sidebar.markdown("---")

min_score = st.sidebar.slider("Min composite score", 0.0, 1.0, 0.45, 0.05)
top_n = st.sidebar.slider("Show top N stocks", 5, 74, 30)

sectors = st.sidebar.multiselect(
    "Filter sectors",
    options=["All"],
    default=["All"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model: japan_deep_value_v1**")
st.sidebar.markdown("Universe: TOPIX Core (74)")

# ─── Load data ─────────────────────────────────────────────────
with st.spinner("Fetching & scoring universe..."):
    df, fundamentals, results = load_data()

# Apply filters
filtered = df[df["Composite"] >= min_score].head(top_n)

# ─── Header metrics ───────────────────────────────────────────
st.title("Japan Value Quant Model")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Stocks Scored", len(df))
col2.metric("Above Threshold", len(df[df["Composite"] >= 0.60]))
col3.metric("Top Score", f"{df['Composite'].max():.3f}")
col4.metric("Median Score", f"{df['Composite'].median():.3f}")
col5.metric("Sectors", df["Sector"].nunique())

st.markdown("---")

# ─── TAB LAYOUT ────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Rankings", "🌐 3D Factor Space", "🔬 Stock Drill-Down",
    "📈 Sector Analysis", "🌈 Factor Heatmap"
])


# ─── TAB 1: Rankings table ────────────────────────────────────
with tab1:
    st.subheader("Ranked Value Picks")

    # Score bar chart
    fig_rank = go.Figure()
    fig_rank.add_trace(go.Bar(
        x=filtered["Ticker"],
        y=filtered["Value"],
        name="Value",
        marker_color="#e94560",
    ))
    fig_rank.add_trace(go.Bar(
        x=filtered["Ticker"],
        y=filtered["Quality"],
        name="Quality",
        marker_color="#0f3460",
    ))
    fig_rank.add_trace(go.Bar(
        x=filtered["Ticker"],
        y=filtered["Strength"],
        name="Strength",
        marker_color="#533483",
    ))
    fig_rank.add_trace(go.Bar(
        x=filtered["Ticker"],
        y=filtered["Momentum"],
        name="Momentum",
        marker_color="#e9c46a",
    ))
    fig_rank.update_layout(
        barmode="group",
        template="plotly_dark",
        height=450,
        title="Factor Score Breakdown",
        xaxis_title="",
        yaxis_title="Score",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    # Data table
    st.dataframe(
        filtered.style.background_gradient(cmap="RdYlGn", subset=["Composite", "Value", "Quality", "Strength", "Momentum"]),
        use_container_width=True,
        height=500,
    )


# ─── TAB 2: 3D Factor Space (the cool rainbow one) ───────────
with tab2:
    st.subheader("3D Factor Space — Interactive")
    st.caption("Each stock plotted in Value × Quality × Momentum space. Size = market cap. Color = composite score.")

    plot_df = filtered.copy()
    plot_df["mcap_billions"] = plot_df["market_cap"].fillna(0) / 1e9
    plot_df["size"] = np.clip(plot_df["mcap_billions"] / plot_df["mcap_billions"].max() * 40, 5, 40)

    fig_3d = px.scatter_3d(
        plot_df,
        x="Value",
        y="Quality",
        z="Momentum",
        color="Composite",
        size="size",
        text="Ticker",
        hover_data=["Name", "Sector", "Composite", "Strength"],
        color_continuous_scale="Rainbow",
        opacity=0.85,
    )
    fig_3d.update_traces(
        textposition="top center",
        textfont_size=9,
        marker=dict(line=dict(width=0.5, color="white")),
    )
    fig_3d.update_layout(
        template="plotly_dark",
        height=700,
        scene=dict(
            xaxis_title="Value Score",
            yaxis_title="Quality Score",
            zaxis_title="Momentum Score",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
            ),
            bgcolor="#0e1117",
        ),
        coloraxis_colorbar=dict(title="Score"),
        # Auto-rotate animation
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0.95,
            x=0.05,
            xanchor="left",
            buttons=[dict(
                label="▶ Rotate",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=50, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=0),
                )],
            )],
        )],
    )

    # Create rotation frames
    frames = []
    for angle in range(0, 360, 3):
        rad = np.radians(angle)
        frames.append(go.Frame(
            layout=dict(
                scene_camera=dict(
                    eye=dict(
                        x=1.8 * np.cos(rad),
                        y=1.8 * np.sin(rad),
                        z=0.8 + 0.3 * np.sin(rad * 2),
                    )
                )
            ),
            name=str(angle),
        ))
    fig_3d.frames = frames

    st.plotly_chart(fig_3d, use_container_width=True)

    # 3D Surface: Value vs Quality → Composite
    st.subheader("Value × Quality Surface")
    st.caption("Interpolated surface showing how value and quality combine into composite score.")

    if len(filtered) >= 4:
        from scipy.interpolate import griddata

        xi = np.linspace(filtered["Value"].min(), filtered["Value"].max(), 30)
        yi = np.linspace(filtered["Quality"].min(), filtered["Quality"].max(), 30)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata(
            (filtered["Value"].values, filtered["Quality"].values),
            filtered["Composite"].values,
            (xi, yi),
            method="cubic",
        )

        fig_surface = go.Figure(data=[
            go.Surface(
                x=xi, y=yi, z=zi,
                colorscale="Rainbow",
                opacity=0.9,
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True),
                ),
            ),
            go.Scatter3d(
                x=filtered["Value"],
                y=filtered["Quality"],
                z=filtered["Composite"],
                mode="markers+text",
                text=filtered["Ticker"],
                textposition="top center",
                marker=dict(size=5, color=filtered["Composite"], colorscale="Rainbow"),
            ),
        ])
        fig_surface.update_layout(
            template="plotly_dark",
            height=600,
            scene=dict(
                xaxis_title="Value",
                yaxis_title="Quality",
                zaxis_title="Composite",
                bgcolor="#0e1117",
            ),
        )
        st.plotly_chart(fig_surface, use_container_width=True)


# ─── TAB 3: Stock Drill-Down ──────────────────────────────────
with tab3:
    st.subheader("Stock Drill-Down")

    selected = st.selectbox("Select stock", filtered["Ticker"].tolist(),
                            format_func=lambda t: f"{t} — {filtered[filtered['Ticker']==t]['Name'].values[0]}")

    if selected:
        row = filtered[filtered["Ticker"] == selected].iloc[0]

        # Radar chart
        categories = ["Value", "Quality", "Strength", "Momentum"]
        values = [row[c] for c in categories]
        values.append(values[0])  # close the polygon

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(233, 69, 96, 0.3)",
            line_color="#e94560",
            name=selected,
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor="#0e1117",
            ),
            template="plotly_dark",
            height=400,
            title=f"{selected} — {row['Name']}",
        )

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_right:
            st.markdown(f"**Composite Score:** `{row['Composite']:.4f}`")
            st.markdown(f"**Sector:** {row['Sector']}")

            metrics = {
                "P/E": row.get("pe_trailing"),
                "P/B": row.get("pb_ratio"),
                "EV/EBITDA": row.get("ev_to_ebitda"),
                "Div Yield": f"{row.get('dividend_yield', 0) * 100:.1f}%" if row.get("dividend_yield") else "N/A",
                "ROE": f"{row.get('roe', 0) * 100:.1f}%" if row.get("roe") else "N/A",
                "D/E": row.get("debt_to_equity"),
                "Cash/MCap": f"{row.get('cash_to_mcap', 0) * 100:.1f}%" if row.get("cash_to_mcap") else "N/A",
                "Market Cap": f"¥{row.get('market_cap', 0) / 1e9:.0f}B" if row.get("market_cap") else "N/A",
                "Beta": row.get("beta"),
            }

            for k, v in metrics.items():
                if v is not None:
                    st.markdown(f"**{k}:** {v}")


# ─── TAB 4: Sector Analysis ───────────────────────────────────
with tab4:
    st.subheader("Sector Breakdown")

    # Treemap — size by count, color by avg composite
    sector_stats = filtered.groupby("Sector").agg(
        count=("Composite", "size"),
        avg_score=("Composite", "mean"),
        avg_value=("Value", "mean"),
        avg_quality=("Quality", "mean"),
    ).reset_index()

    fig_tree = px.treemap(
        sector_stats,
        path=["Sector"],
        values="count",
        color="avg_score",
        color_continuous_scale="Rainbow",
        hover_data=["avg_value", "avg_quality"],
    )
    fig_tree.update_layout(
        template="plotly_dark",
        height=450,
        title="Sector Concentration (size = # stocks, color = avg score)",
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # Sector scatter: avg value vs avg quality
    fig_sector = px.scatter(
        sector_stats,
        x="avg_value",
        y="avg_quality",
        size="count",
        color="avg_score",
        text="Sector",
        color_continuous_scale="Rainbow",
        size_max=40,
    )
    fig_sector.update_traces(textposition="top center")
    fig_sector.update_layout(
        template="plotly_dark",
        height=450,
        title="Sector Value vs Quality",
        xaxis_title="Avg Value Score",
        yaxis_title="Avg Quality Score",
    )
    st.plotly_chart(fig_sector, use_container_width=True)


# ─── TAB 5: Factor Heatmap ────────────────────────────────────
with tab5:
    st.subheader("Full Factor Heatmap")

    heatmap_data = filtered[["Ticker", "Value", "Quality", "Strength", "Momentum", "Composite"]].set_index("Ticker")

    fig_heat = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns.tolist(),
        y=heatmap_data.index.tolist(),
        color_continuous_scale="Rainbow",
        aspect="auto",
    )
    fig_heat.update_layout(
        template="plotly_dark",
        height=max(400, len(filtered) * 22),
        title="Factor Scores by Stock",
        xaxis_title="Factor",
        yaxis_title="",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Correlation matrix
    st.subheader("Factor Correlations")
    corr_cols = ["Composite", "Value", "Quality", "Strength", "Momentum"]
    if "pe_trailing" in df.columns:
        numeric_cols = corr_cols + ["pe_trailing", "pb_ratio", "dividend_yield", "roe", "cash_to_mcap"]
        corr_df = filtered[numeric_cols].corr()
    else:
        corr_df = filtered[corr_cols].corr()

    fig_corr = px.imshow(
        corr_df,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
    )
    fig_corr.update_layout(
        template="plotly_dark",
        height=500,
        title="Factor Correlation Matrix",
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ─── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption("Japan Value Quant v1.0 | Data: yfinance | Model: japan_deep_value_v1")
