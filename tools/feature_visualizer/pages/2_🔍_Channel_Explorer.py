"""
Channel Explorer Page

Deep dive into per-channel statistics for any layer/representation/role.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import (
    FeatureStats,
    get_channel_df,
    load_feature_stats,
)

st.set_page_config(
    page_title="Channel Explorer | DINOv3 Visualizer",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_stats() -> FeatureStats | None:
    """Load stats from session state or default path."""
    if "stats" in st.session_state:
        return st.session_state["stats"]

    base = Path(__file__).parent.parent.parent.parent
    default_path = base / "outputs" / "feature_stats" / "dinov3_feature_stats.pt"
    if default_path.exists():
        stats = load_feature_stats(default_path)
        st.session_state["stats"] = stats
        return stats

    return None


def plot_channel_histogram(df: pd.DataFrame, column: str, title: str, color: str):
    """Create histogram for a channel statistic."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df[column],
            nbinsx=50,
            marker_color=color,
            opacity=0.8,
            hovertemplate=f"{column}: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>",
        )
    )

    # Add vertical lines for mean and std bounds
    mean_val = df[column].mean()
    std_val = df[column].std()

    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#ffffff",
        annotation_text=f"μ = {mean_val:.4f}",
        annotation_position="top",
    )
    fig.add_vline(
        x=mean_val - 2 * std_val,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="-2σ",
        annotation_position="bottom left",
    )
    fig.add_vline(
        x=mean_val + 2 * std_val,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="+2σ",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=column,
        yaxis_title="Channel Count",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        height=350,
        showlegend=False,
    )

    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.1)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.1)")

    return fig


def plot_channel_scatter(df: pd.DataFrame):
    """Create scatter plot of mean vs std for each channel."""
    fig = go.Figure()

    # Color by max_abs
    fig.add_trace(
        go.Scatter(
            x=df["mean"],
            y=df["std"],
            mode="markers",
            marker=dict(
                size=8,
                color=df["max_abs"],
                colorscale="Plasma",
                colorbar=dict(title="Max|x|", thickness=15),
                opacity=0.7,
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
            ),
            text=[f"Channel {i}" for i in df["channel_idx"]],
            hovertemplate="Channel %{text}<br>Mean: %{x:.4f}<br>Std: %{y:.4f}<br>Max|x|: %{marker.color:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="Channel Mean vs Std (colored by Max|x|)", font=dict(size=16)),
        xaxis_title="Mean",
        yaxis_title="Std",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        height=450,
    )

    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.1)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.1)")

    return fig


def plot_channel_bars(df: pd.DataFrame, column: str, top_n: int = 50):
    """Create bar chart for top N channels by a metric."""
    df_sorted = df.sort_values(column, ascending=False).head(top_n)

    # Create color gradient
    colors = px.colors.sample_colorscale(
        "Turbo", np.linspace(0, 1, len(df_sorted))
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_sorted["channel_idx"].astype(str),
            y=df_sorted[column],
            marker_color=colors,
            hovertemplate="Channel %{x}<br>" + column + ": %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"Top {top_n} Channels by {column}", font=dict(size=16)),
        xaxis_title="Channel Index",
        yaxis_title=column,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        height=400,
        xaxis=dict(tickangle=45, dtick=5),
    )

    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.1)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.1)")

    return fig


def plot_all_channels_line(df: pd.DataFrame, column: str, title: str, color: str):
    """Create line plot showing all channels."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["channel_idx"],
            y=df[column],
            mode="lines",
            line=dict(color=color, width=1),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)",
            hovertemplate="Channel %{x}<br>" + column + ": %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Channel Index",
        yaxis_title=column,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        height=300,
    )

    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.1)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.1)")

    return fig


def main():
    st.markdown(
        '<p class="main-header">🔍 Channel Explorer</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Deep dive into per-channel feature statistics</p>',
        unsafe_allow_html=True,
    )

    stats = load_stats()

    if stats is None:
        st.warning("⚠️ No data loaded. Please select a file on the main page first.")
        return

    st.markdown("---")

    # Selection controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        layer_idx = st.selectbox(
            "Layer",
            options=stats.layer_indices,
            index=len(stats.layer_indices) - 1,
            help="Select transformer layer",
        )

    with col2:
        rep_name = st.selectbox(
            "Representation",
            options=["pre", "per_layer_ln", "global_ln"],
            format_func=lambda x: {
                "pre": "Pre-norm",
                "per_layer_ln": "Per-layer LN",
                "global_ln": "Global LN",
            }[x],
        )

    with col3:
        role = st.selectbox(
            "Token Role",
            options=["cls", "patch"],
            format_func=lambda x: {"cls": "CLS Token", "patch": "Patch Tokens"}[x],
        )

    with col4:
        st.markdown("")  # Spacing
        st.markdown(f"**Embedding Dim:** `{stats.embedding_dim}`")

    # Load channel data
    df = get_channel_df(stats, layer_idx, rep_name, role)

    if df.empty:
        st.error("No channel data available for this selection.")
        return

    st.markdown("---")

    # Summary statistics
    st.markdown("### 📊 Summary Statistics")

    cols = st.columns(6)
    summary_stats = [
        ("Mean μ", df["mean"].mean(), "#6366f1"),
        ("Std μ", df["mean"].std(), "#8b5cf6"),
        ("Mean σ", df["std"].mean(), "#06b6d4"),
        ("Std σ", df["std"].std(), "#22c55e"),
        ("Mean Max|x|", df["max_abs"].mean(), "#f59e0b"),
        ("Max Max|x|", df["max_abs"].max(), "#ef4444"),
    ]

    for col, (label, value, color) in zip(cols, summary_stats):
        with col:
            st.markdown(
                f"""
            <div class="stat-card">
                <div class="stat-value" style="color: {color};">{value:.4f}</div>
                <div class="stat-label">{label}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Histograms
    st.markdown("### 📈 Channel Distributions")

    tab1, tab2, tab3 = st.tabs(["Mean", "Std", "Max|x|"])

    with tab1:
        fig_mean = plot_channel_histogram(
            df, "mean", "Distribution of Channel Means", "#6366f1"
        )
        st.plotly_chart(fig_mean, use_container_width=True)

    with tab2:
        fig_std = plot_channel_histogram(
            df, "std", "Distribution of Channel Stds", "#22c55e"
        )
        st.plotly_chart(fig_std, use_container_width=True)

    with tab3:
        fig_max = plot_channel_histogram(
            df, "max_abs", "Distribution of Channel Max|x|", "#f59e0b"
        )
        st.plotly_chart(fig_max, use_container_width=True)

    st.markdown("---")

    # Scatter plot
    st.markdown("### 🎯 Channel Relationships")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_scatter = plot_channel_scatter(df)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.markdown("#### Correlation Matrix")
        corr = df[["mean", "std", "max_abs"]].corr()
        st.dataframe(
            corr.style.format("{:.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
            use_container_width=True,
        )

        st.markdown("#### Quick Stats")
        st.markdown(
            f"""
        - **Total Channels:** {len(df)}
        - **Mean Range:** [{df['mean'].min():.4f}, {df['mean'].max():.4f}]
        - **Std Range:** [{df['std'].min():.4f}, {df['std'].max():.4f}]
        - **Max|x| Range:** [{df['max_abs'].min():.4f}, {df['max_abs'].max():.4f}]
        """
        )

    st.markdown("---")

    # All channels view
    st.markdown("### 📉 All Channels Overview")

    metric_to_plot = st.radio(
        "Select metric",
        ["mean", "std", "max_abs"],
        horizontal=True,
        format_func=lambda x: {"mean": "Mean", "std": "Std", "max_abs": "Max|x|"}[x],
    )

    colors = {"mean": "#6366f1", "std": "#22c55e", "max_abs": "#f59e0b"}
    fig_line = plot_all_channels_line(
        df, metric_to_plot, f"{metric_to_plot.title()} Across All Channels", colors[metric_to_plot]
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # Top channels
    st.markdown("### 🏆 Top Channels")

    col1, col2 = st.columns(2)

    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["max_abs", "std", "mean"],
            format_func=lambda x: {"max_abs": "Max|x|", "std": "Std", "mean": "Mean"}[x],
        )

    with col2:
        top_n = st.slider("Number of channels", 10, 100, 30)

    fig_top = plot_channel_bars(df, sort_by, top_n)
    st.plotly_chart(fig_top, use_container_width=True)

    # Data table
    with st.expander("📋 View Full Channel Data"):
        st.dataframe(
            df.style.format({"mean": "{:.6f}", "std": "{:.6f}", "max_abs": "{:.4f}"}),
            use_container_width=True,
            height=400,
        )


if __name__ == "__main__":
    main()



