"""
Layer-wise Analysis Page

Visualize and compare feature statistics across all transformer layers.
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
    get_layer_summary_df,
    load_feature_stats,
)

st.set_page_config(
    page_title="Layer Analysis | DINOv3 Visualizer",
    page_icon="📈",
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
</style>
""",
    unsafe_allow_html=True,
)


def load_stats() -> FeatureStats | None:
    """Load stats from session state or default path."""
    if "stats" in st.session_state:
        return st.session_state["stats"]

    # Try default path
    base = Path(__file__).parent.parent.parent.parent
    default_path = base / "outputs" / "feature_stats" / "dinov3_feature_stats.pt"
    if default_path.exists():
        stats = load_feature_stats(default_path)
        st.session_state["stats"] = stats
        return stats

    return None


def plot_layer_trends(df: pd.DataFrame, metric: str, title: str):
    """Create layer trend plot for a specific metric."""
    # Color palette for representations
    colors = {
        "pre": "#ef4444",  # Red
        "per_layer_ln": "#22c55e",  # Green
        "global_ln": "#3b82f6",  # Blue
    }

    fig = go.Figure()

    for rep_name in ["pre", "per_layer_ln", "global_ln"]:
        for role in ["cls", "patch"]:
            subset = df[(df["rep_name"] == rep_name) & (df["role"] == role)]
            if subset.empty:
                continue

            line_style = "solid" if role == "cls" else "dash"
            name = f"{rep_name} ({role})"

            fig.add_trace(
                go.Scatter(
                    x=subset["layer_idx"],
                    y=subset[metric],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=colors[rep_name], dash=line_style, width=2),
                    marker=dict(size=6),
                    hovertemplate=f"Layer %{{x}}<br>{metric}: %{{y:.4f}}<extra>{name}</extra>",
                )
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Layer Index",
        yaxis_title=metric,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(30, 41, 59, 0.8)",
        ),
        hovermode="x unified",
        height=450,
    )

    fig.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.1)",
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.1)",
        zeroline=False,
    )

    return fig


def plot_heatmap(df: pd.DataFrame, metric: str, role: str):
    """Create heatmap of metric across layers and representations."""
    subset = df[df["role"] == role].copy()
    pivot = subset.pivot(index="rep_name", columns="layer_idx", values=metric)

    # Reorder rows
    row_order = ["pre", "per_layer_ln", "global_ln"]
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Viridis",
            hovertemplate="Layer: %{x}<br>Rep: %{y}<br>Value: %{z:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"{metric} Heatmap ({role} tokens)", font=dict(size=16)),
        xaxis_title="Layer Index",
        yaxis_title="Representation",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        height=300,
    )

    return fig


def plot_comparison_bars(df: pd.DataFrame, layer_idx: int):
    """Create bar chart comparing all metrics for a specific layer."""
    layer_data = df[df["layer_idx"] == layer_idx].copy()

    metrics = ["mean_of_means", "std_of_means", "mean_of_stds", "mean_max_abs"]
    metric_labels = ["Mean of Means", "Std of Means", "Mean of Stds", "Mean Max|x|"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=metric_labels,
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    colors = {
        "pre": "#ef4444",
        "per_layer_ln": "#22c55e",
        "global_ln": "#3b82f6",
    }

    for idx, metric in enumerate(metrics):
        row = idx // 2 + 1
        col = idx % 2 + 1

        for rep_name in ["pre", "per_layer_ln", "global_ln"]:
            for role in ["cls", "patch"]:
                subset = layer_data[
                    (layer_data["rep_name"] == rep_name) & (layer_data["role"] == role)
                ]
                if subset.empty:
                    continue

                pattern = "" if role == "cls" else "/"
                name = f"{rep_name} ({role})"

                fig.add_trace(
                    go.Bar(
                        x=[name],
                        y=subset[metric].values,
                        name=name,
                        marker_color=colors[rep_name],
                        marker_pattern_shape=pattern,
                        showlegend=(idx == 0),
                        hovertemplate=f"{name}<br>{metric}: %{{y:.4f}}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title=dict(text=f"Layer {layer_idx} Statistics Comparison", font=dict(size=18)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        height=600,
        barmode="group",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(30, 41, 59, 0.8)",
        ),
    )

    return fig


def main():
    st.markdown(
        '<p class="main-header">📈 Layer-wise Analysis</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Compare feature statistics across all transformer layers</p>',
        unsafe_allow_html=True,
    )

    stats = load_stats()

    if stats is None:
        st.warning("⚠️ No data loaded. Please select a file on the main page first.")
        return

    # Get summary dataframe
    df = get_layer_summary_df(stats)

    if df.empty:
        st.error("No layer data available.")
        return

    st.markdown("---")

    # Metric selection
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### 📊 Settings")
        metric = st.selectbox(
            "Select Metric",
            [
                "mean_of_means",
                "std_of_means",
                "mean_of_stds",
                "std_of_stds",
                "mean_max_abs",
                "max_max_abs",
            ],
            format_func=lambda x: {
                "mean_of_means": "Mean of Channel Means",
                "std_of_means": "Std of Channel Means",
                "mean_of_stds": "Mean of Channel Stds",
                "std_of_stds": "Std of Channel Stds",
                "mean_max_abs": "Mean of Max|x|",
                "max_max_abs": "Max of Max|x|",
            }[x],
        )

        show_cls = st.checkbox("Show CLS tokens", value=True)
        show_patch = st.checkbox("Show patch tokens", value=True)

    with col2:
        st.markdown("### 📈 Layer Trend")

        # Filter by role
        filter_df = df.copy()
        if not show_cls:
            filter_df = filter_df[filter_df["role"] != "cls"]
        if not show_patch:
            filter_df = filter_df[filter_df["role"] != "patch"]

        if filter_df.empty:
            st.info("Select at least one token type to display.")
        else:
            metric_title = {
                "mean_of_means": "Mean of Channel Means",
                "std_of_means": "Std of Channel Means",
                "mean_of_stds": "Mean of Channel Stds",
                "std_of_stds": "Std of Channel Stds",
                "mean_max_abs": "Mean of Max|x|",
                "max_max_abs": "Max of Max|x|",
            }[metric]

            fig = plot_layer_trends(filter_df, metric, f"{metric_title} Across Layers")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Heatmaps
    st.markdown("### 🗺️ Layer-Representation Heatmaps")

    tab1, tab2 = st.tabs(["CLS Tokens", "Patch Tokens"])

    with tab1:
        fig_cls = plot_heatmap(df, metric, "cls")
        st.plotly_chart(fig_cls, use_container_width=True)

    with tab2:
        fig_patch = plot_heatmap(df, metric, "patch")
        st.plotly_chart(fig_patch, use_container_width=True)

    st.markdown("---")

    # Layer comparison
    st.markdown("### 🔍 Single Layer Deep Dive")

    layer_idx = st.slider(
        "Select Layer",
        min_value=0,
        max_value=stats.num_layers - 1,
        value=stats.num_layers - 1,
        help="Choose a layer to see detailed statistics comparison",
    )

    fig_bars = plot_comparison_bars(df, layer_idx)
    st.plotly_chart(fig_bars, use_container_width=True)

    # Raw data table
    with st.expander("📋 View Raw Data Table"):
        layer_data = df[df["layer_idx"] == layer_idx].copy()
        st.dataframe(
            layer_data.style.format(
                {
                    "mean_of_means": "{:.6f}",
                    "std_of_means": "{:.6f}",
                    "mean_of_stds": "{:.6f}",
                    "std_of_stds": "{:.6f}",
                    "mean_max_abs": "{:.4f}",
                    "max_max_abs": "{:.4f}",
                }
            ),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()



