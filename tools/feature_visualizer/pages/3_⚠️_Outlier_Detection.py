"""
Outlier Detection Page

Identify anomalous channels with extreme values across layers.
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
    find_outlier_channels,
    get_layer_summary_df,
    load_feature_stats,
)

st.set_page_config(
    page_title="Outlier Detection | DINOv3 Visualizer",
    page_icon="⚠️",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #ef4444 0%, #f59e0b 50%, #eab308 100%);
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
    .outlier-high {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
    }
    .outlier-low {
        background: linear-gradient(145deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.5);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
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


def plot_outlier_scatter(df: pd.DataFrame, threshold: float):
    """Create scatter plot highlighting outliers."""
    df = df.copy()
    df["is_outlier"] = np.abs(df["z_score"]) > threshold
    df["outlier_type"] = df.apply(
        lambda row: "High" if row["z_score"] > threshold else ("Low" if row["z_score"] < -threshold else "Normal"),
        axis=1,
    )

    colors = {"High": "#ef4444", "Low": "#3b82f6", "Normal": "#64748b"}

    fig = go.Figure()

    for otype in ["Normal", "Low", "High"]:
        subset = df[df["outlier_type"] == otype]
        if subset.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=subset["channel_idx"],
                y=subset["z_score"],
                mode="markers",
                name=otype,
                marker=dict(
                    color=colors[otype],
                    size=10 if otype != "Normal" else 6,
                    opacity=0.9 if otype != "Normal" else 0.5,
                    symbol="diamond" if otype != "Normal" else "circle",
                ),
                hovertemplate="Channel %{x}<br>Z-score: %{y:.2f}<extra>" + otype + "</extra>",
            )
        )

    # Add threshold lines
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444", annotation_text=f"+{threshold}σ")
    fig.add_hline(y=-threshold, line_dash="dash", line_color="#3b82f6", annotation_text=f"-{threshold}σ")
    fig.add_hline(y=0, line_dash="solid", line_color="#64748b", line_width=1)

    fig.update_layout(
        title=dict(text="Channel Z-Scores with Outlier Threshold", font=dict(size=16)),
        xaxis_title="Channel Index",
        yaxis_title="Z-Score",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.1)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.1)")

    return fig


def plot_cross_layer_outliers(stats: FeatureStats, metric: str, role: str, threshold: float):
    """Create heatmap showing outlier channels across layers."""
    all_data = []

    for layer_idx in stats.layer_indices:
        for rep_name in ["pre", "per_layer_ln", "global_ln"]:
            df = get_channel_df(stats, layer_idx, rep_name, role)
            if df.empty:
                continue

            values = df[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            z_scores = (values - mean_val) / (std_val + 1e-8)

            for i, (ch, z) in enumerate(zip(df["channel_idx"], z_scores)):
                all_data.append(
                    {
                        "layer_idx": layer_idx,
                        "rep_name": rep_name,
                        "channel_idx": ch,
                        "z_score": z,
                        "value": values[i],
                        "is_outlier": abs(z) > threshold,
                    }
                )

    df_all = pd.DataFrame(all_data)

    # Count outliers per layer
    outlier_counts = (
        df_all[df_all["is_outlier"]]
        .groupby(["layer_idx", "rep_name"])
        .size()
        .reset_index(name="outlier_count")
    )

    # Pivot for heatmap
    pivot = outlier_counts.pivot(index="rep_name", columns="layer_idx", values="outlier_count").fillna(0)

    # Reorder
    row_order = ["pre", "per_layer_ln", "global_ln"]
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="YlOrRd",
            hovertemplate="Layer: %{x}<br>Rep: %{y}<br>Outliers: %{z}<extra></extra>",
            colorbar=dict(title="Count"),
        )
    )

    fig.update_layout(
        title=dict(text=f"Outlier Channel Count by Layer ({metric}, {role})", font=dict(size=16)),
        xaxis_title="Layer Index",
        yaxis_title="Representation",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        height=250,
    )

    return fig, df_all


def identify_persistent_outliers(df_all: pd.DataFrame, threshold: float, min_layers: int):
    """Find channels that are outliers across multiple layers."""
    outlier_df = df_all[df_all["is_outlier"]].copy()

    # Count how many layers each channel is an outlier
    channel_outlier_counts = (
        outlier_df.groupby(["channel_idx", "rep_name"])
        .agg(
            layer_count=("layer_idx", "count"),
            layers=("layer_idx", list),
            mean_z_score=("z_score", "mean"),
        )
        .reset_index()
    )

    # Filter by minimum layer threshold
    persistent = channel_outlier_counts[channel_outlier_counts["layer_count"] >= min_layers].copy()
    persistent = persistent.sort_values("layer_count", ascending=False)

    return persistent


def main():
    st.markdown(
        '<p class="main-header">⚠️ Outlier Detection</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Identify channels with anomalous feature values</p>',
        unsafe_allow_html=True,
    )

    stats = load_stats()

    if stats is None:
        st.warning("⚠️ No data loaded. Please select a file on the main page first.")
        return

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Detection Settings")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric = st.selectbox(
            "Metric",
            ["max_abs", "std", "mean"],
            format_func=lambda x: {"max_abs": "Max|x|", "std": "Std", "mean": "Mean"}[x],
            help="Metric to use for outlier detection",
        )

    with col2:
        threshold = st.slider(
            "Z-Score Threshold",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Number of standard deviations for outlier threshold",
        )

    with col3:
        role = st.selectbox(
            "Token Role",
            ["patch", "cls"],
            format_func=lambda x: {"cls": "CLS Token", "patch": "Patch Tokens"}[x],
        )

    with col4:
        top_k = st.slider(
            "Top K Outliers",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of top outliers to display",
        )

    st.markdown("---")

    # Single layer analysis
    st.markdown("### 🔬 Single Layer Outlier Analysis")

    col1, col2 = st.columns([1, 3])

    with col1:
        layer_idx = st.selectbox(
            "Select Layer",
            options=stats.layer_indices,
            index=len(stats.layer_indices) - 1,
        )

        rep_name = st.selectbox(
            "Representation",
            options=["pre", "per_layer_ln", "global_ln"],
            format_func=lambda x: {
                "pre": "Pre-norm",
                "per_layer_ln": "Per-layer LN",
                "global_ln": "Global LN",
            }[x],
        )

    with col2:
        outlier_df = find_outlier_channels(
            stats, layer_idx, rep_name, role, metric, top_k, threshold
        )

        if not outlier_df.empty:
            fig = plot_outlier_scatter(outlier_df, threshold)
            st.plotly_chart(fig, use_container_width=True)

    # Outlier table
    if not outlier_df.empty:
        n_outliers = outlier_df["is_outlier"].sum()
        st.markdown(f"**Found {n_outliers} outliers** (|z| > {threshold})")

        # Format for display
        display_df = outlier_df[["channel_idx", "mean", "std", "max_abs", "z_score", "is_outlier"]].copy()
        display_df = display_df.rename(columns={"channel_idx": "Channel"})

        st.dataframe(
            display_df.head(top_k).style.format(
                {
                    "mean": "{:.6f}",
                    "std": "{:.6f}",
                    "max_abs": "{:.4f}",
                    "z_score": "{:+.2f}",
                }
            ).applymap(
                lambda x: "background-color: rgba(239, 68, 68, 0.3)" if x else "",
                subset=["is_outlier"],
            ),
            use_container_width=True,
        )

    st.markdown("---")

    # Cross-layer analysis
    st.markdown("### 🌐 Cross-Layer Outlier Analysis")

    fig_heatmap, df_all = plot_cross_layer_outliers(stats, metric, role, threshold)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    # Persistent outliers
    st.markdown("### 🔄 Persistent Outliers")
    st.markdown("Channels that appear as outliers across multiple layers")

    col1, col2 = st.columns([1, 3])

    with col1:
        min_layers = st.slider(
            "Min Layers",
            min_value=2,
            max_value=min(10, stats.num_layers),
            value=min(5, stats.num_layers // 2),
            help="Minimum number of layers where channel must be an outlier",
        )

    persistent_df = identify_persistent_outliers(df_all, threshold, min_layers)

    with col2:
        if persistent_df.empty:
            st.info(f"No channels found that are outliers in ≥{min_layers} layers.")
        else:
            # Create bar chart
            fig = go.Figure()

            colors = {
                "pre": "#ef4444",
                "per_layer_ln": "#22c55e",
                "global_ln": "#3b82f6",
            }

            for rep in ["pre", "per_layer_ln", "global_ln"]:
                subset = persistent_df[persistent_df["rep_name"] == rep]
                if subset.empty:
                    continue

                fig.add_trace(
                    go.Bar(
                        x=subset["channel_idx"].astype(str),
                        y=subset["layer_count"],
                        name=rep,
                        marker_color=colors[rep],
                        hovertemplate="Channel %{x}<br>Outlier in %{y} layers<extra>" + rep + "</extra>",
                    )
                )

            fig.update_layout(
                title=dict(text=f"Persistent Outlier Channels (≥{min_layers} layers)", font=dict(size=16)),
                xaxis_title="Channel Index",
                yaxis_title="Number of Layers",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.8)",
                height=400,
                barmode="group",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

    # Persistent outlier table
    if not persistent_df.empty:
        st.markdown("#### Persistent Outlier Details")

        # Prepare display dataframe
        display_persistent = persistent_df.copy()
        display_persistent["layers"] = display_persistent["layers"].apply(
            lambda x: ", ".join(map(str, sorted(x)[:10])) + ("..." if len(x) > 10 else "")
        )

        st.dataframe(
            display_persistent.style.format({"mean_z_score": "{:+.2f}"}),
            use_container_width=True,
        )

    st.markdown("---")

    # Summary statistics
    st.markdown("### 📊 Outlier Summary")

    total_channels = stats.embedding_dim
    total_outliers = df_all["is_outlier"].sum()
    unique_outlier_channels = df_all[df_all["is_outlier"]]["channel_idx"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Channels", total_channels)
    with col2:
        st.metric("Total Outlier Instances", total_outliers)
    with col3:
        st.metric("Unique Outlier Channels", unique_outlier_channels)
    with col4:
        pct = (unique_outlier_channels / total_channels * 100) if total_channels > 0 else 0
        st.metric("Outlier Rate", f"{pct:.1f}%")


if __name__ == "__main__":
    main()



