"""
Comparison Page

Compare different normalization representations and token types side-by-side.
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
    get_layer_summary_df,
    load_feature_stats,
)

st.set_page_config(
    page_title="Comparison | DINOv3 Visualizer",
    page_icon="📊",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%);
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
    .comparison-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
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


def plot_rep_comparison(stats: FeatureStats, layer_idx: int, role: str, metric: str):
    """Create side-by-side comparison of representations."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Pre-norm", "Per-layer LN", "Global LN"],
        horizontal_spacing=0.08,
    )

    rep_names = ["pre", "per_layer_ln", "global_ln"]
    colors = ["#ef4444", "#22c55e", "#3b82f6"]

    for i, (rep_name, color) in enumerate(zip(rep_names, colors)):
        df = get_channel_df(stats, layer_idx, rep_name, role)
        if df.empty:
            continue

        fig.add_trace(
            go.Histogram(
                x=df[metric],
                nbinsx=40,
                marker_color=color,
                opacity=0.8,
                name=rep_name,
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        title=dict(text=f"Layer {layer_idx} - {metric} Distribution by Representation", font=dict(size=16)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        height=350,
    )

    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor="rgba(148, 163, 184, 0.1)",
            row=1,
            col=i,
            title_text=metric if i == 2 else "",
        )
        fig.update_yaxes(
            gridcolor="rgba(148, 163, 184, 0.1)",
            row=1,
            col=i,
            title_text="Count" if i == 1 else "",
        )

    return fig


def plot_role_comparison(stats: FeatureStats, layer_idx: int, rep_name: str, metric: str):
    """Create overlay comparison of CLS vs Patch tokens."""
    fig = go.Figure()

    colors = {"cls": "#f59e0b", "patch": "#8b5cf6"}

    for role in ["cls", "patch"]:
        df = get_channel_df(stats, layer_idx, rep_name, role)
        if df.empty:
            continue

        fig.add_trace(
            go.Histogram(
                x=df[metric],
                nbinsx=50,
                marker_color=colors[role],
                opacity=0.6,
                name=f"{role.upper()} tokens",
            )
        )

    fig.update_layout(
        title=dict(text=f"CLS vs Patch Tokens - {metric} Distribution", font=dict(size=16)),
        xaxis_title=metric,
        yaxis_title="Count",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        height=350,
        barmode="overlay",
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


def plot_channel_correlation(stats: FeatureStats, layer_idx: int, role: str):
    """Plot correlation between different representations for each channel."""
    df_pre = get_channel_df(stats, layer_idx, "pre", role)
    df_layer = get_channel_df(stats, layer_idx, "per_layer_ln", role)
    df_global = get_channel_df(stats, layer_idx, "global_ln", role)

    if df_pre.empty or df_layer.empty or df_global.empty:
        return None

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Pre vs Per-layer LN",
            "Pre vs Global LN",
            "Per-layer LN vs Global LN",
        ],
        horizontal_spacing=0.08,
    )

    # Pre vs Per-layer
    fig.add_trace(
        go.Scatter(
            x=df_pre["max_abs"],
            y=df_layer["max_abs"],
            mode="markers",
            marker=dict(color="#6366f1", size=5, opacity=0.5),
            name="Pre vs Per-layer",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Pre vs Global
    fig.add_trace(
        go.Scatter(
            x=df_pre["max_abs"],
            y=df_global["max_abs"],
            mode="markers",
            marker=dict(color="#22c55e", size=5, opacity=0.5),
            name="Pre vs Global",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Per-layer vs Global
    fig.add_trace(
        go.Scatter(
            x=df_layer["max_abs"],
            y=df_global["max_abs"],
            mode="markers",
            marker=dict(color="#f59e0b", size=5, opacity=0.5),
            name="Per-layer vs Global",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # Add diagonal reference lines
    for i in range(1, 4):
        fig.add_trace(
            go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=i,
        )

    fig.update_layout(
        title=dict(text=f"Channel Max|x| Correlation Between Representations", font=dict(size=16)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        height=350,
    )

    return fig


def plot_layer_progression(stats: FeatureStats, rep_name: str, role: str, channels: list):
    """Plot how specific channels evolve across layers."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, ch_idx in enumerate(channels):
        values = []
        layers = []

        for layer_idx in stats.layer_indices:
            df = get_channel_df(stats, layer_idx, rep_name, role)
            if df.empty or ch_idx >= len(df):
                continue
            values.append(df.iloc[ch_idx]["max_abs"])
            layers.append(layer_idx)

        if values:
            fig.add_trace(
                go.Scatter(
                    x=layers,
                    y=values,
                    mode="lines+markers",
                    name=f"Channel {ch_idx}",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                )
            )

    fig.update_layout(
        title=dict(text="Channel Evolution Across Layers (Max|x|)", font=dict(size=16)),
        xaxis_title="Layer Index",
        yaxis_title="Max|x|",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        height=400,
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


def compute_stats_diff(stats: FeatureStats, layer_idx: int, role: str):
    """Compute differences between representations."""
    df_pre = get_channel_df(stats, layer_idx, "pre", role)
    df_layer = get_channel_df(stats, layer_idx, "per_layer_ln", role)
    df_global = get_channel_df(stats, layer_idx, "global_ln", role)

    if df_pre.empty:
        return None

    result = {
        "Pre-norm": {
            "Mean μ": df_pre["mean"].mean(),
            "Std μ": df_pre["mean"].std(),
            "Mean σ": df_pre["std"].mean(),
            "Mean Max|x|": df_pre["max_abs"].mean(),
            "Max Max|x|": df_pre["max_abs"].max(),
        },
    }

    if not df_layer.empty:
        result["Per-layer LN"] = {
            "Mean μ": df_layer["mean"].mean(),
            "Std μ": df_layer["mean"].std(),
            "Mean σ": df_layer["std"].mean(),
            "Mean Max|x|": df_layer["max_abs"].mean(),
            "Max Max|x|": df_layer["max_abs"].max(),
        }

    if not df_global.empty:
        result["Global LN"] = {
            "Mean μ": df_global["mean"].mean(),
            "Std μ": df_global["mean"].std(),
            "Mean σ": df_global["std"].mean(),
            "Mean Max|x|": df_global["max_abs"].mean(),
            "Max Max|x|": df_global["max_abs"].max(),
        }

    return pd.DataFrame(result).T


def main():
    st.markdown(
        '<p class="main-header">📊 Comparison View</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Compare representations, token types, and channels side-by-side</p>',
        unsafe_allow_html=True,
    )

    stats = load_stats()

    if stats is None:
        st.warning("⚠️ No data loaded. Please select a file on the main page first.")
        return

    st.markdown("---")

    # Global settings
    col1, col2, col3 = st.columns(3)

    with col1:
        layer_idx = st.selectbox(
            "Layer",
            options=stats.layer_indices,
            index=len(stats.layer_indices) - 1,
        )

    with col2:
        role = st.selectbox(
            "Token Role",
            options=["patch", "cls"],
            format_func=lambda x: {"cls": "CLS Token", "patch": "Patch Tokens"}[x],
        )

    with col3:
        metric = st.selectbox(
            "Metric",
            options=["max_abs", "std", "mean"],
            format_func=lambda x: {"max_abs": "Max|x|", "std": "Std", "mean": "Mean"}[x],
        )

    st.markdown("---")

    # Representation comparison
    st.markdown("### 🔄 Representation Comparison")
    st.markdown("Compare Pre-norm, Per-layer LN, and Global LN distributions")

    fig_rep = plot_rep_comparison(stats, layer_idx, role, metric)
    st.plotly_chart(fig_rep, use_container_width=True)

    # Stats comparison table
    stats_df = compute_stats_diff(stats, layer_idx, role)
    if stats_df is not None:
        st.markdown("#### Summary Statistics")
        st.dataframe(
            stats_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0),
            use_container_width=True,
        )

    st.markdown("---")

    # CLS vs Patch comparison
    st.markdown("### 🎯 CLS vs Patch Token Comparison")

    rep_for_role = st.selectbox(
        "Select Representation",
        options=["pre", "per_layer_ln", "global_ln"],
        format_func=lambda x: {
            "pre": "Pre-norm",
            "per_layer_ln": "Per-layer LN",
            "global_ln": "Global LN",
        }[x],
        key="rep_for_role",
    )

    fig_role = plot_role_comparison(stats, layer_idx, rep_for_role, metric)
    st.plotly_chart(fig_role, use_container_width=True)

    st.markdown("---")

    # Channel correlation
    st.markdown("### 🔗 Cross-Representation Channel Correlation")
    st.markdown("How do channel values correlate across different normalizations?")

    fig_corr = plot_channel_correlation(stats, layer_idx, role)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # Channel progression
    st.markdown("### 📈 Channel Layer Progression")
    st.markdown("Track specific channels across all layers")

    col1, col2 = st.columns([1, 3])

    with col1:
        # Get channel data to find interesting channels
        df_current = get_channel_df(stats, layer_idx, "pre", role)
        if not df_current.empty:
            top_channels = df_current.nlargest(5, "max_abs")["channel_idx"].tolist()
            default_channels = top_channels[:3]
        else:
            default_channels = [0, 1, 2]

        selected_channels = st.multiselect(
            "Select Channels",
            options=list(range(stats.embedding_dim)),
            default=default_channels[:3],
            max_selections=8,
            help="Select up to 8 channels to track",
        )

        rep_for_prog = st.selectbox(
            "Representation",
            options=["pre", "per_layer_ln", "global_ln"],
            format_func=lambda x: {
                "pre": "Pre-norm",
                "per_layer_ln": "Per-layer LN",
                "global_ln": "Global LN",
            }[x],
            key="rep_for_prog",
        )

    with col2:
        if selected_channels:
            fig_prog = plot_layer_progression(stats, rep_for_prog, role, selected_channels)
            st.plotly_chart(fig_prog, use_container_width=True)
        else:
            st.info("Select channels to visualize their progression across layers.")

    st.markdown("---")

    # Quick insights
    st.markdown("### 💡 Quick Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Normalization Effect")

        df_pre = get_channel_df(stats, layer_idx, "pre", role)
        df_global = get_channel_df(stats, layer_idx, "global_ln", role)

        if not df_pre.empty and not df_global.empty:
            pre_max = df_pre["max_abs"].max()
            global_max = df_global["max_abs"].max()
            reduction = (1 - global_max / pre_max) * 100 if pre_max > 0 else 0

            st.markdown(
                f"""
            - **Pre-norm Max|x|:** {pre_max:.4f}
            - **Global LN Max|x|:** {global_max:.4f}
            - **Reduction:** {reduction:.1f}%
            """
            )

    with col2:
        st.markdown("#### Value Ranges")

        if not df_pre.empty:
            st.markdown(
                f"""
            - **Mean Range:** [{df_pre['mean'].min():.4f}, {df_pre['mean'].max():.4f}]
            - **Std Range:** [{df_pre['std'].min():.4f}, {df_pre['std'].max():.4f}]
            - **Max|x| Range:** [{df_pre['max_abs'].min():.4f}, {df_pre['max_abs'].max():.4f}]
            """
            )


if __name__ == "__main__":
    main()

