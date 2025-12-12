"""
DINOv3 Feature Statistics Visualizer

A modern web-based visualization tool for analyzing DINOv3 backbone
intermediate feature statistics across layers and channels.

Run with: streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import discover_stats_files, load_feature_stats

# --- Page Configuration ---
st.set_page_config(
    page_title="DINOv3 Feature Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for modern look ---
st.markdown(
    """
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --background-dark: #0f172a;
        --surface-color: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif;
    }
    
    .sub-header {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(234, 88, 12, 0.1) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Make selectbox options more visible */
    .stSelectbox > div > div {
        background-color: #1e293b;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #94a3b8;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Code blocks */
    code {
        background-color: #1e293b;
        color: #e879f9;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }
    
    /* DataFrame styling */
    .dataframe {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.85rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_default_stats_path() -> Path:
    """Get the default path to feature stats file."""
    # Try common locations relative to this script
    base = Path(__file__).parent.parent.parent  # Go up to CSIRO root
    candidates = [
        base / "outputs" / "feature_stats" / "dinov3_feature_stats.pt",
        base / "outputs" / "train_all" / "feature_stats" / "dinov3_feature_stats.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # Return first candidate even if doesn't exist


def render_sidebar():
    """Render sidebar with file selection and navigation."""
    with st.sidebar:
        st.markdown("## 🔬 Feature Analyzer")
        st.markdown("---")

        # File selection
        st.markdown("### 📁 Data Source")

        # Input method selection
        input_method = st.radio(
            "Select input method",
            ["Default Path", "Browse Directory", "Manual Path"],
            label_visibility="collapsed",
        )

        stats_path = None

        if input_method == "Default Path":
            default_path = get_default_stats_path()
            if default_path.exists():
                stats_path = default_path
                st.success(f"Found: `{default_path.name}`")
            else:
                st.warning("Default file not found. Use other options.")

        elif input_method == "Browse Directory":
            search_dir = st.text_input(
                "Search directory",
                value=str(Path(__file__).parent.parent.parent / "outputs"),
                help="Directory to search for feature stats files",
            )
            if search_dir and Path(search_dir).exists():
                files = discover_stats_files(search_dir)
                if files:
                    selected = st.selectbox(
                        "Select file",
                        files,
                        format_func=lambda x: f".../{x.parent.name}/{x.name}",
                    )
                    stats_path = selected
                else:
                    st.warning("No feature stats files found.")

        else:  # Manual Path
            manual_path = st.text_input(
                "File path",
                value=str(get_default_stats_path()),
                help="Full path to the .pt file",
            )
            if manual_path and Path(manual_path).exists():
                stats_path = Path(manual_path)
            elif manual_path:
                st.error("File not found")

        st.markdown("---")

        # Navigation info
        st.markdown("### 📑 Pages")
        st.markdown(
            """
        - **Overview**: Model & data summary
        - **Layer Analysis**: Cross-layer statistics
        - **Channel Explorer**: Per-channel details
        - **Outlier Detection**: Find anomalous channels
        """
        )

        st.markdown("---")
        st.markdown(
            """
        <div style="text-align: center; color: #64748b; font-size: 0.75rem;">
            DINOv3 Feature Visualizer v1.0<br>
            Built with Streamlit & Plotly
        </div>
        """,
            unsafe_allow_html=True,
        )

        return stats_path


def render_overview(stats):
    """Render the overview/summary page."""
    st.markdown(
        '<p class="main-header">🔬 DINOv3 Feature Statistics Overview</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Comprehensive analysis of backbone intermediate features</p>',
        unsafe_allow_html=True,
    )

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🖼️ Images", f"{stats.num_images:,}")
    with col2:
        st.metric("📊 Batches", f"{stats.num_batches:,}")
    with col3:
        st.metric("🔢 Layers", stats.num_layers)
    with col4:
        st.metric("📐 Embed Dim", stats.embedding_dim)
    with col5:
        st.metric("🎯 Storage Tokens", stats.n_storage_tokens)

    st.markdown("---")

    # Model information
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 Backbone Information")
        st.markdown(
            f"""
        <div class="info-box">
            <strong>Model:</strong> <code>{stats.backbone_name}</code><br>
            <strong>LoRA Enabled:</strong> {'✅ Yes' if stats.used_lora else '❌ No'}<br>
            <strong>Weights:</strong> <code>{Path(stats.dino_weights_pt).name if stats.dino_weights_pt else 'N/A'}</code>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if stats.lora_error:
            st.markdown(
                f"""
            <div class="warning-box">
                <strong>⚠️ LoRA Error:</strong> {stats.lora_error}
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### 📁 Data Paths")
        st.markdown(
            f"""
        <div class="info-box">
            <strong>Project Dir:</strong><br>
            <code style="font-size: 0.75rem;">{stats.project_dir}</code><br><br>
            <strong>Head Weights:</strong><br>
            <code style="font-size: 0.75rem;">{Path(stats.first_head_path).name if stats.first_head_path else 'N/A'}</code>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Statistics structure explanation
    st.markdown("### 📚 Data Structure")

    st.markdown(
        """
    The feature statistics are organized in a hierarchical structure:
    
    | Level | Key | Values |
    |-------|-----|--------|
    | **Layer** | `layer_idx` | 0 to num_layers-1 |
    | **Representation** | `rep_name` | `pre`, `per_layer_ln`, `global_ln` |
    | **Token Role** | `role` | `cls`, `patch` |
    | **Statistics** | - | `count`, `mean`, `std`, `max_abs` |
    """
    )

    with st.expander("📖 Representation Types Explained"):
        st.markdown(
            """
        - **`pre`**: Raw block output before any LayerNorm (直接的 Transformer block 输出)
        - **`per_layer_ln`**: After applying a fresh LayerNorm(affine=False) per layer (每层独立的归一化)
        - **`global_ln`**: After DINOv3's shared output LayerNorm (使用 DINOv3 共享的输出 LayerNorm)
        """
        )

    with st.expander("🔧 Configuration Details"):
        st.json(stats.config)


def render_no_data():
    """Render message when no data is loaded."""
    st.markdown(
        '<p class="main-header">🔬 DINOv3 Feature Analyzer</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Interactive visualization for DINOv3 backbone features</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
    <div class="warning-box">
        <h3>📂 No Data Loaded</h3>
        <p>Please select a feature statistics file from the sidebar to begin analysis.</p>
        <p>Expected file: <code>dinov3_feature_stats.pt</code></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🚀 Quick Start")
    st.markdown(
        """
    1. Generate feature statistics using `analyze_dinov3_features.py`
    2. Select the output `.pt` file from the sidebar
    3. Explore layer-wise and channel-wise statistics
    """
    )

    st.markdown("### 📊 Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        #### 📈 Layer Analysis
        - Cross-layer statistics
        - Trend visualization
        - Normalization comparison
        """
        )

    with col2:
        st.markdown(
            """
        #### 🔍 Channel Explorer
        - Per-channel distributions
        - Interactive histograms
        - Statistical summaries
        """
        )

    with col3:
        st.markdown(
            """
        #### ⚠️ Outlier Detection
        - Z-score analysis
        - Top-k outliers
        - Cross-layer anomalies
        """
        )


def main():
    """Main application entry point."""
    # Render sidebar and get selected file path
    stats_path = render_sidebar()

    # Load data if path is valid
    if stats_path and stats_path.exists():
        try:
            # Cache the loaded data
            @st.cache_data
            def load_cached_stats(path: str):
                return load_feature_stats(path)

            stats = load_cached_stats(str(stats_path))
            render_overview(stats)

            # Store in session state for other pages
            st.session_state["stats"] = stats
            st.session_state["stats_path"] = str(stats_path)

        except Exception as e:
            st.error(f"Error loading file: {e}")
            render_no_data()
    else:
        render_no_data()


if __name__ == "__main__":
    main()



