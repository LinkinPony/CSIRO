"""
Data loader for DINOv3 feature statistics .pt files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class FeatureStats:
    """Container for loaded feature statistics."""

    # Metadata
    project_dir: str = ""
    backbone_name: str = ""
    dino_weights_pt: str = ""
    head_weights_base: str = ""
    first_head_path: str = ""
    used_lora: bool = False
    lora_error: Optional[str] = None

    # Counts
    num_images: int = 0
    num_batches: int = 0
    num_layers: int = 0
    embedding_dim: int = 0
    n_storage_tokens: int = 0

    # Raw config (for display)
    config: Dict[str, Any] = field(default_factory=dict)

    # Layer statistics: layers[layer_idx][rep_name][role] -> {count, mean, std, max_abs}
    layers: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = field(
        default_factory=dict
    )

    @property
    def layer_indices(self) -> List[int]:
        """Return sorted list of layer indices."""
        return sorted(self.layers.keys())

    @property
    def rep_names(self) -> List[str]:
        """Return list of representation names (pre, per_layer_ln, global_ln)."""
        return ["pre", "per_layer_ln", "global_ln"]

    @property
    def roles(self) -> List[str]:
        """Return list of token roles (cls, patch)."""
        return ["cls", "patch"]


def load_feature_stats(path: str | Path) -> FeatureStats:
    """
    Load feature statistics from a .pt file.

    Args:
        path: Path to the .pt file (e.g., dinov3_feature_stats.pt)

    Returns:
        FeatureStats object with all loaded data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature stats file not found: {path}")

    payload = torch.load(str(path), map_location="cpu", weights_only=False)

    meta = payload.get("meta", {})
    backbone_meta = meta.get("backbone", {})
    counts = meta.get("counts", {})

    stats = FeatureStats(
        project_dir=str(meta.get("project_dir", "")),
        backbone_name=backbone_meta.get("backbone_name", ""),
        dino_weights_pt=backbone_meta.get("dino_weights_pt", ""),
        head_weights_base=backbone_meta.get("head_weights_base", ""),
        first_head_path=backbone_meta.get("first_head_path", ""),
        used_lora=backbone_meta.get("used_lora", False),
        lora_error=backbone_meta.get("lora_error"),
        num_images=counts.get("num_images", 0),
        num_batches=counts.get("num_batches", 0),
        num_layers=counts.get("num_layers", 0),
        embedding_dim=counts.get("embedding_dim", 0),
        n_storage_tokens=counts.get("n_storage_tokens", 0),
        config=meta.get("config", {}),
        layers=payload.get("layers", {}),
    )

    return stats


def get_layer_summary_df(stats: FeatureStats) -> pd.DataFrame:
    """
    Create a summary DataFrame with per-layer statistics.

    Columns: layer_idx, rep_name, role, count, mean_of_means, std_of_means,
             mean_of_stds, mean_max_abs, max_max_abs
    """
    rows = []
    for layer_idx in stats.layer_indices:
        for rep_name in stats.rep_names:
            for role in stats.roles:
                layer_data = stats.layers.get(layer_idx, {})
                rep_data = layer_data.get(rep_name, {})
                role_data = rep_data.get(role, {})

                if not role_data:
                    continue

                count = role_data.get("count", 0)
                mean_tensor = role_data.get("mean")
                std_tensor = role_data.get("std")
                max_abs_tensor = role_data.get("max_abs")

                if mean_tensor is None:
                    continue

                # Convert to numpy
                mean_np = (
                    mean_tensor.numpy()
                    if isinstance(mean_tensor, torch.Tensor)
                    else np.array(mean_tensor)
                )
                std_np = (
                    std_tensor.numpy()
                    if isinstance(std_tensor, torch.Tensor)
                    else np.array(std_tensor)
                )
                max_abs_np = (
                    max_abs_tensor.numpy()
                    if isinstance(max_abs_tensor, torch.Tensor)
                    else np.array(max_abs_tensor)
                )

                rows.append(
                    {
                        "layer_idx": layer_idx,
                        "rep_name": rep_name,
                        "role": role,
                        "count": count,
                        "mean_of_means": float(np.mean(mean_np)),
                        "std_of_means": float(np.std(mean_np)),
                        "mean_of_stds": float(np.mean(std_np)),
                        "std_of_stds": float(np.std(std_np)),
                        "mean_max_abs": float(np.mean(max_abs_np)),
                        "max_max_abs": float(np.max(max_abs_np)),
                        "min_max_abs": float(np.min(max_abs_np)),
                    }
                )

    return pd.DataFrame(rows)


def get_channel_df(
    stats: FeatureStats, layer_idx: int, rep_name: str, role: str
) -> pd.DataFrame:
    """
    Get per-channel statistics as a DataFrame for a specific layer/rep/role.

    Returns DataFrame with columns: channel_idx, mean, std, max_abs
    """
    layer_data = stats.layers.get(layer_idx, {})
    rep_data = layer_data.get(rep_name, {})
    role_data = rep_data.get(role, {})

    if not role_data:
        return pd.DataFrame(columns=["channel_idx", "mean", "std", "max_abs"])

    mean_tensor = role_data.get("mean")
    std_tensor = role_data.get("std")
    max_abs_tensor = role_data.get("max_abs")

    if mean_tensor is None:
        return pd.DataFrame(columns=["channel_idx", "mean", "std", "max_abs"])

    mean_np = (
        mean_tensor.numpy()
        if isinstance(mean_tensor, torch.Tensor)
        else np.array(mean_tensor)
    )
    std_np = (
        std_tensor.numpy()
        if isinstance(std_tensor, torch.Tensor)
        else np.array(std_tensor)
    )
    max_abs_np = (
        max_abs_tensor.numpy()
        if isinstance(max_abs_tensor, torch.Tensor)
        else np.array(max_abs_tensor)
    )

    return pd.DataFrame(
        {
            "channel_idx": np.arange(len(mean_np)),
            "mean": mean_np,
            "std": std_np,
            "max_abs": max_abs_np,
        }
    )


def find_outlier_channels(
    stats: FeatureStats,
    layer_idx: int,
    rep_name: str,
    role: str,
    metric: str = "max_abs",
    top_k: int = 10,
    threshold_std: float = 3.0,
) -> pd.DataFrame:
    """
    Find outlier channels based on a specified metric.

    Args:
        stats: FeatureStats object
        layer_idx: Layer index
        rep_name: Representation name (pre, per_layer_ln, global_ln)
        role: Token role (cls, patch)
        metric: Metric to use for outlier detection (mean, std, max_abs)
        top_k: Number of top outliers to return
        threshold_std: Standard deviation threshold for outlier detection

    Returns:
        DataFrame with outlier channels and their statistics
    """
    df = get_channel_df(stats, layer_idx, rep_name, role)
    if df.empty:
        return df

    values = df[metric].values
    mean_val = np.mean(values)
    std_val = np.std(values)

    # Z-score
    df["z_score"] = (values - mean_val) / (std_val + 1e-8)
    df["is_outlier"] = np.abs(df["z_score"]) > threshold_std

    # Sort by absolute z-score and return top-k
    df_sorted = df.reindex(df["z_score"].abs().sort_values(ascending=False).index)
    return df_sorted.head(top_k)


def get_cross_layer_comparison(
    stats: FeatureStats, rep_name: str, role: str, metric: str = "mean_max_abs"
) -> pd.DataFrame:
    """
    Get cross-layer comparison for a specific representation and role.

    Returns DataFrame with layer_idx and the specified metric aggregated across channels.
    """
    summary_df = get_layer_summary_df(stats)
    filtered = summary_df[
        (summary_df["rep_name"] == rep_name) & (summary_df["role"] == role)
    ].copy()
    return filtered[["layer_idx", metric]].reset_index(drop=True)


def discover_stats_files(search_dir: str | Path) -> List[Path]:
    """
    Discover all dinov3_feature_stats.pt files in a directory tree.
    """
    search_dir = Path(search_dir)
    if not search_dir.exists():
        return []

    files = list(search_dir.rglob("*feature_stats*.pt"))
    return sorted(files)

