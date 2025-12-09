#!/usr/bin/env python
"""
Analyze DINOv3 backbone intermediate features on the training dataset.

This script:
  1) Reads configs/train.yaml to build the training datamodule (same as train.py).
  2) Builds a DINOv3 backbone + optional LoRA adapters in the same way as infer_and_submit_pt.py.
  3) Runs a forward pass over the (main) training dataloader.
  4) For each transformer block, collects channel-wise statistics of:
       - CLS token and patch tokens before the global output LayerNorm ("pre")
       - The same tokens after a per-layer LayerNorm (fresh nn.LayerNorm, affine=False) ("per_layer_ln")
       - The same tokens after the shared global LayerNorm(s) used by DINOv3 ("global_ln")
  5) Saves all aggregated statistics to a single torch.save(.pt) file for later visualization.

NOTE: To avoid enormous disk usage, this script stores ONLY aggregated statistics,
      not full tokens for every image.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from loguru import logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute DINOv3 per-layer feature statistics over the training set."
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project root containing configs/ and src/ (default: current directory).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to train.yaml; default: <project-dir>/configs/train.yaml.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to the output .pt file containing aggregated statistics. "
            "If omitted, a default path under outputs/<version>/[train_all/]feature_stats/ is used."
        ),
    )
    parser.add_argument(
        "--dino-weights-pt",
        type=str,
        default=None,
        help=(
            "Path to frozen DINOv3 backbone weights .pt file. "
            "If omitted, falls back to infer_and_submit_pt.DINO_WEIGHTS_PT_PATH."
        ),
    )
    parser.add_argument(
        "--head-weights",
        type=str,
        default=None,
        help=(
            "Path to regression head weights file or directory (for LoRA payload). "
            "If omitted, falls back to infer_and_submit_pt.HEAD_WEIGHTS_PT_PATH."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional limit on number of training images to process (0 = all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run backbone on: 'auto', 'cuda', or 'cpu' (default: auto).",
    )
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    name = str(name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _add_repo_paths(project_dir: Path) -> None:
    """
    Ensure that project src/, third_party/dinov3, and third_party/peft are on sys.path.
    Mirrors infer_and_submit_pt.py behavior where relevant.
    """
    proj_abs = project_dir.resolve()
    if str(proj_abs) not in sys.path:
        sys.path.insert(0, str(proj_abs))

    src_dir = proj_abs / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Local dinov3 source (for offline use)
    dinov3_dir = proj_abs / "third_party" / "dinov3" / "dinov3"
    if dinov3_dir.is_dir() and str(dinov3_dir) not in sys.path:
        sys.path.insert(0, str(dinov3_dir))


def _load_config(project_dir: Path, config_path: Optional[str]) -> Dict:
    """
    Reuse infer_and_submit_pt.load_config when available; otherwise load YAML directly.
    """
    import yaml

    cfg_path = (
        Path(config_path)
        if config_path is not None
        else project_dir / "configs" / "train.yaml"
    )
    cfg_path = cfg_path.expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    logger.info("Loading training config from {}", cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_train_dataloader(cfg: Dict, log_dir: Path):
    """
    Build the main training dataloader using the same PastureDataModule config
    as train_single_split in src/training/single_run.py.
    """
    from src.training.single_run import _build_datamodule, resolve_dataset_area_m2

    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building PastureDataModule and training dataloader...")
    area_m2 = resolve_dataset_area_m2(cfg)
    dm = _build_datamodule(cfg, log_dir, area_m2=area_m2, train_df=None, val_df=None)
    # Ensure z-score stats etc. are computed (mirrors training behavior).
    try:
        dm.setup()
    except Exception:
        # For analysis, continue even if z-score computation fails.
        pass

    train_loader = dm.train_dataloader()
    # When NDVI-dense is enabled, dm.train_dataloader() returns [main_loader, ndvi_loader].
    if isinstance(train_loader, (list, tuple)) and len(train_loader) > 0:
        main_loader = train_loader[0]
    else:
        main_loader = train_loader

    # Log basic dataloader stats
    try:
        num_samples = len(getattr(main_loader, "dataset", []))  # type: ignore[arg-type]
    except Exception:
        num_samples = None
    try:
        num_batches = len(main_loader)
    except Exception:
        num_batches = None
    batch_size = getattr(main_loader, "batch_size", None)
    logger.info(
        "Train dataloader ready: samples={}, batch_size={}, batches={}",
        num_samples,
        batch_size,
        num_batches,
    )
    return main_loader


def _build_backbone_with_lora(
    cfg: Dict,
    dino_weights_pt: Optional[str],
    head_weights_base: Optional[str],
    project_dir: Path,
) -> Tuple[nn.Module, Dict]:
    """
    Build a DINOv3 backbone and (optionally) inject LoRA adapters using the same
    logic as infer_and_submit_pt.py (per-head bundled PEFT payload).

    Only the first discovered head is used to recover LoRA parameters.
    """
    import os

    logger.info("Building DINOv3 backbone (with optional LoRA) using infer_and_submit_pt logic...")
    # Import helper utilities from the existing inference script.
    import infer_and_submit_pt as infer_mod

    # Resolve default weights paths from infer_and_submit_pt when not explicitly provided.
    if dino_weights_pt is None or len(str(dino_weights_pt).strip()) == 0:
        dino_weights_pt = infer_mod.DINO_WEIGHTS_PT_PATH
    if head_weights_base is None or len(str(head_weights_base).strip()) == 0:
        head_weights_base = infer_mod.HEAD_WEIGHTS_PT_PATH

    dino_weights_pt = os.path.abspath(dino_weights_pt)
    if not os.path.isfile(dino_weights_pt):
        raise FileNotFoundError(f"DINOv3 weights .pt not found: {dino_weights_pt}")

    head_weights_base = os.path.abspath(head_weights_base)
    logger.info("Using DINOv3 weights: {}", dino_weights_pt)
    logger.info("Using head weights (for LoRA/meta): {}", head_weights_base)

    # Determine backbone constructor from dinov3.hub.backbones.
    backbone_name = str(cfg["model"]["backbone"]).strip()
    logger.info("Configured DINOv3 backbone: {}", backbone_name)
    try:
        if backbone_name == "dinov3_vith16plus":
            from dinov3.hub.backbones import dinov3_vith16plus as _make_backbone  # type: ignore
        elif backbone_name == "dinov3_vitl16":
            from dinov3.hub.backbones import dinov3_vitl16 as _make_backbone  # type: ignore
        else:
            raise ImportError(f"Unsupported backbone in config: {backbone_name}")
    except Exception as exc:
        raise ImportError(
            "dinov3 is not available locally or backbones import failed. "
            "Ensure third_party/dinov3 is available and on PYTHONPATH."
        ) from exc

    # Discover head weights (may be a single file or a directory, possibly with ensemble manifest).
    ensemble_enabled = infer_mod._read_ensemble_enabled_flag(str(project_dir))
    if ensemble_enabled:
        pkg_manifest = infer_mod._read_packaged_ensemble_manifest_if_exists(
            head_weights_base
        )
        if pkg_manifest is not None:
            head_entries, _agg = pkg_manifest
            head_weight_paths = [p for (p, _w) in head_entries]
        else:
            head_weight_paths = infer_mod.discover_head_weight_paths(head_weights_base)
    else:
        head_weight_paths = infer_mod.discover_head_weight_paths(head_weights_base)
    if not head_weight_paths:
        raise FileNotFoundError(f"No head weights found under: {head_weights_base}")
    logger.info("Discovered {} head weight file(s); first: {}", len(head_weight_paths), head_weight_paths[0])

    # Load frozen DINOv3 backbone weights (same as infer_and_submit_pt).
    dino_state = torch.load(dino_weights_pt, map_location="cpu")
    if isinstance(dino_state, dict) and "state_dict" in dino_state:
        dino_state = dino_state["state_dict"]

    backbone = _make_backbone(pretrained=False)
    try:
        backbone.load_state_dict(dino_state, strict=True)
    except Exception:
        backbone.load_state_dict(dino_state, strict=False)

    backbone.eval()

    # Inspect first head file to see if a PEFT (LoRA) payload is bundled.
    first_head_path = head_weight_paths[0]
    _state, _meta, peft_payload = infer_mod.load_head_state(first_head_path)

    used_lora = False
    lora_error: Optional[str] = None

    if peft_payload is not None and isinstance(peft_payload, dict):
        peft_cfg_dict = peft_payload.get("config", None)
        peft_state = peft_payload.get("state_dict", None)
        if peft_cfg_dict and peft_state:
            try:
                try:
                    from peft.config import PeftConfig  # type: ignore
                    from peft.mapping_func import get_peft_model  # type: ignore
                    from peft.utils.save_and_load import (  # type: ignore
                        set_peft_model_state_dict,
                    )
                except Exception:
                    # Fallback to vendored PEFT via src.models.peft_integration._import_peft
                    from src.models.peft_integration import _import_peft

                    (
                        _LoraConfig,  # noqa: F841
                        get_peft_model_alt,
                        _init_eva,  # noqa: F841
                        _get_peft_state,  # noqa: F841
                    ) = _import_peft()
                    from peft.config import PeftConfig  # type: ignore
                    from peft.utils.save_and_load import (  # type: ignore
                        set_peft_model_state_dict,
                    )

                    get_peft_model = get_peft_model_alt  # type: ignore

                peft_config = PeftConfig.from_peft_type(**peft_cfg_dict)
                backbone = get_peft_model(backbone, peft_config)
                set_peft_model_state_dict(
                    backbone, peft_state, adapter_name="default"
                )
                backbone.eval()
                used_lora = True
            except Exception as exc:
                lora_error = str(exc)

    # Freeze parameters (analysis is inference-only).
    for p in backbone.parameters():
        p.requires_grad = False

    if used_lora:
        logger.info("LoRA adapters successfully injected into backbone.")
    elif lora_error is not None:
        logger.warning("LoRA injection failed or skipped: {}", lora_error)
    else:
        logger.info("No LoRA payload found in head weights; using pure frozen DINOv3 backbone.")

    meta = {
        "backbone_name": backbone_name,
        "dino_weights_pt": dino_weights_pt,
        "head_weights_base": head_weights_base,
        "first_head_path": first_head_path,
        "used_lora": used_lora,
        "lora_error": lora_error,
    }
    return backbone, meta


def _unwrap_dino(backbone: nn.Module) -> nn.Module:
    """
    Return the underlying DinoVisionTransformer, unwrapping PEFT wrappers if necessary.
    """
    # Many PEFT models expose base_model; otherwise assume backbone itself is DINO.
    if hasattr(backbone, "base_model"):
        return getattr(backbone, "base_model")
    return backbone


def _get_block_count(dino: nn.Module) -> int:
    if hasattr(dino, "n_blocks"):
        return int(getattr(dino, "n_blocks"))
    if hasattr(dino, "blocks") and isinstance(dino.blocks, nn.ModuleList):
        return len(dino.blocks)
    raise RuntimeError("Cannot infer DINO backbone depth (number of blocks).")


def _get_n_storage_tokens(dino: nn.Module) -> int:
    return int(getattr(dino, "n_storage_tokens", 0))


def _init_channel_stats(dim: int) -> Dict[str, torch.Tensor]:
    return {
        "count": torch.zeros(1, dtype=torch.long),
        "sum": torch.zeros(dim, dtype=torch.float64),
        "sum_sq": torch.zeros(dim, dtype=torch.float64),
        "max_abs": torch.zeros(dim, dtype=torch.float32),
    }


def _update_channel_stats(stats: Dict[str, torch.Tensor], x: torch.Tensor) -> None:
    """
    Update channel-wise statistics given a tensor x of shape (..., C).
    """
    if x.numel() == 0:
        return
    c = x.shape[-1]
    x_flat = x.reshape(-1, c)
    with torch.no_grad():
        stats["count"] += x_flat.shape[0]
        stats["sum"] += x_flat.sum(dim=0).double().cpu()
        stats["sum_sq"] += (x_flat.double() ** 2).sum(dim=0).cpu()
        max_abs_batch, _ = x_flat.abs().max(dim=0)
        stats["max_abs"] = torch.max(stats["max_abs"], max_abs_batch.cpu().to(stats["max_abs"].dtype))


def _finalize_stats(stats: Dict[int, Dict[str, Dict[str, Dict[str, torch.Tensor]]]]) -> Dict:
    """
    Convert running sums into mean/std and move everything to CPU/float32 where appropriate.
    """
    out: Dict[int, Dict[str, Dict[str, Dict[str, torch.Tensor]]]] = {}
    eps = 1e-8
    for layer_idx, rep_dict in stats.items():
        out[layer_idx] = {}
        for rep_name, role_dict in rep_dict.items():
            out[layer_idx][rep_name] = {}
            for role, ch_stats in role_dict.items():
                count = ch_stats["count"].item()
                if count <= 0:
                    out[layer_idx][rep_name][role] = {
                        "count": int(count),
                        "mean": torch.zeros_like(ch_stats["sum"], dtype=torch.float32),
                        "std": torch.ones_like(ch_stats["sum"], dtype=torch.float32),
                        "max_abs": ch_stats["max_abs"].clone().cpu().to(torch.float32),
                    }
                    continue
                s = ch_stats["sum"]
                ss = ch_stats["sum_sq"]
                mean = (s / float(count)).to(torch.float64)
                var = (ss / float(count)) - mean**2
                var = torch.clamp(var, min=0.0)
                std = torch.sqrt(var + eps)
                out[layer_idx][rep_name][role] = {
                    "count": int(count),
                    "mean": mean.to(torch.float32).cpu(),
                    "std": std.to(torch.float32).cpu(),
                    "max_abs": ch_stats["max_abs"].clone().cpu().to(torch.float32),
                }
    return out


def _analyze_features_over_train(
    backbone: nn.Module,
    train_loader,
    max_images: int,
    device: torch.device,
) -> Tuple[Dict[int, Dict[str, Dict[str, Dict[str, torch.Tensor]]]], Dict[str, int]]:
    """
    Iterate over the (main) training loader and accumulate per-layer statistics.
    """
    dino = _unwrap_dino(backbone)
    # Basic structural info
    num_blocks = _get_block_count(dino)
    n_storage = _get_n_storage_tokens(dino)
    embed_dim = int(getattr(dino, "embed_dim", getattr(dino, "num_features", 0)))
    if embed_dim <= 0:
        raise RuntimeError("Cannot resolve embedding dimension from DINO backbone.")

    logger.info(
        "Backbone structure: num_layers={}, embedding_dim={}, n_storage_tokens={}",
        num_blocks,
        embed_dim,
        n_storage,
    )

    # Prepare per-layer LayerNorms (affine=False) to simulate independent output norms.
    per_layer_lns: List[nn.LayerNorm] = []
    for _ in range(num_blocks):
        ln = nn.LayerNorm(
            embed_dim,
            eps=getattr(dino.norm, "eps", 1e-6),
            elementwise_affine=False,
        )
        per_layer_lns.append(ln.to(device))

    # Allocate statistics: stats[layer_idx][rep_name][role] -> channel stats
    # rep_name in { "pre", "per_layer_ln", "global_ln" }
    # role in { "cls", "patch" }
    stats: Dict[int, Dict[str, Dict[str, Dict[str, torch.Tensor]]]] = {}
    for layer_idx in range(num_blocks):
        stats[layer_idx] = {
            "pre": {
                "cls": _init_channel_stats(embed_dim),
                "patch": _init_channel_stats(embed_dim),
            },
            "per_layer_ln": {
                "cls": _init_channel_stats(embed_dim),
                "patch": _init_channel_stats(embed_dim),
            },
            "global_ln": {
                "cls": _init_channel_stats(embed_dim),
                "patch": _init_channel_stats(embed_dim),
            },
        }

    num_images_seen = 0
    num_batches_seen = 0

    backbone.to(device)
    backbone.eval()

    logger.info("Starting feature analysis over training set on device {}", device)

    # Alias global norms for convenience.
    global_norm = getattr(dino, "norm", None)
    cls_norm = getattr(dino, "cls_norm", None)
    untie_cls_and_patch = bool(getattr(dino, "untie_cls_and_patch_norms", False))

    with torch.no_grad():
        for batch in train_loader:
            # Extract image tensor from batch (dataset returns dicts).
            if isinstance(batch, dict):
                images = batch.get("image")
            elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                # Fallback: assume first element is images
                images = batch[0]
            else:
                images = batch

            if images is None:
                logger.warning("Received batch without 'image' key; skipping batch.")
                continue

            images = images.to(
                device=device,
                dtype=next(backbone.parameters()).dtype,
                non_blocking=True,
            )

            bsz = images.shape[0]
            num_images_seen += bsz
            num_batches_seen += 1
            if max_images > 0 and num_images_seen > max_images:
                # Truncate last batch if we overshoot the limit.
                excess = num_images_seen - max_images
                if excess > 0 and excess < bsz:
                    images = images[: bsz - excess]
                    bsz = images.shape[0]
                    num_images_seen = max_images
                else:
                    break

            # Pre-norm outputs: one tensor per selected block.
            # We request all blocks [0, ..., num_blocks-1].
            layer_indices: Sequence[int] = list(range(num_blocks))
            outs_pre: List[torch.Tensor] = dino._get_intermediate_layers_not_chunked(  # type: ignore[attr-defined]
                images,
                n=layer_indices,
            )
            if len(outs_pre) != len(layer_indices):
                raise RuntimeError(
                    f"Expected {len(layer_indices)} intermediate outputs, got {len(outs_pre)}"
                )

            for idx_in_list, out in enumerate(outs_pre):
                layer_idx = layer_indices[idx_in_list]
                # out: (B, T, C) with tokens [CLS, storage_tokens, patch_tokens]
                if out.dim() != 3 or out.shape[0] != bsz or out.shape[2] != embed_dim:
                    raise RuntimeError(
                        f"Unexpected intermediate shape at layer {layer_idx}: {tuple(out.shape)}"
                    )

                # Split into CLS and patch tokens.
                cls_tokens_pre = out[:, 0, :]  # (B, C)
                patch_tokens_pre = out[:, 1 + n_storage :, :]  # (B, N, C)

                # --- 1) Pre-norm stats ---
                _update_channel_stats(stats[layer_idx]["pre"]["cls"], cls_tokens_pre)
                _update_channel_stats(
                    stats[layer_idx]["pre"]["patch"],
                    patch_tokens_pre,
                )

                # --- 2) Per-layer LayerNorm (fresh, affine=False) ---
                ln = per_layer_lns[layer_idx]
                out_layer_ln = ln(out)  # (B, T, C)
                cls_tokens_layer_ln = out_layer_ln[:, 0, :]
                patch_tokens_layer_ln = out_layer_ln[:, 1 + n_storage :, :]
                _update_channel_stats(
                    stats[layer_idx]["per_layer_ln"]["cls"],
                    cls_tokens_layer_ln,
                )
                _update_channel_stats(
                    stats[layer_idx]["per_layer_ln"]["patch"],
                    patch_tokens_layer_ln,
                )

                # --- 3) Global LayerNorm used by DINOv3 (shared across layers) ---
                if global_norm is None:
                    # If, for some reason, there is no global norm, skip this part.
                    continue

                if untie_cls_and_patch and cls_norm is not None:
                    x_norm_cls_reg = cls_norm(out[:, : 1 + n_storage])
                    x_norm_patch = global_norm(out[:, 1 + n_storage :])
                    out_global_ln = torch.cat((x_norm_cls_reg, x_norm_patch), dim=1)
                else:
                    out_global_ln = global_norm(out)

                cls_tokens_global = out_global_ln[:, 0, :]
                patch_tokens_global = out_global_ln[:, 1 + n_storage :, :]
                _update_channel_stats(
                    stats[layer_idx]["global_ln"]["cls"],
                    cls_tokens_global,
                )
                _update_channel_stats(
                    stats[layer_idx]["global_ln"]["patch"],
                    patch_tokens_global,
                )

            if max_images > 0 and num_images_seen >= max_images:
                break

            # Periodic progress logging
            if num_batches_seen % 50 == 0:
                logger.info(
                    "Progress: processed {} images in {} batches so far...",
                    num_images_seen,
                    num_batches_seen,
                )

    meta_counts = {
        "num_images": int(num_images_seen),
        "num_batches": int(num_batches_seen),
        "num_layers": int(num_blocks),
        "embedding_dim": int(embed_dim),
        "n_storage_tokens": int(n_storage),
    }
    return stats, meta_counts


def _setup_logging(project_dir: Path, output_path: Path) -> None:
    """
    Configure loguru to log both to stderr and to a file under the feature_stats folder.
    """
    log_dir = output_path.parent if output_path.parent != Path("") else project_dir / "outputs" / "feature_stats"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "analyze_dinov3_features.log"

    # Reset default handlers to avoid duplicate logs when re-running in notebooks.
    logger.remove()
    # Console (INFO+)
    logger.add(sys.stderr, level="INFO")
    # File (DEBUG+)
    logger.add(
        str(log_file),
        level="DEBUG",
        encoding="utf-8",
        rotation="10 MB",
        retention=5,
        backtrace=False,
        diagnose=False,
    )
    logger.info("Logging initialized. Log file: {}", log_file)


def main() -> None:
    args = _parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.is_dir():
        raise RuntimeError(f"project-dir does not exist or is not a directory: {project_dir}")
    _add_repo_paths(project_dir)

    # Load config first so we can derive a sensible default output path.
    cfg = _load_config(project_dir, args.config)

    # Resolve versioned log directory (mirrors train.py behavior).
    version = cfg.get("version", None)
    if version in (None, "", "null"):
        version = None
    logging_cfg = cfg.get("logging", {})
    base_log_dir = Path(logging_cfg.get("log_dir", "outputs")).expanduser()
    log_dir = base_log_dir / version if version else base_log_dir

    # Prefer train_all subfolder when enabled.
    train_all_cfg = cfg.get("train_all", {})
    train_all_enabled = bool(train_all_cfg.get("enabled", False))
    analysis_base_dir = log_dir / "train_all" if train_all_enabled else log_dir

    # Resolve output path: CLI path has highest priority; otherwise use
    #   outputs/<version>/[train_all/]feature_stats/dinov3_feature_stats.pt
    if args.output is not None and str(args.output).strip():
        out_path = Path(args.output).expanduser()
    else:
        out_path = analysis_base_dir / "feature_stats" / "dinov3_feature_stats.pt"

    _setup_logging(project_dir, out_path)

    logger.info("Project directory: {}", project_dir)
    logger.info("Version: {}", version)
    logger.info("Train-all enabled: {}", train_all_enabled)
    logger.info("Base log dir (from config): {}", base_log_dir)
    logger.info("Analysis base dir: {}", analysis_base_dir)
    logger.info("Output will be saved to: {}", out_path)

    device = _resolve_device(args.device)
    logger.info("Using device: {}", device)

    # Build training dataloader (main regression stream).
    train_loader = _build_train_dataloader(cfg, analysis_base_dir)

    # Build backbone + optional LoRA (reusing infer_and_submit_pt logic).
    backbone, backbone_meta = _build_backbone_with_lora(
        cfg,
        dino_weights_pt=args.dino_weights_pt,
        head_weights_base=args.head_weights,
        project_dir=project_dir,
    )

    # Run analysis over train set.
    raw_stats, meta_counts = _analyze_features_over_train(
        backbone=backbone,
        train_loader=train_loader,
        max_images=int(args.max_images),
        device=device,
    )
    final_stats = _finalize_stats(raw_stats)

    # Prepare output payload.
    payload: Dict = {
        "meta": {
            "project_dir": str(project_dir),
            "config": cfg,
            "backbone": backbone_meta,
            "counts": meta_counts,
        },
        "layers": final_stats,
    }

    out_dir = out_path.parent
    if out_dir and not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(payload, str(out_path))
    logger.info(
        "Feature statistics saved. images={}, batches={}, num_layers={}, embedding_dim={}",
        meta_counts.get("num_images"),
        meta_counts.get("num_batches"),
        meta_counts.get("num_layers"),
        meta_counts.get("embedding_dim"),
    )
    print(f"[analyze_dinov3_features] Saved feature statistics to: {out_path}")


if __name__ == "__main__":
    main()


