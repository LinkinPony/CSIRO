from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
from loguru import logger

from src.training.logging_utils import init_logging


def resolve_repo_root() -> Path:
    """
    Resolve the project root directory that contains both `configs/` and `src/`.

    This is used to make training robust when invoked from a different working
    directory (for example, Ray Tune trials chdir into per-trial folders).
    """
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "configs").is_dir() and (p / "src").is_dir():
            return p
    # Fallback: repo root = parent of src/
    try:
        src_dir = here.parents[2]  # .../src/training/entrypoint.py -> .../src
        if src_dir.name == "src":
            return src_dir.parent
    except Exception:
        pass
    return Path.cwd().resolve()


def _resolve_dir_under_root(path: Path, *, repo_root: Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def snapshot_training_config(
    cfg: dict,
    *,
    log_dir: Path,
    source_config_path: Optional[str] = None,
) -> None:
    """
    Write a reproducible snapshot of the training config under log_dir/train.yaml.

    - If source_config_path is provided and exists, we copy it verbatim (keeps comments).
    - Otherwise, we dump the resolved cfg dict as YAML.
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = log_dir / "train.yaml"

        if source_config_path:
            src = Path(str(source_config_path)).expanduser()
            if src.is_file():
                import shutil as _shutil

                _shutil.copyfile(str(src), str(snapshot_path))
                return

        import yaml

        with open(snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        logger.warning(f"Failed to snapshot training config to {log_dir}/train.yaml: {e}")


def _maybe_copy_dinov3_weights(cfg: dict, *, repo_root: Path) -> None:
    """
    Best-effort: ensure DINOv3 weights are present under <repo_root>/dinov3_weights/.
    This helps keep a single frozen-backbone weights store across runs.
    """
    try:
        import shutil

        dinov3_src = (cfg.get("model", {}) or {}).get("weights_path", None)
        if not dinov3_src:
            return
        dinov3_src = str(dinov3_src).strip()
        if not dinov3_src:
            return
        if not os.path.isfile(dinov3_src):
            # If cfg uses a relative path, try resolving from repo_root.
            cand = (repo_root / dinov3_src).resolve()
            if cand.is_file():
                dinov3_src = str(cand)
            else:
                return

        dinov3_dir = repo_root / "dinov3_weights"
        dinov3_dir.mkdir(parents=True, exist_ok=True)

        src_name = Path(dinov3_src).name
        if src_name.endswith(".pth"):
            src_name = src_name[:-4] + ".pt"
        dinov3_dst = dinov3_dir / src_name
        if not dinov3_dst.is_file() or os.path.getsize(dinov3_dst) == 0:
            shutil.copyfile(str(dinov3_src), str(dinov3_dst))
    except Exception as e:
        logger.warning(f"Copying DINOv3 weights failed: {e}")


def run_training(
    cfg: dict,
    *,
    log_dir: Path,
    ckpt_dir: Path,
    repo_root: Optional[Path] = None,
    source_config_path: Optional[str] = None,
    extra_callbacks: Optional[list] = None,
    enable_post_kfold_swa_eval: bool = True,
) -> None:
    """
    Unified training entrypoint used by:
      - legacy `train.py --config ...` (PyYAML)
      - Hydra-based launcher
      - Ray Tune trials

    IMPORTANT: This function temporarily `chdir`s into repo_root so relative paths
    in the config (for example `data.root: data`) keep working under Ray Tune.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"run_training expects cfg as dict, got: {type(cfg)}")

    # ---- Global torch perf knobs (safe defaults, no model/hparam changes) ----
    # These typically improve GPU throughput/utilization for mixed-precision training.
    try:
        import torch

        # Enable TF32 on matmul/conv where applicable (AMP is still used for most ops).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Slightly more permissive reductions can improve throughput on some GPUs.
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        except Exception:
            pass
        # Prefer Tensor Cores for fp32 matmul when it happens (Lightning warns about this).
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        # cuDNN autotune for fixed input sizes (your pipeline uses a fixed image_size).
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    repo_root = (repo_root or resolve_repo_root()).resolve()
    log_dir = _resolve_dir_under_root(Path(log_dir), repo_root=repo_root)
    ckpt_dir = _resolve_dir_under_root(Path(ckpt_dir), repo_root=repo_root)

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    snapshot_training_config(cfg, log_dir=log_dir, source_config_path=source_config_path)

    init_logging(log_dir, use_loguru=bool((cfg.get("logging", {}) or {}).get("use_loguru", True)))
    logger.info("Repo root: {}", repo_root)
    if source_config_path:
        logger.info("Loaded config from {}", source_config_path)

    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)
    _maybe_copy_dinov3_weights(cfg, repo_root=repo_root)

    # Optional k-fold / train-all configuration
    kfold_cfg = cfg.get("kfold", {}) or {}
    kfold_enabled = bool(kfold_cfg.get("enabled", False))

    train_all_cfg = cfg.get("train_all", {}) or {}
    train_all_enabled = bool(train_all_cfg.get("enabled", False))

    # Ensure we run from repo_root so relative data paths remain valid.
    cwd = Path.cwd()
    try:
        os.chdir(str(repo_root))

        # Local imports after chdir (keeps relative resource discovery consistent)
        from src.training.kfold_runner import run_kfold
        from src.training.single_run import (
            parse_image_size,
            resolve_dataset_area_m2,
            train_single_split,
        )

        # 1) Run k-fold training if enabled (uses per-fold subdirectories under log_dir/ckpt_dir)
        if kfold_enabled:
            run_kfold(cfg, log_dir, ckpt_dir, extra_callbacks=extra_callbacks)

        # 2) Train-all or plain single-split training
        if train_all_enabled:
            # Build full dataframe once when train_all is enabled to construct splits.
            try:
                from src.data.datamodule import PastureDataModule

                area_m2 = resolve_dataset_area_m2(cfg)
                base_dm = PastureDataModule(
                    data_root=cfg["data"]["root"],
                    train_csv=cfg["data"]["train_csv"],
                    image_size=parse_image_size(cfg["data"]["image_size"]),
                    batch_size=int(cfg["data"]["batch_size"]),
                    val_batch_size=int(cfg["data"].get("val_batch_size", cfg["data"]["batch_size"])),
                    num_workers=int(cfg["data"]["num_workers"]),
                    prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)),
                    val_split=float(cfg["data"]["val_split"]),
                    target_order=list(cfg["data"]["target_order"]),
                    mean=list(cfg["data"]["normalization"]["mean"]),
                    std=list(cfg["data"]["normalization"]["std"]),
                    train_scale=tuple(cfg["data"]["augment"]["random_resized_crop_scale"]),
                    sample_area_m2=float(area_m2),
                    zscore_output_path=str((log_dir / "train_all") / "z_score.json"),
                    log_scale_targets=bool(cfg["model"].get("log_scale_targets", False)),
                    hflip_prob=float(cfg["data"]["augment"]["horizontal_flip_prob"]),
                    shuffle=bool(cfg["data"].get("shuffle", True)),
                )
                full_df = base_dm.build_full_dataframe()
            except Exception as e:
                logger.warning(f"Constructing train_all splits failed: {e}")
                raise

            import pandas as pd  # local import to avoid global dependency if unused

            rng_seed = int(cfg.get("seed", 42))
            if len(full_df) < 1:
                raise ValueError("No samples available to construct train_all splits.")
            dummy_val = full_df.sample(n=1, random_state=rng_seed)
            val_df = pd.concat([dummy_val], ignore_index=True)
            train_df = full_df.reset_index(drop=True)

            train_all_log_dir = log_dir / "train_all"
            train_all_ckpt_dir = ckpt_dir / "train_all"
            train_all_log_dir.mkdir(parents=True, exist_ok=True)
            train_all_ckpt_dir.mkdir(parents=True, exist_ok=True)

            train_single_split(
                cfg,
                train_all_log_dir,
                train_all_ckpt_dir,
                train_df=train_df,
                val_df=val_df,
                train_all_mode=True,
                extra_callbacks=extra_callbacks,
            )
        elif not kfold_enabled:
            # Regular single-split training without k-fold or train_all.
            train_single_split(
                cfg,
                log_dir,
                ckpt_dir,
                train_df=None,
                val_df=None,
                train_all_mode=False,
                extra_callbacks=extra_callbacks,
            )

        # 3) Optional SWA-based k-fold validation evaluation after all training has completed.
        if kfold_enabled and enable_post_kfold_swa_eval:
            try:
                from swa_eval_kfold import run_swa_eval_for_kfold_cfg

                logger.info(
                    "Running post-training SWA k-fold evaluation to compute log-space "
                    "metrics with global baseline (kfold_swa_metrics.json)."
                )
                run_swa_eval_for_kfold_cfg(
                    cfg,
                    device="auto",
                    output_name="kfold_swa_metrics.json",
                )
            except Exception as e:
                logger.warning(f"SWA k-fold evaluation failed (non-fatal): {e}")
    finally:
        try:
            os.chdir(str(cwd))
        except Exception:
            pass


