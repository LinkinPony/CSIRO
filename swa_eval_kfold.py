import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from package_artifacts import (
    load_cfg as load_train_cfg,
    resolve_dirs as resolve_train_dirs,
    _load_checkpoint,
    _find_swa_average_model_state,
)
from sanity_check import compute_metrics
from src.data.datamodule import PastureDataModule
from src.models.regressor import BiomassRegressor
from src.training.single_run import (
    parse_image_size,
    resolve_dataset_area_m2,
    _build_datamodule,
)


SCORE_TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g", "Dry_Total_g"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate SWA-averaged checkpoints on k-fold validation splits and "
            "write per-fold / averaged metrics JSON under outputs/<version>/."
        )
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "train.yaml"),
        help="Path to training YAML config used for this run.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device: 'auto' (default), 'cpu', or 'cuda'.",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="kfold_swa_metrics.json",
        help="Filename for the metrics JSON written under outputs/<version>/",
    )
    return p.parse_args()


def _select_device(device_arg: str) -> torch.device:
    device_arg = str(device_arg or "auto").lower()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[SWA-EVAL] CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(v, device) for v in batch)
    return batch


def _build_full_dataframe(cfg: Dict) -> pd.DataFrame:
    area_m2 = resolve_dataset_area_m2(cfg)
    dm = PastureDataModule(
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
        log_scale_targets=bool(cfg["model"].get("log_scale_targets", False)),
        hflip_prob=float(cfg["data"]["augment"]["horizontal_flip_prob"]),
        shuffle=bool(cfg["data"].get("shuffle", True)),
    )
    return dm.build_full_dataframe()


def _extract_numeric_averages(per_fold: List[Dict[str, Any]]) -> Dict[str, float]:
    keys: set[str] = set()
    for m in per_fold:
        keys.update(m.keys())
    # Exclude metadata keys from averaging
    exclude = {"fold", "epoch", "ckpt_path", "used_swa"}
    out: Dict[str, float] = {}
    for k in keys:
        if k in exclude:
            continue
        vals: List[float] = []
        for m in per_fold:
            v = m.get(k, None)
            if v is None:
                continue
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                vals.append(float(v))
        if vals:
            out[k] = float(sum(vals) / len(vals))
    return out


def run_swa_eval_for_kfold_cfg(
    cfg: Dict,
    *,
    device: str = "auto",
    output_name: str = "kfold_swa_metrics.json",
) -> None:
    """
    Run SWA-based k-fold validation evaluation for an in-memory training config.

    This mirrors the CLI behaviour of this script but can be called directly
    from Python code (e.g., at the end of train.py).
    """
    log_dir, ckpt_dir = resolve_train_dirs(cfg)
    log_dir = log_dir.expanduser()
    ckpt_dir = ckpt_dir.expanduser()

    kfold_cfg = cfg.get("kfold", {})
    k = int(kfold_cfg.get("k", 5))

    fold_splits_root = log_dir / "folds"
    if not fold_splits_root.is_dir():
        raise FileNotFoundError(
            f"Fold splits directory not found: {fold_splits_root}. "
            "Ensure k-fold training has completed."
        )

    print(f"[SWA-EVAL] Using log_dir={log_dir}  ckpt_dir={ckpt_dir}  k={k}")

    # Build full dataframe once to provide global targets for R^2 baseline.
    full_df = _build_full_dataframe(cfg)
    if not all(t in full_df.columns for t in SCORE_TARGETS):
        missing = [t for t in SCORE_TARGETS if t not in full_df.columns]
        raise RuntimeError(f"Full dataframe is missing required targets {missing}")
    global_targets = torch.from_numpy(
        full_df[SCORE_TARGETS].to_numpy(dtype=np.float32)
    )

    dev = _select_device(device)
    print(f"[SWA-EVAL] Using device: {dev}")

    area_m2 = resolve_dataset_area_m2(cfg)

    per_fold_metrics: List[Dict[str, Any]] = []

    for fold_idx in range(k):
        fold_ckpt_dir = ckpt_dir / f"fold_{fold_idx}"
        if not fold_ckpt_dir.is_dir():
            print(
                f"[SWA-EVAL] Skipping fold {fold_idx}: "
                f"checkpoint dir not found: {fold_ckpt_dir}"
            )
            continue

        last_ckpt = fold_ckpt_dir / "last.ckpt"
        if last_ckpt.is_file():
            ckpt_for_swa: Optional[Path] = last_ckpt
        else:
            ckpt_candidates = [p for p in fold_ckpt_dir.glob("*.ckpt") if p.is_file()]
            ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else None

        if ckpt_for_swa is None:
            print(
                f"[SWA-EVAL] Skipping fold {fold_idx}: "
                f"no .ckpt files found in {fold_ckpt_dir}"
            )
            continue

        print(f"[SWA-EVAL] Fold {fold_idx}: loading checkpoint {ckpt_for_swa}")
        try:
            ckpt_obj = _load_checkpoint(ckpt_for_swa)
        except Exception as e:
            print(f"[SWA-EVAL] Fold {fold_idx}: failed to load checkpoint: {e}")
            continue

        try:
            epoch_raw = ckpt_obj.get("epoch", None)
            epoch_val: Optional[float]
            if epoch_raw is None:
                epoch_val = None
            else:
                epoch_val = float(epoch_raw)
        except Exception:
            epoch_val = None

        avg_state = _find_swa_average_model_state(ckpt_obj)
        used_swa = isinstance(avg_state, dict) and len(avg_state) > 0
        if not used_swa:
            print(
                f"[SWA-EVAL] Fold {fold_idx}: no SWA average_model_state found in "
                "checkpoint; evaluating raw checkpoint weights instead."
            )

        # Reconstruct train/val splits from saved CSVs to mirror training exactly.
        fold_split_dir = fold_splits_root / f"fold_{fold_idx}"
        train_csv = fold_split_dir / "train.csv"
        val_csv = fold_split_dir / "val.csv"
        if not train_csv.is_file() or not val_csv.is_file():
            print(
                f"[SWA-EVAL] Fold {fold_idx}: missing train/val split CSVs under "
                f"{fold_split_dir}; skipping this fold."
            )
            continue

        tdf = pd.read_csv(str(train_csv))
        vdf = pd.read_csv(str(val_csv))
        merged_train = tdf.merge(full_df, on=["image_id", "image_path"], how="inner")
        merged_val = vdf.merge(full_df, on=["image_id", "image_path"], how="inner")
        if len(merged_val) == 0 or len(merged_train) == 0:
            print(
                f"[SWA-EVAL] Fold {fold_idx}: empty merged train/val after join with "
                "full_df; skipping."
            )
            continue

        print(
            f"[SWA-EVAL] Fold {fold_idx}: "
            f"train rows={len(merged_train)}  val rows={len(merged_val)}"
        )

        # Build a fresh datamodule with predefined splits for this fold.
        fold_log_dir = log_dir / f"fold_{fold_idx}"
        dm = _build_datamodule(
            cfg,
            fold_log_dir,
            area_m2=area_m2,
            train_df=merged_train,
            val_df=merged_val,
        )
        # Ensure z-score stats are computed before creating loaders.
        try:
            dm.setup()
        except Exception:
            # For safety, continue with whatever stats are available.
            pass

        val_loader = dm.val_dataloader()
        # If multiple val loaders are returned (e.g., NDVI-dense), use the main one.
        if isinstance(val_loader, (list, tuple)):
            if not val_loader:
                print(
                    f"[SWA-EVAL] Fold {fold_idx}: empty val loader list; skipping."
                )
                continue
            val_loader = val_loader[0]

        # Instantiate model from checkpoint hyperparameters, then load SWA-averaged
        # weights when available.
        try:
            model = BiomassRegressor.load_from_checkpoint(
                str(ckpt_for_swa),
                map_location="cpu",
            )
        except Exception as e:
            print(
                "[SWA-EVAL] Fold "
                f"{fold_idx}: failed to construct BiomassRegressor from checkpoint: {e}"
            )
            continue

        if used_swa:
            try:
                missing, unexpected = model.load_state_dict(
                    avg_state,
                    strict=False,  # type: ignore[arg-type]
                )
                if missing or unexpected:
                    print(
                        f"[SWA-EVAL] Fold {fold_idx}: loaded SWA state with "
                        f"missing={len(missing)}, unexpected={len(unexpected)} keys."
                    )
            except Exception as e:
                print(
                    f"[SWA-EVAL] Fold {fold_idx}: failed to load SWA average_model_state, "
                    f"using raw weights. Error: {e}"
                )
                used_swa = False

        model.to(dev)
        model.eval()

        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in val_loader:
                batch_dev = _move_to_device(batch, dev)
                out = model._shared_step(batch_dev, stage="val")  # type: ignore[attr-defined]
                preds = out.get("preds", None)
                targets = out.get("targets", None)
                if preds is None or targets is None:
                    continue
                all_preds.append(preds.detach().cpu())
                all_targets.append(targets.detach().cpu())

        if not all_preds or not all_targets:
            print(
                f"[SWA-EVAL] Fold {fold_idx}: no predictions/targets collected; "
                "skipping."
            )
            continue

        preds_tensor = torch.cat(all_preds, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)

        if preds_tensor.shape[0] != targets_tensor.shape[0]:
            print(
                f"[SWA-EVAL] Fold {fold_idx}: mismatch between preds and targets "
                f"shapes {preds_tensor.shape} vs {targets_tensor.shape}; skipping."
            )
            continue

        # Compute metrics in log space with global baseline and DESCRIPTION weights.
        metrics = compute_metrics(
            preds=preds_tensor,
            targets=targets_tensor,
            global_targets=global_targets,
            target_names=SCORE_TARGETS,
        )

        record: Dict[str, Any] = {
            "fold": float(fold_idx),
            "epoch": epoch_val,
            "ckpt_path": str(ckpt_for_swa),
            "used_swa": bool(used_swa),
        }
        record.update(metrics)
        per_fold_metrics.append(record)

    if not per_fold_metrics:
        print("[SWA-EVAL] No fold metrics were collected; nothing to write.")
        return

    avg_metrics = _extract_numeric_averages(per_fold_metrics)
    payload = {
        "num_folds": len(per_fold_metrics),
        "per_fold": per_fold_metrics,
        "average": avg_metrics,
    }

    out_path = log_dir / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[SWA-EVAL] Saved SWA k-fold metrics JSON -> {out_path}")


def main() -> None:
    args = parse_args()
    cfg = load_train_cfg(args.config)
    run_swa_eval_for_kfold_cfg(
        cfg,
        device=args.device,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()


