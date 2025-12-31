import argparse
import json
import os
import shutil
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
    resolve_repo_root,
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


def _resolve_dino_weights_best_effort(cfg: Dict) -> str:
    """
    Best-effort resolve DINO weights path for inference-style evaluation.
    """
    try:
        p = cfg.get("model", {}).get("weights_path", None)
        if isinstance(p, str) and p.strip() and os.path.isfile(p):
            return os.path.abspath(p)
    except Exception:
        pass

    repo_root = resolve_repo_root()
    # Preferred single shared backbone weights (workspace rule)
    preferred = (repo_root / "dinov3_weights" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pt").resolve()
    if preferred.is_file():
        return str(preferred)

    # Fallback: any .pt under dinov3_weights/
    cand_dir = (repo_root / "dinov3_weights").resolve()
    if cand_dir.is_dir():
        pts = [p for p in cand_dir.glob("*.pt") if p.is_file()]
        pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if pts:
            return str(pts[0].resolve())

    raise FileNotFoundError(
        "Could not resolve DINO weights. Please set cfg.model.weights_path or place weights under dinov3_weights/."
    )


def _write_pseudo_test_csv(
    *,
    out_csv_path: Path,
    rows: pd.DataFrame,
    targets: List[str],
    data_root: str,
) -> None:
    """
    Create an inference-compatible 'test.csv' for val pseudo-test evaluation.

    Columns required by src/inference/pipeline.py:
      - sample_id
      - image_path
      - target_name

    We also include image_id to make post-processing robust (extra columns are ignored by inference).
    """
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure absolute image paths to avoid dependency on dataset_root resolution inside inference.
    def _abs_img_path(rel_or_abs: str) -> str:
        p = str(rel_or_abs)
        if os.path.isabs(p):
            return p
        return os.path.abspath(os.path.join(str(data_root), p))

    recs: List[Dict[str, str]] = []
    for _, r in rows.iterrows():
        image_id = str(r.get("image_id", ""))
        image_path = _abs_img_path(str(r.get("image_path", "")))
        for t in targets:
            sid = f"{image_id}__{t}"
            recs.append(
                {
                    "sample_id": sid,
                    "image_id": image_id,
                    "image_path": image_path,
                    "target_name": str(t),
                }
            )

    df = pd.DataFrame(recs, columns=["sample_id", "image_id", "image_path", "target_name"])
    df.to_csv(str(out_csv_path), index=False)


def _read_submission_as_wide(submission_csv: Path, pseudo_test_csv: Path) -> pd.DataFrame:
    """
    Read a submission.csv (sample_id,target) and pivot into wide rows keyed by image_id.
    """
    sub = pd.read_csv(str(submission_csv))  # sample_id,target
    test_df = pd.read_csv(str(pseudo_test_csv))  # includes sample_id,image_id,target_name
    merged = sub.merge(test_df, on="sample_id", how="inner")
    pv = (
        merged.pivot_table(index="image_id", columns="target_name", values="target", aggfunc="first")
        .reset_index()
    )
    # Prefix prediction columns to avoid collisions with ground-truth target columns.
    for c in list(pv.columns):
        if c == "image_id":
            continue
        pv = pv.rename(columns={c: f"pred_{c}"})
    return pv


def _extract_numeric_averages_nested(per_fold: List[Dict[str, Any]], *, key: str) -> Dict[str, float]:
    """
    Average numeric fields inside per-fold[key] dict.
    """
    out: Dict[str, float] = {}
    vals_by_k: Dict[str, List[float]] = {}
    for m in per_fold:
        obj = m.get(key, None)
        if not isinstance(obj, dict):
            continue
        for kk, vv in obj.items():
            if isinstance(vv, (int, float)) and np.isfinite(float(vv)):
                vals_by_k.setdefault(str(kk), []).append(float(vv))
    for kk, vals in vals_by_k.items():
        if vals:
            out[kk] = float(sum(vals) / len(vals))
    return out


def run_post_train_eval_on_kfold_val_cfg(
    cfg: Dict,
    *,
    device: str = "auto",
    output_name: str = "kfold_swa_post_train_val_metrics.json",
) -> None:
    """
    K-fold 'pseudo-test' evaluation on each fold's validation split:
      - base: evaluate using SWA-exported head (packed infer_head.pt)
      - post_train: adapt LoRA+head on the *val images only* (unlabeled) using src/post_train/ttt.py,
        then re-evaluate on val labels
      - delta: post_train - base for numeric metrics

    Writes a single comparison JSON under outputs/<version>/.
    """
    log_dir, ckpt_dir = resolve_train_dirs(cfg)
    log_dir = log_dir.expanduser()
    ckpt_dir = ckpt_dir.expanduser()

    post_cfg_obj = dict(cfg.get("post_train", {}) or {})
    eval_kfold = dict(post_cfg_obj.get("eval_kfold_val", {}) or {})
    if not bool(eval_kfold.get("enabled", False)):
        print("[POST-TRAIN-EVAL] post_train.eval_kfold_val.enabled=false; skipping.")
        return

    kfold_cfg = cfg.get("kfold", {})
    k = int(kfold_cfg.get("k", 5))

    fold_splits_root = log_dir / "folds"
    if not fold_splits_root.is_dir():
        raise FileNotFoundError(
            f"Fold splits directory not found: {fold_splits_root}. Ensure k-fold training has completed."
        )

    # Global targets baseline for R^2 (log-space) per competition spec.
    full_df = _build_full_dataframe(cfg)
    if not all(t in full_df.columns for t in SCORE_TARGETS):
        missing = [t for t in SCORE_TARGETS if t not in full_df.columns]
        raise RuntimeError(f"Full dataframe is missing required targets {missing}")
    global_targets = torch.from_numpy(full_df[SCORE_TARGETS].to_numpy(dtype=np.float32))

    dino_weights_pt_path = _resolve_dino_weights_best_effort(cfg)
    data_root = str(cfg.get("data", {}).get("root", "") or "")
    if not data_root:
        raise RuntimeError("cfg.data.root is required for post-train eval on val.")

    # Use the main post_train config as the source of truth (eval_kfold_val only provides `enabled`).
    merged_post_train: Dict[str, Any] = dict(post_cfg_obj)
    merged_post_train["enabled"] = True

    from src.post_train.ttt import parse_post_train_config, post_train_single_head
    from src.inference.settings import InferenceSettings
    from src.inference.pipeline import run as run_infer
    from package_artifacts import _export_swa_head_from_checkpoint

    dev = _select_device(device)
    _ = dev  # device selection kept for parity/logging; inference pipeline selects internally.

    out_root = log_dir / "post_train_eval_kfold_val"
    out_root.mkdir(parents=True, exist_ok=True)

    per_fold: List[Dict[str, Any]] = []

    # If output JSON already exists, reuse completed folds and skip re-evaluation.
    out_path = (log_dir / output_name).resolve()
    existing_by_fold: Dict[int, Dict[str, Any]] = {}
    existing_payload: Optional[dict] = None
    if out_path.is_file():
        try:
            existing_payload = json.loads(out_path.read_text(encoding="utf-8"))
            for rec in list(existing_payload.get("per_fold", []) or []):
                try:
                    fi = int(rec.get("fold"))
                    existing_by_fold[fi] = rec
                except Exception:
                    continue
        except Exception:
            existing_payload = None

    # Optional: load existing SWA baseline metrics (for reference only)
    baseline_swa_path = (log_dir / "kfold_swa_metrics.json").resolve()
    baseline_swa_obj: Optional[dict] = None
    if baseline_swa_path.is_file():
        try:
            baseline_swa_obj = json.loads(baseline_swa_path.read_text(encoding="utf-8"))
        except Exception:
            baseline_swa_obj = None

    for fold_idx in range(k):
        if fold_idx in existing_by_fold:
            per_fold.append(existing_by_fold[fold_idx])
            print(f"[POST-TRAIN-EVAL] Fold {fold_idx}: metrics already exist in {out_path.name}; skipping eval.")
            continue

        fold_ckpt_dir = ckpt_dir / f"fold_{fold_idx}"
        if not fold_ckpt_dir.is_dir():
            print(f"[POST-TRAIN-EVAL] Skipping fold {fold_idx}: ckpt dir not found: {fold_ckpt_dir}")
            continue

        # Build fold val df (mirrors swa_eval_kfold.py behaviour)
        fold_split_dir = fold_splits_root / f"fold_{fold_idx}"
        train_csv = fold_split_dir / "train.csv"
        val_csv = fold_split_dir / "val.csv"
        if not train_csv.is_file() or not val_csv.is_file():
            print(f"[POST-TRAIN-EVAL] Skipping fold {fold_idx}: missing split CSVs under {fold_split_dir}")
            continue
        tdf = pd.read_csv(str(train_csv))
        vdf = pd.read_csv(str(val_csv))
        merged_train = tdf.merge(full_df, on=["image_id", "image_path"], how="inner")
        merged_val = vdf.merge(full_df, on=["image_id", "image_path"], how="inner")
        if len(merged_val) == 0 or len(merged_train) == 0:
            print(f"[POST-TRAIN-EVAL] Skipping fold {fold_idx}: empty merged train/val after join")
            continue

        fold_out = out_root / f"fold_{fold_idx}"
        fold_out.mkdir(parents=True, exist_ok=True)

        # Export SWA head for this fold to a dedicated directory (do not clobber other exporters).
        ckpt_for_swa = (fold_ckpt_dir / "last.ckpt").resolve()
        if not ckpt_for_swa.is_file():
            ckpt_candidates = [p for p in fold_ckpt_dir.glob("*.ckpt") if p.is_file()]
            ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else ckpt_for_swa
        if not ckpt_for_swa.is_file():
            print(f"[POST-TRAIN-EVAL] Skipping fold {fold_idx}: no ckpt found under {fold_ckpt_dir}")
            continue

        head_swa_dir = (fold_ckpt_dir / "head_swa_eval").resolve()
        try:
            _src_ckpt, head_base_pt = _export_swa_head_from_checkpoint(ckpt_for_swa, head_swa_dir)
            head_base_pt = Path(head_base_pt).resolve()
        except Exception as e:
            print(f"[POST-TRAIN-EVAL] Fold {fold_idx}: SWA head export failed: {e}")
            continue

        # Ensure z_score.json is colocated with the exported head so inference can invert reg3 z-score.
        # Training writes per-fold z_score.json under outputs/<version>/fold_i/z_score.json.
        try:
            fold_log_dir = (log_dir / f"fold_{fold_idx}").resolve()
            src_z = (fold_log_dir / "z_score.json").resolve()
            dst_z = (head_swa_dir / "z_score.json").resolve()
            if src_z.is_file() and (not dst_z.is_file()):
                shutil.copy2(str(src_z), str(dst_z))
        except Exception:
            pass

        # Build pseudo test.csv for this fold's val set
        pseudo_csv = (fold_out / "val_pseudo_test.csv").resolve()
        _write_pseudo_test_csv(out_csv_path=pseudo_csv, rows=merged_val, targets=SCORE_TARGETS, data_root=data_root)

        # ---------------------------
        # Base evaluation (no post-train)
        # ---------------------------
        base_sub_csv = (fold_out / "base_submission.csv").resolve()
        infer_bsz = int(post_cfg_obj.get("batch_size", cfg.get("data", {}).get("val_batch_size", 4)) or 4)
        settings_base = InferenceSettings(
            head_weights_pt_path=str(head_base_pt),
            dino_weights_pt_path=str(dino_weights_pt_path),
            input_path=str(pseudo_csv),
            output_submission_path=str(base_sub_csv),
            project_dir=str(resolve_repo_root()),
            infer_batch_size=infer_bsz,
            post_train_enabled=False,
            post_train_force_disable=True,
        )
        run_infer(settings_base)
        preds_base_wide = _read_submission_as_wide(base_sub_csv, pseudo_csv)

        # Align preds/targets by image_id
        merged = merged_val[["image_id"] + SCORE_TARGETS].merge(preds_base_wide, on="image_id", how="inner")
        if len(merged) == 0:
            print(f"[POST-TRAIN-EVAL] Fold {fold_idx}: no predictions joined for base eval; skipping.")
            continue
        # Targets
        targets_tensor = torch.from_numpy(merged[SCORE_TARGETS].to_numpy(dtype=np.float32))
        # Predictions (prefixed)
        pred_cols = [f"pred_{t}" for t in SCORE_TARGETS]
        for pc in pred_cols:
            if pc not in merged.columns:
                merged[pc] = 0.0
        preds_tensor = torch.from_numpy(merged[pred_cols].to_numpy(dtype=np.float32))

        base_metrics = compute_metrics(
            preds=preds_tensor,
            targets=targets_tensor,
            global_targets=global_targets,
            target_names=SCORE_TARGETS,
        )

        # ---------------------------
        # Post-train on val images (unlabeled), then re-evaluate
        # ---------------------------
        post_cfg = parse_post_train_config(merged_post_train)

        # Use absolute image paths; post_train loader treats them as paths under dataset_root.
        abs_paths: List[str] = []
        for p in merged_val["image_path"].astype(str).tolist():
            if os.path.isabs(p):
                abs_paths.append(p)
            else:
                abs_paths.append(os.path.abspath(os.path.join(data_root, p)))

        post_dir = (fold_out / "post_train" / "head").resolve()
        post_dir.mkdir(parents=True, exist_ok=True)
        out_head_pt = (post_dir / "infer_head.pt").resolve()
        adapted_head_pt = post_train_single_head(
            cfg_train_yaml=cfg,
            dino_weights_pt_file=str(dino_weights_pt_path),
            head_in_pt=str(head_base_pt),
            dataset_root="",
            image_paths=abs_paths,
            out_head_pt=str(out_head_pt),
            cfg=post_cfg,
        )

        # Also colocate z_score.json with the post-trained head package.
        try:
            fold_log_dir = (log_dir / f"fold_{fold_idx}").resolve()
            src_z = (fold_log_dir / "z_score.json").resolve()
            dst_z = (post_dir / "z_score.json").resolve()
            if src_z.is_file() and (not dst_z.is_file()):
                shutil.copy2(str(src_z), str(dst_z))
        except Exception:
            pass

        post_sub_csv = (fold_out / "post_submission.csv").resolve()
        settings_post = InferenceSettings(
            head_weights_pt_path=str(adapted_head_pt),
            dino_weights_pt_path=str(dino_weights_pt_path),
            input_path=str(pseudo_csv),
            output_submission_path=str(post_sub_csv),
            project_dir=str(resolve_repo_root()),
            infer_batch_size=infer_bsz,
            post_train_enabled=False,
            post_train_force_disable=True,
        )
        run_infer(settings_post)
        preds_post_wide = _read_submission_as_wide(post_sub_csv, pseudo_csv)

        merged2 = merged_val[["image_id"] + SCORE_TARGETS].merge(preds_post_wide, on="image_id", how="inner")
        if len(merged2) == 0:
            print(f"[POST-TRAIN-EVAL] Fold {fold_idx}: no predictions joined for post-train eval; skipping.")
            continue
        targets2 = torch.from_numpy(merged2[SCORE_TARGETS].to_numpy(dtype=np.float32))
        pred_cols2 = [f"pred_{t}" for t in SCORE_TARGETS]
        for pc in pred_cols2:
            if pc not in merged2.columns:
                merged2[pc] = 0.0
        preds2 = torch.from_numpy(merged2[pred_cols2].to_numpy(dtype=np.float32))
        post_metrics = compute_metrics(
            preds=preds2,
            targets=targets2,
            global_targets=global_targets,
            target_names=SCORE_TARGETS,
        )

        # Delta: post - base (numeric metrics only)
        delta: Dict[str, float] = {}
        for kk, vv in base_metrics.items():
            try:
                if kk in post_metrics and isinstance(vv, (int, float)) and isinstance(post_metrics[kk], (int, float)):
                    delta[kk] = float(post_metrics[kk]) - float(vv)
            except Exception:
                continue

        per_fold.append(
            {
                "fold": int(fold_idx),
                "ckpt_path": str(ckpt_for_swa),
                "head_base_pt": str(head_base_pt),
                "head_post_pt": str(adapted_head_pt),
                "base": base_metrics,
                "post_train": post_metrics,
                "delta": delta,
                "post_train_cfg": {
                    "steps": int(post_cfg.steps),
                    "batch_size": int(post_cfg.batch_size),
                    "lr_head": float(post_cfg.lr_head),
                    "lr_lora": float(post_cfg.lr_lora),
                    "weight_reg3": float(post_cfg.weight_reg3),
                    "weight_ratio": float(post_cfg.weight_ratio),
                    "ema_enabled": bool(post_cfg.ema_enabled),
                    "ema_decay": float(post_cfg.ema_decay),
                    "anchor_weight": float(post_cfg.anchor_weight),
                },
            }
        )

    if not per_fold:
        print("[POST-TRAIN-EVAL] No fold metrics collected; nothing to write.")
        return

    # Deduplicate by fold index (if existing payload had duplicates).
    per_fold_by_fold: Dict[int, Dict[str, Any]] = {}
    for rec in per_fold:
        try:
            per_fold_by_fold[int(rec.get("fold"))] = rec
        except Exception:
            continue
    per_fold = [per_fold_by_fold[k] for k in sorted(per_fold_by_fold.keys())]

    avg_base = _extract_numeric_averages_nested(per_fold, key="base")
    avg_post = _extract_numeric_averages_nested(per_fold, key="post_train")
    avg_delta = _extract_numeric_averages_nested(per_fold, key="delta")

    payload = {
        "num_folds": int(len(per_fold)),
        "targets": list(SCORE_TARGETS),
        "per_fold": per_fold,
        "average": {
            "base": avg_base,
            "post_train": avg_post,
            "delta": avg_delta,
        },
        "baseline_kfold_swa_metrics_json": baseline_swa_obj,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[POST-TRAIN-EVAL] Saved k-fold val pseudo-test post-train metrics JSON -> {out_path}")


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

    # If output JSON already exists, reuse completed folds and skip re-evaluation.
    out_path = (log_dir / output_name).resolve()
    existing_by_fold: Dict[int, Dict[str, Any]] = {}
    if out_path.is_file():
        try:
            existing_payload = json.loads(out_path.read_text(encoding="utf-8"))
            for rec in list(existing_payload.get("per_fold", []) or []):
                try:
                    fi = int(rec.get("fold"))
                    existing_by_fold[fi] = rec
                except Exception:
                    continue
        except Exception:
            existing_by_fold = {}

    for fold_idx in range(k):
        if fold_idx in existing_by_fold:
            per_fold_metrics.append(existing_by_fold[fold_idx])
            print(f"[SWA-EVAL] Fold {fold_idx}: metrics already exist in {out_path.name}; skipping eval.")
            continue

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

    # Deduplicate by fold index (if existing payload had duplicates).
    per_fold_by_fold: Dict[int, Dict[str, Any]] = {}
    for rec in per_fold_metrics:
        try:
            per_fold_by_fold[int(rec.get("fold"))] = rec
        except Exception:
            continue
    per_fold_metrics = [per_fold_by_fold[k] for k in sorted(per_fold_by_fold.keys())]

    avg_metrics = _extract_numeric_averages(per_fold_metrics)
    payload = {
        "num_folds": len(per_fold_metrics),
        "per_fold": per_fold_metrics,
        "average": avg_metrics,
    }

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


