import argparse
import csv
import os
import re
import shutil
from pathlib import Path
import math

import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Package head weights and project sources into weights/ folder")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "train.yaml"),
        help="Path to YAML config file (to resolve version and output dirs)",
    )
    p.add_argument(
        "--weights-dir",
        type=str,
        default=str(Path(__file__).parent / "weights"),
        help="Destination weights directory",
    )
    p.add_argument(
        "--best",
        action="store_true",
        help="Select best checkpoint by val_loss (default: use latest-epoch checkpoint)",
    )
    p.add_argument(
        "--no-swa",
        action="store_true",
        help="Disable SWA-averaged weights and use raw head-epoch checkpoints (legacy behaviour).",
    )
    return p.parse_args()


def load_cfg(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_config_snapshot_for_version(
    base_cfg: dict,
    repo_root: Path,
    version: str,
) -> Path | None:
    """
    Locate a snapshot of the training config for a given version.
    Priority:
      1) outputs/<version>/train.yaml
      2) outputs/<version>/fold_0/train.yaml   (train_all or special cases)
    Falls back to None if not found.
    """
    base_log_dir = Path(base_cfg["logging"]["log_dir"]).expanduser()
    # Root version log dir (without train_all folding)
    v_root = base_log_dir / version if version else base_log_dir
    cand1 = v_root / "train.yaml"
    if cand1.is_file():
        return cand1
    cand2 = v_root / "fold_0" / "train.yaml"
    if cand2.is_file():
        return cand2
    return None


def _copy_snapshot_into_weights(
    snapshot_path: Path | None,
    repo_root: Path,
    weights_dir: Path,
) -> None:
    """
    Copy config snapshot into weights/configs/train.yaml, overriding the repo copy.
    If snapshot_path is None or missing, leave the repo copy as-is.
    """
    try:
        dst_cfg_dir = weights_dir / "configs"
        dst_cfg_dir.mkdir(parents=True, exist_ok=True)
        if snapshot_path is not None and snapshot_path.is_file():
            import shutil as _shutil

            dst_cfg_path = dst_cfg_dir / "train.yaml"
            _shutil.copyfile(str(snapshot_path), str(dst_cfg_path))
            print(f"[CFG] Using snapshot config for inference: {snapshot_path} -> {dst_cfg_path}")
    except Exception as e:
        print(f"[CFG] Failed to copy config snapshot into weights/: {e}")


def load_ensemble_cfg(repo_root: Path) -> dict:
    """
    Read configs/ensemble.json if present. Returns dict with at least:
      - enabled: bool
      - versions: list[str]
    """
    try:
        path = repo_root / "configs" / "ensemble.json"
        if not path.is_file():
            return {"enabled": False, "versions": []}
        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            obj = _json.load(f)
        if not isinstance(obj, dict):
            return {"enabled": False, "versions": []}
        enabled = bool(obj.get("enabled", False))
        # Backward-compat: accept single 'version' or list 'versions'
        versions = obj.get("versions", None)
        if versions is None:
            v = obj.get("version", "")
            versions = [str(v)] if isinstance(v, str) and v else []
        else:
            if isinstance(versions, list):
                versions = [str(v) for v in versions if isinstance(v, (str, int, float)) and str(v)]
            else:
                versions = []
        return {"enabled": enabled, "versions": versions}
    except Exception:
        return {"enabled": False, "versions": []}


def resolve_dirs(cfg: dict) -> tuple[Path, Path]:
    base_log_dir = Path(cfg["logging"]["log_dir"]).expanduser()
    base_ckpt_dir = Path(cfg["logging"]["ckpt_dir"]).expanduser()
    version = cfg.get("version", None)
    version = None if version in (None, "", "null") else str(version)
    log_dir = base_log_dir / version if version else base_log_dir
    ckpt_dir = base_ckpt_dir / version if version else base_ckpt_dir
    return log_dir, ckpt_dir


def find_latest_metrics_csv(log_dir: Path) -> Path | None:
    candidate_files: list[Path] = []
    if not log_dir.exists():
        return None
    for root, _, files in os.walk(log_dir):
        for name in files:
            if name == "metrics.csv":
                candidate_files.append(Path(root) / name)
    if not candidate_files:
        return None
    candidate_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidate_files[0]


def pick_best_epoch_from_metrics(metrics_csv: Path) -> int | None:
    try:
        with open(metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            last_val_loss_by_epoch: dict[int, float] = {}
            for row in reader:
                if "epoch" not in row:
                    continue
                try:
                    epoch = int(float(row["epoch"]))
                except Exception:
                    continue
                if "val_loss" in row and row["val_loss"] not in ("", None):
                    try:
                        val_loss = float(row["val_loss"])
                        last_val_loss_by_epoch[epoch] = val_loss
                    except Exception:
                        pass
        if not last_val_loss_by_epoch:
            return None
        # select epoch with minimal val_loss
        best_epoch = min(last_val_loss_by_epoch.items(), key=lambda kv: kv[1])[0]
        return best_epoch
    except Exception:
        return None


def list_head_checkpoints(head_dir: Path) -> list[Path]:
    if not head_dir.exists():
        return []
    files = [p for p in head_dir.iterdir() if p.is_file() and p.name.startswith("head-epoch") and p.suffix == ".pt"]
    return files


def find_head_by_epoch(head_dir: Path, epoch: int) -> Path | None:
    # Support metric-suffixed filenames like head-epoch003-val_loss0.123456-...pt
    candidates = []
    prefix = f"head-epoch{epoch:03d}"
    for p in list_head_checkpoints(head_dir):
        name = p.name
        if name.startswith(prefix) and name.endswith(".pt"):
            candidates.append(p)
    if not candidates:
        return None
    # If multiple, prefer the most recent
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def pick_latest_head(head_files: list[Path]) -> Path | None:
    if not head_files:
        return None
    def parse_epoch(p: Path) -> int:
        m = re.search(r"head-epoch(\d+)(?:[^/]*)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    head_files.sort(key=lambda p: (parse_epoch(p), p.stat().st_mtime), reverse=True)
    return head_files[0]


def copy_head_to_weights(head_src: Path, weights_dir: Path) -> Path:
    dst_dir = weights_dir / "head"
    # Clean destination directory before copying
    if dst_dir.exists() and dst_dir.is_dir():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / "infer_head.pt"
    shutil.copyfile(str(head_src), str(dst_path))
    return dst_path


def copy_tree(src_dir: Path, dst_dir: Path) -> None:
    # Clean destination directory before copying
    if dst_dir.exists() and dst_dir.is_dir():
        shutil.rmtree(dst_dir)
    shutil.copytree(
        src_dir,
        dst_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.pyd", ".DS_Store"),
    )


def copy_optional_third_party(repo_root: Path, weights_dir: Path) -> None:
    # Copy PEFT sources for offline PEFT inference (optional)
    peft_src = repo_root / "third_party" / "peft"
    if peft_src.exists() and peft_src.is_dir():
        dst = weights_dir / "third_party" / "peft"
        if dst.exists() and dst.is_dir():
            shutil.rmtree(dst)
        shutil.copytree(
            peft_src,
            dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.pyd", ".git", ".DS_Store"),
        )


def copy_top_level_scripts(repo_root: Path, weights_dir: Path) -> list[Path]:
    # Copy selected top-level scripts into weights/ for portability
    script_names = [
        "infer_and_submit_pt.py",
        "package_artifacts.py",
        "train.py",
        "sanity_check.py",
    ]
    copied: list[Path] = []
    for name in script_names:
        src = repo_root / name
        if src.is_file():
            dst = weights_dir / name
            shutil.copyfile(str(src), str(dst))
            copied.append(dst)
    return copied


def _load_checkpoint(path: Path):
    import torch

    try:
        # PyTorch 2.6+ defaults to weights_only=True internally, which can fail for
        # Lightning checkpoints that store callback state. If the safe loader fails,
        # retry with weights_only=False (trusted local checkpoint).
        return torch.load(str(path), map_location="cpu")
    except Exception:
        try:
            print(f"[SWA] Retrying torch.load with weights_only=False for checkpoint: {path}")
            return torch.load(str(path), map_location="cpu", weights_only=False)  # type: ignore[call-arg]
        except Exception as e:
            print(f"[SWA] torch.load failed even with weights_only=False: {e}")
            raise


def _find_swa_average_model_state(ckpt_obj: dict) -> dict | None:
    """
    Locate the StochasticWeightAveraging callback state in a Lightning checkpoint
    and return its `average_model_state` (if present).
    """
    callbacks_state = ckpt_obj.get("callbacks", None)
    if not isinstance(callbacks_state, dict):
        return None
    for cb_state in callbacks_state.values():
        if not isinstance(cb_state, dict):
            continue
        avg_state = cb_state.get("average_model_state", None)
        if isinstance(avg_state, dict) and avg_state:
            return avg_state
    return None


def _export_swa_head_from_checkpoint(ckpt_path: Path, dst_dir: Path) -> tuple[Path, Path] | None:
    """
    Build an inference head using SWA-averaged weights stored inside a Lightning
    checkpoint. Returns (ckpt_path, dst_head_path) on success, or None if SWA
    information is not available in the checkpoint.
    """
    from src.models.regressor import BiomassRegressor
    from src.callbacks.head_checkpoint import HeadCheckpoint
    import shutil as _shutil

    if not ckpt_path.is_file():
        print(f"[SWA] Checkpoint not found on disk, skipping SWA export: {ckpt_path}")
        return None

    print(f"[SWA] Attempting to export SWA head from checkpoint: {ckpt_path}")

    try:
        ckpt_obj = _load_checkpoint(ckpt_path)
    except Exception as e:
        print(f"[SWA] Failed to load checkpoint ({ckpt_path}): {e}")
        return None

    avg_state = _find_swa_average_model_state(ckpt_obj)
    if not isinstance(avg_state, dict) or not avg_state:
        # No SWA information in this checkpoint
        print(f"[SWA] No average_model_state found in checkpoint callbacks (SWA may be disabled or not checkpointed).")
        return None

    # Instantiate model from checkpoint hyperparameters and then load SWA-averaged weights
    try:
        model = BiomassRegressor.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    except Exception as e:
        print(f"[SWA] Failed to construct BiomassRegressor from checkpoint ({ckpt_path}): {e}")
        return None

    try:
        missing, unexpected = model.load_state_dict(avg_state, strict=False)
        if missing or unexpected:
            print(f"[SWA] Warning: loading SWA state into model had missing={len(missing)}, unexpected={len(unexpected)} keys.")
    except Exception as e:
        print(f"[SWA] Failed to load SWA average_model_state into model: {e}")
        return None

    # Use the existing HeadCheckpoint logic to compose a packed inference head
    # from the SWA-averaged model.
    # IMPORTANT: place the temporary directory OUTSIDE dst_dir, because we may
    # clear dst_dir before copying the final head file.
    tmp_dir = dst_dir.parent / ".tmp_swa_head"
    if tmp_dir.exists() and tmp_dir.is_dir():
        _shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    head_cb = HeadCheckpoint(output_dir=str(tmp_dir))

    class _DummyTrainer:
        def __init__(self, epoch: int):
            self.current_epoch = epoch
            # No metrics suffix needed for packaging; filenames will be simple.
            self.callback_metrics = {}

    epoch_int = 0
    try:
        epoch_int = int(ckpt_obj.get("epoch", 0))
    except Exception:
        epoch_int = 0

    dummy_trainer = _DummyTrainer(epoch=epoch_int)
    try:
        head_cb.on_validation_epoch_end(dummy_trainer, model)
    except Exception as e:
        print(f"[SWA] HeadCheckpoint export failed when using SWA-averaged model: {e}")
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # Locate the produced head-epoch*.pt under the temporary directory.
    head_files = list_head_checkpoints(tmp_dir)
    if not head_files:
        print("[SWA] No head-epoch*.pt files were produced by HeadCheckpoint in temporary SWA export dir.")
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    chosen = pick_latest_head(head_files)
    if chosen is None:
        print("[SWA] Failed to select a head checkpoint from temporary SWA export dir.")
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # Copy into destination as infer_head.pt
    if dst_dir.exists() and dst_dir.is_dir():
        _shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / "infer_head.pt"
    _shutil.copyfile(str(chosen), str(dst_path))

    # Clean temporary directory
    _shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[SWA] Successfully exported SWA head: {ckpt_path} -> {dst_path}")
    return ckpt_path, dst_path


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    repo_root = Path(__file__).parent
    ensemble_cfg = load_ensemble_cfg(repo_root)
    log_dir, ckpt_dir = resolve_dirs(cfg)

    weights_dir = Path(args.weights_dir).expanduser()
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Multi-version ensemble packaging path
    enabled = bool(ensemble_cfg.get("enabled", False))
    versions: list[str] = list(ensemble_cfg.get("versions", []))
    if enabled and len(versions) > 0:
        print(f"[ENSEMBLE] Multi-version packaging enabled. Versions: {versions}")
        select_best = bool(getattr(args, "best", False))
        use_swa = not bool(getattr(args, "no_swa", False))

        packaged_head_rel_paths: list[str] = []
        # Track a canonical version for config snapshot (first in list)
        canonical_ver: str = versions[0]
        canonical_snapshot: Path | None = _find_config_snapshot_for_version(cfg, repo_root, canonical_ver)
        for ver in versions:
            # Resolve per-version dirs without mutating base cfg
            tmp_cfg = dict(cfg or {})
            tmp_cfg["version"] = ver
            v_log_dir, v_ckpt_dir = resolve_dirs(tmp_cfg)

            # Detect kfold by listing fold_* directories
            fold_dirs = [p for p in v_ckpt_dir.glob("fold_*") if p.is_dir()]
            if fold_dirs:
                fold_dirs.sort(key=lambda p: p.name)
                exported = []
                for fold_dir in fold_dirs:
                    try:
                        fold_idx = int(str(fold_dir.name).split("_", 1)[1])
                    except Exception:
                        # fallback: enumerate
                        fold_idx = None
                    dst_dir = weights_dir / "head" / ver / (f"fold_{fold_idx}" if fold_idx is not None else fold_dir.name)

                    # Try SWA export
                    chosen: Path | None = None
                    dst_path: Path | None = None
                    if use_swa:
                        last_ckpt = fold_dir / "last.ckpt"
                        if last_ckpt.is_file():
                            ckpt_for_swa = last_ckpt
                        else:
                            ckpt_candidates = [p for p in fold_dir.glob("*.ckpt") if p.is_file()]
                            ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                            ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else None
                        if ckpt_for_swa is not None:
                            print(f"[SWA] {ver} fold {fold_dir.name}: attempting SWA export from {ckpt_for_swa}")
                            res = _export_swa_head_from_checkpoint(ckpt_for_swa, dst_dir)
                            if res is not None:
                                chosen, dst_path = res
                                exported.append((chosen, dst_path))
                                try:
                                    rel = dst_path.relative_to(weights_dir / "head")
                                    packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                                except Exception:
                                    pass
                                # Copy z_score.json for this version/fold if present
                                zsrc = v_log_dir / fold_dir.name / "z_score.json"
                                if zsrc.is_file():
                                    try:
                                        shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                                    except Exception:
                                        pass
                                # done this fold
                                continue
                            else:
                                print(f"[SWA] {ver} fold {fold_dir.name}: SWA export failed; fallback to raw head-epoch*.pt.")

                    # Fallback to raw head-epoch*.pt under fold_dir/head
                    fold_ckpt_head_dir = fold_dir / "head"
                    head_files = list_head_checkpoints(fold_ckpt_head_dir)
                    if not head_files:
                        raise FileNotFoundError(f"No head checkpoints found under: {fold_ckpt_head_dir}")
                    if select_best:
                        metrics_csv = find_latest_metrics_csv(v_log_dir / fold_dir.name)
                        chosen = None
                        if metrics_csv is not None:
                            best_epoch = pick_best_epoch_from_metrics(metrics_csv)
                            if best_epoch is not None:
                                p = find_head_by_epoch(fold_ckpt_head_dir, best_epoch)
                                if p is not None:
                                    chosen = p
                        if chosen is None:
                            chosen = pick_latest_head(head_files)
                    else:
                        chosen = pick_latest_head(head_files)
                    if chosen is None:
                        raise FileNotFoundError(f"Failed to determine a head checkpoint for {ver}/{fold_dir.name}.")
                    if dst_dir.exists() and dst_dir.is_dir():
                        shutil.rmtree(dst_dir)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst_path = dst_dir / "infer_head.pt"
                    shutil.copyfile(str(chosen), str(dst_path))
                    exported.append((chosen, dst_path))
                    try:
                        rel = dst_path.relative_to(weights_dir / "head")
                        packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                    except Exception:
                        pass
                    zsrc = v_log_dir / fold_dir.name / "z_score.json"
                    if zsrc.is_file():
                        try:
                            shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                        except Exception:
                            pass

                print(f"[ENSEMBLE] Packaged heads for version '{ver}':")
                for src_path, dst_path in exported:
                    print(f" - {src_path} -> {dst_path}")
            else:
                # Single-run directory
                dst_dir = weights_dir / "head" / ver
                # Try SWA
                chosen: Path | None = None
                if use_swa:
                    main_last_ckpt = v_ckpt_dir / "last.ckpt"
                    if main_last_ckpt.is_file():
                        ckpt_for_swa = main_last_ckpt
                    else:
                        ckpt_candidates = [p for p in v_ckpt_dir.glob("*.ckpt") if p.is_file()]
                        ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else None
                    if ckpt_for_swa is not None:
                        print(f"[SWA] {ver}: attempting SWA export from {ckpt_for_swa}")
                        res = _export_swa_head_from_checkpoint(ckpt_for_swa, dst_dir)
                        if res is not None:
                            chosen, copied_head = res
                            print(f"Copied SWA head checkpoint: {chosen} -> {copied_head}")
                            try:
                                rel = copied_head.relative_to(weights_dir / "head")
                                packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                            except Exception:
                                pass
                            # Copy z_score.json for this version if present
                            zsrc = v_log_dir / "z_score.json"
                            if zsrc.is_file():
                                try:
                                    shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                                except Exception:
                                    pass
                        else:
                            print(f"[SWA] {ver}: SWA export failed; fallback to raw head-epoch*.pt.")
                if chosen is None:
                    # Legacy raw copy
                    head_dir = v_ckpt_dir / "head"
                    head_files = list_head_checkpoints(head_dir)
                    if not head_files:
                        raise FileNotFoundError(f"No head checkpoints found under: {head_dir}")
                    if select_best:
                        metrics_csv = find_latest_metrics_csv(v_log_dir)
                        chosen = None
                        if metrics_csv is not None:
                            best_epoch = pick_best_epoch_from_metrics(metrics_csv)
                            if best_epoch is not None:
                                p = find_head_by_epoch(head_dir, best_epoch)
                                if p is not None:
                                    chosen = p
                        if chosen is None:
                            chosen = pick_latest_head(head_files)
                    else:
                        chosen = pick_latest_head(head_files)
                    if chosen is None:
                        raise FileNotFoundError(f"Failed to determine a head checkpoint for version {ver}.")
                    if dst_dir.exists() and dst_dir.is_dir():
                        shutil.rmtree(dst_dir)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    copied_head = dst_dir / "infer_head.pt"
                    shutil.copyfile(str(chosen), str(copied_head))
                    print(f"Copied head checkpoint: {chosen} -> {copied_head}")
                    try:
                        rel = copied_head.relative_to(weights_dir / "head")
                        packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                    except Exception:
                        pass
                    zsrc = v_log_dir / "z_score.json"
                    if zsrc.is_file():
                        try:
                            shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                        except Exception:
                            pass

        # Write packaged ensemble manifest
        if packaged_head_rel_paths:
            try:
                import json as _json
                manifest = {
                    "aggregation": "mean",
                    "heads": [{"path": p} for p in packaged_head_rel_paths],
                }
                dst = weights_dir / "head" / "ensemble.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "w", encoding="utf-8") as f:
                    _json.dump(manifest, f, indent=2)
                print(f"[ENSEMBLE] Wrote packaged manifest with {len(packaged_head_rel_paths)} heads: {dst}")
            except Exception as e:
                print(f"[ENSEMBLE] Failed to write packaged manifest: {e}")

        # Copy configs and src into weights/
        copy_tree(repo_root / "configs", weights_dir / "configs")
        # Override configs/train.yaml with snapshot from canonical version if available
        _copy_snapshot_into_weights(canonical_snapshot, repo_root, weights_dir)
        copy_tree(repo_root / "src", weights_dir / "src")
        copy_optional_third_party(repo_root, weights_dir)
        scripts_copied = copy_top_level_scripts(repo_root, weights_dir)
        print(f"Copied configs/ and src/ to: {weights_dir}")
        if scripts_copied:
            print("Copied scripts to weights/:")
            for p in scripts_copied:
                print(f" - {p}")
        return

    # If kfold is enabled, export one head per fold under weights/head/fold_*/infer_head.pt.
    # If train_all is enabled, additionally export a preferred single head to
    # weights/head/infer_head.pt (used as the default for offline inference).
    kfold_cfg = cfg.get("kfold", {})
    kfold_enabled = bool(kfold_cfg.get("enabled", False))

    train_all_cfg = cfg.get("train_all", {})
    train_all_enabled = bool(train_all_cfg.get("enabled", False))

    select_best = bool(getattr(args, "best", False))
    use_swa = not bool(getattr(args, "no_swa", False))

    # Collect relative paths (under weights/head) for packaged heads to emit a packaged manifest
    packaged_head_rel_paths: list[str] = []
    emit_packaged_manifest: bool = bool(ensemble_cfg.get("enabled", False))

    # 1) Per-fold heads for k-fold training (if enabled **and** train_all is disabled).
    # When both kfold and train_all are enabled, we now only package the train_all
    # head as the preferred single model for offline inference, and skip exporting
    # individual per-fold heads here.
    if kfold_enabled and not train_all_enabled:
        k = int(kfold_cfg.get("k", 5))
        exported = []
        for fold_idx in range(k):
            fold_root_ckpt_dir = ckpt_dir / f"fold_{fold_idx}"
            fold_ckpt_head_dir = fold_root_ckpt_dir / "head"
            fold_log_dir = log_dir / f"fold_{fold_idx}"

            # When SWA is enabled, attempt to build the head from the SWA-averaged
            # weights stored in this fold's Lightning checkpoint (prefer last.ckpt).
            chosen: Path | None = None
            dst_path: Path | None = None
            if use_swa:
                last_ckpt = fold_root_ckpt_dir / "last.ckpt"
                ckpt_for_swa: Path | None
                if last_ckpt.is_file():
                    ckpt_for_swa = last_ckpt
                else:
                    # Fallback: any .ckpt in the fold directory, preferring the most recent.
                    ckpt_candidates = [p for p in fold_root_ckpt_dir.glob("*.ckpt") if p.is_file()]
                    ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else None

                if ckpt_for_swa is not None:
                    print(f"[SWA] Fold {fold_idx}: attempting SWA export from checkpoint: {ckpt_for_swa}")
                    dst_dir = weights_dir / "head" / f"fold_{fold_idx}"
                    res = _export_swa_head_from_checkpoint(ckpt_for_swa, dst_dir)
                    if res is not None:
                        chosen, dst_path = res
                        exported.append((chosen, dst_path))
                        try:
                            rel = dst_path.relative_to(weights_dir / "head")
                            packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                        except Exception:
                            pass
                        # Copy z_score.json for this fold if present
                        zsrc = fold_log_dir / "z_score.json"
                        if zsrc.is_file():
                            try:
                                shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                            except Exception:
                                pass
                        # SWA export for this fold succeeded; continue to next fold.
                        continue
                    else:
                        print(f"[SWA] Fold {fold_idx}: SWA export failed or unavailable, falling back to raw head-epoch*.pt.")
                else:
                    print(f"[SWA] Fold {fold_idx}: no .ckpt files found for SWA export, falling back to raw head-epoch*.pt.")

            # Legacy behaviour (or SWA unavailable): select a head checkpoint file
            # under fold_{i}/head/ as before.
            head_files = list_head_checkpoints(fold_ckpt_head_dir)
            if not head_files:
                raise FileNotFoundError(f"No head checkpoints found under: {fold_ckpt_head_dir}")

            if select_best:
                # Pick best by val_loss using this fold's latest metrics.csv
                metrics_csv = find_latest_metrics_csv(fold_log_dir)
                chosen = None
                if metrics_csv is not None:
                    best_epoch = pick_best_epoch_from_metrics(metrics_csv)
                    if best_epoch is not None:
                        p = find_head_by_epoch(fold_ckpt_head_dir, best_epoch)
                        if p is not None:
                            chosen = p
                # Fallback to latest-epoch checkpoint if best could not be determined
                if chosen is None:
                    chosen = pick_latest_head(head_files)
            else:
                # Default: purely latest-epoch checkpoint
                chosen = pick_latest_head(head_files)

            if chosen is None:
                raise FileNotFoundError(f"Failed to determine a head checkpoint for fold {fold_idx}.")

            # Copy into weights/head/fold_{i}/infer_head.pt (clean per-fold dir)
            dst_dir = weights_dir / "head" / f"fold_{fold_idx}"
            if dst_dir.exists() and dst_dir.is_dir():
                shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / "infer_head.pt"
            shutil.copyfile(str(chosen), str(dst_path))
            exported.append((chosen, dst_path))
            try:
                rel = dst_path.relative_to(weights_dir / "head")
                packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
            except Exception:
                pass
            # Copy z_score.json for this fold if present
            zsrc = fold_log_dir / "z_score.json"
            if zsrc.is_file():
                try:
                    shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                except Exception:
                    pass

        print("Copied per-fold head checkpoints (src -> dst):")
        for src_path, dst_path in exported:
            print(f" - {src_path} -> {dst_path}")

    # 2) Preferred train_all head exported to weights/head/infer_head.pt (if enabled).
    train_all_head_dst: Path | None = None
    if train_all_enabled:
        train_all_root_ckpt_dir = ckpt_dir / "train_all"
        train_all_head_dir = train_all_root_ckpt_dir / "head"
        train_all_log_dir = log_dir / "train_all"

        chosen: Path | None = None
        # When SWA is enabled, attempt to build the head from SWA-averaged weights first.
        if use_swa:
            main_last_ckpt = train_all_root_ckpt_dir / "last.ckpt"
            ckpt_for_swa: Path | None
            if main_last_ckpt.is_file():
                ckpt_for_swa = main_last_ckpt
            else:
                ckpt_candidates = [p for p in train_all_root_ckpt_dir.glob("*.ckpt") if p.is_file()]
                ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else None

            if ckpt_for_swa is not None:
                print(f"[SWA] train_all: attempting SWA export from checkpoint: {ckpt_for_swa}")
                swa_dst_dir = weights_dir / "head" / "train_all"
                res = _export_swa_head_from_checkpoint(ckpt_for_swa, swa_dst_dir)
                if res is not None:
                    chosen, swa_head_path = res
                    # Copy to the preferred root location without removing existing fold_* dirs.
                    head_dst_dir = weights_dir / "head"
                    head_dst_dir.mkdir(parents=True, exist_ok=True)
                    train_all_head_dst = head_dst_dir / "infer_head.pt"
                    shutil.copyfile(str(swa_head_path), str(train_all_head_dst))
                    # Remove the temporary train_all/ copy to avoid redundant packaging.
                    try:
                        shutil.rmtree(swa_dst_dir, ignore_errors=True)
                    except Exception:
                        pass
                    try:
                        rel = train_all_head_dst.relative_to(weights_dir / "head")
                        packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                    except Exception:
                        pass
                    # Copy z_score.json for train_all if present
                    zsrc = train_all_log_dir / "z_score.json"
                    if zsrc.is_file():
                        try:
                            shutil.copyfile(str(zsrc), str(head_dst_dir / "z_score.json"))
                        except Exception:
                            pass
                else:
                    print("[SWA] train_all: SWA export failed; falling back to raw head-epoch*.pt.")
            else:
                print("[SWA] train_all: no .ckpt files found for SWA export, falling back to raw head-epoch*.pt.")

        # Fallback: latest/best raw head checkpoint under train_all/head
        if train_all_head_dst is None:
            head_files = list_head_checkpoints(train_all_head_dir)
            if not head_files:
                raise FileNotFoundError(f"No head checkpoints found under: {train_all_head_dir}")

            if select_best:
                metrics_csv = find_latest_metrics_csv(train_all_log_dir)
                chosen = None
                if metrics_csv is not None:
                    best_epoch = pick_best_epoch_from_metrics(metrics_csv)
                    if best_epoch is not None:
                        p = find_head_by_epoch(train_all_head_dir, best_epoch)
                        if p is not None:
                            chosen = p
                if chosen is None:
                    chosen = pick_latest_head(head_files)
            else:
                chosen = pick_latest_head(head_files)

            if chosen is None:
                raise FileNotFoundError("Failed to determine a head checkpoint for train_all.")

            head_dst_dir = weights_dir / "head"
            head_dst_dir.mkdir(parents=True, exist_ok=True)
            train_all_head_dst = head_dst_dir / "infer_head.pt"
            shutil.copyfile(str(chosen), str(train_all_head_dst))
            try:
                rel = train_all_head_dst.relative_to(weights_dir / "head")
                packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
            except Exception:
                pass
            # Copy z_score.json for train_all if present
            zsrc = train_all_log_dir / "z_score.json"
            if zsrc.is_file():
                try:
                    shutil.copyfile(str(zsrc), str(head_dst_dir / "z_score.json"))
                except Exception:
                    pass

        if train_all_head_dst is not None:
            print(f"Copied train_all head checkpoint -> {train_all_head_dst}")

    # 3) Single-run export (non-kfold, non-train_all) for backward-compatibility.
    if not kfold_enabled and not train_all_enabled:
        head_dir = ckpt_dir / "head"

        # When SWA is enabled, first attempt to build the head from SWA-averaged
        # weights stored in the main checkpoint directory (prefer last.ckpt).
        if use_swa:
            main_last_ckpt = ckpt_dir / "last.ckpt"
            ckpt_for_swa: Path | None
            if main_last_ckpt.is_file():
                ckpt_for_swa = main_last_ckpt
            else:
                ckpt_candidates = [p for p in ckpt_dir.glob("*.ckpt") if p.is_file()]
                ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else None

            if ckpt_for_swa is not None:
                print(f"[SWA] Single-run: attempting SWA export from checkpoint: {ckpt_for_swa}")
                dst_dir = weights_dir / "head"
                res = _export_swa_head_from_checkpoint(ckpt_for_swa, dst_dir)
                if res is not None:
                    chosen, copied_head = res
                    print(f"Copied SWA head checkpoint: {chosen} -> {copied_head}")
                    try:
                        rel = copied_head.relative_to(weights_dir / "head")
                        packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                    except Exception:
                        pass
                    # Copy z_score.json from run log dir if present
                    zsrc = log_dir / "z_score.json"
                    if zsrc.is_file():
                        try:
                            shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                        except Exception:
                            pass
                    # SWA export succeeded; skip legacy head selection.
                    head_dir = None  # type: ignore[assignment]
                else:
                    # Fall back to legacy behaviour below
                    print("[SWA] Single-run: SWA export failed or unavailable, falling back to raw head-epoch*.pt.")
            else:
                print("[SWA] Single-run: no .ckpt files found for SWA export, falling back to raw head-epoch*.pt.")

        if head_dir is not None:
            head_files = list_head_checkpoints(head_dir)
            if not head_files:
                raise FileNotFoundError(f"No head checkpoints found under: {head_dir}")

            if select_best:
                metrics_csv = find_latest_metrics_csv(log_dir)
                chosen = None
                if metrics_csv is not None:
                    best_epoch = pick_best_epoch_from_metrics(metrics_csv)
                    if best_epoch is not None:
                        p = find_head_by_epoch(head_dir, best_epoch)
                        if p is not None:
                            chosen = p
                if chosen is None:
                    chosen = pick_latest_head(head_files)
            else:
                chosen = pick_latest_head(head_files)

            if chosen is None:
                raise FileNotFoundError("Failed to determine a head checkpoint to package.")

            copied_head = copy_head_to_weights(chosen, weights_dir)
            print(f"Copied head checkpoint: {chosen} -> {copied_head}")
            try:
                rel = copied_head.relative_to(weights_dir / "head")
                packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
            except Exception:
                pass
        # Copy z_score.json from run log dir if present
        zsrc = log_dir / "z_score.json"
        if zsrc.is_file():
            try:
                shutil.copyfile(str(zsrc), str(weights_dir / "z_score.json"))
            except Exception:
                pass

    # If ensemble is enabled, emit a packaged manifest at weights/head/ensemble.json
    if emit_packaged_manifest and packaged_head_rel_paths:
        try:
            import json as _json
            manifest = {
                "aggregation": "mean",
                "heads": [{"path": p} for p in packaged_head_rel_paths],
            }
            dst = weights_dir / "head" / "ensemble.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "w", encoding="utf-8") as f:
                _json.dump(manifest, f, indent=2)
            print(f"[ENSEMBLE] Wrote packaged manifest: {dst}")
        except Exception as e:
            print(f"[ENSEMBLE] Failed to write packaged manifest: {e}")

    # Copy configs and src into weights/
    copy_tree(repo_root / "configs", weights_dir / "configs")
    # For single-version packaging, prefer snapshot from this cfg.version if available
    snapshot_ver = cfg.get("version", None)
    snapshot_ver = None if snapshot_ver in (None, "", "null") else str(snapshot_ver)
    snapshot_path = _find_config_snapshot_for_version(cfg, repo_root, snapshot_ver) if snapshot_ver else None
    _copy_snapshot_into_weights(snapshot_path, repo_root, weights_dir)
    copy_tree(repo_root / "src", weights_dir / "src")
    copy_optional_third_party(repo_root, weights_dir)
    scripts_copied = copy_top_level_scripts(repo_root, weights_dir)
    print(f"Copied configs/ and src/ to: {weights_dir}")
    if scripts_copied:
        print("Copied scripts to weights/:")
        for p in scripts_copied:
            print(f" - {p}")


if __name__ == "__main__":
    main()


