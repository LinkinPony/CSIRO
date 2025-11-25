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
    log_dir, ckpt_dir = resolve_dirs(cfg)

    weights_dir = Path(args.weights_dir).expanduser()
    weights_dir.mkdir(parents=True, exist_ok=True)

    # If kfold is enabled, export one head per fold under weights/head/fold_*/infer_head.pt
    kfold_cfg = cfg.get("kfold", {})
    kfold_enabled = bool(kfold_cfg.get("enabled", False))

    # Train-all mode is treated like kfold with a single fold (fold_0)
    train_all_cfg = cfg.get("train_all", {})
    train_all_enabled = bool(train_all_cfg.get("enabled", False))

    # Special case: k-fold training with an additional train_all run
    # -----------------------------------------------------------------
    # The kfold runner can optionally launch a final "train_all" pass on
    # the full dataset even when cfg.train_all.enabled is False, writing
    # checkpoints under:
    #   <ckpt_dir>/<version>/train_all/
    # and logs under:
    #   <log_dir>/<version>/train_all/
    #
    # When this happens and train_all is *disabled in the config*, we
    # want package_artifacts.py to package the weights from this
    # train_all run instead of the per-fold k-fold heads.
    #
    # Concretely: if the config says train_all.enabled == False but a
    # "train_all" subdirectory exists in the checkpoint directory, we
    # switch to treating this as a single-run export rooted at
    #   ckpt_dir/train_all and log_dir/train_all.
    train_all_ckpt_dir = ckpt_dir / "train_all"
    train_all_log_dir = log_dir / "train_all"
    if (not train_all_enabled) and train_all_ckpt_dir.is_dir():
        # Prefer the train_all run for packaging.
        print(
            "[INFO] Detected 'train_all' checkpoint directory while "
            "train_all.enabled is False in config; packaging this full-data "
            "train_all model instead of per-fold k-fold heads."
        )
        ckpt_dir = train_all_ckpt_dir
        # Restrict metrics/z_score search to the train_all log subtree when present.
        if train_all_log_dir.is_dir():
            log_dir = train_all_log_dir
        # Force the logic below to take the single-run export path.
        kfold_enabled = False
        train_all_enabled = False

    select_best = bool(getattr(args, "best", False))
    use_swa = not bool(getattr(args, "no_swa", False))

    if kfold_enabled or train_all_enabled:
        k = 1 if train_all_enabled else int(kfold_cfg.get("k", 5))
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
    else:
        # Single-run export (non-kfold)
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
        # Copy z_score.json from run log dir if present
        zsrc = log_dir / "z_score.json"
        if zsrc.is_file():
            try:
                shutil.copyfile(str(zsrc), str(weights_dir / "z_score.json"))
            except Exception:
                pass

    # Copy configs and src into weights/
    repo_root = Path(__file__).parent
    copy_tree(repo_root / "configs", weights_dir / "configs")
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


