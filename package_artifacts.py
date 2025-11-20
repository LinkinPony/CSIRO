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

    select_best = bool(getattr(args, "best", False))

    if kfold_enabled or train_all_enabled:
        k = 1 if train_all_enabled else int(kfold_cfg.get("k", 5))
        exported = []
        for fold_idx in range(k):
            fold_ckpt_head_dir = ckpt_dir / f"fold_{fold_idx}" / "head"
            fold_log_dir = log_dir / f"fold_{fold_idx}"
            head_files = list_head_checkpoints(fold_ckpt_head_dir)
            if not head_files:
                raise FileNotFoundError(f"No head checkpoints found under: {fold_ckpt_head_dir}")

            if select_best:
                # Pick best by val_loss using this fold's latest metrics.csv
                metrics_csv = find_latest_metrics_csv(fold_log_dir)
                chosen: Path | None = None
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
        head_files = list_head_checkpoints(head_dir)
        if not head_files:
            raise FileNotFoundError(f"No head checkpoints found under: {head_dir}")

        if select_best:
            metrics_csv = find_latest_metrics_csv(log_dir)
            chosen: Path | None = None
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


