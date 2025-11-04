import argparse
import csv
import os
import re
import shutil
from pathlib import Path

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
    target = head_dir / f"head-epoch{epoch:03d}.pt"
    return target if target.is_file() else None


def pick_latest_head(head_files: list[Path]) -> Path | None:
    if not head_files:
        return None
    def parse_epoch(p: Path) -> int:
        m = re.search(r"head-epoch(\d+)\.pt$", p.name)
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


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    log_dir, ckpt_dir = resolve_dirs(cfg)

    head_dir = ckpt_dir / "head"
    head_files = list_head_checkpoints(head_dir)
    if not head_files:
        raise FileNotFoundError(f"No head checkpoints found under: {head_dir}")

    # Try to pick best by val_loss from latest metrics.csv
    metrics_csv = find_latest_metrics_csv(log_dir)
    chosen: Path | None = None
    if metrics_csv is not None:
        best_epoch = pick_best_epoch_from_metrics(metrics_csv)
        if best_epoch is not None:
            p = find_head_by_epoch(head_dir, best_epoch)
            if p is not None:
                chosen = p

    # Fallback to latest by epoch/mtime
    if chosen is None:
        chosen = pick_latest_head(head_files)
    if chosen is None:
        raise FileNotFoundError("Failed to determine a head checkpoint to package.")

    weights_dir = Path(args.weights_dir).expanduser()
    weights_dir.mkdir(parents=True, exist_ok=True)

    copied_head = copy_head_to_weights(chosen, weights_dir)
    print(f"Copied head checkpoint to: {copied_head}")

    # Copy configs and src into weights/
    repo_root = Path(__file__).parent
    copy_tree(repo_root / "configs", weights_dir / "configs")
    copy_tree(repo_root / "src", weights_dir / "src")
    print(f"Copied configs/ and src/ to: {weights_dir}")


if __name__ == "__main__":
    main()


