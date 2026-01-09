import argparse
from pathlib import Path

import yaml

from src.training.entrypoint import run_training

def resolve_repo_root() -> Path:
    """
    Resolve the project root directory that contains both `configs/` and `src/`.

    This script may live either in the repository root or in a packaged
    `weights/scripts/` directory; in the packaged case, the repo root is the parent
    of the scripts directory.
    """
    here = Path(__file__).resolve().parent
    if (here / "configs").is_dir() and (here / "src").is_dir():
        return here
    if (here.parent / "configs").is_dir() and (here.parent / "src").is_dir():
        return here.parent
    return here


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(resolve_repo_root() / "configs" / "train.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    version = cfg.get("version", None)
    version = None if version in (None, "", "null") else str(version)

    base_log_dir = Path(cfg["logging"]["log_dir"]).expanduser()
    base_ckpt_dir = Path(cfg["logging"]["ckpt_dir"]).expanduser()
    log_dir = base_log_dir / version if version else base_log_dir
    ckpt_dir = base_ckpt_dir / version if version else base_ckpt_dir

    run_training(
        cfg,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        repo_root=resolve_repo_root(),
        source_config_path=args.config,
        extra_callbacks=None,
        enable_post_kfold_swa_eval=True,
            )


if __name__ == "__main__":
    main()


