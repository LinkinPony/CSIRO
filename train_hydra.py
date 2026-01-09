from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.training.entrypoint import resolve_repo_root, run_training


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    # Convert Hydra DictConfig -> plain dict (training code uses isinstance(x, dict) checks).
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Hydra config did not resolve to a dict, got: {type(cfg_dict)}")

    # Remove Hydra internal keys from the snapshot + downstream training config.
    cfg_dict.pop("hydra", None)

    version = cfg_dict.get("version", None)
    version = None if version in (None, "", "null") else str(version)

    base_log_dir = Path(str(cfg_dict["logging"]["log_dir"])).expanduser()
    base_ckpt_dir = Path(str(cfg_dict["logging"]["ckpt_dir"])).expanduser()

    log_dir = base_log_dir / version if version else base_log_dir
    ckpt_dir = base_ckpt_dir / version if version else base_ckpt_dir

    run_training(
        cfg_dict,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        repo_root=resolve_repo_root(),
        source_config_path=None,  # snapshot the resolved Hydra config
        extra_callbacks=None,
        enable_post_kfold_swa_eval=True,
    )


if __name__ == "__main__":
    main()


