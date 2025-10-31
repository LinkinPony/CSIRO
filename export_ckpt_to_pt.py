import argparse
import os
from pathlib import Path

import torch

from src.models.regressor import BiomassRegressor


def parse_args():
    p = argparse.ArgumentParser(description="Export Lightning .ckpt to a standard PyTorch .pt state_dict")
    p.add_argument("--ckpt", required=True, type=str, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .pt path; defaults to <ckpt_dir>/<ckpt_basename>.pt",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path = args.out
    if out_path is None:
        base_dir = os.path.dirname(ckpt_path)
        base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_path = os.path.join(base_dir, base_name + ".pt")
    out_path = os.path.abspath(out_path)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    # Load the Lightning module without downloading any external weights
    model: BiomassRegressor = BiomassRegressor.load_from_checkpoint(ckpt_path, pretrained=False)
    state = model.state_dict()

    # Include a minimal metadata block to help inference reconstruct shapes
    meta = {
        "backbone": model.hparams.get("backbone_name", "dinov3_vitl16") if hasattr(model, "hparams") else "dinov3_vitl16",
        "embedding_dim": int(model.hparams.get("embedding_dim", 1024)) if hasattr(model, "hparams") else 1024,
        "num_outputs": int(model.hparams.get("num_outputs", 3)) if hasattr(model, "hparams") else 3,
    }

    torch.save({"state_dict": state, "meta": meta}, out_path)
    print(f"Exported state_dict to: {out_path}")


if __name__ == "__main__":
    main()


