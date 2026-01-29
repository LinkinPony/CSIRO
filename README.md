# CSIRO Pasture Biomass (DINOv3 + Head)

Train models that predict pasture biomass components from pasture images (plus optional auxiliary signals like NDVI/height).

This repository was built for a competition-style setup where predictions are produced in **long CSV format** with 5 targets per image (grams):
`Dry_Clover_g`, `Dry_Dead_g`, `Dry_Green_g`, `GDM_g`, `Dry_Total_g`.

For detailed training + Ray Tune cluster instructions, see [`README_TRAINING.md`](README_TRAINING.md).

## Highlights

- **Frozen DINOv3 backbone**, trainable regression head (multiple head types; Hydra config system under `conf/`).
- **Competition-aligned metrics** (weighted multi-target scoring; see [`DESCRIPTION.md`](DESCRIPTION.md)).
- **Ray Tune HPO** via [`tune.py`](tune.py) and helper scripts.
- **Inference pipeline** that can load **backbone weights + head-only weights** and write `submission.csv`.

## Repository layout (important files)

- Training
  - [`train.py`](train.py): legacy single-YAML launcher (`configs/train.yaml`)
  - [`train_hydra.py`](train_hydra.py): Hydra launcher (recommended; config in `conf/`)
  - [`configs/`](configs/): legacy training configs
  - [`conf/`](conf/): Hydra modular configs (entrypoint: `conf/train.yaml`)
- Hyperparameter search
  - [`tune.py`](tune.py): Ray Tune entrypoint (Hydra config: choose one of `conf/tune_*.yaml` via `--config-name`)
  - [`tune.sh`](tune.sh): convenience wrapper
- Inference / submission
  - [`infer_and_submit_pt.py`](infer_and_submit_pt.py): run inference and write a Kaggle-style submission CSV
- Source
  - [`src/`](src/): training, data, models, inference
- Extras
  - [`tools/tune_viewer/README.md`](tools/tune_viewer/README.md): local web UI for browsing Ray Tune results
  - [`export_ckpt_to_pt.py`](export_ckpt_to_pt.py): convert Lightning `.ckpt` to a plain `.pt` state dict (utility)
  - [`package_artifacts.py`](package_artifacts.py): package weights + minimal sources into `weights/` (for inference)

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

## Data layout

By default configs assume `data.root: data` and `data.train_csv: train.csv` (see [`configs/train.yaml`](configs/train.yaml)).

Minimum expected structure:

```
data/
  train.csv
  test.csv
  train/            # or another subdir referenced by `image_path`
    IDxxxxxxxxxx.jpg
  test/
    IDxxxxxxxxxx.jpg
```

Key conventions:

- **`image_path` is interpreted relative to `data/`** (the `data.root` directory).
- Training expects a **long-format** `train.csv` and pivots it to image-level rows internally.
- Inference expects a **long-format** `test.csv` and produces a **long-format** `submission.csv` with columns:
  `sample_id,target`.

## Evaluation (competition metric)

This repo uses a competition-style **weighted R² in log-space** (evaluate on `log1p(clamp(x, min=0))` per target).
Target weights:

- `Dry_Clover_g`: 0.1
- `Dry_Dead_g`: 0.1
- `Dry_Green_g`: 0.1
- `GDM_g`: 0.2
- `Dry_Total_g`: 0.5

## Training

### Option A: legacy YAML launcher

```bash
python train.py --config configs/train.yaml
```

### Option B: Hydra launcher (recommended)

```bash
python train_hydra.py
```

Common overrides:

```bash
# set a run name (controls outputs/ subdir)
python train_hydra.py version=my_run

# change training budget
python train_hydra.py trainer.max_epochs=10 trainer.limit_train_batches=200

# change optimizer hyperparameters
python train_hydra.py optimizer.lr=5e-4 optimizer.weight_decay=0.01
```

Outputs (defaults):

- Logs: `outputs/<version>/`
- Checkpoints: `outputs/checkpoints/<version>/`

More details (including Ray cluster setup, NFS notes, and resume behavior) are in [`README_TRAINING.md`](README_TRAINING.md).

## Hyperparameter search (Ray Tune)

Tune configs live under `conf/` (for example: `conf/tune_vitdet.yaml`, `conf/tune_mlp.yaml`, `conf/tune_mamba_v8.yaml`).
Select a config using Hydra's `--config-name`:

```bash
python tune.py --config-name tune_vitdet
```

Common usage (example with shared storage + Ray auto address):

```bash
python tune.py --config-name tune_vitdet tune.storage_path=/mnt/csiro_nfs/ray_results ray.address=auto
```

Or use the wrapper:

```bash
./tune.sh --config-name tune_vitdet tune-run-name tune.storage_path=/mnt/csiro_nfs/ray_results ray.address=auto
```

See [`README_TRAINING.md`](README_TRAINING.md) for the full two-node (head/worker) recipe.

## Weights: backbone vs head (important)

This repo treats the **backbone** and the **regression head** as separate artifacts:

- **Backbone weights**: stored under `dinov3_weights/` (one shared file per backbone variant, e.g. `dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pt`).
- **Head weights**: stored separately (small, head-only checkpoint).

Typical head checkpoint locations:

- Training saves final-epoch head checkpoint under:
  `outputs/checkpoints/<version>/head/head-epochXXX*.pt`
- Packaged for inference at:
  `weights/head/infer_head.pt`
- Lightning also saves a full `last.ckpt` (Lightning checkpoint) for backward compatibility.

Note: k-fold and `train_all` modes write under subdirectories like `fold_0/` or `train_all/` inside `outputs/` and `outputs/checkpoints/`.

Inference requires both:

1) a DINOv3 backbone weights file (or a directory containing official weights) via `DINO_WEIGHTS_PT_PATH` in [`infer_and_submit_pt.py`](infer_and_submit_pt.py)
2) a head weights file or directory via `HEAD_WEIGHTS_PT_PATH` in [`infer_and_submit_pt.py`](infer_and_submit_pt.py) (for example `weights/head/`, which contains `infer_head.pt`)

In directory mode, `infer_and_submit_pt.py` auto-selects the correct backbone weights file based on the configured backbone name.

## Inference / submission

Edit the “Required user variables” at the top of [`infer_and_submit_pt.py`](infer_and_submit_pt.py) to point to:

- `HEAD_WEIGHTS_PT_PATH` (head weights file or directory)
- `DINO_WEIGHTS_PT_PATH` (a `.pt/.pth` file, or a directory like `dinov3_weights/`)
- `INPUT_PATH` (either `data/` or a direct `test.csv` path)
- `OUTPUT_SUBMISSION_PATH` (e.g. `submission.csv`)

Then run:

```bash
python infer_and_submit_pt.py
```

It will write `submission.csv` with the required format (`sample_id,target`).

## Tools

- **Tune results UI**: see [`tools/tune_viewer/README.md`](tools/tune_viewer/README.md).
- **Packaging for inference**: see [`package_artifacts.py`](package_artifacts.py).

## Notes / constraints

- This repository vendors a large amount of code under `third_party/`. **Do not edit `third_party/`**.
- Dataset files and model weights are not included in this repository; you must provide them locally.

