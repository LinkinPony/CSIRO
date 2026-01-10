### Training (DINOv3 ViT-L/16 backbone + linear head)

- Install deps:
```bash
pip install -r requirements.txt
```

- Launch training (legacy single YAML):
```bash
python train.py --config configs/train.yaml
```

Notes:
- Backbone weights are loaded via `torch.hub` from `facebookresearch/dinov3` and kept frozen by default.
- All configurable parameters live in `configs/train.yaml`.
- Data must be under `/home/dl/Git/CSIRO/data` with `train.csv` and images.

---

### Training with Hydra (recommended)

This repo now supports a Hydra-based, modular config system under `conf/`.

- **Default run** (equivalent to the repo's baseline `conf/experiment/default.yaml`):

```bash
python train_hydra.py
```

- **Override config values from CLI** (examples):

```bash
# Change version/output subdir name
python train_hydra.py version=my_run

# Change basic trainer budget
python train_hydra.py trainer.max_epochs=10 trainer.limit_train_batches=200

# Change an optimizer hyperparameter
python train_hydra.py optimizer.lr=5e-4 optimizer.weight_decay=0.01
```

Hydra config layout:
- `conf/train.yaml`: Hydra entry config (keeps cwd unchanged).
- `conf/experiment/default.yaml`: chooses the default config groups.
- `conf/data/*`, `conf/model/*`, ...: modular groups for each top-level section.

---

### Hyperparameter search with Ray Tune (2 machines, 1×RTX4090 each)

Cluster assumptions for this repo:
- **Ray head IP**: `192.168.10.14` (this machine)
- **Ray worker IP**: `192.168.199.241`
- **Shared storage**: NFS (recommended) mounted at the same path on both nodes (example: `/mnt/csiro_nfs`)
- **Training data**: keep on local SSD on each node (don’t put the dataset on NFS)

#### 0) (Recommended) Setup NFS for shared Tune results

On the **head** node (`192.168.10.14`):

```bash
sudo apt update
sudo apt install -y nfs-kernel-server

sudo mkdir -p /srv/nfs/csiro/ray_results
sudo chown -R $USER:$USER /srv/nfs/csiro

# Export to the worker node (edit /etc/exports and add the following line)
echo "/srv/nfs/csiro 192.168.199.241(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports

sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

On the **worker** node (`192.168.199.241`):

```bash
sudo apt update
sudo apt install -y nfs-common

sudo mkdir -p /mnt/csiro_nfs
sudo mount -t nfs4 192.168.10.14:/srv/nfs/csiro /mnt/csiro_nfs -o vers=4,proto=tcp

# Optional: auto-mount on boot
echo "192.168.10.14:/srv/nfs/csiro /mnt/csiro_nfs nfs4 defaults,_netdev,vers=4,proto=tcp 0 0" | sudo tee -a /etc/fstab
```

#### 1) Start a Ray cluster

On the **head** node:

```bash
ray start --head \
  --node-ip-address=192.168.10.14 \
  --port=6379 \
  --dashboard-host=0.0.0.0 --dashboard-port=8265 \
  --num-gpus=1 \
  --min-worker-port=10000 --max-worker-port=10100
```

On the **worker** node:

```bash
ray start \
  --address='192.168.10.14:6379' \
  --node-ip-address=192.168.199.241 \
  --num-gpus=1 \
  --min-worker-port=10000 --max-worker-port=10100
```

#### 2) Run Tune from the head node

Tune uses `conf/tune.yaml` (it composes the training config at the root and defines the search space under `tune.search_space`).

```bash
# Store Tune results on NFS so both nodes write to the same filesystem
python tune.py tune.storage_path=/mnt/csiro_nfs/ray_results ray.address=auto
```

Alternatively, use the convenience wrapper:

```bash
# Start a new run with an explicit name (recommended for versioning)
./tune.sh tune-csiro-20260109_1500 tune.storage_path=/mnt/csiro_nfs/ray_results ray.address=auto
```

Common overrides:

```bash
# More samples
python tune.py tune.num_samples=50

# Change per-trial budget
python tune.py trainer.max_epochs=10 trainer.limit_train_batches=200

# Multi-seed evaluation (recommended after HPO; use FIFO and disable per-epoch ASHA reporting)
python tune.py tune.seeds='[42,43,44]' tune.scheduler.type=fifo tune.report_per_epoch=false

# Dashboard (head node)
# Open: `http://192.168.10.14:8265`
```

Resume after interruption:

```bash
# Re-run after interruption: it will auto-resume by default (keep the same tune.name and search space)
./tune.sh tune-csiro-20260109_1500 tune.storage_path=/mnt/csiro_nfs/ray_results ray.address=auto

# Force a fresh run (do NOT restore)
RESUME=0 ./tune.sh tune-csiro-20260109_1500 tune.storage_path=/mnt/csiro_nfs/ray_results ray.address=auto
```

Outputs:
- Ray results go under `tune.storage_path` (default: `ray_results/`).
- The script writes a convenience resolved config for the best result to:
  `best_train_cfg.yaml` inside the experiment directory on persistent storage
  (typically: `ray_results/<tune.name>/best_train_cfg.yaml`).
  You can feed it into `python train.py --config ...` for a final full training run.

Export best-so-far **during** a running Tune (no need to wait for `tune.py` to finish):
```bash
# Export "best so far" by scanning per-trial progress.csv/params.json under the experiment dir.
# Tip: set --min-epoch to match ASHA grace_period (e.g. 15) to avoid picking very early noisy results.
python tools/export_best_tune_cfg.py \
  --config-name tune_vitdet \
  --exp-dir /mnt/csiro_nfs/ray_results/tune-vitdet-v1-12h \
  --scope best \
  --min-epoch 15

# Or export the config for a specific trial directory (useful when you spot a good run in ray_results/).
python tools/export_best_tune_cfg.py \
  --config-name tune_vitdet \
  --trial-dir /mnt/csiro_nfs/ray_results/tune-vitdet-v1-12h/<trial_dir_name>
```

Versioning guidance:
- Use a unique `tune.name` per search run (include date/time and optionally git commit).
- Do **not** change the search space (or code) and then try to resume the old run. Start a new run instead (new `tune.name`).


python tools/nano_banana_pro/augment_train.py --config configs/nano_banana_pro_augment.yaml --limit 20