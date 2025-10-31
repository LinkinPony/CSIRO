### Training (DINOv3 ViT-L/16 backbone + linear head)

- Install deps:
```bash
pip install -r requirements.txt
```

- Launch training:
```bash
python train.py --config configs/train.yaml
```

Notes:
- Backbone weights are loaded via `torch.hub` from `facebookresearch/dinov3` and kept frozen by default.
- All configurable parameters live in `configs/train.yaml`.
- Data must be under `/home/dl/Git/CSIRO/data` with `train.csv` and images.
