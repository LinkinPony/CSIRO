## Weights Management (Backbone vs Head)

- DINOv3 backbone is fully frozen during training. A single shared file is used for all runs:
  - Path: `dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pt`
  - Purpose: Loaded as the backbone weights across all experiments and inference.

- Regression head weights are saved separately every epoch (small, backbone excluded):
  - Training save path pattern: `outputs/checkpoints/<version>/head/head-epochXXX.pt`
  - Packaged for inference at: `weights/head/infer_head.pt`
  - Contents: `state_dict` for `model.head` and minimal `meta` (embedding_dim, num_outputs, head config).

- Inference requires two inputs (new format):
  1) DINOv3 backbone weights (`dinov3_weights/...pt`)
  2) Regression head weights (`weights/head/infer_head.pt`)

`last.ckpt` continues to be saved unchanged by Lightning for backward compatibility.


