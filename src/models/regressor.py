from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, _LRScheduler
from lightning.pytorch import LightningModule
from loguru import logger

from .backbone import build_feature_extractor
from .peft_integration import inject_lora_into_feature_extractor, get_lora_param_list
from .head_builder import build_head_layer
from src.training.cutmix import CutMixBatchAugment, CMixupBatchAugment


class BiomassRegressor(LightningModule):
    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int,
        num_outputs: int = 3,
        dropout: float = 0.0,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        use_output_softplus: bool = True,
        log_scale_targets: bool = False,
        # Normalization and scaling
        area_m2: float = 1.0,
        reg3_zscore_mean: Optional[List[float]] = None,
        reg3_zscore_std: Optional[List[float]] = None,
        ndvi_zscore_mean: Optional[float] = None,
        ndvi_zscore_std: Optional[float] = None,
        pretrained: bool = True,
        weights_url: Optional[str] = None,
        weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        uw_learning_rate: Optional[float] = None,
        uw_weight_decay: Optional[float] = None,
        scheduler_name: Optional[str] = None,
        scheduler_warmup_epochs: Optional[int] = None,
        scheduler_warmup_start_factor: float = 0.1,
        max_epochs: Optional[int] = None,
        loss_weighting: Optional[str] = None,
        num_species_classes: Optional[int] = None,
        num_state_classes: Optional[int] = None,
        peft_cfg: Optional[Dict[str, Any]] = None,
        mtl_enabled: bool = True,
        ndvi_dense_prob: Optional[float] = None,
        enable_height: bool = False,
        enable_ndvi: bool = False,
        enable_ndvi_dense: bool = False,
        enable_species: bool = False,
        enable_state: bool = False,
        # CutMix / C-Mixup configs (batch-level augmentation)
        cutmix_cfg: Optional[Dict[str, Any]] = None,
        ndvi_dense_cutmix_cfg: Optional[Dict[str, Any]] = None,
        cmixup_cfg: Optional[Dict[str, Any]] = None,
        # MIR (tile-based multi-instance regression) configuration
        mir_enabled: bool = False,
        mir_attn_hidden_dim: int = 256,
        mir_num_heads: int = 1,
        mir_token_normalize: str = "none",
        mir_instance_dropout: float = 0.0,
        mir_tiling_cfg: Optional[Dict[str, Any]] = None,
        mir_stage1_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        feature_extractor = build_feature_extractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
            weights_url=weights_url,
            weights_path=weights_path,
        )
        # Optionally inject LoRA into the frozen backbone
        self._peft_enabled: bool = bool(peft_cfg and peft_cfg.get("enabled", False))
        if self._peft_enabled:
            try:
                # Allow gradient flow for adapters only
                feature_extractor.inference_only = False
            except Exception:
                pass
            feature_extractor, lora_cfg = inject_lora_into_feature_extractor(feature_extractor, peft_cfg)
            # Persist LoRA LR and WD for optimizer grouping
            self._peft_lora_lr: Optional[float] = float(peft_cfg.get("lora_lr", 5e-5))
            self._peft_lora_weight_decay: Optional[float] = float(peft_cfg.get("lora_weight_decay", 0.0))
        else:
            self._peft_lora_lr = None
            self._peft_lora_weight_decay = None
        if not freeze_backbone:
            for parameter in feature_extractor.backbone.parameters():
                parameter.requires_grad = True
            feature_extractor.backbone.train()
        self.feature_extractor = feature_extractor

        # Shared bottleneck: MLP defined by hidden_dims; last hidden dim is bottleneck size
        hidden_dims: List[int] = list(head_hidden_dims or [512, 256])
        act_name = head_activation
        layers: List[nn.Module] = []
        # Backbone feature now returns CLS concat mean(patch) → 2 * embedding_dim
        in_dim = embedding_dim * 2
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        def _act():
            name = (act_name or "").lower()
            if name == "relu":
                return nn.ReLU(inplace=True)
            if name == "gelu":
                return nn.GELU()
            if name in ("silu", "swish"):
                return nn.SiLU(inplace=True)
            return nn.ReLU(inplace=True)
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            layers.append(_act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hd
        self.shared_bottleneck = nn.Sequential(*layers)

        # Task heads
        bottleneck_dim = hidden_dims[-1] if hidden_dims else embedding_dim
        # Main reg3 heads: three independent 1-d regressors (one per biomass target)
        self.num_outputs: int = int(num_outputs)
        if self.num_outputs < 1:
            raise ValueError("num_outputs must be >= 1 for reg3 head")
        self.reg3_heads = nn.ModuleList(
            [nn.Linear(bottleneck_dim, 1) for _ in range(self.num_outputs)]
        )

        self.mtl_enabled: bool = bool(mtl_enabled)
        # Per-task toggles (gated by global MTL flag)
        self.enable_height = bool(enable_height) and self.mtl_enabled
        self.enable_ndvi = bool(enable_ndvi) and self.mtl_enabled
        self.enable_ndvi_dense = bool(enable_ndvi_dense) and self.mtl_enabled
        self.enable_species = bool(enable_species) and self.mtl_enabled
        self.enable_state = bool(enable_state) and self.mtl_enabled
        # Probability to include NDVI-dense step on each training step (approximate per-epoch ratio)
        if ndvi_dense_prob is None:
            base_prob = 1.0 if self.enable_ndvi_dense else 0.0
        else:
            base_prob = float(ndvi_dense_prob)
        # clamp to [0,1]
        self._ndvi_dense_prob: float = float(max(0.0, min(1.0, base_prob)))

        # Instantiate only enabled heads
        self.height_head = nn.Linear(bottleneck_dim, 1) if self.enable_height else None  # type: ignore[assignment]
        self.ndvi_head = nn.Linear(bottleneck_dim, 1) if self.enable_ndvi else None  # type: ignore[assignment]
        if self.enable_species:
            if num_species_classes is None or int(num_species_classes) <= 1:
                raise ValueError("num_species_classes must be provided (>1) when species task is enabled")
            self.num_species_classes = int(num_species_classes)
            self.species_head = nn.Linear(bottleneck_dim, self.num_species_classes)
        else:
            self.num_species_classes = 0
            self.species_head = None
        if self.enable_state:
            if num_state_classes is None or int(num_state_classes) <= 1:
                raise ValueError("num_state_classes must be provided (>1) when state task is enabled")
            self.num_state_classes = int(num_state_classes)
            self.state_head = nn.Linear(bottleneck_dim, self.num_state_classes)
        else:
            self.num_state_classes = 0
            self.state_head = None
        # Log-scale control applies only to main reg3 outputs
        self.log_scale_targets: bool = bool(log_scale_targets)
        # Per-task z-score params
        self._area_m2: float = float(max(1e-12, area_m2))
        self._reg3_mean: Optional[Tensor] = torch.tensor(reg3_zscore_mean, dtype=torch.float32) if reg3_zscore_mean is not None else None
        self._reg3_std: Optional[Tensor] = torch.tensor(reg3_zscore_std, dtype=torch.float32) if reg3_zscore_std is not None else None
        self._ndvi_mean: Optional[float] = float(ndvi_zscore_mean) if ndvi_zscore_mean is not None else None
        self._ndvi_std: Optional[float] = float(ndvi_zscore_std) if ndvi_zscore_std is not None else None
        self._use_reg3_zscore: bool = (self._reg3_mean is not None and self._reg3_std is not None)
        # Softplus is disabled for reg3 when predicting in log-domain or when using z-score
        self.out_softplus = nn.Softplus() if (use_output_softplus and (not self.log_scale_targets) and (not self._use_reg3_zscore)) else None
        # Optional overrides for UW optimizer hyperparameters
        self._uw_learning_rate: Optional[float] = float(uw_learning_rate) if uw_learning_rate is not None else None
        self._uw_weight_decay: Optional[float] = float(uw_weight_decay) if uw_weight_decay is not None else None

        # Uncertainty Weighting (UW) parameters (optional)
        self.loss_weighting: Optional[str] = (loss_weighting.lower() if loss_weighting else None)
        # Per-task UW parameters (dynamic per batch to avoid size mismatch)
        self._uw_task_params: Optional[nn.ParameterDict] = None
        if self.mtl_enabled and self.loss_weighting == "uw":
            task_names: List[str] = ["reg3"]
            if self.enable_height:
                task_names.append("height")
            if self.enable_ndvi:
                task_names.append("ndvi")
            if self.enable_species:
                task_names.append("species")
            if self.enable_state:
                task_names.append("state")
            pdict = nn.ParameterDict({name: nn.Parameter(torch.zeros(1)) for name in task_names})
            self._uw_task_params = pdict

        # buffers for epoch-wise metrics on validation set
        self._val_preds: list[Tensor] = []
        self._val_targets: list[Tensor] = []

        # --- Batch-level augmentations (C-Mixup, CutMix) ---
        self._cmixup_main = CMixupBatchAugment.from_cfg(cmixup_cfg)
        self._cutmix_main = CutMixBatchAugment.from_cfg(cutmix_cfg)

        # --- MIR (tile-based multi-instance regression) configuration ---
        self.mir_enabled: bool = bool(mir_enabled)
        self._mir_stage1_epochs: int = max(0, int(mir_stage1_epochs))
        self._mir_attn_hidden_dim: int = int(mir_attn_hidden_dim)
        self._mir_num_heads: int = max(1, int(mir_num_heads))
        self._mir_token_normalize: str = str(mir_token_normalize or "none").lower()
        self._mir_instance_dropout: float = float(max(0.0, min(1.0, mir_instance_dropout)))

        # Per-dataset tiling config, keyed by dataset_id (e.g., "csiro", "irish_glass_clover").
        # YAML uses [W, H]; here we normalize to dicts with (tile_h, tile_w, stride_h, stride_w, max_tiles).
        self._mir_tiling_cfg: Dict[str, Dict[str, Any]] = {}
        if mir_tiling_cfg:
            try:
                for ds_name, tcfg in mir_tiling_cfg.items():
                    if tcfg is None:
                        continue
                    size = tcfg.get("tile_size", None)
                    if size is None or len(size) != 2:
                        continue
                    stride = tcfg.get("tile_stride", size)
                    if stride is None or len(stride) != 2:
                        stride = size
                    # YAML is [W, H]; convert to (H, W)
                    tile_w, tile_h = int(size[0]), int(size[1])
                    stride_w, stride_h = int(stride[0]), int(stride[1])
                    max_tiles = tcfg.get("max_tiles_per_image", None)
                    max_tiles_int = int(max_tiles) if max_tiles is not None else None
                    if max_tiles_int is not None and max_tiles_int <= 0:
                        max_tiles_int = None
                    self._mir_tiling_cfg[str(ds_name)] = {
                        "tile_h": max(1, tile_h),
                        "tile_w": max(1, tile_w),
                        "stride_h": max(1, stride_h),
                        "stride_w": max(1, stride_w),
                        "max_tiles": max_tiles_int,
                    }
            except Exception:
                # If anything goes wrong, fall back to empty tiling config.
                self._mir_tiling_cfg = {}

        # Attention parameters for MIR pooling (operate on 2*embedding_dim tile features)
        in_dim_mir = embedding_dim * 2
        if self.mir_enabled:
            self._mir_ln = nn.LayerNorm(in_dim_mir)
            self._mir_attn_proj = nn.Linear(in_dim_mir, self._mir_attn_hidden_dim)
            self._mir_attn_out = nn.Linear(self._mir_attn_hidden_dim, self._mir_num_heads)
        else:
            self._mir_ln = None  # type: ignore[assignment]
            self._mir_attn_proj = None  # type: ignore[assignment]
            self._mir_attn_out = None  # type: ignore[assignment]

    def _forward_reg3_logits(self, z: Tensor) -> Tensor:
        """
        Compute main reg3 prediction in normalized domain (g/m^2 or z-score),
        by aggregating three independent scalar heads into a (B, num_outputs) tensor.
        """
        preds: List[Tensor] = []
        for head in self.reg3_heads:
            preds.append(head(z))
        return torch.cat(preds, dim=-1)

    def forward(self, images: Tensor) -> Tensor:
        # Return main 3-d regression prediction in original grams (g)
        if self.mir_enabled:
            features = self._extract_mir_bag_features(images, dataset_ids=None)
        else:
            features = self.feature_extractor(images)
        z = self.shared_bottleneck(features)
        out = self._forward_reg3_logits(z)
        if self.out_softplus is not None:
            out = self.out_softplus(out)
        # Invert normalization and scaling to grams
        out_g = self._invert_reg3_to_grams(out)
        return out_g

    def _invert_reg3_to_g_per_m2(self, vals: Tensor) -> Tensor:
        x = vals
        if self._use_reg3_zscore and self._reg3_mean is not None and self._reg3_std is not None:
            safe_std = torch.clamp(self._reg3_std.to(x.device, dtype=x.dtype), min=1e-8)
            x = x * safe_std + self._reg3_mean.to(x.device, dtype=x.dtype)
        if self.log_scale_targets:
            x = torch.expm1(x).clamp_min(0.0)
        return x

    def _invert_reg3_to_grams(self, vals: Tensor) -> Tensor:
        gm2 = self._invert_reg3_to_g_per_m2(vals)
        return gm2 * float(self._area_m2)

    def _uw_sum(self, named_losses: List[Tuple[str, Tensor]]) -> Tensor:
        if self.loss_weighting != "uw" or self._uw_task_params is None or len(named_losses) == 0:
            return torch.stack([loss for _, loss in named_losses]).mean()
        total = 0.0
        for name, loss in named_losses:
            s = self._uw_task_params.get(name, None)
            if s is None:
                # If this task wasn't registered at init, fall back to equal weighting
                total = total + loss
            else:
                total = total + 0.5 * (torch.exp(-s) * loss + s)
        # Normalize by number of tasks for scale stability
        return total if len(named_losses) == 1 else (total / len(named_losses))

    def _log_uw_parameters(self, stage: str) -> None:
        """
        Log UW parameters per task:
          - {stage}_uw_logvar_{task}: s (log variance)
          - {stage}_uw_sigma_{task}: sigma = exp(0.5 * s)
        """
        if self.loss_weighting != "uw" or self._uw_task_params is None:
            return
        for name, s_param in self._uw_task_params.items():
            try:
                s = s_param.detach()
            except Exception:
                s = s_param
            sigma = torch.exp(0.5 * s)
            # Ensure scalar logging
            self.log(f"{stage}_uw_logvar_{name}", s.squeeze(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f"{stage}_uw_sigma_{name}", sigma.squeeze(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

    # --- MIR / LoRA stage helpers ---
    def _mir_in_stage1(self) -> bool:
        """
        Returns True during the first stage of MIR training (before LoRA fine-tuning).
        Controlled by mir_stage1_epochs in the config.
        """
        if not (self.mir_enabled and self._mir_stage1_epochs > 0):
            return False
        try:
            cur_epoch = int(getattr(self, "current_epoch", 0))
        except Exception:
            cur_epoch = 0
        return cur_epoch < self._mir_stage1_epochs

    def on_train_epoch_start(self) -> None:  # type: ignore[override]
        """
        Two-stage training in a single run:
          - Stage 1 (epoch < mir_stage1_epochs): MIR head is trained with full tiles; LoRA params are frozen.
          - Stage 2 (epoch >= mir_stage1_epochs): LoRA params are unfrozen and trained together with MIR/MLP.
        Also logs the stage and transitions explicitly.
        """
        super().on_train_epoch_start()
        # Determine current MIR stage
        in_stage1 = self._mir_in_stage1()

        # Log stage information (only when MIR is enabled)
        if self.mir_enabled:
            try:
                cur_epoch = int(getattr(self, "current_epoch", 0))
            except Exception:
                cur_epoch = 0
            prev = getattr(self, "_mir_prev_stage1_flag", None)
            if prev is None:
                # First epoch
                if in_stage1:
                    logger.info(
                        f"[MIR] Epoch {cur_epoch}: Stage 1 (frozen LoRA, all tiles, ignore max_tiles_per_image & instance_dropout)."
                    )
                else:
                    logger.info(
                        f"[MIR] Epoch {cur_epoch}: Stage 2 (LoRA trainable, max_tiles_per_image & instance_dropout active if configured)."
                    )
            elif prev and not in_stage1:
                # Stage 1 -> Stage 2 transition
                logger.info(
                    f"[MIR] Switching to Stage 2 at epoch {cur_epoch}: enabling LoRA training and applying max_tiles_per_image / instance_dropout."
                )
            elif (not prev) and in_stage1:
                # Unexpected Stage 2 -> Stage 1 transition (should not normally happen)
                logger.info(
                    f"[MIR] Switched back to Stage 1 at epoch {cur_epoch}: freezing LoRA and using all tiles."
                )
            self._mir_prev_stage1_flag = in_stage1

        # If no LoRA is configured, nothing to freeze/unfreeze
        if not self._peft_enabled:
            return
        # Freeze LoRA parameters in stage 1, unfreeze afterwards.
        try:
            backbone = getattr(self.feature_extractor, "backbone", None)
        except Exception:
            backbone = None
        if backbone is None:
            return
        try:
            lora_params = get_lora_param_list(backbone)
        except Exception:
            lora_params = []
        for p in lora_params:
            p.requires_grad = not in_stage1

    def _get_mir_tiling_for_dataset(self, dataset_id: Any) -> Optional[Dict[str, Any]]:
        if not self.mir_enabled or not self._mir_tiling_cfg:
            return None
        try:
            key = str(dataset_id) if dataset_id is not None else "csiro"
        except Exception:
            key = "csiro"
        cfg = self._mir_tiling_cfg.get(key, None)
        if cfg is None:
            # Fallback to csiro if defined
            cfg = self._mir_tiling_cfg.get("csiro", None)
        return cfg

    def _build_mir_tiles(self, images: Tensor, dataset_ids: Optional[Any]) -> Tuple[Tensor, Tensor]:
        """
        Build tiled crops for MIR.

        Args:
            images: (B, C, H, W)
            dataset_ids: optional sequence/list of per-sample identifiers (e.g., "csiro", "irish_glass_clover").

        Returns:
            tiles: (N_total, C, Th, Tw)
            owners: (N_total,) long tensor mapping each tile to its originating sample index in [0, B-1].
        """
        B, C, H, W = images.shape
        device = images.device
        tiles: list[Tensor] = []
        owners: list[Tensor] = []

        ds_ids_seq: Optional[list[Any]] = None
        if dataset_ids is not None:
            if isinstance(dataset_ids, (list, tuple)):
                ds_ids_seq = list(dataset_ids)
            else:
                # Fallback: wrap scalar / tensor into list with repetition if needed
                try:
                    ds_ids_seq = [dataset_ids for _ in range(B)]
                except Exception:
                    ds_ids_seq = None

        for i in range(B):
            img = images[i : i + 1]  # (1, C, H, W)
            ds_id = None
            if ds_ids_seq is not None and i < len(ds_ids_seq):
                ds_id = ds_ids_seq[i]
            tcfg = self._get_mir_tiling_for_dataset(ds_id)
            if tcfg is None:
                # Fallback: treat full image as a single tile
                tiles.append(img)
                owners.append(torch.full((1,), i, dtype=torch.long, device=device))
                continue

            tile_h = int(tcfg.get("tile_h", H))
            tile_w = int(tcfg.get("tile_w", W))
            stride_h = int(tcfg.get("stride_h", tile_h))
            stride_w = int(tcfg.get("stride_w", tile_w))
            max_tiles = tcfg.get("max_tiles", None)

            tile_h = max(1, min(tile_h, H))
            tile_w = max(1, min(tile_w, W))
            stride_h = max(1, stride_h)
            stride_w = max(1, stride_w)

            positions: list[Tuple[int, int]] = []
            y = 0
            while y + tile_h <= H:
                x = 0
                while x + tile_w <= W:
                    positions.append((y, x))
                    x += stride_w
                y += stride_h

            if not positions:
                # As a safety fallback, use the full image
                tiles.append(img)
                owners.append(torch.full((1,), i, dtype=torch.long, device=device))
                continue

            # Apply max_tiles_per_image only in LoRA-enabled second stage training.
            if (
                max_tiles is not None
                and len(positions) > max_tiles
                and self._peft_enabled
                and self.training
                and (not self._mir_in_stage1())
            ):
                # Randomly sample a subset of tile positions for this image.
                perm = torch.randperm(len(positions), device=device)
                keep_idx = perm[:max_tiles].tolist()
                positions = [positions[j] for j in keep_idx]

            for (yy, xx) in positions:
                tiles.append(img[..., yy : yy + tile_h, xx : xx + tile_w])
                owners.append(torch.full((1,), i, dtype=torch.long, device=device))

        if not tiles:
            # Degenerate fallback: treat each image as a single tile
            owners_tensor = torch.arange(B, dtype=torch.long, device=device)
            return images, owners_tensor

        tiles_tensor = torch.cat(tiles, dim=0)
        owners_tensor = torch.cat(owners, dim=0)
        return tiles_tensor, owners_tensor

    def _mir_pool(self, tile_feats: Tensor, tile_to_image: Tensor) -> Tensor:
        """
        Attention-based MIR pooling over tile features.

        Args:
            tile_feats: (N_total, D) tensor of tile-level features.
            tile_to_image: (N_total,) long tensor mapping each tile to its sample index.

        Returns:
            bag_feats: (B, D) tensor of per-sample aggregated features.
        """
        if tile_feats.numel() == 0 or tile_feats.dim() != 2:
            return tile_feats
        N, D = tile_feats.shape
        B = int(tile_to_image.max().item()) + 1 if tile_to_image.numel() > 0 else 0
        if B <= 0:
            return tile_feats.new_zeros((0, D))

        h = tile_feats
        if self._mir_token_normalize == "layernorm" and self._mir_ln is not None:
            h = self._mir_ln(h)
        elif self._mir_token_normalize == "l2":
            h = F.normalize(h, p=2.0, dim=-1)

        # Projection to attention hidden space and per-head scores
        if self._mir_attn_proj is None or self._mir_attn_out is None:
            return h.view(B, -1, D).mean(dim=1)

        u = torch.tanh(self._mir_attn_proj(h))  # (N, H_dim)
        scores = self._mir_attn_out(u)  # (N, num_heads)

        bag_feats = tile_feats.new_zeros((B, D))
        num_heads = scores.shape[1]

        for b in range(B):
            mask = (tile_to_image == b)
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            # Instance dropout during training: randomly drop some tiles but keep at least one.
            # Only used in LoRA-enabled second stage; MIR pretraining uses all tiles.
            if (
                self.training
                and self._peft_enabled
                and (not self._mir_in_stage1())
                and self._mir_instance_dropout > 0.0
                and idx.numel() > 1
            ):
                keep_mask = torch.rand(idx.shape[0], device=idx.device) > self._mir_instance_dropout
                if not bool(keep_mask.any()):
                    # Ensure at least one tile remains
                    keep_mask[torch.randint(low=0, high=idx.shape[0], size=(1,), device=idx.device)] = True
                idx = idx[keep_mask]

            h_b = h[idx]  # (T_b, D)
            scores_b = scores[idx]  # (T_b, num_heads)

            if h_b.shape[0] == 0:
                continue

            head_outputs: list[Tensor] = []
            for k in range(num_heads):
                attn_raw = scores_b[:, k]  # (T_b,)
                attn = F.softmax(attn_raw, dim=0)
                head_outputs.append(torch.sum(attn.unsqueeze(-1) * h_b, dim=0))

            # Average outputs from all heads to keep dimensionality D
            bag_feats[b] = torch.stack(head_outputs, dim=0).mean(dim=0)

        return bag_feats

    def _extract_mir_bag_features(self, images: Tensor, dataset_ids: Optional[Any]) -> Tensor:
        """
        Construct tile bags and apply MIR pooling to obtain per-image features.
        """
        if not self.mir_enabled:
            return self.feature_extractor(images)
        tiles, owners = self._build_mir_tiles(images, dataset_ids)
        if tiles.size(0) == 0:
            return self.feature_extractor(images)
        tile_feats = self.feature_extractor(tiles)
        bag_feats = self._mir_pool(tile_feats, owners)
        return bag_feats

    def _shared_step(self, batch: Any, stage: str) -> Dict[str, Tensor]:
        # When multiple dataloaders are used, Lightning may deliver a list/tuple of batches
        # For alternating training, process only ONE sub-batch per step to avoid holding multiple graphs.
        if isinstance(batch, (list, tuple)):
            # Flatten potential nested structures
            flat_batches: List[Any] = []
            for sub in batch:
                if isinstance(sub, (list, tuple)):
                    for sb in sub:
                        flat_batches.append(sb)
                else:
                    flat_batches.append(sub)
            # Decide whether to use NDVI-only (from dense dataset) this step (train only)
            use_ndvi_only = False
            if stage == "train" and self.enable_ndvi_dense:
                try:
                    use_ndvi_only = bool(torch.rand(()) < self._ndvi_dense_prob)
                except Exception:
                    use_ndvi_only = False
            # Select one sub-batch according to the decision
            selected: Optional[Any] = None
            if use_ndvi_only:
                selected = next((x for x in flat_batches if isinstance(x, dict) and bool(x.get("ndvi_only", False))), None)
            else:
                selected = next((x for x in flat_batches if isinstance(x, dict) and not bool(x.get("ndvi_only", False))), None)
            # Fallback: pick any dict-like batch if preferred type not found
            if selected is None:
                selected = next((x for x in flat_batches if isinstance(x, dict)), flat_batches[0])
            return self._shared_step(selected, stage)

        # batch is a dict from the dataset
        # Optional C-Mixup then CutMix for main regression tasks (train only)
        if stage == "train":
            if self._cmixup_main is not None:
                try:
                    batch = self._cmixup_main.apply_main_batch(batch)  # type: ignore[assignment]
                except Exception:
                    pass
            if self._cutmix_main is not None:
                try:
                    batch = self._cutmix_main.apply_main_batch(batch)  # type: ignore[assignment]
                except Exception:
                    pass
        images: Tensor = batch["image"]
        is_ndvi_only: bool = bool(batch.get("ndvi_only", False))

        if self.mir_enabled and not is_ndvi_only:
            dataset_ids = batch.get("dataset_id", None)
            features = self._extract_mir_bag_features(images, dataset_ids)
        else:
            features = self.feature_extractor(images)
        z = self.shared_bottleneck(features)
        if is_ndvi_only:
            # NDVI-only batch (no reg3 supervision). Optimize NDVI scalar head only.
            if not self.enable_ndvi or self.ndvi_head is None:
                # If NDVI task is disabled, skip by returning zero loss
                zero = (z.sum() * 0.0)
                self.log(f"{stage}_loss_ndvi", zero, on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"{stage}_loss", zero, on_step=False, on_epoch=True, prog_bar=True)
                return {"loss": zero}
            y_ndvi_only: Tensor = batch["y_ndvi"]  # (B,1)
            # Log NDVI supervision count for this batch / epoch (scalar NDVI-only branch)
            ndvi_count_batch = torch.tensor(
                float(y_ndvi_only.shape[0]),
                device=y_ndvi_only.device,
                dtype=y_ndvi_only.dtype,
            )
            # Per-batch NDVI supervised sample count
            self.log(f"{stage}_ndvi_batch_count", ndvi_count_batch, on_step=True, on_epoch=False, prog_bar=False)
            # Per-epoch total NDVI supervised sample count
            self.log(
                f"{stage}_ndvi_epoch_count",
                ndvi_count_batch,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                reduce_fx="sum",
            )
            pred_ndvi_only = self.ndvi_head(z)
            loss_ndvi_only = F.mse_loss(pred_ndvi_only, y_ndvi_only)
            self.log(f"{stage}_loss_ndvi", loss_ndvi_only, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse_ndvi", loss_ndvi_only, on_step=False, on_epoch=True, prog_bar=False)
            mae_ndvi_only = F.l1_loss(pred_ndvi_only, y_ndvi_only)
            self.log(f"{stage}_mae_ndvi", mae_ndvi_only, on_step=False, on_epoch=True, prog_bar=False)
            total_ndvi = self._uw_sum([("ndvi", loss_ndvi_only)])
            self.log(f"{stage}_loss", total_ndvi, on_step=False, on_epoch=True, prog_bar=True)
            return {"loss": total_ndvi}

        # y_reg3 provided by dataset is already in normalized domain (z-score on g/m^2 when enabled)
        y_reg3: Tensor = batch["y_reg3"]  # (B,3)
        reg3_mask: Optional[Tensor] = batch.get("reg3_mask", None)
        y_height: Tensor = batch["y_height"]  # (B,1)
        y_ndvi: Tensor = batch["y_ndvi"]  # (B,1)
        y_species: Tensor = batch["y_species"]  # (B,)
        y_state: Tensor = batch["y_state"]  # (B,)

        pred_reg3 = self._forward_reg3_logits(z)
        if self.out_softplus is not None:
            pred_reg3 = self.out_softplus(pred_reg3)

        # Optional per-dimension supervision mask for reg3 (to support partial targets)
        if reg3_mask is not None:
            mask = reg3_mask.to(device=y_reg3.device, dtype=y_reg3.dtype)
        else:
            mask = torch.ones_like(y_reg3, dtype=y_reg3.dtype, device=y_reg3.device)
        diff_reg3 = pred_reg3 - y_reg3
        diff2_reg3 = (diff_reg3 * diff_reg3) * mask
        mask_sum_reg3 = mask.sum().clamp_min(1.0)
        # If MTL is disabled, optimize only the main regression task
        loss_reg3 = diff2_reg3.sum() / mask_sum_reg3
        if (not self.mtl_enabled) or (self.enable_height is False and self.enable_ndvi is False and self.enable_species is False and self.enable_state is False):
            self.log(f"{stage}_loss_reg3", loss_reg3, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse_reg3", loss_reg3, on_step=False, on_epoch=True, prog_bar=False)
            mae_reg3 = (diff_reg3.abs() * mask).sum() / mask_sum_reg3
            self.log(f"{stage}_mae_reg3", mae_reg3, on_step=False, on_epoch=True, prog_bar=False)
            with torch.no_grad():
                per_dim_den = mask.sum(dim=0).clamp_min(1.0)
                per_dim_mse = diff2_reg3.sum(dim=0) / per_dim_den
                for i in range(per_dim_mse.shape[0]):
                    self.log(f"{stage}_mse_reg3_{i}", per_dim_mse[i], on_step=False, on_epoch=True, prog_bar=False)

            total_loss = loss_reg3
            self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_mae", mae_reg3, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse", loss_reg3, on_step=False, on_epoch=True, prog_bar=False)
            # For external metrics (e.g., epoch-end R^2), return original-scale grams
            preds_out = self._invert_reg3_to_grams(pred_reg3.detach())
            targets_out = batch.get("y_reg3_g", None)
            if targets_out is None:
                # Fallback: invert from g/m^2 stored if present
                y_gm2 = batch.get("y_reg3_g_m2", None)
                if y_gm2 is not None:
                    targets_out = y_gm2 * float(self._area_m2)
                else:
                    # Last resort: invert from normalized y (approximate)
                    targets_out = self._invert_reg3_to_grams(y_reg3.detach())
            return {
                "loss": total_loss,
                "mae": mae_reg3,
                "mse": loss_reg3,
                "preds": preds_out,
                "targets": targets_out,
            }

        # Otherwise, compute enabled auxiliary task heads and losses
        pred_height = self.height_head(z) if self.enable_height else None  # type: ignore[assignment]
        pred_ndvi = self.ndvi_head(z) if self.enable_ndvi else None  # type: ignore[assignment]
        logits_species = self.species_head(z) if self.enable_species else None  # type: ignore[assignment]
        logits_state = self.state_head(z) if self.enable_state else None  # type: ignore[assignment]

        # Always log reg3
        self.log(f"{stage}_loss_reg3", loss_reg3, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse_reg3", loss_reg3, on_step=False, on_epoch=True, prog_bar=False)

        # Collect losses in consistent order for UW/equal weighting
        named_losses: List[Tuple[str, Tensor]] = [("reg3", loss_reg3)]
        mae_reg3 = (diff_reg3.abs() * mask).sum() / mask_sum_reg3
        self.log(f"{stage}_mae_reg3", mae_reg3, on_step=False, on_epoch=True, prog_bar=False)
        with torch.no_grad():
            per_dim_den = mask.sum(dim=0).clamp_min(1.0)
            per_dim_mse = diff2_reg3.sum(dim=0) / per_dim_den
            for i in range(per_dim_mse.shape[0]):
                self.log(f"{stage}_mse_reg3_{i}", per_dim_mse[i], on_step=False, on_epoch=True, prog_bar=False)

        if self.enable_height:
            loss_height = F.mse_loss(pred_height, y_height)  # type: ignore[arg-type]
            self.log(f"{stage}_loss_height", loss_height, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse_height", loss_height, on_step=False, on_epoch=True, prog_bar=False)
            mae_height = F.l1_loss(pred_height, y_height)  # type: ignore[arg-type]
            self.log(f"{stage}_mae_height", mae_height, on_step=False, on_epoch=True, prog_bar=False)
            named_losses.append(("height", loss_height))
        if self.enable_ndvi:
            ndvi_mask: Optional[Tensor] = batch.get("ndvi_mask", None)
            if ndvi_mask is not None:
                m_nd = ndvi_mask.to(device=y_ndvi.device, dtype=y_ndvi.dtype)  # type: ignore[arg-type]
            else:
                m_nd = torch.ones_like(y_ndvi, dtype=y_ndvi.dtype, device=y_ndvi.device)  # type: ignore[arg-type]

            diff_ndvi = pred_ndvi - y_ndvi  # type: ignore[operator]
            diff2_ndvi = (diff_ndvi * diff_ndvi) * m_nd
            mask_sum_ndvi = m_nd.sum().clamp_min(0.0)

            # Log NDVI supervision count for this batch / epoch (main multi-task branch)
            # mask_sum_ndvi is the number of supervised NDVI positions (typically equals batch size).
            self.log(f"{stage}_ndvi_batch_count", mask_sum_ndvi, on_step=True, on_epoch=False, prog_bar=False)
            self.log(
                f"{stage}_ndvi_epoch_count",
                mask_sum_ndvi,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                reduce_fx="sum",
            )

            if mask_sum_ndvi > 0:
                loss_ndvi = diff2_ndvi.sum() / mask_sum_ndvi
                mae_ndvi = (diff_ndvi.abs() * m_nd).sum() / mask_sum_ndvi
            else:
                # No valid NDVI supervision in this batch; use zero loss to keep graph consistent.
                zero = diff2_ndvi.sum() * 0.0
                loss_ndvi = zero
                mae_ndvi = zero

            self.log(f"{stage}_loss_ndvi", loss_ndvi, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse_ndvi", loss_ndvi, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mae_ndvi", mae_ndvi, on_step=False, on_epoch=True, prog_bar=False)
            named_losses.append(("ndvi", loss_ndvi))
        if self.enable_species and logits_species is not None:
            loss_species = F.cross_entropy(logits_species, y_species)
            self.log(f"{stage}_loss_species", loss_species, on_step=False, on_epoch=True, prog_bar=False)
            with torch.no_grad():
                acc_species = (logits_species.argmax(dim=-1) == y_species).float().mean()
            self.log(f"{stage}_acc_species", acc_species, on_step=False, on_epoch=True, prog_bar=False)
            named_losses.append(("species", loss_species))
        if self.enable_state and logits_state is not None:
            loss_state = F.cross_entropy(logits_state, y_state)
            self.log(f"{stage}_loss_state", loss_state, on_step=False, on_epoch=True, prog_bar=False)
            with torch.no_grad():
                acc_state = (logits_state.argmax(dim=-1) == y_state).float().mean()
            self.log(f"{stage}_acc_state", acc_state, on_step=False, on_epoch=True, prog_bar=False)
            named_losses.append(("state", loss_state))

        total_loss = self._uw_sum(named_losses)

        # overall metrics for backward-compat
        mae = F.l1_loss(pred_reg3, y_reg3)
        mse = F.mse_loss(pred_reg3, y_reg3)
        self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse", mse, on_step=False, on_epoch=True, prog_bar=False)
        # For external validation epoch R^2, return original-scale grams
        preds_out = self._invert_reg3_to_grams(pred_reg3.detach())
        targets_out = batch.get("y_reg3_g", None)
        if targets_out is None:
            y_gm2 = batch.get("y_reg3_g_m2", None)
            if y_gm2 is not None:
                targets_out = y_gm2 * float(self._area_m2)
            else:
                targets_out = self._invert_reg3_to_grams(y_reg3.detach())
        return {
            "loss": total_loss,
            "mae": mae,
            "mse": mse,
            "preds": preds_out,
            "targets": targets_out,
        }

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return self._shared_step(batch, stage="train")["loss"]

    # Guard optimizer stepping to avoid AMP GradScaler assertion when no grads were produced
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_closure,
        **kwargs: Any,
    ) -> None:
        # Run closure to compute backward
        closure_out = optimizer_closure()
        # Check if any parameter has gradients; if not, skip stepping
        has_grad = False
        for group in optimizer.param_groups:
            for p in group.get("params", []):
                if getattr(p, "grad", None) is not None:
                    has_grad = True
                    break
            if has_grad:
                break
        if not has_grad:
            # No gradients this step (e.g., auxiliary-only step got skipped); avoid scaler.step assertion
            return
        # Proceed with default stepping
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad(set_to_none=True)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        out = self._shared_step(batch, stage="val")
        # Only aggregate main regression predictions for val_r2
        if "preds" in out and "targets" in out:
            self._val_preds.append(out["preds"].detach().float().cpu())
            self._val_targets.append(out["targets"].detach().float().cpu())

    def on_validation_epoch_start(self) -> None:
        self._val_preds.clear()
        self._val_targets.clear()

    def on_validation_epoch_end(self) -> None:
        # Always log UW parameters for validation epoch
        self._log_uw_parameters("val")
        if len(self._val_preds) == 0:
            return
        preds = torch.cat(self._val_preds, dim=0)
        targets = torch.cat(self._val_targets, dim=0)
        # overall R^2 across all outputs
        ss_res = torch.sum((targets - preds) ** 2)
        mean_t = torch.mean(targets)
        ss_tot = torch.sum((targets - mean_t) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        self.log("val_r2", r2, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        # Log UW parameters for training epoch
        self._log_uw_parameters("train")

    def configure_optimizers(self):
        # Separate parameter groups: LoRA adapters (smaller LR) and the rest (head, optionally others)
        lora_params: List[torch.nn.Parameter] = []
        try:
            lora_params = get_lora_param_list(self.feature_extractor.backbone)
        except Exception:
            lora_params = []

        all_params = [p for p in self.parameters() if p.requires_grad]
        lora_set = set(lora_params)
        # Uncertainty weighting parameters (separate group if present)
        uw_params: List[torch.nn.Parameter] = []
        try:
            if self._uw_task_params is not None:
                uw_params = [p for p in self._uw_task_params.parameters() if p.requires_grad]
        except Exception:
            uw_params = []
        uw_set = set(uw_params)
        other_params = [p for p in all_params if p not in lora_set and p not in uw_set]

        param_groups: List[Dict[str, Any]] = []
        if len(other_params) > 0:
            param_groups.append({
                "params": other_params,
                "lr": self.hparams.learning_rate,
                "weight_decay": self.hparams.weight_decay,
            })
        if len(uw_params) > 0:
            uw_lr = float(self._uw_learning_rate) if self._uw_learning_rate is not None else float(self.hparams.learning_rate)
            uw_wd = float(self._uw_weight_decay) if self._uw_weight_decay is not None else float(self.hparams.weight_decay)
            param_groups.append({
                "params": uw_params,
                "lr": uw_lr,
                "weight_decay": uw_wd,
            })
        if len(lora_params) > 0:
            lora_lr = float(self._peft_lora_lr or (self.hparams.learning_rate * 0.1))
            lora_wd = float(self._peft_lora_weight_decay if self._peft_lora_weight_decay is not None else self.hparams.weight_decay)
            param_groups.append({
                "params": lora_params,
                "lr": lora_lr,
                "weight_decay": lora_wd,
            })

        optimizer: Optimizer = AdamW(param_groups)

        if self.hparams.scheduler_name and self.hparams.scheduler_name.lower() == "cosine":
            max_epochs: int = int(self.hparams.max_epochs or 10)
            warmup_epochs: int = int(getattr(self.hparams, "scheduler_warmup_epochs", 0) or 0)
            start_factor: float = float(getattr(self.hparams, "scheduler_warmup_start_factor", 0.1))

            if warmup_epochs > 0:
                # Linear warmup for the first N epochs, then cosine annealing
                warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
                cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs))
                scheduler: _LRScheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }
        return optimizer

    # --- Lightweight checkpointing: exclude frozen backbone weights ---
    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):  # type: ignore[override]
        # Get the full state dict from the parent
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # If the backbone is frozen, strip its heavy weights while keeping LoRA params
        try:
            freeze_backbone: bool = bool(getattr(self.hparams, "freeze_backbone", True))
        except Exception:
            freeze_backbone = True
        if not freeze_backbone:
            return full
        filtered = type(full)()
        for key, value in full.items():
            # Keep LoRA adapter parameters inside the backbone
            if key.startswith(f"{prefix}feature_extractor.backbone"):
                if ("lora_" in key) or ("lora_magnitude_vector" in key):
                    filtered[key] = value
                # else: drop heavy backbone weights
                continue
            # Keep everything else (heads, bottleneck, buffers, optimizer hooks, etc.)
            filtered[key] = value
        return filtered

    def load_state_dict(self, state_dict: dict, strict: bool = True):  # type: ignore[override]
        """
        Load with tolerance for missing backbone weights (since we exclude them from ckpt).
        """
        try:
            # Always allow missing/unexpected keys to avoid failures due to stripped backbone
            return super().load_state_dict(state_dict, strict=False)
        except Exception:
            # Fallback to default behavior if anything goes wrong
            return super().load_state_dict(state_dict, strict=strict)


