from __future__ import annotations

from typing import Any, Dict, Optional, List, Sequence

import torch
from torch import Tensor, nn
from lightning.pytorch import LightningModule
from loguru import logger

from ..backbone import build_feature_extractor
from ..peft_integration import inject_lora_into_feature_extractor
from ..layer_utils import normalize_layer_indices
from src.training.cutmix import CutMixBatchAugment

from .manifold_mixup import ManifoldMixup
from .forward import RegressorForwardMixin
from .steps import RegressorStepsMixin
from .optim import RegressorOptimMixin
from .checkpointing import RegressorCheckpointingMixin
from .heads import init_head_by_type


class BiomassRegressor(
    RegressorForwardMixin,
    RegressorStepsMixin,
    RegressorOptimMixin,
    RegressorCheckpointingMixin,
    LightningModule,
):
    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int,
        # Head architecture selector
        head_type: str = "mlp",
        # ViTDet (SimpleFeaturePyramid-style) settings
        vitdet_dim: int = 256,
        vitdet_patch_size: int = 16,
        vitdet_scale_factors: Optional[List[float]] = None,
        # FPN (Phase A) settings
        fpn_dim: int = 256,
        fpn_num_levels: int = 3,
        fpn_patch_size: int = 16,
        # When True (default), assign deeper backbone layers (larger indices) to higher-resolution
        # FPN levels (less downsampling). See `FPNHeadConfig.reverse_level_order`.
        fpn_reverse_level_order: bool = True,
        # DPT-style dense prediction head settings
        dpt_features: int = 256,
        dpt_patch_size: int = 16,
        dpt_readout: str = "ignore",
        num_outputs: int = 1,
        dropout: float = 0.0,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        use_output_softplus: bool = True,
        # Whether to include CLS token in the global feature vector.
        # If True:  global feature = [CLS ; mean(patch)] with 2 * embedding_dim channels.
        # If False: global feature = mean(patch) with embedding_dim channels.
        use_cls_token: bool = True,
        # Optional patch-based main regression path (per-patch prediction then average)
        use_patch_reg3: bool = False,
        # Ratio head coupling mode (preferred). When provided, it overrides the legacy
        # boolean flags below. Supported values: "shared" | "separate_mlp" | "separate_spatial".
        ratio_head_mode: Optional[str] = None,
        # If True, use an independent ratio MLP branch (no shared scalar MLP trunk with reg3).
        separate_ratio_head: bool = False,
        # If True, duplicate the entire spatial head for ratio (pyramid/conv + MLP),
        # so ratio does not share any head parameters with reg3.
        separate_ratio_spatial_head: bool = False,
        log_scale_targets: bool = False,
        # Normalization and scaling
        area_m2: float = 1.0,
        reg3_zscore_mean: Optional[List[float]] = None,
        reg3_zscore_std: Optional[List[float]] = None,
        ndvi_zscore_mean: Optional[float] = None,
        ndvi_zscore_std: Optional[float] = None,
        biomass_5d_zscore_mean: Optional[List[float]] = None,
        biomass_5d_zscore_std: Optional[List[float]] = None,
        pretrained: bool = True,
        weights_url: Optional[str] = None,
        weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
        # Cast *frozen* backbone weights to a lower-precision dtype (VRAM saver).
        # NOTE: this is different from Lightning AMP `precision: 16-mixed`, which does
        # not change parameter storage dtype by default.
        backbone_weights_dtype: str = "fp32",
        # Activation / gradient checkpointing for backbone blocks (memory saver; slower backward)
        gradient_checkpointing: bool = False,
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
        # CutMix configs (batch-level augmentation)
        cutmix_cfg: Optional[Dict[str, Any]] = None,
        ndvi_dense_cutmix_cfg: Optional[Dict[str, Any]] = None,
        # Manifold mixup on shared bottleneck representation
        manifold_mixup_cfg: Optional[Dict[str, Any]] = None,
        # Biomass decomposition / ratio head configuration
        enable_ratio_head: bool = True,
        ratio_kl_weight: float = 1.0,
        enable_5d_loss: bool = True,
        loss_5d_weight: float = 1.0,
        biomass_5d_weights: Optional[List[float]] = None,
        # Multi-layer backbone features / layer-wise heads
        use_layerwise_heads: bool = False,
        backbone_layer_indices: Optional[List[int]] = None,
        # When True, each selected backbone layer uses its own bottleneck MLP.
        # When False, all layers share a single bottleneck (legacy behavior).
        use_separate_bottlenecks: bool = True,
        # How to fuse predictions/features across selected backbone layers (when enabled).
        # Options: "mean" (default) | "learned"
        backbone_layers_fusion: str = "mean",
        # Optimizer / SAM configuration
        optimizer_name: Optional[str] = None,
        use_sam: bool = False,
        sam_rho: float = 0.05,
        sam_adaptive: bool = False,
        # --- Debug utilities ---
        # Normalization used for input images (for de-normalizing debug dumps).
        input_image_mean: Optional[Sequence[float]] = None,
        input_image_std: Optional[Sequence[float]] = None,
        # Base run log dir to write debug artifacts under (e.g., outputs/<version>).
        run_log_dir: Optional[str] = None,
        # Debug config to optionally dump final model input images to disk.
        debug_input_dump_cfg: Optional[Dict[str, Any]] = None,
        # AugMix consistency configuration (used when train transform returns multi-view images).
        augmix_consistency_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # Normalize optimizer hyperparameters before saving them.
        if optimizer_name is None:
            optimizer_name = "adamw"
        opt_name = str(optimizer_name).lower()
        if opt_name in ("sam", "sam_adamw", "adamw_sam"):
            use_sam = True
        optimizer_name = opt_name
        # Normalize head_type early for consistent behavior and export meta.
        head_type_norm = str(head_type or "mlp").strip().lower()
        if head_type_norm in ("fpn", "fpn_scalar", "spatial_fpn"):
            head_type_norm = "fpn"
        elif head_type_norm in ("dpt", "dpt_scalar", "dpt_head", "dense_dpt", "dpt_dense"):
            head_type_norm = "dpt"
        elif head_type_norm in ("vitdet", "vitdet_head", "vitdet_scalar", "simple_feature_pyramid"):
            head_type_norm = "vitdet"
        elif head_type_norm in ("mlp", "linear", "head", ""):
            head_type_norm = "mlp"
        else:
            logger.warning(f"Unknown head_type={head_type_norm!r}, falling back to 'mlp'.")
            head_type_norm = "mlp"
        self._head_type: str = head_type_norm
        self.save_hyperparameters()

        # --- Debug image dump configuration ---
        self._run_log_dir: Optional[str] = str(run_log_dir) if run_log_dir not in (None, "", "null") else None
        self._debug_input_dump_cfg: Dict[str, Any] = dict(debug_input_dump_cfg or {})
        self._augmix_consistency_cfg: Dict[str, Any] = dict(augmix_consistency_cfg or {})
        self._input_image_mean: Optional[torch.Tensor] = None
        self._input_image_std: Optional[torch.Tensor] = None
        try:
            if input_image_mean is not None and input_image_std is not None:
                mean = [float(x) for x in list(input_image_mean)]
                std = [float(x) for x in list(input_image_std)]
                if len(mean) == 3 and len(std) == 3 and all(s > 0.0 for s in std):
                    self._input_image_mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
                    self._input_image_std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        except Exception:
            # Best-effort: keep None and fall back to saving normalized tensors (or skip denorm).
            self._input_image_mean = None
            self._input_image_std = None

        feature_extractor = build_feature_extractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
            weights_url=weights_url,
            weights_path=weights_path,
            gradient_checkpointing=bool(gradient_checkpointing),
        )
        # Optionally inject LoRA into the frozen backbone
        if peft_cfg and bool(peft_cfg.get("enabled", False)):
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

        # Optionally cast *frozen* backbone params to fp16/bf16 to save VRAM.
        # Keep trainable params (LoRA adapters when enabled) in fp32 for optimizer stability.
        dtype_key = str(backbone_weights_dtype or "fp32").strip().lower()
        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        cast_dtype = dtype_map.get(dtype_key, torch.float32)
        if cast_dtype != torch.float32:
            try:
                for p in feature_extractor.backbone.parameters():
                    if (not p.requires_grad) and p.is_floating_point():
                        p.data = p.data.to(dtype=cast_dtype)
            except Exception:
                # Best-effort; if casting fails, keep fp32.
                pass
        self.feature_extractor = feature_extractor

        # --------------------
        # Head construction
        # --------------------
        # Keep existing hparams names for checkpoint compatibility.
        self.use_cls_token: bool = bool(use_cls_token)
        self.num_outputs: int = int(num_outputs)
        if self.num_outputs < 1:
            raise ValueError("num_outputs must be >= 1 for reg3 head")

        # Multi-layer backbone configuration is defined below; we need it to build the FPN head.

        # For backward compatibility, keep the flag but interpret it as:
        # - mlp head: same meaning as before
        # - fpn head: always consumes patch tokens (effectively True)
        self.use_patch_reg3: bool = bool(use_patch_reg3) if self._head_type == "mlp" else True
        # Ratio head coupling mode (resolved from enum or legacy flags).
        try:
            from .heads.ratio_mode import resolve_ratio_head_mode, flags_from_ratio_head_mode

            self.ratio_head_mode: str = resolve_ratio_head_mode(
                ratio_head_mode,
                separate_ratio_head=separate_ratio_head,
                separate_ratio_spatial_head=separate_ratio_spatial_head,
            )
            sep_mlp, sep_spatial = flags_from_ratio_head_mode(self.ratio_head_mode)
        except Exception:
            # Safe fallback: keep historical behavior (shared).
            self.ratio_head_mode = "shared"
            sep_mlp, sep_spatial = False, False
        self.separate_ratio_head: bool = bool(sep_mlp)
        self.separate_ratio_spatial_head: bool = bool(sep_spatial)

        # --- Multi-layer backbone configuration ---
        self.use_layerwise_heads: bool = bool(use_layerwise_heads)
        self.backbone_layers_fusion: str = str(backbone_layers_fusion or "mean").strip().lower()
        if self.use_layerwise_heads:
            indices = normalize_layer_indices(backbone_layer_indices or [])
            if len(indices) == 0:
                raise ValueError(
                    "When use_layerwise_heads is True, backbone_layer_indices must be non-empty."
                )
            self.backbone_layer_indices: List[int] = indices
            self.num_layers: int = len(indices)
        else:
            self.backbone_layer_indices = []
            self.num_layers = 0

        # Optional learnable fusion weights for the MLP multi-layer path.
        # For other head types (vitdet/fpn/dpt), multi-layer fusion is implemented inside the head module.
        if (
            self.use_layerwise_heads
            and self._head_type == "mlp"
            and self.num_layers > 0
            and self.backbone_layers_fusion == "learned"
        ):
            # Stored as logits and converted to weights via softmax during fusion.
            self.mlp_layer_logits = nn.Parameter(torch.zeros(self.num_layers, dtype=torch.float32))
        else:
            self.mlp_layer_logits = None  # type: ignore[assignment]

        # Optional per-layer bottlenecks for multi-layer heads
        self.use_separate_bottlenecks: bool = bool(use_separate_bottlenecks)
        self.mtl_enabled: bool = bool(mtl_enabled)
        # Per-task toggles (gated by global MTL flag)
        self.enable_height = bool(enable_height) and self.mtl_enabled
        self.enable_ndvi = bool(enable_ndvi) and self.mtl_enabled
        self.enable_ndvi_dense = bool(enable_ndvi_dense) and self.mtl_enabled
        self.enable_species = bool(enable_species) and self.mtl_enabled
        self.enable_state = bool(enable_state) and self.mtl_enabled

        # Persist auxiliary class counts (needed by head builders).
        if self.enable_species:
            if num_species_classes is None or int(num_species_classes) <= 1:
                raise ValueError(
                    "num_species_classes must be provided (>1) when species task is enabled"
                )
            self.num_species_classes = int(num_species_classes)
        else:
            self.num_species_classes = 0
        if self.enable_state:
            if num_state_classes is None or int(num_state_classes) <= 1:
                raise ValueError(
                    "num_state_classes must be provided (>1) when state task is enabled"
                )
            self.num_state_classes = int(num_state_classes)
        else:
            self.num_state_classes = 0
        # Probability to include NDVI-dense step on each training step (approximate per-epoch ratio)
        if ndvi_dense_prob is None:
            base_prob = 1.0 if self.enable_ndvi_dense else 0.0
        else:
            base_prob = float(ndvi_dense_prob)
        # clamp to [0,1]
        self._ndvi_dense_prob: float = float(max(0.0, min(1.0, base_prob)))

        # Instantiate auxiliary heads (height/ndvi/species/state/ratio and layer-wise heads)
        # is delegated to head-type builders in `src/models/regressor/heads/`.
        # Optional biomass ratio head: predicts proportions of
        # (Dry_Clover_g, Dry_Dead_g, Dry_Green_g) which are later combined with
        # Dry_Total_g for 5D weighted MSE loss.
        self.enable_ratio_head: bool = bool(enable_ratio_head)
        # Weight for the ratio loss (currently MSE in probability domain).
        # Kept argument name `ratio_kl_weight` for backward compatibility with existing checkpoints/configs.
        self.ratio_loss_weight: float = float(max(0.0, ratio_kl_weight))
        self.enable_5d_loss: bool = bool(enable_5d_loss)
        self.loss_5d_weight: float = float(max(0.0, loss_5d_weight))
        self.ratio_components: List[str] = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]
        self.num_ratio_outputs: int = len(self.ratio_components)
        # NOTE: ratio_head is created inside head-type init for MLP only; for FPN/DPT it is None.

        # 5D biomass weights (Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g)
        import math as _math  # local import to avoid polluting global namespace
        default_5d = [0.1, 0.1, 0.1, 0.2, 0.5]
        if biomass_5d_weights is not None and len(biomass_5d_weights) == 5:
            try:
                weights_list = [float(v) for v in biomass_5d_weights]
                if not _math.isfinite(sum(weights_list)):
                    weights_list = default_5d
            except Exception:
                weights_list = default_5d
        else:
            weights_list = default_5d
        self.register_buffer(
            "biomass_5d_weights",
            torch.tensor(weights_list, dtype=torch.float32),
            persistent=False,
        )

        # Log-scale control applies only to main reg3 outputs
        self.log_scale_targets: bool = bool(log_scale_targets)
        # Per-task z-score params
        self._area_m2: float = float(max(1e-12, area_m2))
        self._reg3_mean: Optional[Tensor] = torch.tensor(reg3_zscore_mean, dtype=torch.float32) if reg3_zscore_mean is not None else None
        self._reg3_std: Optional[Tensor] = torch.tensor(reg3_zscore_std, dtype=torch.float32) if reg3_zscore_std is not None else None
        self._ndvi_mean: Optional[float] = float(ndvi_zscore_mean) if ndvi_zscore_mean is not None else None
        self._ndvi_std: Optional[float] = float(ndvi_zscore_std) if ndvi_zscore_std is not None else None
        # Enable reg3 z-score only when stats are present and match expected dimensionality.
        self._use_reg3_zscore: bool = False
        if self._reg3_mean is not None and self._reg3_std is not None:
            try:
                if int(self._reg3_mean.numel()) == int(self.num_outputs) and int(self._reg3_std.numel()) == int(self.num_outputs):
                    self._use_reg3_zscore = True
                else:
                    logger.warning(
                        "Invalid reg3 z-score stats: expected num_outputs={} but got mean_len={}, std_len={}. "
                        "Disabling reg3 z-score normalization.",
                        int(self.num_outputs),
                        int(self._reg3_mean.numel()),
                        int(self._reg3_std.numel()),
                    )
                    self._reg3_mean = None
                    self._reg3_std = None
            except Exception:
                self._use_reg3_zscore = False
        # Optional 5D biomass z-score (g/m^2, possibly log-transformed)
        self._biomass_5d_mean: Optional[Tensor] = (
            torch.tensor(biomass_5d_zscore_mean, dtype=torch.float32)
            if biomass_5d_zscore_mean is not None
            else None
        )
        self._biomass_5d_std: Optional[Tensor] = (
            torch.tensor(biomass_5d_zscore_std, dtype=torch.float32)
            if biomass_5d_zscore_std is not None
            else None
        )
        self._use_biomass_5d_zscore: bool = (
            self._biomass_5d_mean is not None and self._biomass_5d_std is not None
        )
        # Softplus is disabled for reg3 when predicting in log-domain or when using z-score
        self.out_softplus = nn.Softplus() if (use_output_softplus and (not self.log_scale_targets) and (not self._use_reg3_zscore)) else None
        # Optional overrides for UW optimizer hyperparameters
        self._uw_learning_rate: Optional[float] = float(uw_learning_rate) if uw_learning_rate is not None else None
        self._uw_weight_decay: Optional[float] = float(uw_weight_decay) if uw_weight_decay is not None else None

        # Uncertainty Weighting (UW) parameters (optional)
        self.loss_weighting: Optional[str] = (loss_weighting.lower() if loss_weighting else None)
        # Per-task UW parameters (dynamic per batch to avoid size mismatch)
        self._uw_task_params: Optional[nn.ParameterDict] = None
        if self.loss_weighting == "uw":
            # Treat main biomass reg3, ratio loss and 5D loss as three separate UW tasks.
            task_names: List[str] = ["reg3"]
            if self.enable_ratio_head:
                task_names.append("ratio")
            if self.enable_5d_loss and self.enable_ratio_head:
                task_names.append("biomass_5d")
            # Optional: treat AugMix consistency regularization as an independent UW task.
            try:
                cons_cfg = dict(getattr(self, "_augmix_consistency_cfg", {}) or {})
                if bool(cons_cfg.get("enabled", False)) and bool(cons_cfg.get("uw_task", True)):
                    task_names.append("consistency")
            except Exception:
                pass
            # Keep auxiliary tasks for MTL when enabled
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

        # buffers for epoch-wise metrics on validation set (store predictions in grams)
        self._val_preds: list[Tensor] = []
        self._val_targets: list[Tensor] = []

        # --- CutMix batch-level augmentation ---
        self._cutmix_main = CutMixBatchAugment.from_cfg(cutmix_cfg)
        # --- Manifold mixup on bottleneck representation ---
        self._manifold_mixup = ManifoldMixup.from_cfg(manifold_mixup_cfg)

        # --------------------
        # Head-type initialization (mlp/fpn/dpt)
        # --------------------
        hidden_dims: List[int] = list(head_hidden_dims or [512, 256])
        bottleneck_dim = init_head_by_type(
            self,
            head_type=str(self._head_type),
            embedding_dim=int(embedding_dim),
            hidden_dims=hidden_dims,
            head_activation=str(head_activation),
            dropout=float(dropout or 0.0),
            fpn_dim=int(fpn_dim),
            fpn_num_levels=int(fpn_num_levels),
            fpn_patch_size=int(fpn_patch_size),
            fpn_reverse_level_order=bool(fpn_reverse_level_order),
            dpt_features=int(dpt_features),
            dpt_patch_size=int(dpt_patch_size),
            dpt_readout=str(dpt_readout),
            vitdet_dim=int(vitdet_dim),
            vitdet_patch_size=int(vitdet_patch_size),
            vitdet_scale_factors=list(vitdet_scale_factors or [2.0, 1.0, 0.5]),
        )


