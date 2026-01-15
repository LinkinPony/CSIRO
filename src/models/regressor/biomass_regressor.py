from __future__ import annotations

from typing import Any, Dict, Optional, List, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from lightning.pytorch import LightningModule
from loguru import logger

from ..backbone import build_feature_extractor
from ..peft_integration import inject_lora_into_feature_extractor
from ..layer_utils import normalize_layer_indices
from src.training.cutmix import CutMixBatchAugment
from src.training.pcgrad import pcgrad_project, pcgrad_project_primary_anchored

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
        # EoMT-style query pooling head settings
        eomt_num_queries: int = 16,
        eomt_num_layers: int = 2,
        eomt_num_heads: int = 8,
        eomt_ffn_dim: int = 2048,
        eomt_query_pool: str = "mean",
        # EoMT pooled feature sources + projection (concat -> proj_dim)
        eomt_use_mean_query: bool = True,
        eomt_use_mean_patch: bool = False,
        eomt_use_cls_token: bool = False,
        eomt_proj_dim: int = 0,
        eomt_proj_activation: str = "relu",
        eomt_proj_dropout: float = 0.0,
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
        # Optional dual-branch fusion for MLP patch-mode:
        # fuse patch-based main prediction with a global prediction from CLS+mean(patch).
        dual_branch_enabled: bool = False,
        # Initial weight for the global branch in fusion:
        #   y = a*y_global + (1-a)*y_patch
        # Stored as a learnable logit parameter when dual_branch_enabled=True.
        dual_branch_alpha_init: float = 0.2,
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
        enable_date: bool = False,
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
        # Gradient surgery (multi-task optimization)
        pcgrad_cfg: Optional[Dict[str, Any]] = None,
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
        elif head_type_norm in ("eomt", "eomt_query", "query_pool", "qpool", "query"):
            head_type_norm = "eomt"
        elif head_type_norm in ("mlp", "linear", "head", ""):
            head_type_norm = "mlp"
        else:
            logger.warning(f"Unknown head_type={head_type_norm!r}, falling back to 'mlp'.")
            head_type_norm = "mlp"
        self._head_type: str = head_type_norm
        self.save_hyperparameters()

        # --- PCGrad (Projected Conflicting Gradients) configuration ---
        self._pcgrad_cfg: Dict[str, Any] = dict(pcgrad_cfg or {})
        self._pcgrad_enabled: bool = bool(self._pcgrad_cfg.get("enabled", False))
        try:
            self._pcgrad_eps: float = float(self._pcgrad_cfg.get("eps", 1.0e-8))
        except Exception:
            self._pcgrad_eps = 1.0e-8
        self._pcgrad_reduction: str = str(self._pcgrad_cfg.get("reduction", "sum") or "sum").lower().strip()
        self._pcgrad_shuffle_tasks: bool = bool(self._pcgrad_cfg.get("shuffle_tasks", True))
        # Mode: symmetric (default) | primary_anchored
        mode_raw = str(self._pcgrad_cfg.get("mode", "symmetric") or "symmetric").lower().strip()
        if mode_raw in ("primary", "anchored", "primary_anchored", "primary-anchored", "primaryanchored"):
            self._pcgrad_mode: str = "primary_anchored"
        elif mode_raw in ("symmetric", "pcgrad", "default"):
            self._pcgrad_mode = "symmetric"
        else:
            self._pcgrad_mode = "symmetric"
        seed_v = self._pcgrad_cfg.get("seed", None)
        try:
            self._pcgrad_seed: Optional[int] = int(seed_v) if seed_v is not None else None
        except Exception:
            self._pcgrad_seed = None
        # Which optimizer param groups to apply PCGrad to (by `group_type` from configure_optimizers).
        apply_to = self._pcgrad_cfg.get("apply_to_group_types", ["head", "lora"])
        self._pcgrad_apply_to_group_types: List[str] = (
            [str(x).lower().strip() for x in (apply_to or [])] if isinstance(apply_to, (list, tuple)) else []
        )
        exclude_groups = self._pcgrad_cfg.get("exclude_group_types", ["uw"])
        self._pcgrad_exclude_group_types: List[str] = (
            [str(x).lower().strip() for x in (exclude_groups or [])] if isinstance(exclude_groups, (list, tuple)) else []
        )
        excl_names = self._pcgrad_cfg.get("exclude_param_name_substrings", [])
        self._pcgrad_exclude_param_name_substrings: List[str] = (
            [str(x) for x in (excl_names or [])] if isinstance(excl_names, (list, tuple)) else []
        )
        # Task filtering (optional): include_tasks has priority over exclude_tasks.
        inc_tasks = self._pcgrad_cfg.get("include_tasks", None)
        exc_tasks = self._pcgrad_cfg.get("exclude_tasks", None)
        self._pcgrad_include_tasks: Optional[List[str]] = (
            [str(x) for x in inc_tasks] if isinstance(inc_tasks, (list, tuple)) else None
        )
        self._pcgrad_exclude_tasks: List[str] = (
            [str(x) for x in exc_tasks] if isinstance(exc_tasks, (list, tuple)) else []
        )
        # Primary/Aux task lists (used when mode=primary_anchored).
        prim = self._pcgrad_cfg.get("primary_tasks", None)
        aux = self._pcgrad_cfg.get("aux_tasks", None)
        self._pcgrad_primary_tasks: Optional[List[str]] = (
            [str(x) for x in prim] if isinstance(prim, (list, tuple)) else None
        )
        self._pcgrad_aux_tasks: Optional[List[str]] = (
            [str(x) for x in aux] if isinstance(aux, (list, tuple)) else None
        )
        # Stashed by training_step for the backward hook (per-step).
        self._pcgrad_terms = None
        self._pcgrad_unscaled_loss = None

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

        # Optional dual-branch fusion is supported only for MLP patch-mode.
        try:
            dual_on = bool(dual_branch_enabled)
        except Exception:
            dual_on = False
        try:
            alpha_init = float(dual_branch_alpha_init)
        except Exception:
            alpha_init = 0.2
        self.dual_branch_enabled: bool = bool(
            dual_on and self._head_type == "mlp" and bool(self.use_patch_reg3)
        )
        # Persist init value for export/debug; the learnable logit is created in head init.
        self.dual_branch_alpha_init: float = float(alpha_init)

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
        self.enable_date = bool(enable_date) and self.mtl_enabled

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
            if self.enable_date:
                task_names.append("date")
            pdict = nn.ParameterDict({name: nn.Parameter(torch.zeros(1)) for name in task_names})
            self._uw_task_params = pdict

        # buffers for epoch-wise metrics on validation set (store predictions in grams)
        self._val_preds: list[Tensor] = []
        self._val_targets: list[Tensor] = []

        # --- CutMix batch-level augmentation ---
        self._cutmix_main = CutMixBatchAugment.from_cfg(cutmix_cfg)
        # --- Manifold mixup on bottleneck representation ---
        self._manifold_mixup = ManifoldMixup.from_cfg(manifold_mixup_cfg)
        # Control where manifold mixup is applied in the forward path.
        #
        # - "features" (default): mix the global feature vectors (CLS/patch-mean concat) or
        #   patch-mean features depending on `mix_cls_token`.
        # - "tokens": mix backbone patch tokens (v186 behavior for multi-layer MLP path),
        #   then downstream code computes global features from the mixed tokens.
        try:
            mm_cfg = dict(manifold_mixup_cfg or {})
            apply_on = str(mm_cfg.get("apply_on", mm_cfg.get("apply", "features")) or "features")
            self._manifold_mixup_apply_on = apply_on.strip().lower()
        except Exception:
            self._manifold_mixup_apply_on = "features"
        try:
            if self._manifold_mixup is not None:
                logger.info(
                    "ManifoldMixup enabled (prob={}, alpha={}, mix_cls_token={}, detach_pair={}, apply_on={})",
                    getattr(self._manifold_mixup, "prob", None),
                    getattr(self._manifold_mixup, "alpha", None),
                    getattr(self._manifold_mixup, "mix_cls_token", None),
                    getattr(self._manifold_mixup, "detach_pair", None),
                    getattr(self, "_manifold_mixup_apply_on", None),
                )
        except Exception:
            # Best-effort only: do not fail model init if logging fails.
            pass

        # --------------------
        # Head-type initialization (mlp/fpn/dpt)
        # --------------------
        # IMPORTANT semantic note:
        # - For ViTDet heads, an explicit empty list `[]` is meaningful and disables the bottleneck MLP
        #   (linear heads on pooled pyramid features).
        # - For other head types, we keep legacy behavior where "unset" (None/[]) falls back to [512, 256].
        ht = str(self._head_type or "mlp").strip().lower()
        if head_hidden_dims is None:
            hidden_dims: List[int] = [512, 256]
        elif ht == "vitdet":
            hidden_dims = list(head_hidden_dims)
        else:
            hidden_dims = list(head_hidden_dims or [512, 256])
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
            # EoMT-style query pooling (optional)
            eomt_num_queries=int(eomt_num_queries),
            eomt_num_layers=int(eomt_num_layers),
            eomt_num_heads=int(eomt_num_heads),
            eomt_ffn_dim=int(eomt_ffn_dim),
            eomt_query_pool=str(eomt_query_pool),
            eomt_use_mean_query=bool(eomt_use_mean_query),
            eomt_use_mean_patch=bool(eomt_use_mean_patch),
            eomt_use_cls_token=bool(eomt_use_cls_token),
            eomt_proj_dim=int(eomt_proj_dim),
            eomt_proj_activation=str(eomt_proj_activation),
            eomt_proj_dropout=float(eomt_proj_dropout or 0.0),
        )


    def _pcgrad_get_optimizer(self) -> Optional[Optimizer]:
        """
        Best-effort access to the (single) optimizer during automatic optimization.
        """
        try:
            tr = getattr(self, "trainer", None)
            opts = getattr(tr, "optimizers", None) if tr is not None else None
            if isinstance(opts, (list, tuple)) and len(opts) >= 1:
                opt0 = opts[0]
                if isinstance(opt0, Optimizer):
                    return opt0
        except Exception:
            return None
        return None

    def _pcgrad_select_params(self, optimizer: Optimizer) -> List[nn.Parameter]:
        """
        Select parameters to apply PCGrad to, based on optimizer param group metadata.

        We intentionally support excluding UW parameters (`group_type: uw`) to avoid
        cross-task gradient injection into log-variance weights.
        """
        include = set(self._pcgrad_apply_to_group_types or [])
        exclude = set(self._pcgrad_exclude_group_types or [])
        params: List[nn.Parameter] = []
        for group in getattr(optimizer, "param_groups", []):
            try:
                gt = str(group.get("group_type", "") or "").lower().strip()
            except Exception:
                gt = ""
            if gt and gt in exclude:
                continue
            if include and (gt not in include):
                continue
            for p in group.get("params", []):
                if isinstance(p, nn.Parameter) and bool(getattr(p, "requires_grad", False)):
                    params.append(p)
        # De-duplicate while preserving order.
        seen: set[int] = set()
        uniq: List[nn.Parameter] = []
        for p in params:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            uniq.append(p)

        # Optional name-based exclusion (substrings).
        if self._pcgrad_exclude_param_name_substrings:
            name_by_id: Dict[int, str] = {}
            try:
                for n, p in self.named_parameters():
                    name_by_id[id(p)] = str(n)
            except Exception:
                name_by_id = {}
            filtered: List[nn.Parameter] = []
            for p in uniq:
                n = name_by_id.get(id(p), "")
                if any(sub in n for sub in self._pcgrad_exclude_param_name_substrings):
                    continue
                filtered.append(p)
            uniq = filtered
        return uniq

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        """
        PCGrad integration point.

        We run a standard backward pass to populate gradients for *all* parameters, then
        replace gradients for a configured subset of parameters with PCGrad-projected
        gradients computed from per-task loss terms stashed in `training_step`.

        This preserves Lightning AMP scaling, gradient clipping, SAM, and gradient accumulation.
        """
        if not bool(getattr(self, "_pcgrad_enabled", False)):
            return super().backward(loss, *args, **kwargs)

        terms_any = getattr(self, "_pcgrad_terms", None)
        unscaled_total = getattr(self, "_pcgrad_unscaled_loss", None)
        if not isinstance(terms_any, list) or len(terms_any) == 0:
            return super().backward(loss, *args, **kwargs)

        # Sanitize terms list: expect [(name, Tensor), ...]
        terms_all: List[tuple[str, Tensor]] = []
        for item in terms_any:
            try:
                name, t = item
            except Exception:
                continue
            if isinstance(t, Tensor):
                terms_all.append((str(name), t))
        if not terms_all:
            return super().backward(loss, *args, **kwargs)

        if not isinstance(unscaled_total, Tensor):
            try:
                unscaled_total = torch.stack([t for _, t in terms_all]).sum()
            except Exception:
                unscaled_total = None
        if not isinstance(unscaled_total, Tensor):
            return super().backward(loss, *args, **kwargs)

        include_set: Optional[set[str]] = (
            set(self._pcgrad_include_tasks) if isinstance(self._pcgrad_include_tasks, list) else None
        )
        exclude_set: set[str] = set(self._pcgrad_exclude_tasks or [])

        mode = str(getattr(self, "_pcgrad_mode", "symmetric") or "symmetric").lower().strip()

        # Split tasks into (primary, aux, excluded) for primary-anchored mode, or (included/excluded) for symmetric.
        terms_primary: List[tuple[str, Tensor]] = []
        terms_aux: List[tuple[str, Tensor]] = []
        terms_included: List[tuple[str, Tensor]] = []
        terms_excluded: List[tuple[str, Tensor]] = []

        if mode == "primary_anchored":
            names_all = [n for n, _ in terms_all]
            # Default primary: reg3 when present, else first task.
            prim_cfg = self._pcgrad_primary_tasks
            if isinstance(prim_cfg, list) and len(prim_cfg) > 0:
                primary_set = set(str(x) for x in prim_cfg)
            else:
                primary_set = {"reg3"} if "reg3" in names_all else ({names_all[0]} if names_all else set())
            aux_cfg = self._pcgrad_aux_tasks
            if isinstance(aux_cfg, list) and len(aux_cfg) > 0:
                aux_set = set(str(x) for x in aux_cfg)
            else:
                aux_set = set(names_all) - set(primary_set)

            # Apply include/exclude filters.
            if include_set is not None:
                primary_set = primary_set & include_set
                aux_set = aux_set & include_set
            primary_set = primary_set - exclude_set
            aux_set = aux_set - exclude_set - primary_set

            for name, t in terms_all:
                if name in primary_set:
                    terms_primary.append((name, t))
                elif name in aux_set:
                    terms_aux.append((name, t))
                else:
                    terms_excluded.append((name, t))

            # Need at least 1 primary and 1 aux to do anything meaningful.
            if len(terms_primary) == 0 or len(terms_aux) == 0:
                return super().backward(loss, *args, **kwargs)
        else:
            # symmetric (legacy)
            for name, t in terms_all:
                if include_set is not None:
                    (terms_included if name in include_set else terms_excluded).append((name, t))
                else:
                    (terms_excluded if name in exclude_set else terms_included).append((name, t))

            if len(terms_included) <= 1:
                return super().backward(loss, *args, **kwargs)

        optimizer = self._pcgrad_get_optimizer()
        if optimizer is None:
            return super().backward(loss, *args, **kwargs)
        pc_params = self._pcgrad_select_params(optimizer)
        if not pc_params:
            return super().backward(loss, *args, **kwargs)

        # Compute AMP scale factor (scaled_loss / unscaled_loss) so per-task grads match `loss.backward()`.
        try:
            scale = (loss.detach() / (unscaled_total.detach() + 1.0e-12)).detach()
        except Exception:
            scale = None
        if scale is None or not torch.isfinite(scale).all():
            return super().backward(loss, *args, **kwargs)

        # Preserve accumulated grads before this backward (for accumulate_grad_batches > 1).
        old_grads: List[Optional[Tensor]] = []
        for p in pc_params:
            g = getattr(p, "grad", None)
            old_grads.append(g.detach().clone() if isinstance(g, Tensor) else None)

        if mode == "primary_anchored":
            # Primary gradient (anchor):
            # - If there is only one primary task, use its gradient directly.
            # - If there are multiple primary tasks, first apply *symmetric PCGrad among primaries*
            #   (treat primaries equally), then use the resulting combined primary gradient as the anchor.
            if len(terms_primary) == 1:
                prim_term = terms_primary[0][1]
                g_list_p = torch.autograd.grad(
                    prim_term * scale,
                    pc_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                primary_grads = [g if isinstance(g, Tensor) else None for g in g_list_p]
            else:
                primary_grads_per_task: List[List[Optional[Tensor]]] = []
                for _name, term in terms_primary:
                    g_list = torch.autograd.grad(
                        term * scale,
                        pc_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    primary_grads_per_task.append([g if isinstance(g, Tensor) else None for g in g_list])
                # NOTE: reduction="sum" so the anchor has the same overall scale as summing primary tasks,
                # while still resolving conflicts between them via PCGrad.
                primary_grads = pcgrad_project(
                    primary_grads_per_task,
                    eps=float(self._pcgrad_eps),
                    reduction="sum",
                    shuffle_tasks=bool(self._pcgrad_shuffle_tasks),
                    seed=self._pcgrad_seed,
                )

            # Aux gradients (per-task).
            aux_grads_per_task: List[List[Optional[Tensor]]] = []
            for _name, term in terms_aux:
                g_list = torch.autograd.grad(
                    term * scale,
                    pc_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                aux_grads_per_task.append([g if isinstance(g, Tensor) else None for g in g_list])

            pcgrad_grads = pcgrad_project_primary_anchored(
                primary_grads=primary_grads,
                aux_grads_per_task=aux_grads_per_task,
                primary_count=int(len(terms_primary)),
                eps=float(self._pcgrad_eps),
                reduction=str(self._pcgrad_reduction),
                shuffle_tasks=bool(self._pcgrad_shuffle_tasks),
                seed=self._pcgrad_seed,
            )
        else:
            # Symmetric PCGrad: per-task gradients for all included tasks.
            grads_per_task = []
            for _name, term in terms_included:
                g_list = torch.autograd.grad(
                    term * scale,
                    pc_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                grads_per_task.append([g if isinstance(g, Tensor) else None for g in g_list])
            pcgrad_grads = pcgrad_project(
                grads_per_task,
                eps=float(self._pcgrad_eps),
                reduction=str(self._pcgrad_reduction),
                shuffle_tasks=bool(self._pcgrad_shuffle_tasks),
                seed=self._pcgrad_seed,
            )

        # Gradients for excluded tasks are kept "as-is" (no surgery) and added back after projection.
        excluded_grads: Optional[List[Optional[Tensor]]] = None
        if terms_excluded:
            try:
                excl_sum = torch.stack([t for _, t in terms_excluded]).sum()
                g_list = torch.autograd.grad(
                    excl_sum * scale,
                    pc_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                excluded_grads = [g if isinstance(g, Tensor) else None for g in g_list]
            except Exception:
                excluded_grads = None

        # Run the default backward to populate grads for all parameters (and trigger AMP/DDP hooks).
        super().backward(loss, *args, **kwargs)

        # Replace gradients for selected parameters: old_accum + excluded + pcgrad(included).
        for idx, p in enumerate(pc_params):
            g_acc = old_grads[idx]
            g_new: Optional[Tensor] = None
            if excluded_grads is not None and isinstance(excluded_grads[idx], Tensor):
                g_new = excluded_grads[idx]
            if isinstance(pcgrad_grads[idx], Tensor):
                g_new = pcgrad_grads[idx] if g_new is None else (g_new + pcgrad_grads[idx])  # type: ignore[operator]
            if g_new is None:
                continue
            if isinstance(g_acc, Tensor):
                g_new = g_acc + g_new
            p.grad = g_new  # type: ignore[assignment]

        # Clear step-local stash to avoid accidental reuse.
        try:
            self._pcgrad_terms = None
            self._pcgrad_unscaled_loss = None
        except Exception:
            pass
