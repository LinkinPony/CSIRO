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
from .head_builder import build_head_layer, SwiGLU
from .layer_utils import average_layerwise_predictions, normalize_layer_indices
from src.training.cutmix import CutMixBatchAugment
from src.training.sam import SAM


class ManifoldMixup:
    """
    Feature-level (\"manifold\") mixup applied on the shared bottleneck representation z,
    together with consistent mixing of regression and 5D/ratio targets.
    """

    def __init__(self, *, enabled: bool, prob: float, alpha: float) -> None:
        self.enabled: bool = bool(enabled)
        self.prob: float = float(prob)
        self.alpha: float = float(alpha)
        # Previous-sample cache to support batch_size == 1 manifold mixup
        self._prev: Optional[Dict[str, Tensor]] = None

    @staticmethod
    def from_cfg(cfg: Optional[Dict[str, Any]]) -> Optional["ManifoldMixup"]:
        if cfg is None:
            return None
        enabled = bool(cfg.get("enabled", False))
        prob = float(cfg.get("prob", 0.0))
        alpha = float(cfg.get("alpha", 1.0))
        if (not enabled) or prob <= 0.0:
            return None
        return ManifoldMixup(enabled=enabled, prob=prob, alpha=alpha)

    def apply(self, z: Tensor, batch: Dict[str, Tensor], *, force: bool = False) -> Tuple[Tensor, Dict[str, Tensor], bool]:
        """
        Apply manifold mixup on feature tensor z and relevant regression/5D/ratio targets.

        Returns:
            z_mixed, batch_mixed, applied_flag
        """
        if not self.enabled:
            return z, batch, False
        if z.dim() < 2 or z.size(0) <= 0:
            return z, batch, False
        if not force:
            if self.prob <= 0.0:
                return z, batch, False
            if torch.rand(()) > self.prob:
                return z, batch, False

        bsz = z.size(0)
        lam = 1.0
        if self.alpha > 0.0:
            lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample().item())

        # Case 1: standard in-batch mixup for batch_size >= 2 (unchanged behavior).
        if bsz >= 2:
            perm = torch.randperm(bsz, device=z.device)
            z_mixed = lam * z + (1.0 - lam) * z[perm]

            # Mix main scalar regression targets (already in normalized space when applicable)
            for key in ("y_reg3", "y_height", "y_ndvi"):
                if key in batch:
                    y = batch[key]
                    if isinstance(y, torch.Tensor) and y.dim() >= 1 and y.size(0) == bsz:
                        batch[key] = lam * y + (1.0 - lam) * y[perm]

            # Mix 5D biomass grams and recompute ratio targets from the mixed grams.
            if "y_biomass_5d_g" in batch:
                y_5d = batch["y_biomass_5d_g"]
                if (
                    isinstance(y_5d, torch.Tensor)
                    and y_5d.dim() == 2
                    and y_5d.size(0) == bsz
                    and y_5d.size(1) >= 5
                ):
                    y_5d_perm = y_5d[perm]
                    mixed_5d = lam * y_5d + (1.0 - lam) * y_5d_perm
                    batch["y_biomass_5d_g"] = mixed_5d

                    # Mix 5D masks conservatively (only keep supervision where both were valid)
                    mask_5d = batch.get("biomass_5d_mask", None)
                    if isinstance(mask_5d, torch.Tensor) and mask_5d.dim() == mixed_5d.dim():
                        mask_5d_perm = mask_5d[perm]
                        batch["biomass_5d_mask"] = mask_5d * mask_5d_perm

                    # Recompute ratio labels from mixed grams to keep physical consistency
                    mixed_total = mixed_5d[:, 4].clamp(min=1e-6)
                    if "y_ratio" in batch:
                        batch["y_ratio"] = torch.stack(
                            [
                                mixed_5d[:, 0] / mixed_total,
                                mixed_5d[:, 1] / mixed_total,
                                mixed_5d[:, 2] / mixed_total,
                            ],
                            dim=-1,
                        )
                    if "ratio_mask" in batch:
                        ratio_mask = batch["ratio_mask"]
                        if isinstance(ratio_mask, torch.Tensor) and ratio_mask.size(0) == bsz:
                            batch["ratio_mask"] = ratio_mask * ratio_mask[perm]

            return z_mixed, batch, True

        # Case 2: batch_size == 1. Use previous cached sample to perform mixing, similar to CutMix.
        # Cache current sample (detached) for potential use on the next step.
        current: Dict[str, Tensor] = {
            "z": z.detach().clone(),
        }
        for key in ("y_reg3", "y_height", "y_ndvi", "y_biomass_5d_g", "biomass_5d_mask", "y_ratio", "ratio_mask"):
            val = batch.get(key, None)
            if isinstance(val, torch.Tensor):
                current[key] = val.detach().clone()

        # If no previous sample or incompatible shape, only update cache and skip mixing this time.
        if self._prev is None or "z" not in self._prev or self._prev["z"].shape != z.shape:
            self._prev = current
            return z, batch, False

        prev = self._prev
        z_prev = prev["z"].to(z.device, dtype=z.dtype)
        z_mixed = lam * z + (1.0 - lam) * z_prev

        # Mix main scalar regression targets with the cached sample.
        for key in ("y_reg3", "y_height", "y_ndvi"):
            if key in batch and key in prev:
                y = batch[key]
                y_prev = prev[key].to(y.device, dtype=y.dtype)
                if (
                    isinstance(y, torch.Tensor)
                    and isinstance(y_prev, torch.Tensor)
                    and y.dim() >= 1
                    and y_prev.shape == y.shape
                ):
                    batch[key] = lam * y + (1.0 - lam) * y_prev

        # Mix 5D biomass grams and recompute ratio targets using the cached sample.
        if "y_biomass_5d_g" in batch and "y_biomass_5d_g" in prev:
            y_5d = batch["y_biomass_5d_g"]
            y_5d_prev = prev["y_biomass_5d_g"].to(y_5d.device, dtype=y_5d.dtype)
            if (
                isinstance(y_5d, torch.Tensor)
                and isinstance(y_5d_prev, torch.Tensor)
                and y_5d.dim() == 2
                and y_5d_prev.shape == y_5d.shape
                and y_5d.size(1) >= 5
            ):
                mixed_5d = lam * y_5d + (1.0 - lam) * y_5d_prev
                batch["y_biomass_5d_g"] = mixed_5d

                mask_5d = batch.get("biomass_5d_mask", None)
                mask_5d_prev = prev.get("biomass_5d_mask", None)
                if (
                    isinstance(mask_5d, torch.Tensor)
                    and isinstance(mask_5d_prev, torch.Tensor)
                    and mask_5d.shape == mixed_5d.shape
                ):
                    batch["biomass_5d_mask"] = mask_5d * mask_5d_prev.to(
                        mask_5d.device, dtype=mask_5d.dtype
                    )

                # Recompute ratio labels from mixed grams to keep physical consistency
                mixed_total = mixed_5d[:, 4].clamp(min=1e-6)
                if "y_ratio" in batch:
                    batch["y_ratio"] = torch.stack(
                        [
                            mixed_5d[:, 0] / mixed_total,
                            mixed_5d[:, 1] / mixed_total,
                            mixed_5d[:, 2] / mixed_total,
                        ],
                        dim=-1,
                    )
                if "ratio_mask" in batch and "ratio_mask" in prev:
                    ratio_mask = batch["ratio_mask"]
                    ratio_mask_prev = prev["ratio_mask"].to(
                        ratio_mask.device, dtype=ratio_mask.dtype
                    )
                    if (
                        isinstance(ratio_mask, torch.Tensor)
                        and isinstance(ratio_mask_prev, torch.Tensor)
                        and ratio_mask_prev.shape == ratio_mask.shape
                    ):
                        batch["ratio_mask"] = ratio_mask * ratio_mask_prev

        # Update cache with current sample for the next step.
        self._prev = current
        return z_mixed, batch, True


class BiomassRegressor(LightningModule):
    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int,
        num_outputs: int = 1,
        dropout: float = 0.0,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        use_output_softplus: bool = True,
        # Optional patch-based main regression path (per-patch prediction then average)
        use_patch_reg3: bool = False,
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
        # Optimizer / SAM configuration
        optimizer_name: Optional[str] = None,
        use_sam: bool = False,
        sam_rho: float = 0.05,
        sam_adaptive: bool = False,
    ) -> None:
        super().__init__()
        # Normalize optimizer hyperparameters before saving them.
        if optimizer_name is None:
            optimizer_name = "adamw"
        opt_name = str(optimizer_name).lower()
        if opt_name in ("sam", "sam_adamw", "adamw_sam"):
            use_sam = True
        optimizer_name = opt_name
        self.save_hyperparameters()

        feature_extractor = build_feature_extractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
            weights_url=weights_url,
            weights_path=weights_path,
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
        self.feature_extractor = feature_extractor

        # Shared bottleneck: MLP defined by hidden_dims; last hidden dim is bottleneck size.
        # Supports a legacy activation-based MLP and a SwiGLU variant.
        hidden_dims: List[int] = list(head_hidden_dims or [512, 256])
        act_name = (head_activation or "").lower()
        # For legacy heads, the backbone feature is CLS concat mean(patch) → 2 * embedding_dim.
        # When use_patch_reg3 is enabled, the main regression path and packed head operate
        # directly on patch-token dimensionality (embedding_dim), and global features are
        # reduced to this size before entering the bottleneck.
        use_patch = bool(use_patch_reg3)

        def _build_bottleneck() -> nn.Sequential:
            layers: List[nn.Module] = []
            if use_patch:
                in_dim = embedding_dim
            else:
                in_dim = embedding_dim * 2
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))

            if act_name == "swiglu":
                # SwiGLU bottleneck: for each hidden_dim we use Linear(in_dim, 2 * hidden_dim)
                # followed by a SwiGLU gate, which halves the dimension back to hidden_dim.
                for hd in hidden_dims:
                    layers.append(nn.Linear(in_dim, hd * 2))
                    layers.append(SwiGLU())
                    if dropout and dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    in_dim = hd
            else:
                # Legacy MLP: Linear + pointwise activation (ReLU/GELU/SiLU).
                def _act():
                    name = act_name
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

            return nn.Sequential(*layers)

        # Default shared bottleneck (used when not using per-layer bottlenecks, and
        # also as a fallback when layer-wise bottlenecks are not constructed).
        self.shared_bottleneck = _build_bottleneck()

        # Task heads
        bottleneck_dim = hidden_dims[-1] if hidden_dims else embedding_dim
        # Main reg3 heads: one or more independent 1-d regressors (e.g., Dry_Total_g only)
        self.num_outputs: int = int(num_outputs)
        if self.num_outputs < 1:
            raise ValueError("num_outputs must be >= 1 for reg3 head")
        self.reg3_heads = nn.ModuleList(
            [nn.Linear(bottleneck_dim, 1) for _ in range(self.num_outputs)]
        )

        # Whether to use per-patch regression (scheme A: only main task uses patch path).
        # When enabled, the main reg3 prediction is obtained by:
        #   1) Extracting CLS and patch tokens from the backbone,
        #   2) Concatenating CLS with each patch token -> (B, N, 2C),
        #   3) Applying shared_bottleneck + reg3_heads per patch,
        #   4) Averaging predictions over patches to obtain a per-image output.
        # Auxiliary tasks (height/NDVI/species/state) and ratio/5D losses continue
        # to use the global CLS + mean(patch) bottleneck.
        self.use_patch_reg3: bool = bool(use_patch_reg3)

        # --- Multi-layer backbone configuration ---
        self.use_layerwise_heads: bool = bool(use_layerwise_heads)
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

        # Optional per-layer bottlenecks for multi-layer heads
        self.use_separate_bottlenecks: bool = bool(use_separate_bottlenecks)
        if self.use_layerwise_heads and self.use_separate_bottlenecks:
            # One bottleneck MLP per selected backbone layer.
            self.layer_bottlenecks = nn.ModuleList(
                [_build_bottleneck() for _ in range(self.num_layers)]
            )
        else:
            # Either multi-layer heads are disabled or we keep using the shared bottleneck.
            self.layer_bottlenecks = None  # type: ignore[assignment]

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
        if self.enable_ratio_head:
            self.ratio_head = nn.Linear(bottleneck_dim, self.num_ratio_outputs)
        else:
            self.ratio_head = None  # type: ignore[assignment]

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
        self._use_reg3_zscore: bool = (self._reg3_mean is not None and self._reg3_std is not None)
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
        if self.mtl_enabled and self.loss_weighting == "uw":
            # Treat main biomass reg3, ratio loss and 5D loss as three separate UW tasks.
            task_names: List[str] = ["reg3"]
            if self.enable_ratio_head:
                task_names.append("ratio")
            if self.enable_5d_loss and self.enable_ratio_head:
                task_names.append("biomass_5d")
            # Keep auxiliary tasks for MTL
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

        # --- Layer-wise heads per backbone layer (optional) ---
        # These heads share the same shared_bottleneck but have independent final linear layers.
        if self.use_layerwise_heads:
            L = self.num_layers
            # Main reg3 heads: [L][num_outputs] scalar heads
            self.layer_reg3_heads = nn.ModuleList(
                nn.ModuleList([nn.Linear(bottleneck_dim, 1) for _ in range(self.num_outputs)])
                for _ in range(L)
            )
            # Ratio heads (if enabled): one per layer
            if self.enable_ratio_head:
                self.layer_ratio_heads = nn.ModuleList(
                    nn.Linear(bottleneck_dim, self.num_ratio_outputs) for _ in range(L)
                )
            else:
                self.layer_ratio_heads = None  # type: ignore[assignment]
            # Auxiliary tasks
            self.layer_height_heads = (
                nn.ModuleList(nn.Linear(bottleneck_dim, 1) for _ in range(L))
                if self.enable_height
                else None  # type: ignore[assignment]
            )
            self.layer_ndvi_heads = (
                nn.ModuleList(nn.Linear(bottleneck_dim, 1) for _ in range(L))
                if self.enable_ndvi
                else None  # type: ignore[assignment]
            )
            self.layer_species_heads = (
                nn.ModuleList(nn.Linear(bottleneck_dim, self.num_species_classes) for _ in range(L))
                if self.enable_species and self.num_species_classes > 0
                else None  # type: ignore[assignment]
            )
            self.layer_state_heads = (
                nn.ModuleList(nn.Linear(bottleneck_dim, self.num_state_classes) for _ in range(L))
                if self.enable_state and self.num_state_classes > 0
                else None  # type: ignore[assignment]
            )
        else:
            self.layer_reg3_heads = None  # type: ignore[assignment]
            self.layer_ratio_heads = None  # type: ignore[assignment]
            self.layer_height_heads = None  # type: ignore[assignment]
            self.layer_ndvi_heads = None  # type: ignore[assignment]
            self.layer_species_heads = None  # type: ignore[assignment]
            self.layer_state_heads = None  # type: ignore[assignment]

    def _forward_reg3_logits_for_heads(self, z: Tensor, heads: List[nn.Linear]) -> Tensor:
        """
        Compute main reg3 prediction in normalized domain (g/m^2 or z-score),
        by aggregating three independent scalar heads into a (B, num_outputs) tensor.
        """
        preds: List[Tensor] = []
        for head in heads:
            preds.append(head(z))
        return torch.cat(preds, dim=-1)

    def _forward_reg3_logits(self, z: Tensor) -> Tensor:
        """
        Convenience wrapper for single set of reg3 heads (no layer-wise structure).
        """
        return self._forward_reg3_logits_for_heads(z, list(self.reg3_heads))

    def _compute_reg3_from_images(
        self,
        images: Optional[Tensor] = None,
        pt_tokens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute main reg3 prediction logits (before optional Softplus) and global bottleneck
        features from input images.

        Returns:
            pred_reg3_logits: (B, num_outputs) in normalized domain (g/m^2 or z-score)
            z_global:         (B, bottleneck_dim) shared bottleneck from CLS+mean(patch)
        """
        if not self.use_patch_reg3:
            # Legacy path: CLS concat mean(patch) -> shared_bottleneck -> reg3_heads
            if images is None:
                raise ValueError("images must be provided when use_patch_reg3 is False")
            features = self.feature_extractor(images)
            z = self.shared_bottleneck(features)
            pred_reg3_logits = self._forward_reg3_logits(z)
            return pred_reg3_logits, z

        # Patch-based path for main regression. When pt_tokens are provided, reuse them
        # directly (e.g., after manifold mixup on backbone patch tokens); otherwise,
        # obtain CLS and patch tokens from the backbone.
        if pt_tokens is None:
            if images is None:
                raise ValueError("Either images or pt_tokens must be provided in patch-mode reg3")
            _, pt = self.feature_extractor.forward_cls_and_tokens(images)
        else:
            pt = pt_tokens
        if pt.dim() != 3:
            raise RuntimeError(f"Unexpected patch tokens shape in patch-mode reg3: {tuple(pt.shape)}")
        B, N, C = pt.shape
        # Global bottleneck uses only the mean patch token (patch-only, C-dim).
        patch_mean = pt.mean(dim=1)  # (B, C)
        z_global = self.shared_bottleneck(patch_mean)  # (B, bottleneck_dim)

        # Build per-patch features for the main regression path: each patch token (C-dim)
        # is fed through the shared bottleneck, and predictions are averaged over patches.
        patch_features_flat = pt.reshape(B * N, C)  # (B*N, C)
        z_patches_flat = self.shared_bottleneck(patch_features_flat)  # (B*N, bottleneck_dim)
        pred_patches_flat = self._forward_reg3_logits(z_patches_flat)  # (B*N, num_outputs)
        pred_patches = pred_patches_flat.view(B, N, self.num_outputs)  # (B, N, num_outputs)
        # Average over patches to obtain per-image logits.
        pred_reg3_logits = pred_patches.mean(dim=1)  # (B, num_outputs)
        return pred_reg3_logits, z_global

    def _compute_reg3_and_z_multilayer(
        self,
        images: Optional[Tensor] = None,
        cls_list: Optional[List[Tensor]] = None,
        pt_list: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Multi-layer main regression path:
          - For each selected backbone layer, build a bottleneck feature z_l,
          - Apply a layer-specific reg3 head on z_l (or per-patch z_l when use_patch_reg3 is enabled),
          - Average predictions and bottleneck features over layers.

        Returns:
            pred_reg3_logits: (B, num_outputs) averaged over layers
            z_global:         (B, bottleneck_dim) averaged bottleneck over layers
            z_layers:         list of (B, bottleneck_dim) per-layer bottlenecks
        """
        if not self.use_layerwise_heads:
            raise RuntimeError(
                "_compute_reg3_and_z_multilayer called but use_layerwise_heads is False"
            )
        if len(self.backbone_layer_indices) == 0:
            raise RuntimeError("backbone_layer_indices is empty in multi-layer path")

        # Obtain CLS and patch tokens for all requested layers in a single backbone forward,
        # unless they were already provided (e.g., after manifold mixup on backbone outputs).
        if cls_list is None or pt_list is None:
            if images is None:
                raise ValueError(
                    "Either images or (cls_list, pt_list) must be provided for multi-layer reg3"
                )
            cls_list, pt_list = self.feature_extractor.forward_layers_cls_and_tokens(
                images, self.backbone_layer_indices
            )
        if len(cls_list) != len(pt_list):
            raise RuntimeError("Mismatch between CLS and patch token lists in multi-layer path")

        z_layers: List[Tensor] = []
        pred_layers: List[Tensor] = []

        if not self.use_patch_reg3:
            # CLS + mean(patch) per layer
            for layer_idx, (cls, pt) in enumerate(zip(cls_list, pt_list)):
                if pt.dim() != 3:
                    raise RuntimeError(f"Unexpected patch tokens shape in multi-layer reg3: {tuple(pt.shape)}")
                patch_mean = pt.mean(dim=1)  # (B, C)
                feats = torch.cat([cls, patch_mean], dim=-1)  # (B, 2C)
                # Select per-layer bottleneck if available; otherwise fall back to shared one.
                if self.layer_bottlenecks is not None:
                    bottleneck = self.layer_bottlenecks[layer_idx]
                else:
                    bottleneck = self.shared_bottleneck
                z_l = bottleneck(feats)  # (B, bottleneck_dim)
                z_layers.append(z_l)
                # Select layer-specific reg3 heads if available; otherwise fall back to shared heads.
                if self.layer_reg3_heads is not None:
                    heads_l = list(self.layer_reg3_heads[layer_idx])
                else:
                    heads_l = list(self.reg3_heads)
                pred_l = self._forward_reg3_logits_for_heads(z_l, heads_l)  # (B, num_outputs)
                pred_layers.append(pred_l)
        else:
            # Patch-based reg3 per layer: per-patch predictions then averaged, then averaged over layers.
            for layer_idx, (cls, pt) in enumerate(zip(cls_list, pt_list)):
                if pt.dim() != 3:
                    raise RuntimeError(f"Unexpected patch tokens shape in multi-layer patch-mode reg3: {tuple(pt.shape)}")
                B, N, C = pt.shape
                patch_mean = pt.mean(dim=1)  # (B, C)
                # Select per-layer bottleneck if available; otherwise fall back to shared one.
                if self.layer_bottlenecks is not None:
                    bottleneck = self.layer_bottlenecks[layer_idx]
                else:
                    bottleneck = self.shared_bottleneck
                z_global_l = bottleneck(patch_mean)  # (B, bottleneck_dim)
                z_layers.append(z_global_l)

                patch_features_flat = pt.reshape(B * N, C)
                z_patches_flat = bottleneck(patch_features_flat)  # (B*N, bottleneck_dim)
                if self.layer_reg3_heads is not None:
                    heads_l = list(self.layer_reg3_heads[layer_idx])
                else:
                    heads_l = list(self.reg3_heads)
                pred_patches_flat = self._forward_reg3_logits_for_heads(z_patches_flat, heads_l)  # (B*N, num_outputs)
                pred_patches = pred_patches_flat.view(B, N, self.num_outputs)
                pred_l = pred_patches.mean(dim=1)  # (B, num_outputs)
                pred_layers.append(pred_l)

        pred_reg3_logits = average_layerwise_predictions(pred_layers)
        z_global = average_layerwise_predictions(z_layers)
        return pred_reg3_logits, z_global, z_layers

    def forward(self, images: Tensor) -> Tensor:
        # Return main regression prediction in original grams (g).
        # When use_patch_reg3 is enabled, this corresponds to the per-patch prediction
        # averaged over all patches; otherwise it is the legacy CLS+mean(patch) head.
        if self.use_layerwise_heads:
            pred_reg3_logits, _, _ = self._compute_reg3_and_z_multilayer(images)
        else:
            pred_reg3_logits, _ = self._compute_reg3_from_images(images)
        out = pred_reg3_logits
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
        is_ndvi_only: bool = bool(batch.get("ndvi_only", False))
        # Decide which augmentation (CutMix / manifold mixup) to apply for this batch.
        use_cutmix = False
        use_mixup = False
        if stage == "train" and (not is_ndvi_only):
            # CutMix gating
            cutmix_enabled = self._cutmix_main is not None
            cutmix_prob = 0.0
            if cutmix_enabled:
                try:
                    cutmix_enabled = bool(getattr(self._cutmix_main, "cfg", None) and self._cutmix_main.cfg.enabled)
                    cutmix_prob = float(self._cutmix_main.cfg.prob)
                except Exception:
                    cutmix_enabled = False
                    cutmix_prob = 0.0
            # Manifold mixup gating
            mixup_enabled = self._manifold_mixup is not None and bool(self._manifold_mixup.enabled)
            mixup_prob = 0.0
            if mixup_enabled:
                try:
                    mixup_prob = float(self._manifold_mixup.prob)
                except Exception:
                    mixup_prob = 0.0
            cut_trigger = False
            mix_trigger = False
            if cutmix_enabled and cutmix_prob > 0.0:
                try:
                    cut_trigger = bool(torch.rand(()) < cutmix_prob)
                except Exception:
                    cut_trigger = False
            if mixup_enabled and mixup_prob > 0.0:
                try:
                    mix_trigger = bool(torch.rand(()) < mixup_prob)
                except Exception:
                    mix_trigger = False
            if cut_trigger and mix_trigger:
                # Both selected by their own probabilities: choose exactly one with equal chance.
                try:
                    choose_cut = bool(torch.rand(()) < 0.5)
                except Exception:
                    choose_cut = True
                if choose_cut:
                    use_cutmix, use_mixup = True, False
                else:
                    use_cutmix, use_mixup = False, True
            elif cut_trigger:
                use_cutmix = True
            elif mix_trigger:
                use_mixup = True

        images: Tensor = batch["image"]
        # Apply CutMix on images + scalar/dense labels before backbone if chosen
        if use_cutmix and self._cutmix_main is not None:
            try:
                batch, _ = self._cutmix_main.apply_main_batch(batch, force=True)  # type: ignore[assignment]
                images = batch["image"]
            except Exception:
                pass

        # Main regression path (reg3). When use_patch_reg3 is enabled, this returns
        # per-patch predictions averaged over patches together with a global bottleneck
        # feature built from CLS + mean(patch). Auxiliary heads and ratio/5D always
        # use the global bottleneck z. When use_layerwise_heads is enabled, both
        # reg3 and z are obtained by averaging per-layer predictions/features.
        pred_reg3: Tensor
        z: Tensor
        z_layers: Optional[List[Tensor]] = None
        if self.use_layerwise_heads:
            # Multi-layer path: obtain per-layer CLS and patch tokens from the backbone,
            # optionally apply manifold mixup on the DINO patch tokens, then build
            # bottleneck features and heads on top.
            cls_list, pt_list = self.feature_extractor.forward_layers_cls_and_tokens(
                images, self.backbone_layer_indices
            )
            if (
                use_mixup
                and self._manifold_mixup is not None
                and (stage == "train")
                and (not is_ndvi_only)
            ):
                try:
                    # Stack per-layer patch tokens into a single tensor of shape (B, L, N, C)
                    # so that manifold mixup operates on DINO outputs along the batch dim.
                    pt_stack = torch.stack(pt_list, dim=1)
                    pt_stack, batch, _ = self._manifold_mixup.apply(pt_stack, batch, force=True)
                    # Unstack back into a list of (B, N, C) tensors per layer.
                    pt_list = [pt_stack[:, i] for i in range(pt_stack.shape[1])]
                except Exception:
                    pass
            pred_reg3, z, z_layers = self._compute_reg3_and_z_multilayer(
                images=None, cls_list=cls_list, pt_list=pt_list
            )
        else:
            if self.use_patch_reg3:
                # Patch-based main regression path: obtain backbone patch tokens, apply
                # manifold mixup on these DINO features, then compute bottleneck features.
                _, pt_tokens = self.feature_extractor.forward_cls_and_tokens(images)
                if (
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                ):
                    try:
                        pt_tokens, batch, _ = self._manifold_mixup.apply(
                            pt_tokens, batch, force=True
                        )
                    except Exception:
                        pass
                pred_reg3, z = self._compute_reg3_from_images(images=None, pt_tokens=pt_tokens)
            else:
                # Legacy CLS+mean(patch) path: apply manifold mixup directly on the
                # DINO global features (CLS concat mean(patch)) before the bottleneck.
                features = self.feature_extractor(images)
                if (
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                ):
                    try:
                        features, batch, _ = self._manifold_mixup.apply(
                            features, batch, force=True
                        )
                    except Exception:
                        pass
                z = self.shared_bottleneck(features)
                pred_reg3 = self._forward_reg3_logits(z)
        if is_ndvi_only:
            # NDVI-only batch (no reg3 supervision). Optimize NDVI scalar head only.
            if not self.enable_ndvi or (self.ndvi_head is None and (not self.use_layerwise_heads or self.layer_ndvi_heads is None)):
                # If NDVI task is disabled, skip by returning zero loss
                zero = (z.sum() * 0.0)
                self.log(f"{stage}_loss_ndvi", zero, on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"{stage}_loss", zero, on_step=False, on_epoch=True, prog_bar=True)
                return {"loss": zero}
            y_ndvi_only: Tensor = batch["y_ndvi"]  # (B,1)
            if self.use_layerwise_heads and self.layer_ndvi_heads is not None and z_layers is not None:
                preds_layers_ndvi: List[Tensor] = []
                for idx, head in enumerate(self.layer_ndvi_heads):
                    preds_layers_ndvi.append(head(z_layers[idx]))
                pred_ndvi_only = average_layerwise_predictions(preds_layers_ndvi)
            else:
                pred_ndvi_only = self.ndvi_head(z)  # type: ignore[operator]
            loss_ndvi_only = F.mse_loss(pred_ndvi_only, y_ndvi_only)
            self.log(f"{stage}_loss_ndvi", loss_ndvi_only, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse_ndvi", loss_ndvi_only, on_step=False, on_epoch=True, prog_bar=False)
            mae_ndvi_only = F.l1_loss(pred_ndvi_only, y_ndvi_only)
            self.log(f"{stage}_mae_ndvi", mae_ndvi_only, on_step=False, on_epoch=True, prog_bar=False)
            total_ndvi = self._uw_sum([("ndvi", loss_ndvi_only)])
            self.log(f"{stage}_loss", total_ndvi, on_step=False, on_epoch=True, prog_bar=True)
            return {"loss": total_ndvi}

        # y_reg3 provided by dataset is already in normalized domain (z-score on g/m^2 when enabled)
        y_reg3: Tensor = batch["y_reg3"]  # (B, num_outputs)
        reg3_mask: Optional[Tensor] = batch.get("reg3_mask", None)
        y_height: Tensor = batch["y_height"]  # (B,1)
        y_ndvi: Tensor = batch["y_ndvi"]  # (B,1)
        y_species: Tensor = batch["y_species"]  # (B,)
        y_state: Tensor = batch["y_state"]  # (B,)

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
        # Base MSE on main reg3 outputs (e.g., Dry_Total_g).
        loss_reg3_mse = diff2_reg3.sum() / mask_sum_reg3

        # Always log base reg3 metrics (independent of MTL / auxiliary tasks)
        self.log(f"{stage}_loss_reg3_mse", loss_reg3_mse, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse_reg3", loss_reg3_mse, on_step=False, on_epoch=True, prog_bar=False)
        mae_reg3 = (diff_reg3.abs() * mask).sum() / mask_sum_reg3
        self.log(f"{stage}_mae_reg3", mae_reg3, on_step=False, on_epoch=True, prog_bar=False)
        with torch.no_grad():
            per_dim_den = mask.sum(dim=0).clamp_min(1.0)
            per_dim_mse = diff2_reg3.sum(dim=0) / per_dim_den
            for i in range(per_dim_mse.shape[0]):
                self.log(f"{stage}_mse_reg3_{i}", per_dim_mse[i], on_step=False, on_epoch=True, prog_bar=False)

        # --- Ratio MSE loss (CSIRO only; masked via ratio_mask) ---
        loss_ratio_mse: Optional[Tensor] = None
        if self.enable_ratio_head:
            y_ratio: Optional[Tensor] = batch.get("y_ratio", None)  # (B,3)
            ratio_mask: Optional[Tensor] = batch.get("ratio_mask", None)  # (B,1)
            if y_ratio is not None and ratio_mask is not None:
                # Predict logits for (Dry_Clover_g, Dry_Dead_g, Dry_Green_g)
                if self.use_layerwise_heads and self.layer_ratio_heads is not None and z_layers is not None:
                    logits_per_layer: List[Tensor] = []
                    for idx, head in enumerate(self.layer_ratio_heads):
                        logits_per_layer.append(head(z_layers[idx]))
                    ratio_logits = average_layerwise_predictions(logits_per_layer)
                else:
                    ratio_logits = self.ratio_head(z)  # type: ignore[operator]
                # Predicted probabilities
                p_pred = F.softmax(ratio_logits, dim=-1)
                # Ensure target is a proper distribution
                p_true = y_ratio.clamp_min(0.0)
                denom = p_true.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                p_true = p_true / denom
                # Per-sample MSE over the 3 ratio components, then masked average
                diff_ratio = p_pred - p_true
                mse_per_sample = (diff_ratio * diff_ratio).sum(dim=-1, keepdim=True)  # (B,1)
                m = ratio_mask.to(device=mse_per_sample.device, dtype=mse_per_sample.dtype)
                num = (mse_per_sample * m).sum()
                den = m.sum().clamp_min(1.0)
                loss_ratio_mse = (num / den)
                self.log(f"{stage}_loss_ratio_mse", loss_ratio_mse, on_step=False, on_epoch=True, prog_bar=False)

        # --- 5D weighted MSE loss over physical components ---
        loss_5d: Optional[Tensor] = None
        y_5d_g: Optional[Tensor] = batch.get("y_biomass_5d_g", None)  # (B,5) grams
        mask_5d: Optional[Tensor] = batch.get("biomass_5d_mask", None)  # (B,5)
        # For metrics, prefer canonical 5D predictions/targets when available.
        metrics_preds_5d: Optional[Tensor] = None
        metrics_targets_5d: Optional[Tensor] = None
        if self.enable_5d_loss and y_5d_g is not None and mask_5d is not None and self.enable_ratio_head:
            # Convert main reg3 prediction back to g/m^2 (Dry_Total_g)
            pred_total_gm2 = self._invert_reg3_to_g_per_m2(pred_reg3)  # (B,1)
            # Ratio predictions (probabilities over 3 components)
            if self.use_layerwise_heads and self.layer_ratio_heads is not None and z_layers is not None:
                logits_per_layer_5d: List[Tensor] = []
                for idx, head in enumerate(self.layer_ratio_heads):
                    logits_per_layer_5d.append(head(z_layers[idx]))
                ratio_logits = average_layerwise_predictions(logits_per_layer_5d)
            else:
                ratio_logits = self.ratio_head(z)  # type: ignore[operator]
            p_pred = F.softmax(ratio_logits, dim=-1)  # (B,3)
            # Component g/m^2
            comp_gm2 = p_pred * pred_total_gm2  # (B,3)
            clover_pred = comp_gm2[:, 0]
            dead_pred = comp_gm2[:, 1]
            green_pred = comp_gm2[:, 2]
            gdm_pred = clover_pred + green_pred
            total_pred = pred_total_gm2.squeeze(-1)
            pred_5d_gm2 = torch.stack([clover_pred, dead_pred, green_pred, gdm_pred, total_pred], dim=-1)

            # For external metrics, work in grams in the fixed 5D order.
            metrics_preds_5d = pred_5d_gm2 * float(self._area_m2)
            metrics_targets_5d = y_5d_g.to(device=metrics_preds_5d.device, dtype=metrics_preds_5d.dtype)

            # Convert targets to g/m^2
            y_5d_gm2 = y_5d_g.to(device=pred_5d_gm2.device, dtype=pred_5d_gm2.dtype) / float(
                self._area_m2
            )

            # Optionally move to z-scored space using precomputed 5D stats
            pred_5d_input = pred_5d_gm2
            target_5d_input = y_5d_gm2

            if self.log_scale_targets:
                pred_5d_input = torch.log1p(torch.clamp(pred_5d_input, min=0.0))
                target_5d_input = torch.log1p(torch.clamp(target_5d_input, min=0.0))

            if self._use_biomass_5d_zscore:
                mean_5d = self._biomass_5d_mean.to(device=pred_5d_gm2.device, dtype=pred_5d_gm2.dtype)  # type: ignore[union-attr]
                std_5d = torch.clamp(
                    self._biomass_5d_std.to(device=pred_5d_gm2.device, dtype=pred_5d_gm2.dtype),  # type: ignore[union-attr]
                    min=1e-8,
                )
                pred_5d = (pred_5d_input - mean_5d) / std_5d
                target_5d = (target_5d_input - mean_5d) / std_5d
            else:
                pred_5d = pred_5d_input
                target_5d = target_5d_input

            # Weighted MSE across components with masks
            w = self.biomass_5d_weights.to(device=pred_5d.device, dtype=pred_5d.dtype)  # (5,)
            m5 = mask_5d.to(device=pred_5d.device, dtype=pred_5d.dtype)
            diff_5d = (pred_5d - target_5d) * m5
            diff2_5d = diff_5d * diff_5d
            per_dim_den = m5.sum(dim=0).clamp_min(1.0)
            mse_per_dim = diff2_5d.sum(dim=0) / per_dim_den  # (5,)
            valid_weight = w * (per_dim_den > 0).to(dtype=w.dtype)
            total_w = valid_weight.sum().clamp_min(1e-8)
            loss_5d = (w * mse_per_dim).sum() / total_w

            # Log per-component MSE and aggregated loss
            names_5d = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g", "Dry_Total_g"]
            for i, name in enumerate(names_5d):
                self.log(f"{stage}_mse_5d_{name}", mse_per_dim[i], on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_loss_5d_weighted", loss_5d, on_step=False, on_epoch=True, prog_bar=False)

        # Aggregate reg3-related losses for logging (independent of UW task structure)
        loss_reg3_total = loss_reg3_mse
        if loss_ratio_mse is not None:
            loss_reg3_total = loss_reg3_total + loss_ratio_mse
        if loss_5d is not None:
            loss_reg3_total = loss_reg3_total + loss_5d
        self.log(f"{stage}_loss_reg3", loss_reg3_total, on_step=False, on_epoch=True, prog_bar=False)

        # If MTL is disabled or all auxiliary tasks are off, optimize only the reg3 path
        if (not self.mtl_enabled) or (
            self.enable_height is False
            and self.enable_ndvi is False
            and self.enable_species is False
            and self.enable_state is False
        ):
            total_loss = loss_reg3_total
            self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
            # Overall mae/mse are computed on normalized reg3 space for backward compatibility
            self.log(f"{stage}_mae", mae_reg3, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse", loss_reg3_mse, on_step=False, on_epoch=True, prog_bar=False)
            # For external metrics (e.g., epoch-end R^2), prefer 5D grams when available.
            if metrics_preds_5d is not None and metrics_targets_5d is not None:
                preds_out = metrics_preds_5d.detach()
                targets_out = metrics_targets_5d.detach()
            else:
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
                "mse": loss_reg3_mse,
                "preds": preds_out,
                "targets": targets_out,
            }

        # Otherwise, compute enabled auxiliary task heads and losses
        pred_height = None
        pred_ndvi = None
        logits_species = None
        logits_state = None
        if self.use_layerwise_heads and z_layers is not None:
            if self.enable_height and self.layer_height_heads is not None:
                height_preds_layers: List[Tensor] = []
                for idx, head in enumerate(self.layer_height_heads):
                    height_preds_layers.append(head(z_layers[idx]))
                pred_height = average_layerwise_predictions(height_preds_layers)
            if self.enable_ndvi and self.layer_ndvi_heads is not None:
                ndvi_preds_layers: List[Tensor] = []
                for idx, head in enumerate(self.layer_ndvi_heads):
                    ndvi_preds_layers.append(head(z_layers[idx]))
                pred_ndvi = average_layerwise_predictions(ndvi_preds_layers)
            if self.enable_species and self.layer_species_heads is not None:
                species_logits_layers: List[Tensor] = []
                for idx, head in enumerate(self.layer_species_heads):
                    species_logits_layers.append(head(z_layers[idx]))
                logits_species = average_layerwise_predictions(species_logits_layers)
            if self.enable_state and self.layer_state_heads is not None:
                state_logits_layers: List[Tensor] = []
                for idx, head in enumerate(self.layer_state_heads):
                    state_logits_layers.append(head(z_layers[idx]))
                logits_state = average_layerwise_predictions(state_logits_layers)
        else:
            pred_height = self.height_head(z) if self.enable_height else None  # type: ignore[assignment]
            pred_ndvi = self.ndvi_head(z) if self.enable_ndvi else None  # type: ignore[assignment]
            logits_species = self.species_head(z) if self.enable_species else None  # type: ignore[assignment]
            logits_state = self.state_head(z) if self.enable_state else None  # type: ignore[assignment]

        # Collect losses in consistent order for UW/equal weighting
        named_losses: List[Tuple[str, Tensor]] = []
        # Treat reg3, ratio and 5D as separate UW tasks when enabled.
        named_losses.append(("reg3", loss_reg3_mse))
        if loss_ratio_mse is not None:
            named_losses.append(("ratio", loss_ratio_mse))
        if loss_5d is not None:
            named_losses.append(("biomass_5d", loss_5d))

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

            if mask_sum_ndvi > 0:
                loss_ndvi = diff2_ndvi.sum() / mask_sum_ndvi
                mae_ndvi = (diff_ndvi.abs() * m_nd).sum() / mask_sum_ndvi
            else:
                # No NDVI supervision in this batch (e.g., Irish-only). Use zero loss.
                zero_nd = diff2_ndvi.sum() * 0.0
                loss_ndvi = zero_nd
                mae_ndvi = zero_nd
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

        # overall metrics for backward-compat (on normalized reg3 space)
        mae = F.l1_loss(pred_reg3, y_reg3)
        mse = F.mse_loss(pred_reg3, y_reg3)
        self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse", mse, on_step=False, on_epoch=True, prog_bar=False)
        # For external validation epoch R^2, prefer 5D grams when available.
        if metrics_preds_5d is not None and metrics_targets_5d is not None:
            preds_out = metrics_preds_5d.detach()
            targets_out = metrics_targets_5d.detach()
        else:
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

        # SAM requires a two-step update with an extra forward-backward pass.
        if isinstance(optimizer, SAM):
            optimizer.first_step(zero_grad=True)
            optimizer_closure()
            optimizer.second_step(zero_grad=True)
        else:
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
        preds = torch.cat(self._val_preds, dim=0)   # grams, shape (N, D)
        targets = torch.cat(self._val_targets, dim=0)  # grams, shape (N, D)

        # R^2 in log space with baseline mean defined over the training
        # distribution when available, otherwise over the current val subset.
        eps = 1e-8
        preds_clamp = preds.clamp_min(0.0)
        targets_clamp = targets.clamp_min(0.0)
        preds_log = torch.log1p(preds_clamp)
        targets_log = torch.log1p(targets_clamp)

        # Baseline mean per dimension in log-space
        if preds_log.shape[1] == 5 and self._biomass_5d_mean is not None:
            # Use global 5D means in g/m^2, converted to grams.
            try:
                mean_gm2 = self._biomass_5d_mean.to(  # type: ignore[union-attr]
                    device=targets_log.device, dtype=targets_log.dtype
                )
                mean_g = mean_gm2 * float(self._area_m2)
                mean_log = torch.log1p(mean_g.clamp_min(0.0))
            except Exception:
                mean_log = torch.mean(targets_log, dim=0)
        elif self._reg3_mean is not None:
            try:
                mean_gm2 = self._reg3_mean.to(device=targets_log.device, dtype=targets_log.dtype)  # type: ignore[union-attr]
                mean_g = mean_gm2 * float(self._area_m2)
                mean_log = torch.log1p(mean_g.clamp_min(0.0))
            except Exception:
                mean_log = torch.mean(targets_log, dim=0)
        else:
            mean_log = torch.mean(targets_log, dim=0)

        # Per-dimension R^2 in log space
        ss_res_per = torch.sum((targets_log - preds_log) ** 2, dim=0)
        ss_tot_per = torch.sum((targets_log - mean_log) ** 2, dim=0)
        r2_per = 1.0 - (ss_res_per / (ss_tot_per + eps))

        # Aggregate across 5D with DESCRIPTION weights when applicable
        if preds_log.shape[1] == 5:
            weights = torch.tensor(
                [0.1, 0.1, 0.1, 0.2, 0.5],
                device=r2_per.device,
                dtype=r2_per.dtype,
            )
            valid = torch.isfinite(r2_per)
            w_eff = weights * valid.to(dtype=weights.dtype)
            denom = w_eff.sum().clamp_min(eps)
            r2 = (w_eff * r2_per).sum() / denom
        else:
            # Fallback: mean R^2 across dimensions (for non-5D configs)
            r2 = torch.mean(r2_per)

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

        # Optimizer selection: plain AdamW or SAM-wrapped AdamW.
        opt_name = str(getattr(self.hparams, "optimizer_name", "adamw")).lower()
        use_sam_flag = bool(getattr(self.hparams, "use_sam", False))
        if opt_name in ("sam", "sam_adamw", "adamw_sam"):
            use_sam_flag = True

        if use_sam_flag:
            sam_rho = float(getattr(self.hparams, "sam_rho", 0.05))
            sam_adaptive = bool(getattr(self.hparams, "sam_adaptive", False))
            optimizer: Optimizer = SAM(param_groups, AdamW, rho=sam_rho, adaptive=sam_adaptive)
        else:
            optimizer = AdamW(param_groups)

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


