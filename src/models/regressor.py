from typing import Any, Dict, Optional, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from lightning.pytorch import LightningModule
from loguru import logger

from .backbone import build_feature_extractor
from .peft_integration import inject_lora_into_feature_extractor, get_lora_param_list
from .head_builder import build_head_layer


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
        pretrained: bool = True,
        weights_url: Optional[str] = None,
        weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_name: Optional[str] = None,
        max_epochs: Optional[int] = None,
        loss_weighting: Optional[str] = None,
        num_species_classes: Optional[int] = None,
        num_state_classes: Optional[int] = None,
        peft_cfg: Optional[Dict[str, Any]] = None,
        mtl_enabled: bool = True,
        enable_height: bool = False,
        enable_ndvi: bool = False,
        enable_species: bool = False,
        enable_state: bool = False,
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

        # Shared bottleneck: MLP defined by hidden_dims; last hidden dim is bottleneck size
        hidden_dims: List[int] = list(head_hidden_dims or [512, 256])
        act_name = head_activation
        layers: List[nn.Module] = []
        in_dim = embedding_dim
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
        self.reg_head = nn.Linear(bottleneck_dim, num_outputs)  # main 3-d regression

        self.mtl_enabled: bool = bool(mtl_enabled)
        # Per-task toggles (gated by global MTL flag)
        self.enable_height = bool(enable_height) and self.mtl_enabled
        self.enable_ndvi = bool(enable_ndvi) and self.mtl_enabled
        self.enable_species = bool(enable_species) and self.mtl_enabled
        self.enable_state = bool(enable_state) and self.mtl_enabled

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
        self.out_softplus = nn.Softplus() if use_output_softplus else None

        # Uncertainty Weighting (UW) parameters (optional)
        self.loss_weighting: Optional[str] = (loss_weighting.lower() if loss_weighting else None)
        # Determine active tasks for UW/equal weighting (always include reg3 if any aux enabled)
        active_aux: List[str] = []
        if self.enable_height:
            active_aux.append("height")
        if self.enable_ndvi:
            active_aux.append("ndvi")
        if self.enable_species:
            active_aux.append("species")
        if self.enable_state:
            active_aux.append("state")
        self._uw_task_names: List[str] = ["reg3", *active_aux] if len(active_aux) > 0 else ["reg3"]
        if self.mtl_enabled and len(active_aux) > 0 and self.loss_weighting == "uw":
            self.uw_logvars = nn.Parameter(torch.zeros(len(self._uw_task_names)))
        else:
            self.uw_logvars = None

        # buffers for epoch-wise metrics on validation set
        self._val_preds: list[Tensor] = []
        self._val_targets: list[Tensor] = []

    def forward(self, images: Tensor) -> Tensor:
        # Return only main 3-d regression prediction (for compatibility and inference)
        features = self.feature_extractor(images)
        z = self.shared_bottleneck(features)
        out = self.reg_head(z)
        if self.out_softplus is not None:
            out = self.out_softplus(out)
        return out

    def _shared_step(self, batch: Any, stage: str) -> Dict[str, Tensor]:
        # batch is a dict from the dataset
        images: Tensor = batch["image"]
        y_reg3: Tensor = batch["y_reg3"]  # (B,3)
        y_height: Tensor = batch["y_height"]  # (B,1)
        y_ndvi: Tensor = batch["y_ndvi"]  # (B,1)
        y_species: Tensor = batch["y_species"]  # (B,)
        y_state: Tensor = batch["y_state"]  # (B,)

        features = self.feature_extractor(images)
        z = self.shared_bottleneck(features)
        pred_reg3 = self.reg_head(z)
        if self.out_softplus is not None:
            pred_reg3 = self.out_softplus(pred_reg3)

        # If MTL is disabled, optimize only the main regression task
        loss_reg3 = F.mse_loss(pred_reg3, y_reg3)
        if (not self.mtl_enabled) or (self.enable_height is False and self.enable_ndvi is False and self.enable_species is False and self.enable_state is False):
            self.log(f"{stage}_loss_reg3", loss_reg3, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse_reg3", loss_reg3, on_step=False, on_epoch=True, prog_bar=False)
            mae_reg3 = F.l1_loss(pred_reg3, y_reg3)
            self.log(f"{stage}_mae_reg3", mae_reg3, on_step=False, on_epoch=True, prog_bar=False)
            with torch.no_grad():
                per_dim_mse = torch.mean((pred_reg3 - y_reg3) ** 2, dim=0)
                for i in range(per_dim_mse.shape[0]):
                    self.log(f"{stage}_mse_reg3_{i}", per_dim_mse[i], on_step=False, on_epoch=True, prog_bar=False)

            total_loss = loss_reg3
            self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_mae", mae_reg3, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse", loss_reg3, on_step=False, on_epoch=True, prog_bar=False)
            return {
                "loss": total_loss,
                "mae": mae_reg3,
                "mse": loss_reg3,
                "preds": pred_reg3,
                "targets": y_reg3,
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
        losses: List[Tensor] = [loss_reg3]
        mae_reg3 = F.l1_loss(pred_reg3, y_reg3)
        self.log(f"{stage}_mae_reg3", mae_reg3, on_step=False, on_epoch=True, prog_bar=False)
        with torch.no_grad():
            per_dim_mse = torch.mean((pred_reg3 - y_reg3) ** 2, dim=0)
            for i in range(per_dim_mse.shape[0]):
                self.log(f"{stage}_mse_reg3_{i}", per_dim_mse[i], on_step=False, on_epoch=True, prog_bar=False)

        if self.enable_height:
            loss_height = F.mse_loss(pred_height, y_height)  # type: ignore[arg-type]
            self.log(f"{stage}_loss_height", loss_height, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse_height", loss_height, on_step=False, on_epoch=True, prog_bar=False)
            mae_height = F.l1_loss(pred_height, y_height)  # type: ignore[arg-type]
            self.log(f"{stage}_mae_height", mae_height, on_step=False, on_epoch=True, prog_bar=False)
            losses.append(loss_height)
        if self.enable_ndvi:
            loss_ndvi = F.mse_loss(pred_ndvi, y_ndvi)  # type: ignore[arg-type]
            self.log(f"{stage}_loss_ndvi", loss_ndvi, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse_ndvi", loss_ndvi, on_step=False, on_epoch=True, prog_bar=False)
            mae_ndvi = F.l1_loss(pred_ndvi, y_ndvi)  # type: ignore[arg-type]
            self.log(f"{stage}_mae_ndvi", mae_ndvi, on_step=False, on_epoch=True, prog_bar=False)
            losses.append(loss_ndvi)
        if self.enable_species and logits_species is not None:
            loss_species = F.cross_entropy(logits_species, y_species)
            self.log(f"{stage}_loss_species", loss_species, on_step=False, on_epoch=True, prog_bar=False)
            with torch.no_grad():
                acc_species = (logits_species.argmax(dim=-1) == y_species).float().mean()
            self.log(f"{stage}_acc_species", acc_species, on_step=False, on_epoch=True, prog_bar=False)
            losses.append(loss_species)
        if self.enable_state and logits_state is not None:
            loss_state = F.cross_entropy(logits_state, y_state)
            self.log(f"{stage}_loss_state", loss_state, on_step=False, on_epoch=True, prog_bar=False)
            with torch.no_grad():
                acc_state = (logits_state.argmax(dim=-1) == y_state).float().mean()
            self.log(f"{stage}_acc_state", acc_state, on_step=False, on_epoch=True, prog_bar=False)
            losses.append(loss_state)

        # UW over active tasks or equal weighting
        if self.mtl_enabled and len(losses) > 1 and self.loss_weighting == "uw":
            s = self.uw_logvars  # type: ignore[operator]
            losses_vec = torch.stack(losses)
            total_loss = 0.5 * torch.sum(torch.exp(-s) * losses_vec + s)
            with torch.no_grad():
                w_eff = 0.5 * torch.exp(-s)
                sigma = torch.exp(0.5 * s)
                for i, name in enumerate(self._uw_task_names):
                    self.log(f"{stage}_uw_w_{name}", w_eff[i], on_step=False, on_epoch=True, prog_bar=False)
                    self.log(f"{stage}_uw_logvar_{name}", s[i], on_step=False, on_epoch=True, prog_bar=False)
                    self.log(f"{stage}_uw_sigma_{name}", sigma[i], on_step=False, on_epoch=True, prog_bar=False)
        else:
            total_loss = torch.stack(losses).mean()

        # overall metrics for backward-compat
        mae = F.l1_loss(pred_reg3, y_reg3)
        mse = F.mse_loss(pred_reg3, y_reg3)
        self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse", mse, on_step=False, on_epoch=True, prog_bar=False)
        return {
            "loss": total_loss,
            "mae": mae,
            "mse": mse,
            "preds": pred_reg3,
            "targets": y_reg3,
        }

    def training_step(self, batch: Any, batch_idx: int):
        return self._shared_step(batch, stage="train")["loss"]

    def validation_step(self, batch: Any, batch_idx: int):
        out = self._shared_step(batch, stage="val")
        self._val_preds.append(out["preds"].detach().float().cpu())
        self._val_targets.append(out["targets"].detach().float().cpu())

    def on_validation_epoch_start(self) -> None:
        self._val_preds.clear()
        self._val_targets.clear()

    def on_validation_epoch_end(self) -> None:
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

    def configure_optimizers(self):
        # Separate parameter groups: LoRA adapters (smaller LR) and the rest (head, optionally others)
        lora_params: List[torch.nn.Parameter] = []
        try:
            lora_params = get_lora_param_list(self.feature_extractor.backbone)
        except Exception:
            lora_params = []

        all_params = [p for p in self.parameters() if p.requires_grad]
        lora_set = set(lora_params)
        other_params = [p for p in all_params if p not in lora_set]

        param_groups: List[Dict[str, Any]] = []
        if len(other_params) > 0:
            param_groups.append({
                "params": other_params,
                "lr": self.hparams.learning_rate,
                "weight_decay": self.hparams.weight_decay,
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
            max_epochs = self.hparams.max_epochs or 10
            scheduler: _LRScheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
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


